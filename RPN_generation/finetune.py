import torch
import math
import json
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup

from torch.utils.tensorboard import SummaryWriter

import time
from pathlib import Path


from finetune_utils import (
    stream_jsonl, find_last_checkpoint, resume_training, save_checkpoint,
    load_model_tokenizer, prepare_encodings, find_micro_batch_size,
    build_optimizer_and_scheduler, log_jsonl, reconcile_metrics, evaluate,
)



def finetune(
    model_name: str,
    train_data: str,
    train_data_size: int,
    eval_data: str,
    max_length: int = 256,
    output_dir: str = "/scratch/ctisseau/finetuned-models",
    epochs: int = 3,
    batch_size: int = 32,
    lr: float = 5e-5,
    checkpoint_steps=32,
    device: str = None,
):
    # Setup device
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset
    print("Loading dataset")
    train_dataset = list(stream_jsonl(train_data))
    if train_data_size is not None: train_dataset = train_dataset[:train_data_size]
    eval_dataset = list(stream_jsonl(eval_data))
    print("Dataset loaded")

    max_memory3 = {
        0: "7GB",
        1: "7GB",
        2: "7GB"
    }
    
    # Load Model and Tokenizer
    model, tokenizer = load_model_tokenizer(output_dir, model_name, max_memory3)
    model.config.use_cache = False          # Disable KV-cache, which is useless during training
    first_device = next(model.parameters()).device

    # Prepare input_ids, attention_mask, labels for training
    input_ids, attention_mask, labels = prepare_encodings(tokenizer, train_dataset)
    eval_input_ids, eval_attention_mask, eval_labels = prepare_encodings(tokenizer, eval_dataset)

    # Find the perfect micro_batch_size
    micro_batch_size = find_micro_batch_size(model, max_length)
    torch.cuda.empty_cache()    # Is this necessary here? 

    if batch_size % micro_batch_size != 0:
        raise ValueError(
                f"The micro_batch_size ({micro_batch_size}) should be a divisor of the batch_size ({batch_size}). "
                f"batch_size % micro_batch_size = {batch_size % micro_batch_size}"
            )    
    accum_steps = batch_size // micro_batch_size        # The number of micro_batch to process to obtain a real batch

    # DataLoader
    train_dataset = TensorDataset(input_ids, attention_mask, labels)
    dataloader = DataLoader(train_dataset, batch_size=micro_batch_size, shuffle=False)  # We need shuffle = False if we want to track the input of each sentence
                                                                                        # By default drop_last=False, meaning the last batch is smaller than the other ones
    eval_dataset = TensorDataset(eval_input_ids, eval_attention_mask, eval_labels)
    eval_dataloader = DataLoader(eval_dataset, batch_size=micro_batch_size, shuffle=False)
    eval_every = 8   # We will evaluate every 8 optimizer steps (1 epoch is 32 optimizer steps)

    # Optimizer & scheduler
    num_update_steps_per_epoch = math.ceil(len(dataloader) / accum_steps)
    total_steps = num_update_steps_per_epoch * epochs
    optimizer, scheduler = build_optimizer_and_scheduler(model, lr, total_steps)

    # Find information to resume training
    start_epoch, resume_step, global_optimizer_step = resume_training(output_dir, optimizer, scheduler)

    # To monitor training
    metrics_path = Path(output_dir) / "metrics.jsonl"
    reconcile_metrics(metrics_path, last_global_step=global_optimizer_step)
    
    # Loss ignoring padding
    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="sum")    #sum is important here. We are doing gradient accumulation, with sentences of varying lengths.

    model.train()
    optimizer.zero_grad()

    for epoch in range(start_epoch, epochs):
        # Accumulators for the current accumulation window
        window_valid_tokens = 0           # total valid tokens (non padded) across micro-batches in this accumulation window
        window_loss_sum = 0.0             # python float for logging (sum of per-token losses)
        window_mb = 0                     # number of micro-batches seen in this accumulation window

        for step, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", miniters=10)):
            # --- skip ahead if resuming in the middle of an epoch ---
            if epoch == start_epoch and step < resume_step:
                continue

            batch_input_ids, batch_attention, batch_labels = [x.to(first_device) for x in batch]

            outputs = model(
                input_ids=batch_input_ids,
                attention_mask=batch_attention,
                use_cache=False,        # disable KV-cache during training
                return_dict=True,
            )
            logits = outputs.logits  # shape (batch, seq_len, vocab_size)                
            # Shift for teacher forcing
            shift_logits = logits[..., :-1, :].contiguous()     # we exclude the last logits vector because we don't know the ground truth next token  
            shift_labels = batch_labels[..., 1:].clone()     # we exclude the first token because we can't predict it from nothing

            # Count valid tokens in this micro-batch
            valid_mask = (shift_labels != -100)
            tokens_i = int(valid_mask.sum().item())    

            # Compute loss over non-padded tokens
            loss = loss_fct(
                shift_logits.float().view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )                       # a scalar torch.Tensor

            # --- accumulate for logging (detach to avoid keeping graphs) ---
            window_valid_tokens += tokens_i
            window_loss_sum += float(loss.item())
            window_mb += 1

            # UNNORMALIZED loss (important, we'll normalize later before optimizer step)
            loss.backward()         # Backpropagate: compute gradients of loss w.r.t. all model parameters
                                    # Stores the results in param.grad for each nn.Parameter in the model, adds it to the previous gradient

            # Update the weights of the model. We have accumulated enough micro_batch for a real batch
            if window_mb == accum_steps:
                denom = max(window_valid_tokens, 1)
                # NORMALIZE grads once to get mean-per-token gradient
                for p in model.parameters():
                    if p.grad is not None:
                        p.grad.div_(denom)

                optimizer.step()        # Applies an update using the accumulated gradients and updates AdamW internal state.
                scheduler.step()        # Advances the LR schedule by one optimizer *step*. 
                optimizer.zero_grad()   # Clears grads so the next accumulation window starts fresh.
                global_optimizer_step += 1

                # Save checkpoint
                if (checkpoint_steps > 0) and (global_optimizer_step % checkpoint_steps == 0):
                    torch.cuda.synchronize()  # be safe before saving
                    save_checkpoint(
                        model, tokenizer, optimizer, scheduler,
                        epoch=epoch, step=step, global_step=global_optimizer_step,
                        output_dir=output_dir
                    )

                mean_loss = window_loss_sum / denom
                print(f"Epoch {epoch+1}, loss over the (real) batch {(step + 1) // accum_steps}: {(mean_loss):.4f}, ppl: {ppl:.2f}") 
                
                log_jsonl(
                    metrics_path,
                    ts=time.time(),
                    epoch=epoch + 1,
                    global_step=global_optimizer_step,
                    batch=(step + 1) // accum_steps,
                    split="train",
                    tokens_in_batch=denom,
                    loss_token_avg=mean_loss,
                    ppl=math.exp(mean_loss),
                    lr=scheduler.get_last_lr()[0],
                )

                # reset window accumulators
                window_valid_tokens = 0
                window_loss_sum = 0.0
                window_mb = 0

                if eval_every and (global_optimizer_step % eval_every == 0):
                    eval_loss, eval_ppl, eval_tokens = evaluate(model, eval_dataloader, loss_fct, first_device)
                    log_jsonl(metrics_path,
                            ts=time.time(),
                            epoch=epoch + 1,
                            global_step=global_optimizer_step,
                            split="eval",
                            tokens_in_batch=eval_tokens,
                            loss_token_avg=eval_loss,
                            ppl=eval_ppl,
                            lr=scheduler.get_last_lr()[0])
                    print(f"[EVAL] step {global_optimizer_step}: loss/token={eval_loss:.4f}, ppl={eval_ppl:.2f}")

                
        # --- Flush leftover micro-batches at end of epoch (if any) ---
        if window_mb > 0:
            denom = max(window_valid_tokens, 1)
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.div_(denom)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_optimizer_step += 1

            mean_loss = window_loss_sum / denom
            print(f"Epoch {epoch+1}, loss over the (real) batch {(step + 1) // accum_steps}: {(mean_loss):.4f}, ppl: {math.exp(mean_loss):.2f}") 

            log_jsonl(
                    metrics_path,
                    ts=time.time(),
                    epoch=epoch + 1,
                    global_step=global_optimizer_step,
                    batch=(step + 1) // accum_steps,
                    tokens_in_batch=denom,
                    loss_token_avg=mean_loss,
                    ppl=math.exp(mean_loss),
                    lr=scheduler.get_last_lr()[0],
                )
            
            if eval_every and (global_optimizer_step % eval_every == 0):
                    eval_loss, eval_ppl, eval_tokens = evaluate(model, eval_dataloader, loss_fct, first_device)
                    log_jsonl(metrics_path,
                            ts=time.time(),
                            epoch=epoch + 1,
                            global_step=global_optimizer_step,
                            split="eval",
                            tokens_in_batch=eval_tokens,
                            loss_token_avg=eval_loss,
                            ppl=eval_ppl,
                            lr=scheduler.get_last_lr()[0])
                    print(f"[EVAL] step {global_optimizer_step}: loss/token={eval_loss:.4f}, ppl={eval_ppl:.2f}")

    # # Save the fine-tuned model and tokenizer
    # model.save_pretrained(output_dir)
    # tokenizer.save_pretrained(output_dir)   # Why do we need to save the tokenizer too?

if __name__ == "__main__":
    import argparse
    import json
    import hashlib
    from pathlib import Path
    from datetime import datetime

    HERE = Path(__file__).resolve().parent  # directory containing finetune.py
    DATA = HERE / "RPN_benchmark" / "train" / "train_dataset1024.jsonl"     # dataset used for training
    EVAL = HERE / "RPN_benchmark" / "eval" / "eval_dataset128.jsonl"

    OUTPUT_ROOT = Path("/scratch/ctisseau/finetuned-models")

    def slugify(s: str) -> str:
        return (
            s.replace("/", "-")
             .replace("\\", "-")
             .replace(" ", "-")
             .replace(":", "-")
        )

    def args_hash(d: dict, digest_size: int = 10) -> str:
        payload = json.dumps(d, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.blake2b(payload, digest_size=digest_size).hexdigest()


    parser = argparse.ArgumentParser(description="Fine-tune a LLM.")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--train_path", type=Path, default=DATA)
    parser.add_argument("--train_data_size", type=int, default=None,
                        help="Number of samples of the whole dataset to train on.")
    parser.add_argument("--eval_path", type=Path, default=EVAL)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--output_root", type=str, default=OUTPUT_ROOT)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--checkpoint_steps", type=int, default=32)     # save every N optimizer steps. There is dataset_size / batch_size steps in one epoch. 1024 / 32 = 32.
    args = parser.parse_args()

    training_args = {
        "base_model": args.model_name,
        "training_method": "classical_sft",
        "train_dataset": str(args.train_path),
        "train_dataset_size": args.train_data_size,
        "is_shuffled": "no",
        "eval_dataset": str(args.eval_path),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "checkpoint_steps": args.checkpoint_steps,
        "lr": args.lr,
        "warmup_ratio": 0.1,
    }
    
    run_id = args_hash(training_args, digest_size=10)  # 20 hex chars
    run_name = f"{slugify(training_args['base_model'])}_" \
               f"{slugify(training_args['training_method'])}_" \
               f"{run_id}"
    output_dir = args.output_root / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    config_path = output_dir / "training_args.json"
    if not config_path.exists():
        payload = {
            **training_args,
            "run_id": run_id,
            "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        }
        with open(config_path, "w") as f:
            json.dump(payload, f, indent=4)

    finetune(
        model_name=args.model_name,
        train_data=args.train_path,
        train_data_size=args.train_data_size,
        eval_data=args.eval_path,
        max_length=args.max_length,
        output_dir=output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        checkpoint_steps=args.checkpoint_steps,
    )
