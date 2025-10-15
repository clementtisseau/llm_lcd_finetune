import torch
import math
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup

from torch.utils.tensorboard import SummaryWriter

from pathlib import Path



# --- Utility Functions for File I/O ---
def stream_jsonl(filename: str):
    """
    Parses a JSONL file and yields each line as a dictionary.
    """
    with open(filename, "r") as fp:
        for line in fp:
            if any(not x.isspace() for x in line):
                yield json.loads(line)
# ----- Helper functions -----
import os, re, time, json, shutil, signal
from typing import Optional

def _ckpt_step(p: Path) -> int:
    m = re.search(r"checkpoint-(\d+)$", p.name)
    return int(m.group(1)) if m else -1

def find_last_checkpoint(output_dir: str) -> Optional[Path]:
    d = Path(output_dir)
    if not d.exists(): 
        return None
    cks = [p for p in d.glob("checkpoint-*") if p.is_dir()]
    return max(cks, key=_ckpt_step) if cks else None

def save_checkpoint(model, tokenizer, optimizer, scheduler, epoch, step, global_step, output_dir):
    ckpt_dir = Path(output_dir) / f"checkpoint-{global_step:08d}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    # save model/tokenizer shards as safetensors (works with device_map sharding)
    model.save_pretrained(ckpt_dir, safe_serialization=True, max_shard_size="5GB")
    tokenizer.save_pretrained(ckpt_dir)
    # small file with training state
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,          # current epoch index
            "step": step + 1,        # next dataloader step to run in this epoch
            "global_step": global_step,
        },
        ckpt_dir / "training_state.pt",
    )

def load_training_state(ckpt_dir: Path):
    path = ckpt_dir / "training_state.pt"
    return torch.load(path, map_location="cpu") if path.exists() else {}
# ----- End of Helper functions -----


def finetune(
    model_name: str,
    dataset_name: str,
    dataset_size: int,
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
    dataset = list(stream_jsonl(dataset_name))
    if dataset_size is not None: dataset = dataset[:dataset_size]
    print("Dataset loaded")

    max_memory3 = {
        0: "40GB",
        1: "40GB",
        2: "40GB"
    }
    max_memory2 = {
        0: "75GB",   
        1: "75GB",
    }

    resume_dir = find_last_checkpoint(output_dir)
    if resume_dir is not None:
        print(f"Resuming from {resume_dir}")
        model_to_load = resume_dir
    else: 
        print("No checkpoint found — starting from base model")
        model_to_load = model_name
    # Load tokenizer and model
    bf16_ok = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if bf16_ok else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(model_to_load)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_to_load, 
        use_safetensors=True,
        torch_dtype=dtype, 
        device_map="balanced",
        max_memory=max_memory3,
        low_cpu_mem_usage=True,
        )
    print("Tokenizer and Model loaded")
    model.config.use_cache = False          # Disable KV-cache, which is useless during training

    first_device = next(model.parameters()).device

    # --- Process dataset ---
    SYSTEM = (
        "You are an expert at converting arithmetic expressions into Reverse Polish "
        "Notation (RPN). You always output only the RPN expression, with tokens "
        "separated by a single space. Do not include explanations or extra text."
    )
    def build_messages(infix: str, rpn: str | None):
        msgs = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Infix: {infix}\nRPN:"},
        ]
        if rpn is not None:
            msgs.append({"role": "assistant", "content": rpn})
        return msgs
    
    rows = dataset
    # Prompt-only strings (stop at assistant start) — used to compute prompt lengths
    prompt_texts = [
        tokenizer.apply_chat_template(
            build_messages(r["infix"], None),
            tokenize=False,
            add_generation_prompt=True,   # ends right after assistant-start
            enable_thinking=False
        )
        for r in rows
    ]# example: (4 additional newlines when enable_thinking=False)
    # "<|im_start|>system\nYou are an expert at converting arithmetic expressions into Reverse Polish Notation (RPN). You always output only the RPN expression, with tokens separated by a single space. Do not include explanations or extra text.<|im_end|>\n<|im_start|>user\nInfix: 3 + 4 * 2\nRPN:<|im_end|>\n<|im_start|>assistant\n\n\n\n\n"

    # Full training strings (prompt + target)
    full_texts = [
        tokenizer.apply_chat_template(
            build_messages(r["infix"], r["rpn"]),
            tokenize=False,
            add_generation_prompt=False,  # includes assistant message + proper end token
            enable_thinking=False
        )
        for r in rows
    ]
    # "<|im_start|>system\nYou are an expert at converting arithmetic expressions into Reverse Polish Notation (RPN). You always output only the RPN expression, with tokens separated by a single space. Do not include explanations or extra text.<|im_end|>\n<|im_start|>user\nInfix: 3 + 4 * 2\nRPN:<|im_end|>\n<|im_start|>assistant\n\n\n\n\n3 4 2 * +<|im_end|>\n"


    # Tokenize full sequences
    encodings = tokenizer(
        full_texts,
        add_special_tokens=False,        # template already added them
        padding=True,                    # left-padding
        return_tensors="pt",
    )

    input_ids = encodings["input_ids"]
    attention_mask = encodings["attention_mask"]
    labels = input_ids.clone()
    # Compute per-row full (non-pad) lengths
    full_lengths = attention_mask.sum(dim=1)
    padded_seq_len = input_ids.size(1)
    # Prompt lengths (tokenized without padding)
    prompt_tokenized = tokenizer(
        prompt_texts, add_special_tokens=False, padding=False, return_tensors=None
    )
    prompt_lengths = [len(ids) for ids in prompt_tokenized["input_ids"]]
    # Mask the prompt region (left-padded batches)
    for i, l_prompt in enumerate(prompt_lengths):
        start_prompt = padded_seq_len - full_lengths[i]     # index where non-pad starts
        end_prompt = min(start_prompt + l_prompt, padded_seq_len)
        labels[i, start_prompt:end_prompt] = -100
    # Mask padding
    labels[attention_mask == 0] = -100
    # --- END Process dataset ---

    # Find the perfect micro_batch_size
    vocab_size = model.config.vocab_size
    micro_batch_size = 1
    for bs in [32, 16, 8, 4, 2, 1]:
        try:
            model.zero_grad(set_to_none=True)
            dummy = torch.randint(0, vocab_size, (bs, max_length), device=first_device)
            # backward to test peak memory, but NO optimizer step
            test_out = model(dummy, labels=dummy)
            test_out.loss.backward()
            model.zero_grad(set_to_none=True)
            micro_batch_size = bs
            print(f"fits on GPU: micro_batch_size = {bs}")
            break
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"OOM when trying bs={bs}, trying smaller…")
                torch.cuda.empty_cache()
            else:
                raise
    torch.cuda.empty_cache()

    if batch_size % micro_batch_size != 0:
        raise ValueError(
                f"The micro_batch_size ({micro_batch_size}) should be a divisor of the batch_size ({batch_size}). "
                f"batch_size % micro_batch_size = {batch_size % micro_batch_size}"
            )    
    accum_steps = batch_size // micro_batch_size        # The number of micro_batch to process to obtain a real batch

    # DataLoader
    dataset = TensorDataset(input_ids, attention_mask, labels)
    dataloader = DataLoader(dataset, batch_size=micro_batch_size, shuffle=False)    # We need shuffle = False if we want to track the input of each sentence
                                                                                    # By default drop_last=False, meaning the last batch is smaller than the other ones
    # Optimizer & scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    num_update_steps_per_epoch = math.ceil(len(dataloader) / accum_steps)
    total_steps = num_update_steps_per_epoch * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )
    # To monitor training
    # tensorboard --logdir /scratch/ctisseau/finetuned-models/Qwen3-1.7B-RPN-ds1024-e2-ds1024-bs32-testdeletelater/tb --host 127.0.0.1 --port 7007    
    # ssh -N -L 7007:127.0.0.1:7007 -J ctisseau@cleps.paris.inria.fr ctisseau@gpu014
    # Then open http://localhost:7007 on the browser    
    writer = SummaryWriter(log_dir=str(Path(output_dir) / "tb"))

    # Find information to resume training
    # A step is an update of the weights. Each epoch comprise several batch (one batch is multiple micro_batch) and each batch means a step
    start_epoch = 0             # the epoch at which the checkpoint model is
    resume_step = 0             # the optimizer step within the current epoch at which the checkpoint model is
    global_optimizer_step = 0   # the optimizer step at which the checkpoint model is
    if resume_dir is not None:
        state = load_training_state(resume_dir)
        if state:
            try:
                optimizer.load_state_dict(state["optimizer"])
                scheduler.load_state_dict(state["scheduler"])
            except Exception as e:
                print(f"Warning: could not load optimizer/scheduler state: {e}")
            start_epoch = state.get("epoch", 0)
            resume_step = state.get("step", 0)
            global_optimizer_step = state.get("global_step", 0)
    

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

                mean_loss = window_loss_sum / denom
                ppl = math.exp(mean_loss)
                lr_now = scheduler.get_last_lr()[0]
                print(f"Epoch {epoch+1}, loss over the (real) batch {(step + 1) // accum_steps}: {(mean_loss):.4f}, ppl: {ppl:.2f}") 
                writer.add_scalar("train/loss_token_avg", mean_loss, global_optimizer_step)
                writer.add_scalar("train/ppl", ppl, global_optimizer_step)
                writer.add_scalar("train/lr", lr_now, global_optimizer_step)
                writer.flush()

                # reset window accumulators
                window_valid_tokens = 0
                window_loss_sum = 0.0
                window_mb = 0

                # Save checkpoint
                if (checkpoint_steps > 0) and (global_optimizer_step % checkpoint_steps == 0):
                    torch.cuda.synchronize()  # be safe before saving
                    save_checkpoint(
                        model, tokenizer, optimizer, scheduler,
                        epoch=epoch, step=step, global_step=global_optimizer_step,
                        output_dir=output_dir
                    )
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


    # # Save the fine-tuned model and tokenizer
    # model.save_pretrained(output_dir)
    # tokenizer.save_pretrained(output_dir)   # Why do we need to save the tokenizer too?

if __name__ == "__main__":
    import argparse

    HERE = Path(__file__).resolve().parent  # directory containing finetune.py
    DATA = HERE / "RPN_benchmark" / "train" / "train_dataset1024.jsonl"     # dataset used for training

    parser = argparse.ArgumentParser(description="Fine-tune a LLM.")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--dataset_name", type=str, default=DATA)
    parser.add_argument("--dataset_size", type=int, default=None,
                        help="Number of samples of the whole dataset to train on.")
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--output_dir", type=str, default=f"/scratch/ctisseau/finetuned-models/Qwen3-1.7B-RPN-ds1024-e2-ds1024-bs32-testdeletelater")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--checkpoint_steps", type=int, default=32)     # save every N optimizer steps. There is dataset_size / batch_size steps in one epoch. 1024 / 32 = 32.
    args = parser.parse_args()



    finetune(
        model_name=args.model_name,
        dataset_name=args.dataset_name,
        dataset_size=args.dataset_size,
        max_length=args.max_length,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        checkpoint_steps=args.checkpoint_steps,
    )
