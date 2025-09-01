import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup

from pathlib import Path

from datasets import load_dataset

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
    max_length: int = 1024,
    output_dir: str = "/scratch/ctisseau/finetuned-models/Qwen3-1.7B-OCI-e1-ds65536-bs256-ckps64",
    epochs: int = 1,
    batch_size: int = 256,
    lr: float = 5e-5,
    checkpoint_steps = 64,
    device: str = None,
):

    # Setup device
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    initial_ckpt_dir = Path(output_dir) / "checkpoint-00000000"
    if not initial_ckpt_dir.exists():
        print(f"Base model checkpoint not found. Downloading '{model_name}' and saving as checkpoint-00000000.")
        # Download from Hugging Face Hub
        tokenizer_base = AutoTokenizer.from_pretrained(model_name)
        model_base = AutoModelForCausalLM.from_pretrained(model_name, use_safetensors=True, low_cpu_mem_usage=True)

        initial_ckpt_dir.mkdir(parents=True, exist_ok=True)
        model_base.save_pretrained(initial_ckpt_dir, safe_serialization=True, max_shard_size="5GB")
        tokenizer_base.save_pretrained(initial_ckpt_dir)
        # Free up memory
        del model_base
        del tokenizer_base
        torch.cuda.empty_cache()

    # Now, find the latest checkpoint to resume from (it will be checkpoint-00000000 on the first run).
    resume_dir = find_last_checkpoint(output_dir)
    if resume_dir is None:
        raise RuntimeError(f"Could not find any checkpoints in {output_dir}, not even the base one we just tried to create.")
    print(f"Training from {resume_dir}")
    # Load tokenizer and model from the latest checkpoint
    tokenizer = AutoTokenizer.from_pretrained(resume_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    max_memory3 = {
    0: "40GB",
    1: "40GB",
    2: "40GB"
    }
    max_memory2 = {
        0: "75GB",   
        1: "75GB",
    }
    model = AutoModelForCausalLM.from_pretrained(
            resume_dir, 
            use_safetensors=True,
            torch_dtype=torch.float16, 
            device_map="balanced", 
            max_memory=max_memory2, # Choose the map that matches your hardware
            low_cpu_mem_usage=True,
    )
    print("Tokenizer and Model loaded")

    # Load dataset
    print("Loading dataset")
    raw_ds = load_dataset(dataset_name, split=f"train[:{dataset_size}]")   #5 million coding question-answer pairs. 6.4GB. 
    print("Dataset loaded")

    first_device = next(model.parameters()).device
    
    # Filter dataset
    filtered_ds = []
    # Iterate through the raw dataset
    for inp, outp in zip(raw_ds["input"], raw_ds["output"]):
        # Construct the full text
        text = f"### Instruction:\n{inp}\n### Response:\n{outp}"
        
        # Tokenize a single example without padding to check its length
        tokenized_text = tokenizer(text, add_special_tokens=True)["input_ids"]
        
        # If it's within the limit, keep it
        if len(tokenized_text) <= max_length:
            filtered_ds.append({"input": inp, "output": outp})
    
    # Convert raw dataset to list of strings
    filtered_texts = [f"### Instruction:\n{inp}\n### Response:\n{outp}" for inp, outp in zip(filtered_ds["input"], filtered_ds["output"])]

    encodings = tokenizer(
    filtered_texts,
    add_special_tokens=True,    # Adds <s>, </s>
    truncation=False,
    padding=True,               # Adds <pad>
    return_tensors="pt",
    )

    # Create input_ids, attention_mask, labels
    input_ids = encodings["input_ids"]
    attention_mask = encodings["attention_mask"]
    lengths = encodings["attention_mask"].sum(dim=1)
    labels = input_ids.clone()
    # prompts_inp = [f"### Instruction:\n{inp}\n### Response:\n```python\n" for inp in raw_ds["input"]]
    prompts_inp = [f"### Instruction:\n{inp}\n### Response:\n" for inp in raw_ds["input"]]
    prompt_inp_lengths = [len(tokenizer(p, add_special_tokens=True)["input_ids"]) - 1 for p in prompts_inp]
    for i, L in enumerate(prompt_inp_lengths):
        labels[i, :L] = -100                # Question are not taken into account for the loss
    labels[attention_mask == 0] = -100      # Padding tokens are not taken into account for the loss

    # Find the perfect micro_batch_size
    vocab_size = model.config.vocab_size
    seq_len = max_length
    test_micro_batch_size = 8
    for bs in [64,32,16,8]:
        try:
            model.zero_grad()
            dummy = torch.randint(0, vocab_size, (bs, seq_len), device=first_device)
            # use dummy labels so we get a backward pass too
            loss = model(dummy, labels=dummy).loss
            loss.backward()

            # if bs == 32:                         # create optimiser once
            #     optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
            # optimizer.step()                     # exp_avg, exp_avg_sq are allocated
            # optimizer.zero_grad(set_to_none=True)

            test_micro_batch_size = bs
            print(f"→ fits on GPU: micro_batch_size = {bs}")
            break
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"OOM when trying bs={bs}, trying smaller…")
                torch.cuda.empty_cache()
            else:
                raise
    torch.cuda.empty_cache()

    micro_batch_size = test_micro_batch_size
    accum_steps = batch_size // micro_batch_size

    dataset = TensorDataset(input_ids, attention_mask, labels)
    dataloader = DataLoader(dataset, batch_size=micro_batch_size, shuffle=False)    # We need shuffle = False if we want to track the input of each sentence
                                                                                    # By default drop_last=False, meaning the last batch is smaller than the other ones

    # Optimizer & scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    total_steps = len(dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    # Find information to resume training
    start_epoch = 0             # the epoch at which the checkpoint model is
    resume_step = 0             # the optimizer step within the current epoch at which the checkpoint model is
    global_optimizer_step = 0   # the optimizer setp at which the checkpoint model is
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
    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)

    model.train()
    optimizer.zero_grad()
    optimizer_step = 0   # one step per real batch (after gradient accumulation)
    for epoch in range(start_epoch, epochs):
        epoch_loss = 0.0
        n_mini_batches = 0
        for step, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", miniters=10)):
            # --- skip ahead if resuming in the middle of an epoch ---
            if epoch == start_epoch and step < resume_step:
                continue

            batch_input_ids, batch_attention, batch_labels = [x.to(first_device) for x in batch]

            # Forward pass
            outputs = model(
                input_ids=batch_input_ids,
                attention_mask=batch_attention,
                return_dict=True,
            )
            logits = outputs.logits  # shape (batch, seq_len, vocab_size)
            
            # Shift for teacher forcing
            shift_logits = logits[..., :-1, :].contiguous()      
            shift_labels = batch_input_ids[..., 1:].clone()   

            # Compute loss
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )

            raw_loss = loss.item()
            epoch_loss += raw_loss
            n_mini_batches += 1

            loss = loss / accum_steps
            loss.backward()

            if (step + 1) % accum_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_optimizer_step += 1
                optimizer_step += 1

                # Save checkpoint
                if (args.checkpoint_steps > 0) and (global_optimizer_step % args.checkpoint_steps == 0):
                    torch.cuda.synchronize()  # be safe before saving
                    save_checkpoint(
                        model, tokenizer, optimizer, scheduler,
                        epoch=epoch, step=step, global_step=global_optimizer_step,
                        output_dir=output_dir
                    )
                    print(f"Epoch {epoch+1} average loss so far: {(epoch_loss/n_mini_batches):.4f}")

        last_step_was_checkpoint = (optimizer_step > 0) and (global_optimizer_step % args.checkpoint_steps == 0)
        if not last_step_was_checkpoint:
            save_checkpoint(
                model, tokenizer, optimizer, scheduler,
                epoch=epoch, step=len(dataloader)-1, global_step=global_optimizer_step,
                output_dir=output_dir
            )             
        print(f"Epoch {epoch+1} average loss: {(epoch_loss/n_mini_batches):.4f}")

    # Save the fine-tuned model and tokenizer
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fine-tune a constrained LLM with biased logits.")
    #parser.add_argument("--model_name", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--dataset_name", type=str, default="nvidia/OpenCodeInstruct")
    parser.add_argument("--dataset_size", type=int, default=65536,
                        help="Number of samples of the whole dataset to train on.")
    parser.add_argument("--max_length", type=int, default=1024)         # we don't change this
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=256)      
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--checkpoint_steps", type=int, default=64)      # save every N optimizer steps
                                                                        # 16384*4 / 256 = 64*4 optimizer steps per epoch
                                                                        # one checkpoint every 64 optimizer steps means 4 checkpoints + 1 base model
    args = parser.parse_args()



    finetune(
        model_name=args.model_name,
        dataset_name=args.dataset_name,
        dataset_size=args.dataset_size,
        max_length=args.max_length,
        #output_dir=f"/scratch/ctisseau/finetuned-models/Llama-2-7b-hf-OCI-test-32",
        output_dir="/scratch/ctisseau/finetuned-models/Qwen3-1.7B-OCI-e1-ds65536-bs256-ckps64",
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        checkpoint_steps=args.checkpoint_steps,
    )
