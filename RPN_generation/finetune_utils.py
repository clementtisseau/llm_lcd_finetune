import torch
import math
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup
from pathlib import Path


# ---------- File I/O ----------
def stream_jsonl(filename: str):
    """
    Parses a JSONL file and yields each line as a dictionary.
    """
    with open(filename, "r") as fp:
        for line in fp:
            if any(not x.isspace() for x in line):
                yield json.loads(line)




# ---------- Load / Save a ckpt ----------
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

def resume_training(output_dir, optimizer, scheduler):
    # Find information to resume training
    # A step is an update of the weights. Each epoch comprise several batch (one batch is multiple micro_batch) and each batch means a step
    resume_dir = find_last_checkpoint(output_dir)
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
    return start_epoch, resume_step, global_optimizer_step




# ---------- Model and Tokenizer load ----------
def load_model_tokenizer(output_dir, model_name, max_memory):
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
        max_memory=max_memory,
        low_cpu_mem_usage=True,
        )
    print("Tokenizer and Model loaded")
    return model, tokenizer




# ---------- Process dataset ----------
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

def prepare_encodings(tokenizer, dataset):
    rows = dataset

    # prompt only
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

    # prompt + target
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

    return input_ids, attention_mask, labels




# ---------- Micro-batch finder ----------
def find_micro_batch_size(model, max_length: int) -> int:
    first_device = next(model.parameters()).device
    vocab_size = model.config.vocab_size
    for bs in [32, 16, 8, 4, 2, 1]:
        try:
            model.zero_grad(set_to_none=True)
            dummy = torch.randint(0, vocab_size, (bs, max_length), device=first_device)
            out = model(dummy, labels=dummy)
            out.loss.backward()
            model.zero_grad(set_to_none=True)
            print(f"fits on GPU: micro_batch_size = {bs}")
            return bs
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"OOM when trying bs={bs}, trying smaller…")
                torch.cuda.empty_cache()
                continue
            raise
    return 1




# ---------- Optim & sched ----------
def build_optimizer_and_scheduler(model, lr: float, steps_total: int, warmup_ratio: float = 0.1):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(warmup_ratio * steps_total),
        num_training_steps=steps_total,
    )
    return optimizer, scheduler



# ---------- Write metrics ----------
def log_jsonl(path: Path, **fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(fields, separators=(",", ":")) + "\n")

def reconcile_metrics(metrics_path: Path, last_global_step: int):
    """Keep only records with global_step <= last_global_step.
       Silently drops malformed/truncated lines (common after crashes)."""
    if not metrics_path.exists():
        return
    kept = []
    with metrics_path.open("r") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                # drop partial/garbled lines
                continue
            s = int(rec.get("global_step", -1))
            if s <= last_global_step:
                kept.append(line)

    tmp = metrics_path.with_suffix(".jsonl.tmp")
    with tmp.open("w") as f:
        f.writelines(kept)
        f.flush(); os.fsync(f.fileno())
    tmp.replace(metrics_path)  # atomic on POSIX



# ---------- Evaluate model (unbiased logits) ----------
@torch.inference_mode()
def evaluate(model, loader, loss_fct, device):
    was_training = model.training
    model.eval()

    total_loss_sum = 0.0          # python float for logging (sum of per-token losses)
    total_valid_tokens = 0            # total valid tokens (non padded) across each batch

    for input_ids, attention_mask, labels in loader:
        input_ids = input_ids.to(device, non_blocking=True)
        attention_mask = attention_mask.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            use_cache=False,    # disable KV-cache during evaluation
            return_dict=True).logits
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        total_loss_sum += loss_fct(
            shift_logits.float().view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        ).item()
        total_valid_tokens += int((shift_labels != -100).sum())

    mean_loss = total_loss_sum / max(total_valid_tokens, 1)
    ppl = math.exp(mean_loss) if mean_loss < 100 else float("inf")

    if was_training:
        model.train()
    return mean_loss, ppl, total_valid_tokens

