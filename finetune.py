import constraintlm as clm

import outlines

import torch
import bitsandbytes as bnb
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup

from pathlib import Path


from datasets import load_dataset

python_grammar = Path("python.lark").read_text(encoding="utf-8")


def finetune(
    model_name: str,
    dataset_name: str,
    dataset_size: int,
    max_length: int = 1024,
    output_dir: str = "/scratch/ctisseau/finetuned-models",
    epochs: int = 3,
    batch_size: int = 64,
    lr: float = 5e-5,
    device: str = None,
):
    # Setup device
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset
    print("Loading dataset")
    raw_ds = load_dataset(dataset_name, split=f"train[:{dataset_size}]")   #5 million coding question-answer pairs. 6.4GB. 
    #ds = ds.shuffle(seed=42)
    #raw_ds = ds[:dataset_size]
    print("Dataset loaded")

    # Load tokenizer and model
    print("Loading tokenizer and model")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("Tokenizer loaded")
    
    max_memory3 = {
        0: "40GB",   # carve out a bit of safety below the 24 GB physical
        1: "40GB",
        2: "40GB"
    }
    max_memory2 = {
        0: "75GB",   
        1: "75GB",
    }

    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        use_safetensors=True,
        torch_dtype=torch.float16, 
        device_map="balanced",
        max_memory=max_memory3,
        low_cpu_mem_usage=True,
        )
    print("Model loaded")
    first_device = next(model.parameters()).device

    # Create the CFGLogitsProcessor (we need to initialize an outlines and a clm model)
    model_outlines = outlines.models.transformers(model_name)
    print("Outlines model loaded")
    qwen_clm = clm.TransformersLM(model_name)
    print("CLM model loaded")
    cfg = clm.CLMCFGLogitsProcessor(python_grammar, model_outlines.tokenizer, qwen_clm)
    print("CFG loaded")


    # Convert raw dataset to list of strings
    texts = [f"### Instruction:\n{inp}\n### Response:\n{outp}" for inp, outp in zip(raw_ds["input"], raw_ds["output"])]

    
    # Filter dataset

    # #1st method
    # tokenized = tokenizer(
    # texts,
    # add_special_tokens=True,
    # truncation=False,
    # padding=False,           # don’t pad, so each input_ids is its true length
    # return_tensors=None      # get Python lists, not tensors
    # )
    # lengths = [len(ids) for ids in tokenized["input_ids"]]
    # keep_idx = [i for i, l in enumerate(lengths) if l <= max_length]
    # filtered_texts = [texts[i] for i in keep_idx]
    
    # # Encode filtered dataset
    # encodings = tokenizer(
    #     filtered_texts,
    #     return_tensors="pt",
    #     padding=True,
    #     truncation=True,
    #     max_length=tokenizer.model_max_length,
    # )

    #2nd method (shorter)
    encodings = tokenizer(
    texts,
    add_special_tokens=True,    # Adds <s>, </s>
    truncation=False,
    padding=True,               # Adds <pad>
    return_tensors="pt",
    )
    full_lengths = encodings["attention_mask"].sum(dim=1)
    keep = full_lengths <= max_length
    if v.dim() == 2:
        v = v[:, :maxlen]
    encodings[k] = v

    # Create input_ids, attention_mask, labels
    input_ids = encodings["input_ids"]
    attention_mask = encodings["attention_mask"]
    lengths = encodings["attention_mask"].sum(dim=1)
    labels = input_ids.clone()
    filtered_inputs  = [raw_ds["input"][i]  for i in keep]
    prompts_inp = [f"### Instruction:\n{inp}\n### Response:\n" for inp in filtered_inputs]
    prompt_inp_lengths = [len(tokenizer(p, add_special_tokens=True)["input_ids"]) - 1 for p in prompts_inp]
    for i, L in enumerate(prompt_inp_lengths):
        labels[i, :L] = -100                # Question are not taken into account for the loss
    labels[attention_mask == 0] = -100      # Padding tokens are not taken into account for the loss


    # Find the perfect micro_batch_size
    vocab_size = model.config.vocab_size
    seq_len = max_length
    micro_batch_size = 1
    for bs in [32,16,8,4,2,1]:
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

            micro_batch_size = bs
            print(f"→ fits on GPU: micro_batch_size = {bs}")
            break
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"OOM when trying bs={bs}, trying smaller…")
                torch.cuda.empty_cache()
            else:
                raise
    torch.cuda.empty_cache()

    micro_batch_size = 4

    effective_batch = batch_size
    accum_steps = effective_batch // micro_batch_size

    dataset = TensorDataset(input_ids, attention_mask, labels)
    dataloader = DataLoader(dataset, batch_size=micro_batch_size, shuffle=False)    # We need shuffle = False if we want to track the input of each sentence
                                                                                    # By default drop_last=False, meaning the last batch is smaller than the other ones


    # Optimizer & scheduler
    optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=lr)
    total_steps = len(dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )
    

    # Loss ignoring padding
    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    model.train()

    optimizer.zero_grad()

    for epoch in range(epochs):
        epoch_loss = 0.0
        total_batches = 0
        accum_loss = 0.0
        real_batch = 0

        for step, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", miniters=10)):
            batch_input_ids, batch_attention, batch_labels = [x.to(first_device) for x in batch]

            # Forward pass
            outputs = model(
                input_ids=batch_input_ids,
                attention_mask=batch_attention,
                return_dict=True,
            )
            logits = outputs.logits  # shape (batch, seq_len, vocab_size)

            biased_logits = logits.clone()
            for i in range(logits.shape[0]):
                l = prompt_inp_lengths[step + i] - 1 # - 1 is very important here, it makes sure that we can predict the first token of the answer from the last token of the question.
                L = lengths[step + i]
                for j in range(l, L):
                    biased_logits[i, j, :] = cfg.process_logits(batch_input_ids[i, :j].unsqueeze(0), logits[i, j, :].unsqueeze(0)).squeeze(0)  # returns shape (batch, seq_len, vocab)

            # Shift for teacher forcing
            shift_logits = biased_logits[..., :-1, :].contiguous()      
            shift_labels = batch_input_ids[..., 1:].clone()   
            print(shift_logits.shape, shift_labels.shape)          

            # Compute loss
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )

            raw_loss = loss.item()
            epoch_loss += raw_loss
            total_batches += 1
            accum_loss += raw_loss

            loss = loss / accum_steps
            loss.backward()

            if (step + 1) % accum_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                real_batch += 1

                # compute & print the average over those N micro-batches
                avg_group_loss = accum_loss / accum_steps
                print(f"[Step (real batch) {real_batch}] avg loss of last this real batch: {avg_group_loss:.4f}")
                accum_loss = 0.0
                
        print(f"Epoch {epoch+1} average loss: {(epoch_loss/total_batches):.4f}")

    # Save the fine-tuned model and tokenizer
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fine-tune a constrained LLM with biased logits.")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--dataset_name", type=str, default="nvidia/OpenCodeInstruct")
    parser.add_argument("--dataset_size", type=int, default=16,
                        help="Number of samples of the whole dataset to train on.")
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-5)
    args = parser.parse_args()



    finetune(
        model_name=args.model_name,
        dataset_name=args.dataset_name,
        dataset_size=args.dataset_size,
        max_length=args.max_length,
        output_dir=f"/scratch/ctisseau/finetuned-models/Llama-2-7b-hf-OCI-test",
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )


# dataset_name = "nvidia/OpenCodeInstruct"
# model_name = 
# output-dir = f"/scratch/ctisseau/finetuned-models/{args.model_name}"
