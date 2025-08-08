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

python_grammar = Path("python2.lark").read_text(encoding="utf-8")

def finetune_test(
    model_name: str,
    output_dir: str = "/scratch/ctisseau/finetuned-models",
    epochs: int = 3,
    lr: float = 5e-5,
    device: str = None,
):

    # Load tokenizer and model
    print("Loading tokenizer and model")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("Tokenizer loaded")


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
        model_name, 
        use_safetensors=True,
        torch_dtype=torch.float16, 
        device_map="balanced",
        max_memory=max_memory2,
        low_cpu_mem_usage=True,
        )
    print("Model loaded")
    first_device = next(model.parameters()).device
    print("test")

    # Create the CFGLogitsProcessor (we need to initialize an outlines and a clm model)
    model_outlines = outlines.from_transformers(
        AutoModelForCausalLM.from_pretrained(model_name),
        AutoTokenizer.from_pretrained(model_name)
    )
    #model_outlines = outlines.models.transformers(model_name)
    print("Outlines model loaded")
    qwen_clm = clm.TransformersLM(model_name)
    print("CLM model loaded")
    cfg = clm.CLMCFGLogitsProcessor(python_grammar, model_outlines.tokenizer, qwen_clm, tensor_library_name='torch')
    print("CFG loaded")

    prompt = ["In the programming language Python, a code that do : x=10, y=0, while x>0: y = y + 2, x = x - 1, would be in Python:\nx=10 \ny=0 \nwhile x>0: \n    y = y + 2 \n    x = x - 1",
        "In the programming language Python, a code that do : x=2, for i from 1 to 5: x = x**2, would be in Python:\nx=2 \nfor i in range(1,6): \n    x = x**2"]
    encoding = tokenizer(
        prompt,
        add_special_tokens=True,    # Adds <s>, </s>
        truncation=False,
        padding=True,
        return_tensors="pt",
        )
    input_ids = encoding["input_ids"]
    attention_mask = encoding["attention_mask"]
    labels = input_ids.clone()
    L1 = len(tokenizer("In the programming language Python, a code that do : x=10, y=0, while x>0: y = y + 2, x = x - 1, would be in Python:\n", add_special_tokens=True, return_tensors="pt")["input_ids"]) - 1    # Here w/ -1 we exclude the last <\s> token
    L2 = len(tokenizer("In the programming language Python, a code that do : x=2, for i from 1 to 5: x = x**2, would be in Python:\n", add_special_tokens=True, return_tensors="pt")["input_ids"]) - 1
    labels[0,:L1] = -100        
    labels[1,:L2] = -100
    labels[attention_mask == 0] = -100      # Padding tokens are not taken into account for the loss
    prompt_inp_lengths = [L1, L2]
    lengths = [len(tokenizer(p, add_special_tokens=True, padding=False, return_tensors="pt")["input_ids"]) for p in prompt]


    dataset = TensorDataset(input_ids, attention_mask, labels)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False)

    epochs = 3
    # Optimizer & scheduler
    optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=lr)
    total_steps = len(dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )


    # Loss ignoring padding
    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)

    model.train()

    optimizer.zero_grad()
    step = 0

    print("Beginning of training:")
    for epoch in range(epochs):
        
        for step, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")):
            batch_input_ids, batch_attention, batch_labels = [x.to(first_device) for x in batch]

            # Forward pass
            outputs = model(
                input_ids=batch_input_ids,
                attention_mask=batch_attention,
                return_dict=True,
            )
            logits = outputs.logits  # shape (batch, seq_len, vocab_size)

            biased_logits = logits.clone()
            # Is it a good solution to create biased_logits with empty values instead of modifying logits?
            # Apply bias mask
            for i in range(logits.shape[0]):
                # What if prompt_inp_lengths[i] = 0 ? Could be dangerous but it will never happen
                l = prompt_inp_lengths[i] - 1 # - 1 is very important here, it makes sure that we can predict the first token of the answer from the last token of the question.
                L = lengths[i]
                for j in range(l, L):
                    biased_logits[i, j, :] = cfg.process_logits(batch_input_ids[i, :j].unsqueeze(0), logits[i, j, :].unsqueeze(0)).squeeze(0)  # returns shape (batch, seq_len, vocab)
                cfg._seq_start_idx = None
            # Shift for teacher forcing
            shift_logits = biased_logits[..., :-1, :].contiguous()      
            shift_labels = batch_input_ids[..., 1:].clone()   
            print(shift_logits.shape, shift_labels.shape)          

            # Compute loss
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
            
            loss.backward()
            step+=1
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            print(f"Epoch {epoch+1}, step {step} loss: {(loss):.4f}")

    # Save the fine-tuned model and tokenizer
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fine-tune test a constrained LLM with biased logits.")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5e-5)
    args = parser.parse_args()



    finetune_test(
        model_name=args.model_name,
        output_dir=f"/scratch/ctisseau/finetuned-models/Llama-2-7b-hf-OCI-test-testv0",
        epochs=args.epochs,
        lr=args.lr,
    )