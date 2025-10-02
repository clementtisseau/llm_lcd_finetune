import json
import os
import fire
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from pathlib import Path

max_memory3 = {
    0: "40GB",
    1: "40GB",
    2: "40GB"
}

model_name = "/scratch/ctisseau/finetuned-models/Qwen3-1.7B-OCI-e1-ds65536-bs256-ckps64/checkpoint-00000000"            #this need to change
readable_model_name = "Qwen3-1.7B"

HERE = Path(__file__).resolve().parent  # directory containing generate.py
DATA = HERE / "data" / "dataset.jsonl"

# --- Utility Functions for File I/O ---

def stream_jsonl(filename: str):
    """
    Parses a JSONL file and yields each line as a dictionary.
    """
    with open(filename, "r") as fp:
        for line in fp:
            if any(not x.isspace() for x in line):
                yield json.loads(line)

def write_jsonl(filename: str, data, append: bool = False):
    """
    Writes an iterable of dictionaries to a JSONL file.
    """
    mode = 'a' if append else 'w'
    with open(filename, mode) as fp:
        for x in data:
            fp.write(json.dumps(x) + "\n")

# --- Core Generation Function ---

def generate_rpn(model, tokenizer, infix: str, num_samples: int, max_new_tokens: int) -> list[str]:
    """
    Generates a batch of rpn expressions for a given infix.

    Args:
        model: The loaded Hugging Face model.
        tokenizer: The loaded Hugging Face tokenizer.
        infix: The infix notation to prompt the model with.
        num_samples: The number of completions to generate.
        max_new_tokens: The maximum number of tokens for each completion.

    Returns:
        A list of generated rpn expressions.
    """
    messages = [
        {"role": "system", "content": "You are an expert at converting arithmetic expressions into Reverse Polish Notation (RPN). You always output only the RPN expression, with tokens separated by a single space. Do not include explanations or extra text."},
        {"role": "user", "content": f"""Examples:

Infix: 63 - 46
RPN: 63 46 -

Infix: 44 - 85 + 57 + 76 + 88
RPN: 44 85 - 57 + 76 + 88 +

Infix: 95 - 45
RPN: 95 45 -

Now convert this expression:

Infix: {infix}
RPN: """}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False # Switches between thinking and non-thinking modes. Default is True.
    )
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    prompt_len = inputs["input_ids"].shape[-1]
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.2, # Using a lower temperature for more deterministic outputs
            top_p=0.95,
            num_return_sequences=num_samples
        )
    
    # Decode each generated sequence, skipping the prompt part
    completions = [
        tokenizer.decode(g[prompt_len:], skip_special_tokens=True) for g in outputs
    ]
    
    return completions

# --- Main Execution Block ---

def main(
    n_samples: int = 4,
    max_new_tokens: int = 64,
    dataset_file: str = DATA,
):
    """
    Generates code samples for RPN problems using a specified model.

    :param model_name: The name or local path of the Hugging Face model.
    :param n_samples: The number of samples to generate per problem.
    :param max_new_tokens: The maximum number of new tokens for generation.
    :param problem_file: The path to the MBPP JSONL file.
    """
    print(f"Loading model: {readable_model_name}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="balanced",
        max_memory=max_memory3,
    )
    model.eval()
    print("Model loaded.")
    
    print("Reading problems...")
    dataset = list(stream_jsonl(dataset_file))
    # problems = problems[:10]
    print(f"Found {len(dataset)} problems.")
    
    # Create the output directory if it doesn't exist
    output_dir = "samples"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{readable_model_name}-{n_samples}.jsonl")

    print(f"Generating {n_samples} samples for each of the {len(dataset)} problems...")
    samples = []
    for data in tqdm(dataset, desc="Generating samples"):
        generated_completions = generate_rpn(
            model=model,
            tokenizer=tokenizer,
            infix=data["infix"], 
            num_samples=n_samples,
            max_new_tokens=max_new_tokens
        )
        samples.append(
            dict(task_id=data["id"], completion=generated_completions)
        )

    print(f"Writing {len(samples)} sampled solutions to {output_file}")
    write_jsonl(output_file, samples)
    print("Done.")

if __name__ == "__main__":
    fire.Fire(main)