import json
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from pathlib import Path

max_memory3 = {
    0: "7GB",
    1: "7GB",
    2: "7GB"
}


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
def generate_constrained_rpn(model, tokenizer, clm_sampler, infix: str, num_samples: int, max_new_tokens: int, temperature = 1.0, top_k = 20, top_p = 0.95, few_shot=1) -> list[str]:
    """
    Generates a batch of rpn expressions for a given infix, using a constraint.

    Args:
        model: The loaded Hugging Face model.
        tokenizer: The loaded Hugging Face tokenizer.
        clm_sampler: The chosen sampler (Multinomial, SMC) chosen from ConstraintLM.
        infix: The infix notation to prompt the model with.
        num_samples: The number of completions to generate.
        max_new_tokens: The maximum number of tokens for each completion.

    Returns:
        A list of generated rpn expressions.
    """
    if few_shot == 1:
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
RPN:"""}
        ]
    elif few_shot==0:
        messages = [
            {"role": "system", "content": "You are an expert at converting arithmetic expressions into Reverse Polish Notation (RPN). You always output only the RPN expression, with tokens separated by a single space. Do not include explanations or extra text."},
            {"role": "user", "content": f"""Infix: {infix}
RPN:"""}
        ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False # Switches between thinking and non-thinking modes. Default is True.
    )
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    input_ids = inputs["input_ids"]
    attn_mask = inputs.get("attention_mask", None)
    
    outputs = clm_sampler.sample(input_ids, attention_mask=attn_mask, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k, top_p=top_p, num_return_sequences=num_samples)
    # The Multinomial Sampler returns only generated tokens, contrary to transformers.generate()
    
    # Decode each generated sequence, skipping the prompt part
    completions = [
        tokenizer.decode(g.tolist(), skip_special_tokens=True) for g in outputs
    ]
    
    return completions

# --- Main Execution Block ---

import constraintlm as clm
import outlines
from outlines.processors import RegexLogitsProcessor
# We will use this regex to express RPN expressions. It doesn't enforce syntactic validit: "4 3 + -", "+ - * /" respect the regex.
# (?:\d+|[+\-*/])(?: (?:\d+|[+\-*/]))*    a single space *between* two digits/operators. 

def main(
    model_name,
    readable_model_name,
    dataset_file: str,
    n_samples: int = 128,
    max_new_tokens: int = 64,
    temperature = 1.0, 
    top_k = None, 
    top_p = None,
    few_shot=1,
):
    """
    Generates code samples for RPN problems using a specified model.

    :param model_name: The name or local path of the Hugging Face model.
    :param n_samples: The number of samples to generate per problem.
    :param max_new_tokens: The maximum number of new tokens for generation.
    :param problem_file: The path to the MBPP JSONL file.
    """    
    print(f"Loading CLM model: {readable_model_name}...")
    clm_model = clm.TransformersLM(
        model_name,
        torch_dtype=torch.float16,
        device_map="balanced",
        max_memory=max_memory3,
    )
    tokenizer = clm_model.tokenizer
    model = clm_model.model
    model.eval()
    print("CLM model loaded.")
    print("Creating Logits Processor and Sequence Sampler...")
    outlines_model = outlines.from_transformers(clm_model.model, clm_model.tokenizer)
    rpn_logits_processor = RegexLogitsProcessor(
        r"(?:\d+|[+\-*/])(?: (?:\d+|[+\-*/]))*",
        outlines_model.tokenizer,
        outlines_model.tensor_library_name,
    )
    rpn_multinomial_sampler = clm.MultinomialSeqSampler(clm_model, logits_processor=rpn_logits_processor)
    print("Logits Processor and Sequence Sampler created.")
    
    print("Reading problems...")
    dataset = list(stream_jsonl(dataset_file))
    print(f"Found {len(dataset)} problems.")
    
    # Create the output directory if it doesn't exist
    output_dir = "samples"
    os.makedirs(output_dir, exist_ok=True)
    if few_shot == 1:
        output_file = os.path.join(output_dir, f"{readable_model_name}-lcdgen-n{n_samples}-t{f'{temperature:g}'.replace('.', '_')}-p{f'{top_p:g}'.replace('.', '_')}-k{top_k}.jsonl")     # lcd: locally constrained decoding, t1: temperature=1, p08: top-p=0.8
    elif few_shot == 0:
        output_file = os.path.join(output_dir, f"{readable_model_name}-nofewshot-lcdgen-n{n_samples}-t{f'{temperature:g}'.replace('.', '_')}-p{f'{top_p:g}'.replace('.', '_')}-k{top_k}.jsonl")     # lcd: locally constrained decoding, t1: temperature=1, p08: top-p=0.8
    else:
        print("--few_shot should be 0 or 1")
    print(f"Generating {n_samples} samples with constraint for each of the {len(dataset)} problems...")
    samples = []
    for data in tqdm(dataset, desc="Generating samples"):
        generated_completions = generate_constrained_rpn(
            model=model,
            tokenizer=tokenizer,
            clm_sampler=rpn_multinomial_sampler,
            infix=data["infix"], 
            num_samples=n_samples,
            max_new_tokens=max_new_tokens, 
            temperature = temperature, 
            top_k = top_k, 
            top_p = top_p,
            few_shot=few_shot,
        )
        for completion in generated_completions:
            samples.append(
                dict(task_id=data["id"], completion=completion)
            )

    print(f"Writing {len(samples)} sampled solutions to {output_file}")
    write_jsonl(output_file, samples)
    print("Done.")

if __name__ == "__main__":
    import argparse

    HERE = Path(__file__).resolve().parent  # directory containing generate.py
    DATA = HERE / "data" / "dataset900.jsonl"

    parser = argparse.ArgumentParser(description="Fine-tune a LLM.")

    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--readable_model_name", type=str, default="Qwen3-1.7B")
    parser.add_argument("--dataset_file", type=str, default=DATA)
    parser.add_argument("--n_samples", type=int, default=128)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--top_p", type=float, default=1.0)    # 0.95
    parser.add_argument("--few_shot", type=int, default=1) 
    args = parser.parse_args()

    # model_name = "/scratch/ctisseau/finetuned-models/Qwen3-1.7B-RPN-ds1024-e2-ds1024-bs32/checkpoint-00000032"
    # model_name = "/scratch/ctisseau/finetuned-models/Qwen-Qwen3-1.7B_classical_sft_7c8926f8324c983f7990/checkpoint-00000128"
    # model_name = "/scratch/ctisseau/finetuned-models/Qwen-Qwen3-1.7B_lcd_sft_9fdbc593c01affa7ad7a/checkpoint-00000128"
    # model_name = "Qwen/Qwen3-1.7B"

    # readable_model_name = "Qwen3-1.7B"
    # readable_model_name = "Qwen3-1.7B_classical_sft_7c8926f8324c983f7990-ckpt128"
    # readable_model_name = "Qwen3-1.7B_lcd_sft_9fdbc593c01affa7ad7a-ckpt128"

    main(**vars(args))