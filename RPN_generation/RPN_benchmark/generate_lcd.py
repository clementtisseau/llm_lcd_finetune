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

# model_name = "Qwen/Qwen3-1.7B"
# model_name = "/scratch/ctisseau/finetuned-models/Qwen3-1.7B-lcd-RPN-ds1024-e2-ds1024-bs32/checkpoint-00000032"
model_name = "/scratch/ctisseau/finetuned-models/Qwen3-1.7B-lcd-RPN-ds1024-e2-ds1024-bs32-testdeletelater/checkpoint-00000032"
readable_model_name = "Qwen3-1.7B-testdeletelater-dtrain1024-ckpt32-lcd"

HERE = Path(__file__).resolve().parent  # directory containing generate.py
DATA = HERE / "data" / "dataset1000.jsonl"
# DATA = HERE / "data" / "dataset1000.jsonl"

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
def generate_constrained_rpn(model, tokenizer, clm_sampler, infix: str, num_samples: int, max_new_tokens: int, temperature = 1.0, top_k = None, top_p = None) -> list[str]:
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
    input_ids = inputs["input_ids"]
    attn_mask = inputs.get("attention_mask", None)
    
    outputs = clm_sampler.sample(input_ids, attention_mask=attn_mask, max_new_tokens=max_new_tokens, top_p=0.8, temperature=1, num_return_sequences=num_samples)
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
    n_samples: int = 128,
    max_new_tokens: int = 64,
    dataset_file: str = DATA,
    temperature = 1.0, 
    top_k = None, 
    top_p = None,
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
    output_file = os.path.join(output_dir, f"{readable_model_name}-n{n_samples}-t1.jsonl")     # lcd: locally constrained decoding, t1: temperature=1, p08: top-p=0.8

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
        )
        for completion in generated_completions:
            samples.append(
                dict(task_id=data["id"], completion=completion)
            )

    print(f"Writing {len(samples)} sampled solutions to {output_file}")
    write_jsonl(output_file, samples)
    print("Done.")

if __name__ == "__main__":
    fire.Fire(main)