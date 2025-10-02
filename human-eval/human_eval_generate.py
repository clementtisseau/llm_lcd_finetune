# from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
import torch
import argparse

# 1. Load your fine-tuned model
max_memory2 = {
    0: "75GB",
    1: "75GB"
}
max_memory3 = {
    0: "40GB",
    1: "40GB",
    2: "40GB"
}
# model_name = "/scratch/ctisseau/finetuned-models/Qwen3-1.7B-OCI-e1-ds65536-bs256-ckps64/checkpoint-00000000"            #this need to change
# model_name = "/scratch/ctisseau/finetuned-models/Qwen3-1.7B-OCI-testvanillamodel-M4Max/checkpoint-00000002"
model_name = "Qwen/Qwen3-30B-A3B"

parser = argparse.ArgumentParser(add_help=True)
parser.add_argument("--model_name", type=str, default=model_name, help="HF model ID or local path")
parser.add_argument("--subset_start", type=int, default=None, help="Start index (inclusive) of problems subset. Default: None (start at 0).")
parser.add_argument("--subset_end", type=int, default=None, help="End index (exclusive) of problems subset. Default: None (use all to the end).")
parser.add_argument("--max_memory", type=int, default=2, help="Should be 2 or 3, meaning max_memory2 or max_memory3")
args, _ = parser.parse_known_args()
model_name = args.model_name
if args.max_memory == 2:
    max_memory=max_memory2
else:
    max_memory=max_memory3

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="balanced",
    max_memory=max_memory,
)
print("Model loaded.")
model.eval()

#add the generate_regex_code and generate_cfg_code functions
def generate_code(prompt: str, num_samples: int, max_new_tokens: int = 512) -> list[str]:  # keep default=512
    """
    - prompt: the HumanEval prompt, which already contains the 'def fn(...):' line.
    - returns: a list of texts generated after that prompt (function bodies / completion).
    """
    print(f"call to generate_code() for a batch of {num_samples} samples")

    # Wrap the raw HumanEval prompt in the same chat format you use for MBPP
    # messages = [
    #         {"role": "system",
    #         "content": ("Return only Python code for the function body. "
    #                     "No explanations, no markdown fences, no re-defining the function, "
    #                     "start directly after the docstring with a newline and an identation.")},
    #         {"role": "user", "content": prompt},
    #     ]
    # 
    # # Apply chat template and disable "thinking" mode
    # text = tokenizer.apply_chat_template(
    #     messages,
    #     tokenize=False,
    #     add_generation_prompt=True,
    #     enable_thinking=False  
    # )
    # 
    # inputs = tokenizer([text], return_tensors="pt").to(model.device)

    inst = (
        "You are an expert Python developer.\n"
        "Task: complete the function described below.\n"
        "Output ONLY valid Python code (no explanations, tests, or markdown fences).\n\n"
    )
    full_prompt = inst + prompt
    inputs = tokenizer(
        full_prompt,
        return_tensors="pt",
        add_special_tokens=False  # important for raw continuation
    ).to(model.device)

    prompt_len = inputs["input_ids"].shape[-1]

    with torch.no_grad():
        # outputs = model.generate(
        #     **inputs,
        #     max_new_tokens=max_new_tokens,
        #     do_sample=True,
        #     temperature=0.2,
        #     top_p=0.95,
        #     num_return_sequences=num_samples,
        #     stopping_criteria=StoppingCriteriaList([StopOnSubstrings(stops, tokenizer)])
        # )
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,         
            temperature=0.2,
            top_p=0.95,
            num_return_sequences=num_samples,
        )


    completions = [
        tokenizer.decode(g[prompt_len:], skip_special_tokens=True) for g in outputs
    ]

    return completions

#3
from human_eval.data import write_jsonl, read_problems

print("Reading problems...")
if args.subset_start is not None or args.subset_end is not None:
    _all = read_problems()
    _all_items = list(_all.items())
    _start = 0 if args.subset_start is None else max(0, args.subset_start)
    _end = len(_all_items) if args.subset_end is None else min(len(_all_items), args.subset_end)
    problems = dict(_all_items[_start:_end])
    print(f"Using subset of problems: [{_start}:{_end}) out of {len(_all_items)}")
else:
    # If both are None, select ALL problems (ignore the fixed [:20] preview above)
    problems = read_problems()
    print(f"Using all problems: {len(problems)}")
print("Problems read.")

# The instruction prefix purpose is to avoid repetitions in the LLM answers.
# instruction_prefix = "You are a Python programmer. Your task is to complete the function below. Provide only the code for the function body, starting from the first line of implementation after the docstring.\n\n"
num_samples_per_task = 128
samples = []
for task_id in problems:
    # original_prompt = problems[task_id]["prompt"]
    # full_prompt = f"{instruction_prefix}{original_prompt}"
    prompt = problems[task_id]["prompt"]
    
    completions = generate_code(
        prompt=prompt, 
        num_samples=num_samples_per_task
    )
    
    for completion in completions:

    #     cleaned_completion = completion.strip()
        
    #     if cleaned_completion.startswith("```python"):
    #         cleaned_completion = cleaned_completion[len("```python"):].lstrip()
    #     elif cleaned_completion.startswith("```"):
    #         cleaned_completion = cleaned_completion[len("```"):].lstrip()
            
    #     if cleaned_completion.endswith("```"):
    #         cleaned_completion = cleaned_completion[:-len("```")].rstrip()
    #    samples.append(dict(task_id=task_id, completion=completion))
        
        samples.append(dict(task_id=task_id, completion=completion))



from pathlib import Path
import re

# Make model_name safe for filenames (drops directories and removes odd chars)
safe_model = re.sub(r'[^A-Za-z0-9._-]+', '_', Path(model_name).name)

# Build a readable subset tag
if args.subset_start is None and args.subset_end is None:
    subset_tag = "ALL"
else:
    start_tag = 0 if args.subset_start is None else args.subset_start
    end_tag = "END" if args.subset_end is None else args.subset_end
    subset_tag = f"{start_tag}-{end_tag}"

print("Writing sampled solutions in .jsonl file")
write_jsonl(f"{safe_model}-128samples-prompted-problems{subset_tag}.jsonl", samples)


# To test the samples very fast:
# salloc --ntasks=8 --cpus-per-task=1 --mem-per-cpu=6G --time=01:00:00
# evaluate_functional_correctness samples/Qwen3-30B-A3B-128samples-20problems.jsonl --k="1,2,4,8,16,32,64,128"
# evaluate_functional_correctness samples/checkpoint-00000002-128samples-prompted-problemsALL.jsonl --k="1,2,4,8,16,32,64,128"