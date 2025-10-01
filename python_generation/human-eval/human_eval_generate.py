from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 1. Load your fine-tuned model
max_memory3 = {
    0: "40GB",
    1: "40GB",
    2: "40GB"
}
model_name = "/scratch/ctisseau/finetuned-models/Qwen3-1.7B-OCI-e1-ds65536-bs256-ckps64/checkpoint-00000000"            #this need to change
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="balanced",
    max_memory=max_memory3,
)
print("Model loaded.")
model.eval()

#add the generate_regex_code and generate_cfg_code functions
def generate_code(prompt: str, num_samples: int, max_new_tokens: int = 512) -> str: #Always keep max_new_tokens=512 to compare the results
    """
    - prompt: the HumanEval prompt, which already contains the 'def fn(...):' line.
    - returns: the text generated _after_ that prompt, which should complete the function.
    """
    print(f"call to generate_code() for a batch of {num_samples} samples")
    # The tokenizer still processes a single prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # The key change is here: num_return_sequences tells the model
    # to generate 'num_samples' different outputs for the single input.
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=0.2,
        top_p=0.95,
        do_sample=True,
        num_return_sequences=num_samples  # Generate a batch of samples
    )

    # The output is now a batch. We decode each sequence in the batch separately.
    prompt_len = inputs["input_ids"].shape[-1]
    completions = [

        tokenizer.decode(g[prompt_len:], skip_special_tokens=True) for g in outputs
    ]
    
    return completions

#3
from human_eval.data import write_jsonl, read_problems

print("Reading problems...")
problems = read_problems()
print("Problems read.")

num_samples_per_task = 128
samples = [
    dict(task_id=task_id, completion=completion)
    for task_id in problems
    for completion in generate_code(
        prompt=problems[task_id]["prompt"],
        num_samples=num_samples_per_task
    )
]
print("Writing sampled solutions in .jsonl file")
write_jsonl("Qwen3-1.7B-128samples.jsonl", samples)


# To test the samples very fast:
# salloc --ntasks=8 --cpus-per-task=1 --mem-per-cpu=6G --time=01:00:00
# evaluate_functional_correctness humaneval_samples/Qwen3-1.7B-8samples.jsonl --k="1,4,16,64,128"