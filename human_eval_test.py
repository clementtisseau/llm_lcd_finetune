from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 1. Load your fine-tuned model
model_name = "/scratch/ctisseau/finetuned-models/Llama-2-7b-hf-OCI-test"            #this need to change
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)
model.eval()

#add the generate_regex_code and generate_cfg_code functions
def generate_code(prompt: str, max_new_tokens: int = 256) -> str:
    """
    - prompt: the HumanEval prompt, which already contains the 'def fn(...):' line.
    - returns: the text generated _after_ that prompt, which should complete the function.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=0.2,     # sampling temperature
        top_p=0.95,          # nucleus sampling
        do_sample=True,      # enable sampling
        num_return_sequences=1
    )
    # Remove the prompt from the output and return only the completion:
    generated = outputs[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)

# 3. Plug into eval_pass_at_k
from human_eval.eval import eval_pass_at_k
import json

# Load all problems
with open("human_eval.jsonl") as f:
    problems = [json.loads(line) for line in f]

# Run the evaluation
results = eval_pass_at_k(
    problems,
    generate_code,
    ks=[1, 5, 10],
    timeout=120.0  # seconds per test
)

print("Pass@k results:", results)