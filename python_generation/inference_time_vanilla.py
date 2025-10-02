
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datetime import datetime
from time import perf_counter

max_memory2 = {
    0: "75GB",
    1: "75GB"
}
max_memory3 = {
    0: "10GB",
    1: "10GB",
    2: "10GB"
}
# Tokenizer and model
# model_name = "meta-llama/Llama-2-7b-hf"     #32000 tokens
model_name = "Qwen/Qwen3-1.7B"               #151936 tokens

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16, 
    device_map="balanced", 
    max_memory=max_memory3
)
print("Model loaded")





prompt = "In the programming language Python, a code that do : x=10, y=0, while x>0: y = y + 2, x = x - 1, would be in Python:"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)






# ---- timing starts here ----
if torch.cuda.is_available():
    torch.cuda.synchronize()  # flush any queued CUDA work

t_wall_start = datetime.now()
t_perf_start = perf_counter()

# Generate exactly 30 new tokens (no chat template)
outputs = model.generate(
    **inputs,
    max_new_tokens=100,
    do_sample=True,          # optional; remove for deterministic
    temperature=0.7,         # tweak as you like
    pad_token_id=tokenizer.eos_token_id
)

if torch.cuda.is_available():
    torch.cuda.synchronize()  # ensure sampling is finished before stopping timer

t_perf_end = perf_counter()
t_wall_end = datetime.now()

print(f"[Timing] start: {t_wall_start.strftime('%Y-%m-%d %H:%M:%S')}, "
      f"end: {t_wall_end.strftime('%Y-%m-%d %H:%M:%S')}, "
      f"elapsed: {t_perf_end - t_perf_start:.3f}s")
# ---- timing ends here ----


gen_ids = outputs[0][inputs["input_ids"].shape[1]:]
print(tokenizer.decode(gen_ids, skip_special_tokens=True))



# ----------- Results for Classical generation -----------
# "Qwen/Qwen3-1.7B" for 30 tokens              #151936 vocabulary size
# [Timing] start: 2025-09-11 16:38:13, end: 2025-09-11 16:38:17, elapsed: 4.324s
# Average time per token: 0.043

# "meta-llama/Llama-2-7b-hf" for 30 tokens              #32000 vocabulary size
# [Timing] start: 2025-09-11 16:37:24, end: 2025-09-11 16:37:28, elapsed: 4.044s
# Average time per token: 0.04