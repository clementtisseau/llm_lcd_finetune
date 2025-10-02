import constraintlm as clm
import outlines

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from datetime import datetime
from time import perf_counter

from pathlib import Path
# Grammar
python_grammar = Path("python2.lark").read_text(encoding="utf-8")

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
model_name = "meta-llama/Llama-2-7b-hf"     #32000 tokens
# model_name = "Qwen/Qwen2.5-0.5B"               #151936 tokens
# model_name = "Qwen/Qwen3-1.7B"               #151936 tokens

# tokenizer = AutoTokenizer.from_pretrained(model_name)
# if tokenizer.pad_token is None:
#     tokenizer.pad_token = tokenizer.eos_token
# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     torch_dtype=torch.float16,
#     device_map="balanced",
#     max_memory=max_memory3,
# )
clmmodel = clm.TransformersLM(model_name, torch_dtype=torch.float16, device_map="balanced", max_memory=max_memory3)
print("Tokenizer and Model loaded")


# Create the CFGLogitsProcessor (we need to initialize an outlines model's tokenizer)
model_outlines = outlines.from_transformers(
    AutoModelForCausalLM.from_pretrained(model_name, device_map="cpu"),
    AutoTokenizer.from_pretrained(model_name)
)
#model_outlines = outlines.models.transformers(model_name)
print("Outlines model loaded")
cfg = clm.CLMCFGLogitsProcessor(python_grammar, model_outlines.tokenizer, tensor_library_name='torch')
print("CFG loaded", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

cfg_multinomial = clm.MultinomialSeqSampler(clmmodel, logits_processor=cfg)
prompt = ["In the programming language Python, a code that do : x=10, y=0, while x>0: y = y + 2, x = x - 1, would be in Python:"]
batch = clmmodel.tokenizer(prompt, padding=True, return_tensors="pt")

# ---- timing starts here ----
if torch.cuda.is_available():
    torch.cuda.synchronize()  # flush any queued CUDA work

t_wall_start = datetime.now()
t_perf_start = perf_counter()

cons_generated_token_ids = cfg_multinomial.sample(batch.input_ids, max_length=30, top_k=5)

if torch.cuda.is_available():
    torch.cuda.synchronize()  # ensure sampling is finished before stopping timer

t_perf_end = perf_counter()
t_wall_end = datetime.now()

print(f"[Timing] start: {t_wall_start.strftime('%Y-%m-%d %H:%M:%S')}, "
      f"end: {t_wall_end.strftime('%Y-%m-%d %H:%M:%S')}, "
      f"elapsed: {t_perf_end - t_perf_start:.3f}s")
# ---- timing ends here ----


print(clmmodel.tokenizer.batch_decode(torch.cat([batch.input_ids, cons_generated_token_ids], dim=-1)))







# ----------- Results fo Constrained generation -----------
# "Qwen/Qwen3-1.7B" for 30 tokens              #151936 vocabulary size
# CFG loaded 2025-09-11 15:52:32
# [Timing] start: 2025-09-11 15:52:32, end: 2025-09-11 16:00:57, elapsed: 504.682s
# Average time per token: 16.8227 s/token
# Average time per token / vocabulary_size: 1.107225e-04

# "meta-llama/Llama-2-7b-hf" for 30 tokens              #32000 vocabulary size
# CFG loaded 2025-09-11 16:05:03
# [Timing] start: 2025-09-11 16:05:03, end: 2025-09-11 16:06:16, elapsed: 72.831s
# Average time per token: 2.4277 s/token
# Average time per token / vocabulary_size: 7.586563e-05
