import outlines

import torch
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


# Create the CFGLogitsProcessor (we need to initialize an outlines model's tokenizer)
model_outlines = outlines.from_transformers(
    AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="balanced", max_memory=max_memory3),
    AutoTokenizer.from_pretrained(model_name)
)
print("Outlines model loaded")

from outlines.types import CFG
from outlines import Generator

generator = Generator(model_outlines, CFG(python_grammar))
print("CFG Loaded", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))


# ---- timing starts here ----
if torch.cuda.is_available():
    torch.cuda.synchronize()  # flush any queued CUDA work

t_wall_start = datetime.now()
t_perf_start = perf_counter()

sequence = generator("In the programming language Python, a code that do : x=10, y=0, while x>0: y = y + 2, x = x - 1, would be in Python:", max_new_tokens=30)

if torch.cuda.is_available():
    torch.cuda.synchronize()  # ensure sampling is finished before stopping timer

t_perf_end = perf_counter()
t_wall_end = datetime.now()

print(f"[Timing] start: {t_wall_start.strftime('%Y-%m-%d %H:%M:%S')}, "
      f"end: {t_wall_end.strftime('%Y-%m-%d %H:%M:%S')}, "
      f"elapsed: {t_perf_end - t_perf_start:.3f}s")
# ---- timing ends here ----


print(sequence)




# ----------- Results fo Constrained generation -----------
# "Qwen/Qwen3-1.7B" for 30 tokens              #151936 vocabulary size
# CFG Loaded 2025-09-11 16:13:56
# [Timing] start: 2025-09-11 16:13:56, end: 2025-09-11 16:14:08, elapsed: 11.796s
# Average time per token: 0.3932 s/token
# Average time per token / vocabulary_size: 2.587932e-06

# "meta-llama/Llama-2-7b-hf" for 30 tokens              #32000 vocabulary size
# CFG Loaded 2025-09-11 16:15:15
# [Timing] start: 2025-09-11 16:15:15, end: 2025-09-11 16:15:19, elapsed: 3.802s
# Average time per token: 0.1267 s/token
# Average time per token / vocabulary_size: 3.960417e-06
