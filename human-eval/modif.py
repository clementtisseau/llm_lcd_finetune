# import os
# import re
# import json
# import textwrap
# from pathlib import Path

# # Paths (as you specified)
# HERE = Path(__file__).resolve().parent
# data_dir = HERE / "samples"
# os.makedirs(data_dir, exist_ok=True)

# input_path = data_dir / "Qwen3-1.7B-128samples-prompted3.jsonl"
# output_path = data_dir / "Qwen3-1.7B-128samples-prompted3_new4.jsonl"

# # --- Step 1: code fences (leading/trailing) ----------------------------------
# FENCE_START = re.compile(r'^\s*```(?:python|py)?\s*\n?', re.IGNORECASE)
# FENCE_END   = re.compile(r'\n?\s*```\s*$', re.IGNORECASE)

# def strip_fences(s: str) -> str:
#     s = s.strip("\n")
#     s = FENCE_START.sub("", s)
#     s = FENCE_END.sub("", s)
#     return s

# # --- Step 2: remove any import lines (anywhere) ------------------------------
# IMPORT_LINE = re.compile(r'^\s*(?:from\s+\S+\s+import\b|import\b)\s')

# def remove_imports(s: str) -> str:
#     return "\n".join(ln for ln in s.splitlines() if not IMPORT_LINE.match(ln))

# # --- Step 3: remove the first function signature line ------------------------
# # Case A: normal def line
# DEF_SIG = re.compile(r'^\s*def\b[^\n]*:\s*$')
# # Case B: “dangling tail” (no 'def', but ends with ')' and ':' and optional '-> T')
# TAIL_SIG = re.compile(r'^\s*[^#\n]*\)\s*(?:->\s*[^:]+)?\s*:\s*$')

# def remove_first_signature_line(s: str) -> str:
#     lines = s.splitlines()
#     # Remove leading blank lines for robust matching
#     i0 = 0
#     while i0 < len(lines) and not lines[i0].strip():
#         i0 += 1

#     for i in range(i0, len(lines)):
#         ln = lines[i]
#         if DEF_SIG.match(ln) or TAIL_SIG.match(ln):
#             del lines[i]            # drop the entire signature line
#             break

#     # Dedent whatever remains (so body starts at col 0 if it was indented)
#     body = "\n".join(lines)
#     body = textwrap.dedent(body)
#     return body.strip("\n")

# # --- Glue --------------------------------------------------------------------
# def clean_completion(raw: str) -> str:
#     if not raw:
#         return ""
#     s = strip_fences(raw)
#     s = remove_imports(s)
#     s = remove_first_signature_line(s)
#     return s

# def main():
#     processed = 0
#     with open(input_path, "r", encoding="utf-8") as fin, \
#          open(output_path, "w", encoding="utf-8") as fout:
#         for line_no, line in enumerate(fin, 1):
#             if not line.strip():
#                 continue
#             try:
#                 obj = json.loads(line)
#             except json.JSONDecodeError:
#                 print(f"[warn] Skipping invalid JSON on line {line_no}")
#                 continue

#             comp = obj.get("completion", "")
#             obj["completion"] = clean_completion(comp)
#             fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
#             processed += 1

#     print(f"Done. Wrote {processed} cleaned records to {output_path}")

# if __name__ == "__main__":
#     main()


import os
import re
import json
from pathlib import Path

# Paths (as you specified)
HERE = Path(__file__).resolve().parent
data_dir = HERE / "samples"
os.makedirs(data_dir, exist_ok=True)

input_path = data_dir / "checkpoint-00000002-128samples-prompted-problemsALL.jsonl"
output_path = data_dir / "checkpoint-00000002-128samples-prompted-problemsALL-add_indent.jsonl"
# input_path = data_dir / "Qwen3-30B-A3B-128samples-20problems.jsonl"
# output_path = data_dir / "Qwen3-30B-A3B-128samples-20problems-add_indent.jsonl"

# --- Only add indentation ----------------------------------------------------
# Add 4 spaces at the start of lines that don't already start with whitespace.
INDENT = "    "
ADD_INDENT_IF_NONE = re.compile(r"(?m)^(?=\S)")  # line start followed by non-whitespace

def add_indent(raw: str) -> str:
    if not raw:
        return ""
    return ADD_INDENT_IF_NONE.sub(INDENT, raw)

# --- Glue --------------------------------------------------------------------
def clean_completion(raw: str) -> str:
    # Now this only adds indentation; no other modifications.
    return add_indent(raw)

def main():
    processed = 0
    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        for line_no, line in enumerate(fin, 1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                print(f"[warn] Skipping invalid JSON on line {line_no}")
                continue

            comp = obj.get("completion", "")
            obj["completion"] = clean_completion(comp)
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            processed += 1

    print(f"Done. Wrote {processed} cleaned records to {output_path}")

if __name__ == "__main__":
    main()