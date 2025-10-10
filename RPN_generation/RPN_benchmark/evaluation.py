# The evaluation should be in 3 parts. 
# First we evaluate if the output respect the regex (for constrained model it will 100% of the time)
# Among answers that respect the regex, we check if it respects the syntax (must contain one more digit than operator)
# Among answers that respect the syntax, we evaluate the value and see if it is equal (with some error because of * / ^) to the true answer

#!/usr/bin/env python3
import sys
import json
import math
import itertools
from collections import defaultdict, Counter
from typing import Iterable, Dict, List, Union

import fire
import numpy as np
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "dataset1000.jsonl"
SAMPLES = HERE / "samples" / "data1000" / "Qwen3-1.7B-d1000-n128-t1-p08.jsonl"

# --- I/O helpers ---
def stream_jsonl(filename: Path) -> Iterable[Dict]:
    with open(filename, "r", encoding="utf-8") as fp:
        for line in fp:
            if any(not x.isspace() for x in line):
                yield json.loads(line)

def write_jsonl(filename: Path, data: Iterable[Dict]):
    with open(filename, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

def read_dataset(dataset_file: Path) -> Dict[str, Dict]:
    return {ex["id"]: ex for ex in stream_jsonl(dataset_file)}

# --- RPN evaluation ---
def eval_rpn(expr: str) -> float:
    if not expr:
        raise ValueError("Empty RPN expression")
    tokens = expr.strip().split()
    stack: List[float] = []
    ops = {"+", "-", "*", "/"}
    for t in tokens:
        if t in ops:
            if len(stack) < 2:
                raise ValueError(f"Not enough operands for '{t}'")
            b = stack.pop()
            a = stack.pop()
            if t == "+": stack.append(a + b)
            elif t == "-": stack.append(a - b)
            elif t == "*": stack.append(a * b)
            else:
                if b == 0:
                    stack.append(math.inf if a > 0 else (-math.inf if a < 0 else math.nan))
                else:
                    stack.append(a / b)
        else:
            try:
                stack.append(float(t))
            except ValueError:
                stack.append(float(t.replace("−", "-")))
    if len(stack) != 1:
        raise ValueError("Malformed RPN: stack has multiple values at end")
    return stack[0]

# --- pass@k ---
def estimate_pass_at_k(
    num_samples: Union[int, List[int], np.ndarray],
    num_correct: Union[List[int], np.ndarray],
    k: int
) -> np.ndarray:
    def estimator(n: int, c: int, k: int) -> float:
        if n <= 0: return 0.0
        if n - c < k: return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))
    if isinstance(num_samples, int):
        num_samples_it = itertools.repeat(num_samples, len(num_correct))
    else:
        assert len(num_samples) == len(num_correct)
        num_samples_it = iter(num_samples)
    return np.array([estimator(int(n), int(c), k) for n, c in zip(num_samples_it, num_correct)])

def close_enough(pred: float, truth: float, atol: float, rtol: float) -> bool:
    if math.isnan(pred) or math.isnan(truth):
        return False
    return abs(pred - truth) <= (atol + rtol * abs(truth))

# --- Main ---
def evaluate_rpn_correctness(
    sample_file: str = str(SAMPLES),
    dataset_file: str = str(DATA),
    k: str = "1,2,4,8,16,32,64,128",
    atol: float = 1e-6,
    rtol: float = 1e-6,
):
    """
    Evaluation of RPN completions vs ground-truth values.
    """
    dataset = read_dataset(Path(dataset_file))
    k_list = list(map(int, k.split(",")))

    results: Dict[str, List[Dict]] = defaultdict(list)
    completion_id_counter = Counter()

    print(f"Reading and evaluating samples from {sample_file}...")
    for sample in stream_jsonl(Path(sample_file)):
        task_id = sample["task_id"]
        completion = sample["completion"]
        comp_id = completion_id_counter[task_id]
        completion_id_counter[task_id] += 1

        if task_id not in dataset:
            results[task_id].append({
                "task_id": task_id,
                "completion_id": comp_id,
                "completion": completion,
                "pred_value": None,
                "true_value": None,
                "abs_err": None,
                "passed": False,
                "error": f"Unknown task_id '{task_id}'",
            })
            continue

        truth_value = float(dataset[task_id]["value"])
        try:
            pred_val = eval_rpn(completion)
            passed = close_enough(pred_val, truth_value, atol=atol, rtol=rtol)
            err = None
        except Exception as e:
            pred_val = None
            passed = False
            err = f"{type(e).__name__}: {e}"

        results[task_id].append({
            "task_id": task_id,
            "completion_id": comp_id,
            "completion": completion,
            "pred_value": pred_val,
            "true_value": truth_value,
            "abs_err": (None if pred_val is None else abs(pred_val - truth_value)),
            "passed": bool(passed),
            "error": err,
        })

    # Compute pass@k
    total, correct = [], []
    for task_id, task_results in results.items():
        flags = [r["passed"] for r in task_results]
        total.append(len(flags))
        correct.append(sum(flags))
    total = np.array(total if total else [0])
    correct = np.array(correct if correct else [0])

    pass_at_k = {f"pass@{kk}": float(estimate_pass_at_k(total, correct, kk).mean()) for kk in k_list}
    print("pass@k results:")
    print(pass_at_k)

    # Save detailed results
    out_file = Path(sample_file).with_suffix(Path(sample_file).suffix + "_results.jsonl")
    all_results = list(itertools.chain.from_iterable(results.values()))
    write_jsonl(out_file, all_results)
    print(f"Wrote detailed results to: {out_file}")

    return pass_at_k

def main():
    fire.Fire(evaluate_rpn_correctness)

if __name__ == "__main__":
    sys.exit(main())
