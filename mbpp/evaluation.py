import fire
import sys
import json
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Union, Iterable, Dict
import itertools
import numpy as np
from tqdm import tqdm
from pathlib import Path

from execution import check_correctness

HERE = Path(__file__).resolve().parent  # directory containing generate.py
DATA_PROBLEM = HERE / "data" / "mbpp.jsonl"
DATA_EVAL = HERE / "samples" / "Qwen3-1.7B-128samples-test-all.jsonl"

# --- Utility functions for data handling ---

def stream_jsonl(filename: str) -> Iterable[Dict]:
    with open(filename, "r") as fp:
        for line in fp:
            if any(not x.isspace() for x in line):
                yield json.loads(line)

def write_jsonl(filename: str, data: Iterable[Dict]):
    with open(filename, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")

def read_problems(evalset_file: str) -> Dict[int, Dict]:
    """Reads the MBPP dataset into a dictionary mapping task_id to the problem."""
    return {task["task_id"]: task for task in stream_jsonl(evalset_file)}

# --- pass@k estimation logic from human_eval ---

def estimate_pass_at_k(
    num_samples: Union[int, List[int], np.ndarray],
    num_correct: Union[List[int], np.ndarray],
    k: int
) -> np.ndarray:
    def estimator(n: int, c: int, k: int) -> float:
        if n - c < k: return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))
    
    if isinstance(num_samples, int):
        num_samples_it = itertools.repeat(num_samples, len(num_correct))
    else:
        assert len(num_samples) == len(num_correct)
        num_samples_it = iter(num_samples)
    
    return np.array([estimator(int(n), int(c), k) for n, c in zip(num_samples_it, num_correct)])

# --- Main evaluation function ---

def evaluate_functional_correctness(
    sample_file: str = DATA_EVAL,
    k: str = "1,2,4",
    n_workers: int = 8,
    timeout: float = 5.0,
    problem_file: str = DATA_PROBLEM,
):
    """
    Evaluates functional correctness of generated samples and writes results.
    """
    problems = read_problems(problem_file)
    k_list = list(map(int, k.split(',')))

    # Process samples and run tests
    results = defaultdict(list)
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = []
        completion_id_counter = Counter()
        n_samples = 0

        print(f"Reading samples from {sample_file}...")
        for sample in stream_jsonl(sample_file):
            task_id = sample["task_id"]
            completion = sample["completion"]
            problem = problems[task_id]
            
            completion_id = completion_id_counter[task_id]
            args = (problem, completion, timeout, completion_id)
            future = executor.submit(check_correctness, *args)
            futures.append(future)
            
            completion_id_counter[task_id] += 1
            n_samples += 1

        print(f"Running {len(futures)} tests with {n_workers} workers...")
        for future in tqdm(as_completed(futures), total=len(futures)):
            result = future.result()
            task_id = result["task_id"]
            results[task_id].append(result)

    # Calculate pass@k
    total, correct = [], []
    for task_id in results:
        task_results = results[task_id]
        passed = [r["passed"] for r in task_results]
        total.append(len(passed))
        correct.append(sum(passed))

    total = np.array(total)
    correct = np.array(correct)

    pass_at_k = {f"pass@{k}": estimate_pass_at_k(total, correct, k).mean() for k in k_list}
    print("pass@k results:")
    print(pass_at_k)

    # Save detailed results to a new file
    out_file = str(sample_file) + "_results.jsonl"
    print(f"Writing detailed results to {out_file}...")
    
    # Flatten results for writing
    all_results = []
    for res_list in results.values():
        all_results.extend(res_list)
        
    write_jsonl(out_file, all_results)
    
    return pass_at_k

def main():
    fire.Fire(evaluate_functional_correctness)

if __name__ == "__main__":
    sys.exit(main())