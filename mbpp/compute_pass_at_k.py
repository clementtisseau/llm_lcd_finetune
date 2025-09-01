import fire
import sys
import json
from collections import defaultdict
from typing import List, Union, Iterable, Dict
import itertools
import numpy as np

from pathlib import Path

HERE = Path(__file__).resolve().parent 
RESULT = HERE / "samples" / "Qwen3-1.7B-128samples-test-all.jsonl_results.jsonl"

# --- Utility functions for data handling ---

def stream_jsonl(filename: str) -> Iterable[Dict]:
    """Reads a JSONL file and yields each line as a dictionary."""
    with open(filename, "r") as fp:
        for line in fp:
            if any(not x.isspace() for x in line):
                yield json.loads(line)

# --- pass@k estimation logic from human_eval ---

def estimate_pass_at_k(
    num_samples: Union[int, List[int], np.ndarray],
    num_correct: Union[List[int], np.ndarray],
    k: int
) -> np.ndarray:
    """
    Calculates 1 - C(n - c, k) / C(n, k) for each problem.
    """
    def estimator(n: int, c: int, k:int) -> float:
        """
        A single problem estimator.
        """
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

    if isinstance(num_samples, int):
        num_samples_it = itertools.repeat(num_samples, len(num_correct))
    else:
        assert len(num_samples) == len(num_correct)
        num_samples_it = iter(num_samples)

    return np.array([estimator(int(n), int(c), k) for n, c in zip(num_samples_it, num_correct)])

# --- Main calculation function ---

def calculate_pass_at_k(
    result_file: str = RESULT,
    k_values: str = "1,2,4,8,16,32,64,128",
):
    """
    Calculates pass@k from a JSONL file containing execution results.
    
    :param result_file: Path to the .jsonl file with execution results.
                        Each line must have 'task_id' and 'passed' (boolean) fields.
    :param k_values: A comma-separated string of integers for k values (e.g., "1,2,4").
    """
    k_list = list(map(int, k_values.split(',')))

    # Group results by task_id
    results_by_task = defaultdict(list)
    print(f"Reading results from {result_file}...")
    for sample in stream_jsonl(result_file):
        task_id = sample["task_id"]
        passed = sample["passed"]
        results_by_task[task_id].append(passed)

    if not results_by_task:
        print("Error: No data found in the result file.")
        return

    # Calculate total samples (n) and correct samples (c) for each task
    total_samples_per_task = []
    correct_samples_per_task = []
    
    for task_id in sorted(results_by_task.keys()):
        outcomes = results_by_task[task_id]
        total_samples_per_task.append(len(outcomes))
        # In Python, sum([True, False, True]) == 2
        correct_samples_per_task.append(sum(outcomes))

    total_samples_per_task = np.array(total_samples_per_task)
    correct_samples_per_task = np.array(correct_samples_per_task)

    # Calculate pass@k for each k
    pass_at_k_results = {}
    for k in k_list:
        # The .mean() calculates the average pass@k across all tasks
        pass_at_k_results[f"pass@{k}"] = estimate_pass_at_k(
            total_samples_per_task, correct_samples_per_task, k
        ).mean()

    # Print the final results
    print("\n--- pass@k Results ---")
    for key, value in pass_at_k_results.items():
        print(f"{key}: {value:.4f}")
    print("----------------------")
    
    return pass_at_k_results

def main():
    fire.Fire(calculate_pass_at_k)

if __name__ == "__main__":
    sys.exit(main())