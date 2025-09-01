import json
from pathlib import Path
from collections import Counter

def analyze_error_distribution(min_length=None):
    """
    Reads a JSONL file containing sample results, counts the distribution of
    error types, computes the average completion length, and prints the analysis.

    Args:
        min_length (int, optional): If specified, only considers lines where
                                    the character count of "completion" is
                                    greater than or equal to this value.
                                    Defaults to None.
    """
    try:
        # Define the path to the data file relative to the script's location.
        here = Path(__file__).resolve().parent
        # data_file = here / "mbpp" / "samples" / "Qwen3-1.7B-128samples-test-all.jsonl_results.jsonl"
        data_file = here / "human-eval" / "samples" / "Qwen3-1.7B-128samples.jsonl_results-1-128.jsonl"

        # Check if the file exists before attempting to open it.
        if not data_file.exists():
            print(f"Error: The file was not found at the expected path: {data_file}")
            print("Please ensure the script is in the correct directory and the file exists.")
            # Create a dummy file for demonstration purposes if it doesn't exist.
            print("Creating a dummy file for demonstration purposes...")
            data_file.parent.mkdir(exist_ok=True, parents=True)
            with open(data_file, 'w') as f:
                f.write('{"task_id": 1, "passed": false, "result": "failed", "error_type": "Runtime Error", "completion_id": 0, "completion": "def example(): return 1/0"}\n')
                f.write('{"task_id": 1, "passed": true, "result": "passed", "error_type": null, "completion_id": 1, "completion": "def example(): return 42"}\n')
                f.write('{"task_id": 1, "passed": true, "result": "passed", "error_type": null, "completion_id": 2, "completion": "def example(): pass"}\n')
                f.write('{"task_id": 2, "passed": false, "result": "failed", "error_type": "Syntax Error", "completion_id": 0, "completion": "def example() return"}\n')
                f.write('{"task_id": 2, "passed": false, "result": "failed", "error_type": "Runtime Error", "completion_id": 1, "completion": "def example():\\n  a = []\\n  return a[0]"}\n')
            print(f"Dummy file created at: {data_file}")

        error_counts = Counter()
        total_samples = 0
        total_completion_chars = 0
        completion_count = 0

        # Open the file and process it line by line.
        with open(data_file, 'r', encoding='utf-8') as f:
            for line in f:
                # Ensure the line is not empty before parsing.
                if line.strip():
                    try:
                        data = json.loads(line)
                        completion = data.get("completion")

                        # ⬇️ --- NEW: Filter by completion length --- ⬇️
                        if min_length is not None and isinstance(completion, str):
                            if len(completion) < min_length:
                                continue  # Skip this sample if it's too short.

                        # The 'error_type' can be null, which json.loads converts to None.
                        error_type = data.get("error_type")
                        error_counts[error_type] += 1
                        total_samples += 1

                        # Calculate the length of the completion string.
                        if isinstance(completion, str):
                            total_completion_chars += len(completion)
                            completion_count += 1

                    except json.JSONDecodeError:
                        print(f"Warning: Skipping a malformed JSON line: {line.strip()}")
        
        if total_samples == 0:
            print("The file is empty or no samples matched the specified min_length. No analysis to perform.")
            return

        # --- Calculations ---

        # 1. Average number of characters in the "completion" key.
        average_completion_length = 0
        if completion_count > 0:
            average_completion_length = total_completion_chars / completion_count

        # 2. Raw counts of each error type.
        final_counts = {str(k) if k is not None else "Success (null)": v for k, v in error_counts.items()}
        
        # 3. Percentages including successes.
        percentages_all = {
            error_type: (count / total_samples) * 100 
            for error_type, count in final_counts.items()
        }

        # 4. Percentages excluding successes.
        num_successes = error_counts.get(None, 0)
        total_failures = total_samples - num_successes
        
        percentages_failures_only = {}
        if total_failures > 0:
            percentages_failures_only = {
                error_type: (count / total_failures) * 100
                for error_type, count in final_counts.items()
                if error_type != "Success (null)"
            }

        # --- Output ---

        print("\n--- Analysis Results ---")
        if min_length is not None:
            print(f"Filter applied: Showing results for completions with >= {min_length} characters.")
        
        print(f"\nAverage Completion Length: {average_completion_length:.2f} characters")

        print("\n1. Distribution of All Result Types (Raw Counts):")
        print(final_counts)

        print("\n2. Distribution of All Result Types (Percentages):")
        print({k: f"{v:.2f}%" for k, v in percentages_all.items()})

        if total_failures > 0:
            print("\n3. Distribution of Error Types Only (Percentages of Failures):")
            print({k: f"{v:.2f}%" for k, v in percentages_failures_only.items()})
        else:
            print("\nCongratulations! No failures were found in the dataset.")
            
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    print("--- Running analysis on all samples ---")
    analyze_error_distribution()

    print("\n" + "="*60 + "\n")

    print("--- Running analysis for completions with at least min_length characters ---")
    analyze_error_distribution(min_length=1500)