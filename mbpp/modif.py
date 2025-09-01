import json
import os
import re  # Import the regular expressions module
from pathlib import Path

def add_python_prompt_prefix_and_suffix(input_file_path, output_file_path):
    """
    Reads a .jsonl file, adds a descriptive prefix (based on the function
    signature in the 'code' field) and a Python code fence hint to the 'text'
    field of each JSON object, and writes the result to a new .jsonl file.

    Args:
        input_file_path (str): The path to the source .jsonl file.
        output_file_path (str): The path for the new, modified .jsonl file.
    """
    # The suffix to append to the 'text' field of each line.
    suffix_to_add = "\n\nDon't write any natural language, write python code only: \n\n```python\n"

    try:
        # Ensure the output directory exists
        output_dir = os.path.dirname(output_file_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # Open the source file for reading and the destination file for writing
        with open(input_file_path, 'r', encoding='utf-8') as infile, \
             open(output_file_path, 'w', encoding='utf-8') as outfile:

            # Iterate over each line in the source file
            for line_num, line in enumerate(infile, 1):
                try:
                    # Remove leading/trailing whitespace and parse the JSON
                    data = json.loads(line.strip())

                    # --- MODIFICATION START ---
                    prefix_to_add = "" # Default to an empty prefix
                    
                    # Check if 'code' key exists and is a string to extract the function signature
                    if 'code' in data and isinstance(data['code'], str):
                        # Use regex to find the function definition, e.g., "def my_func(arg1, arg2):"
                        # The pattern captures the function name and its parameters.
                        match = re.search(r'def (\w+\s*\(.*\))\s*:', data['code'])
                        if match:
                            # match.group(1) contains the captured part: "function_name(params)"
                            function_signature = match.group(1).strip()
                            prefix_to_add = f"The function name is {function_signature}. "
                        else:
                            print(f"Warning: Could not find a function signature in 'code' on line {line_num}.")
                    else:
                         print(f"Warning: 'code' key not found or not a string in line {line_num}.")


                    # Check if the 'text' key exists before modifying
                    if 'text' in data and isinstance(data['text'], str):
                        # Combine the new prefix, the original text, and the suffix
                        data['text'] = prefix_to_add + data['text'] + suffix_to_add
                    else:
                        print(f"Warning: 'text' key not found or not a string in line {line_num}.")
                    # --- MODIFICATION END ---

                    # Convert the modified dictionary back to a JSON string and write it
                    outfile.write(json.dumps(data) + '\n')

                except json.JSONDecodeError:
                    print(f"Warning: Could not decode JSON on line {line_num}. Skipping.")

        print("Processing complete!")
        print(f"Original file: {input_file_path}")
        print(f"Modified file saved to: {output_file_path}")

    except FileNotFoundError:
        print(f"Error: The input file was not found at '{input_file_path}'")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# --- Main execution block ---
if __name__ == "__main__":
    # --- Configuration ---
    # Create a dummy data directory and file for demonstration
    HERE = Path(__file__).resolve().parent
    data_dir = HERE / "data"
    os.makedirs(data_dir, exist_ok=True)
    
    input_path = data_dir / "mbpp_old.jsonl"
    
    # Set the desired path for the new, modified file
    output_path = data_dir / "mbpp_new.jsonl"

    print("Starting the script...")
    # I've updated the function name to be more descriptive
    add_python_prompt_prefix_and_suffix(input_path, output_path)