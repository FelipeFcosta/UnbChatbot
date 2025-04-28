#!/usr/bin/env python3
"""
Converts a DPO dataset from a simple question/preferred/rejected JSON list
format to the JSONL format expected by the training script (with message lists).

Input JSON format (list of dicts):
[
  {
    "question": "User question string",
    "rejected": "Rejected answer string",
    "preferred": "Preferred answer string",
    "sft_temperature": 0.5,
    "hallucination_score": 1.27,
    "original_qa_hash": "some_hash", # This will be excluded
    "item_hash": "another_hash"
  },
  ...
]

Output JSONL format (one JSON object per line):
{"chosen": [{"role": "user", "content": "..."}, {"role": "model", "content": "..."}], "rejected": [{"role": "user", "content": "..."}, {"role": "model", "content": "..."}], "sft_temperature": 0.5, "hallucination_score": 1.27, "item_hash": "..."}
{"chosen": [...], "rejected": [...], ...}
...
"""

import argparse
import json
import os

def main():
    parser = argparse.ArgumentParser(
        description="Convert raw DPO JSON data to JSONL format suitable for Unsloth DPO training."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to the input JSON file (e.g., dpo_dataset.json)"
    )
    parser.add_argument(
        "--output",
        default="dpo_data",
        help="Base name for the output JSONL file (e.g., 'dpo_data' -> 'dpo_data_train.jsonl')"
    )
    # Add argument for split type if you plan to process train/eval separately later
    parser.add_argument(
        "--split",
        default="train",
        help="Specify the split type (train, validation, test) for the output filename."
    )
    args = parser.parse_args()

    input_file = args.input
    output_basename = args.output
    split_name = args.split
    output_file = f"{output_basename}_{split_name}.jsonl"

    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")

    processed_count = 0
    skipped_count = 0

    try:
        with open(input_file, 'r', encoding='utf-8') as infile, \
             open(output_file, 'w', encoding='utf-8') as outfile:

            # Load the entire JSON list from the input file
            data = json.load(infile)

            if not isinstance(data, list):
                raise ValueError("Input JSON is not a list of objects.")

            print(f"Loaded {len(data)} items from {input_file}.")

            for item in data:
                # Validate required keys exist
                if not all(k in item for k in ["question", "preferred", "rejected"]):
                    print(f"Warning: Skipping item due to missing key (question/preferred/rejected): {item.get('item_hash', 'N/A')}")
                    skipped_count += 1
                    continue

                question = item["question"]
                preferred_answer = item["preferred"]
                rejected_answer = item["rejected"]

                # Create the structured format for chosen/rejected
                chosen_messages = [
                    {"role": "user", "content": question},
                    {"role": "model", "content": preferred_answer}
                ]
                rejected_messages = [
                    {"role": "user", "content": question},
                    {"role": "model", "content": rejected_answer}
                ]

                # Create the output dictionary, starting with structured messages
                output_item = {
                    "chosen": chosen_messages,
                    "rejected": rejected_messages
                }

                # Copy other desired keys, excluding 'original_qa_hash' and the ones we just processed
                keys_to_copy = ["sft_temperature", "hallucination_score", "item_hash"]
                for key in keys_to_copy:
                    if key in item:
                        output_item[key] = item[key]
                    else:
                        # Optional: Add a warning if expected metadata is missing
                        print(f"Warning: Metadata key '{key}' not found in item: {item.get('item_hash', 'N/A')}")


                # Write the transformed item as a JSON line to the output file
                json_line = json.dumps(output_item, ensure_ascii=False)
                outfile.write(json_line + '\n')
                processed_count += 1

    except FileNotFoundError:
        print(f"Error: Input file not found at {input_file}")
        return
    except json.JSONDecodeError:
        print(f"Error: Failed to decode JSON from {input_file}. Please check the file format.")
        return
    except ValueError as ve:
        print(f"Error: {ve}")
        return
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return

    print("-" * 30)
    print(f"Conversion complete.")
    print(f"Processed items: {processed_count}")
    print(f"Skipped items  : {skipped_count}")
    print(f"Output saved to: {output_file}")

if __name__ == "__main__":
    main()