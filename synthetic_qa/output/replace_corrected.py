import json
import os

# --- Description ---
# This script updates a large JSON file by replacing items with corrected versions from another JSON file.
# The script uses a hash key ('qa_pair_hash') to identify which items to replace.
# It loads both files, and then iterates through the original data,
# replacing items where a match is found in the corrected data.

# The corrections were made using another prompt to a large context LLM (gemini 2.5 pro) and are stored in a separate JSON file (corrected_data.json).

# --- Configuration ---
ORIGINAL_FILE = 'synthetic_qa_data.json'  # Path to your original large JSON file
CORRECTED_FILE = 'corrected_data.json' # Path to the JSON file with corrections
OUTPUT_FILE = 'synthetic_qa_data2.json' # Path for the updated output file
# Set to True to overwrite ORIGINAL_FILE instead of creating OUTPUT_FILE
OVERWRITE_ORIGINAL = False

# --- Helper Function to Load JSON ---
def load_json_file(filepath):
    """Loads data from a JSON file."""
    if not os.path.exists(filepath):
        print(f"Error: File not found - {filepath}")
        return None
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if not isinstance(data, list):
            print(f"Error: Expected a JSON list in {filepath}, but found {type(data)}")
            return None
        print(f"Successfully loaded {len(data)} items from {filepath}")
        return data
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from {filepath}: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while loading {filepath}: {e}")
        return None

# --- Helper Function to Save JSON ---
def save_json_file(filepath, data):
    """Saves data to a JSON file."""
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2) # Using indent=2 for readability
        print(f"Successfully saved {len(data)} items to {filepath}")
    except Exception as e:
        print(f"An error occurred while saving to {filepath}: {e}")

# --- Main Script Logic ---
def update_json_data():
    print("--- Starting JSON Update Process ---")

    # 1. Load data from both files
    original_data = load_json_file(ORIGINAL_FILE)
    corrected_data = load_json_file(CORRECTED_FILE)

    if original_data is None:
        print("Aborting due to original_data errors.")
        return

    if corrected_data is None:
        print("Aborting due to corrected_data errors.")
        return

    # 2. Create a lookup dictionary from corrected data
    print("Creating lookup map from corrected data...")
    corrected_lookup = {}
    hashes_in_corrected = set()
    for item in corrected_data:
        if not isinstance(item, dict):
            print(f"Warning: Found non-dictionary item in {CORRECTED_FILE}, skipping: {item}")
            continue
        try:
            item_hash = item.get('qa_pair_hash')
            if item_hash:
                if item_hash in hashes_in_corrected:
                     print(f"Warning: Duplicate qa_pair_hash '{item_hash}' found in {CORRECTED_FILE}. Using the last occurrence.")
                corrected_lookup[item_hash] = item
                hashes_in_corrected.add(item_hash)
            else:
                print(f"Warning: Item in {CORRECTED_FILE} missing 'qa_pair_hash', skipping: {item}")
        except Exception as e:
             print(f"Warning: Error processing item in {CORRECTED_FILE}, skipping: {item}. Error: {e}")

    print(f"Created lookup map with {len(corrected_lookup)} unique correctable hashes.")
    # *** ADDED DEBUG PRINT ***
    if corrected_lookup:
        print(f"  -> Hashes found in corrected data: {list(corrected_lookup.keys())[:10]}...") # Print first 10 keys


    # 3. Iterate through the original data and update if a correction exists
    print("Processing original data and applying corrections...")
    updated_count = 0
    output_data = [] # Store results in a new list

    for index, original_item in enumerate(original_data):
        if not isinstance(original_item, dict):
            print(f"Warning: Found non-dictionary item in {ORIGINAL_FILE} at index {index}, keeping as is.")
            output_data.append(original_item)
            continue

        item_hash = original_item.get('qa_pair_hash')

        if item_hash and item_hash in corrected_lookup:
            print(f"  -> Found match for hash: {item_hash}. Replacing original.")
            # *** ADDED DEBUG PRINT ***
            corrected_item_to_append = corrected_lookup[item_hash]
            print(f"     Corrected item content (first 100 chars): {str(corrected_item_to_append)[:100]}...")
            output_data.append(corrected_item_to_append) # Append the corrected item
            updated_count += 1
            # Optional: remove from lookup to track unused corrections
            # del corrected_lookup[item_hash]
        else:
            # Keep the original item if no hash or no correction found
             output_data.append(original_item)

    print(f"Processing complete. Updated {updated_count} items.")

    # *** ADDED DEBUG PRINT ***
    print("Preview of final output_data (first 2 items):")
    print(json.dumps(output_data[:2], indent=2, ensure_ascii=False))


    # 4. Save the updated data
    target_file = ORIGINAL_FILE if OVERWRITE_ORIGINAL else OUTPUT_FILE
    print(f"Saving updated data to: {target_file}")
    save_json_file(target_file, output_data)

    print("--- JSON Update Process Finished ---")

# --- Run the script ---
if __name__ == "__main__":
    # Create dummy files if they don't exist for testing
    if not os.path.exists(ORIGINAL_FILE):
        print(f"Creating dummy {ORIGINAL_FILE} for testing.")
        dummy_orig = [
            {"qa_pair_hash": "hash1", "question": "Q1 Orig", "answer": "A1 Orig"},
            {"qa_pair_hash": "hash2", "question": "Q2 Orig", "answer": "A2 Orig"},
            {"qa_pair_hash": "hash3", "question": "Q3 Orig", "answer": "A3 Orig"}
        ]
        save_json_file(ORIGINAL_FILE, dummy_orig)

    if not os.path.exists(CORRECTED_FILE):
         print(f"Creating dummy {CORRECTED_FILE} for testing.")
         dummy_corr = [
             {"qa_pair_hash": "hash2", "question": "Q2 Corrected", "answer": "A2 Corrected NEW"},
             {"qa_pair_hash": "hash4", "question": "Q4 Corrected", "answer": "A4 Corrected NEW (not in orig)"}
         ]
         save_json_file(CORRECTED_FILE, dummy_corr)

    update_json_data()