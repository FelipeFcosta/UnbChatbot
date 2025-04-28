import json
import requests
import time
import logging
import os
from tqdm import tqdm
import concurrent.futures
import argparse

# --- Configuration (remains the same) ---
API_URL = "https://liteofspace--unb-chatbot-gguf-web-endpoint-modelendpoint-0fa22e.modal.run"
INPUT_JSON_FILE = "/home/farias/tcc/qa_generation/fine_tuning/dpo/generate_data/input_qa.jsonl"  # Updated extension
OUTPUT_DIR = "/home/farias/tcc/qa_generation/fine_tuning/dpo/generate_data/output/sft_qa_pairs"
FINAL_OUTPUT_DIR = "/home/farias/tcc/qa_generation/fine_tuning/dpo/generate_data/output"
FINAL_OUTPUT_FILENAME = "sft_output_qa.json"
HEADERS = {"Content-Type": "application/json"}
MAX_TOKENS = 4096
TEMPERATURES = [0.5, 1.0, 1.5]
TOP_P = 0.9
ITERATIONS = 3
MAX_CONCURRENT_REQUESTS = 24
REQUEST_DELAY_SECONDS = 0
# --- End Configuration ---

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- call_model_api (remains the same) ---
def call_model_api(api_url, question, max_tokens, temperature, top_p):
    """ Sends a question to the model API and returns the model's response. """
    payload = {
        "prompt": question,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p
    }
    try:
        response = requests.post(api_url, headers=HEADERS, json=payload, timeout=180*2)
        response.raise_for_status()
        response_data = response.json()
        if "response" in response_data:
            return response_data["response"]
        else:
            logging.error(f"API response missing 'response' key: {response_data} for q: {question[:30]}...")
            return None
    except requests.exceptions.Timeout:
        logging.error(f"API request timed out for q: {question[:30]}...")
        return None
    except requests.exceptions.RequestException as e:
        logging.error(f"API request failed for q: {question[:30]}... Error: {e}")
        return None
    except json.JSONDecodeError as e:
        logging.error(f"Failed to decode JSON for q: {question[:30]}... Error: {e}. Response: {response.text[:100]}")
        return None
    except Exception as e:
        logging.error(f"Unexpected error in call_model_api for q: {question[:30]}... Error: {e}")
        return None


# --- Modified process_questions for JSONL input format ---
def process_questions(api_url, input_file, output_dir, parameters, current_iteration, max_workers, combined_results_list):
    """
    Reads questions from JSONL, checks if output exists, submits tasks up to max_workers,
    submits new tasks as old ones complete, saves results individually,
    AND appends results to combined_results_list.
    Progress bar tracks COMPLETED items.
    """
    try:
        # Load JSONL file
        input_data = []
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                input_data.append(json.loads(line))
        total_items = len(input_data)
    except FileNotFoundError:
        logging.error(f"Input file not found: {input_file}")
        return 0, 0
    except json.JSONDecodeError:
        logging.error(f"Error decoding JSON from file: {input_file}")
        return 0, 0

    successful_saves_this_iter = 0
    skipped_existing_this_iter = 0
    loaded_and_added_this_iter = 0
    items_considered_count = 0 # Track items pulled from iterator

    logging.info(f"[Iteration {current_iteration}/{ITERATIONS}] Starting processing for {total_items} questions with controlled concurrency ({max_workers}).")

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_data = {}
        active_futures = set()
        input_iterator = iter(input_data)

        # --- Function to submit the next available task ---
        def submit_next_task():
            # Use nonlocal only for variables modified *within this function* and defined outside
            nonlocal items_considered_count, skipped_existing_this_iter, loaded_and_added_this_iter
            try:
                while True:
                    item = next(input_iterator)
                    items_considered_count += 1 # Increment when we start considering an item

                    # Extract the question from the user message
                    messages = item.get("messages", [])
                    original_question = next((msg["content"] for msg in messages if msg["role"] == "user"), None)
                    original_url = item.get("url")
                    original_hash = item.get("qa_pair_hash")

                    if not original_question or not original_hash:
                        logging.warning(f"Skipping item considered #{items_considered_count} due to missing data: {item}")
                        continue # Try next item

                    new_hash = f"{original_hash}_{current_iteration-1}"
                    # Corrected filename logic again - should likely use current_iteration
                    potential_output_filename = os.path.join(output_dir, f"{new_hash}.json")

                    if os.path.exists(potential_output_filename):
                        skipped_existing_this_iter += 1
                        try:
                            with open(potential_output_filename, 'r', encoding='utf-8') as f_existing:
                                # Optional: log loading
                                logging.info(f"Loading existing file {potential_output_filename}...")
                                existing_data = json.load(f_existing)
                            combined_results_list.append(existing_data)
                            loaded_and_added_this_iter += 1
                            # ***** UPDATE PBAR HERE for loaded item *****
                            pbar.update(1)
                        except (IOError, json.JSONDecodeError) as e:
                            logging.warning(f"Found file {potential_output_filename} but failed to load/decode: {e}. Skipping aggregation & pbar update for this.")
                        continue # Already processed (loaded or failed load), try next item

                    # --- File doesn't exist, submit for generation ---
                    logging.info(f"Submitting API call for item #{items_considered_count}: {original_question[:30]}...")
                    future = executor.submit(
                        call_model_api,
                        api_url,
                        original_question,
                        parameters["max_tokens"],
                        parameters["temperature"],
                        parameters["top_p"]
                    )
                    future_data = {
                        "output_filename": potential_output_filename,
                        "original_question": original_question,
                        "original_url": original_url,
                        "new_hash": new_hash,
                        "original_hash": original_hash,
                        "item_number": items_considered_count # Store item number for context
                    }
                    future_to_data[future] = future_data
                    active_futures.add(future)
                    return True # Task submitted

            except StopIteration:
                 return False # No more items
            except Exception as e:
                 # Log error associated with the item being considered
                 logging.error(f"Error occurred while trying to submit task for item considered #{items_considered_count}: {e}")
                 return False # Stop submitting for now

        # --- Main processing loop: Use progress bar and manage submission ---
        with tqdm(total=total_items, desc=f"Processing Iter {current_iteration}") as pbar:
            # --- Initial submission phase ---
            logging.info(f"Submitting initial batch of tasks (up to {max_workers})...")
            initial_submitted_count = 0
            for _ in range(max_workers):
                if submit_next_task():
                    initial_submitted_count += 1
                else:
                    break # Stop if no more tasks can be submitted or error occurred
            logging.info(f"Initially submitted {initial_submitted_count} tasks for processing.")

            # --- Wait loop ---
            while active_futures:
                done, active_futures_set = concurrent.futures.wait(
                    active_futures, return_when=concurrent.futures.FIRST_COMPLETED
                )
                active_futures = active_futures_set

                for future in done:
                    data = future_to_data.pop(future)
                    output_filename = data["output_filename"]
                    original_hash = data["original_hash"]
                    item_num = data.get("item_number", "N/A") # Get item number for logging

                    task_completed = False # Flag to ensure pbar update happens once per task outcome

                    try:
                        model_answer = future.result()
                        if model_answer is not None:
                            output_item = {
                                "question": data["original_question"],
                                "answer": model_answer,
                                "parameters": parameters.copy(),
                                "url": data["original_url"],
                                "qa_pair_hash": data["new_hash"],
                                "type": "sft"
                            }
                            combined_results_list.append(output_item)
                            try:
                                with open(output_filename, 'w', encoding='utf-8') as f_out:
                                    # Optional: log saving
                                    logging.info(f"Saving individual file {output_filename} (Item #{item_num})...")
                                    json.dump(output_item, f_out, ensure_ascii=False, indent=4)
                                successful_saves_this_iter += 1
                                task_completed = True # Mark as completed successfully
                            except IOError as e:
                                logging.error(f"Error writing individual file {output_filename} (Item #{item_num}): {e}")
                            except Exception as e:
                                logging.error(f"Unexpected error saving individual file {output_filename} (Item #{item_num}): {e}")
                        else:
                            logging.warning(f"API call failed/returned None for hash: {original_hash} (Item #{item_num}). No file saved/aggregated.")
                            task_completed = True # Task finished, albeit unsuccessfully
                    except Exception as e:
                        logging.error(f"Task for hash {original_hash} (Item #{item_num}) generated exception: {e}")
                        task_completed = True # Task finished with exception
                    finally:
                         if task_completed:
                             pbar.update(1)

                    # --- Try to submit a new task ---
                    submit_next_task()

            # Ensure progress bar completes if loop finishes early
            if pbar.n < total_items:
                 logging.info(f"Completing progress bar. Processed {pbar.n}/{total_items}.")
                 pbar.update(total_items - pbar.n)

    logging.info(f"[Iteration {current_iteration}/{ITERATIONS}] Finished processing iteration. "
                 f"Saved {successful_saves_this_iter} new *individual* QA pairs. "
                 f"Skipped/Loaded {skipped_existing_this_iter} existing files. "
                 f"Added {loaded_and_added_this_iter + successful_saves_this_iter} total items to combined list.")

    return loaded_and_added_this_iter, successful_saves_this_iter


# --- Main Execution (remains the same) ---
def main():
    parser = argparse.ArgumentParser(description="Generate SFT answers using a model API.")
    parser.add_argument("--input", type=str, default=INPUT_JSON_FILE, help="Path to the input JSONL file.")
    parser.add_argument("--output-filename", type=str, default=FINAL_OUTPUT_FILENAME, help="Filename for the final aggregated output.")
    parser.add_argument("--api-url", type=str, default=API_URL, help="API URL for the model.")
    args = parser.parse_args()
    api_url = args.api_url
    input_path = args.input
    output_filename = args.output_filename
    
    model_parameters = {
        "max_tokens": MAX_TOKENS,
        "temperatures": TEMPERATURES,
        "top_p": TOP_P
    }

    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        logging.info(f"Individual output directory '{OUTPUT_DIR}' ensured.")
    except OSError as e:
        logging.error(f"Error creating individual output directory {OUTPUT_DIR}: {e}")
        exit(1)
    try:
        os.makedirs(FINAL_OUTPUT_DIR, exist_ok=True)
        logging.info(f"Final aggregated output directory '{FINAL_OUTPUT_DIR}' ensured.")
    except OSError as e:
        logging.error(f"Error creating final aggregated output directory {FINAL_OUTPUT_DIR}: {e}")
        exit(1)

    all_results_combined = []
    total_loaded_aggregated = 0
    total_saved_new_individual = 0

    for i in range(1, ITERATIONS + 1):
        model_parameters["temperature"] = model_parameters["temperatures"][i-1]
        logging.info(f"--- Starting Iteration {i} of {ITERATIONS} ---")
        loaded_count, saved_count = process_questions(
            api_url,
            input_path,
            OUTPUT_DIR,
            model_parameters,
            i,
            MAX_CONCURRENT_REQUESTS,
            all_results_combined
        )
        total_loaded_aggregated += loaded_count
        total_saved_new_individual += saved_count
        logging.info(f"--- Completed Iteration {i} of {ITERATIONS} ---")
        logging.info(f"Total results accumulated in combined list so far: {len(all_results_combined)}")

    final_output_path = os.path.join(FINAL_OUTPUT_DIR, output_filename)
    logging.info(f"All iterations finished.")
    logging.info(f"Total across all iterations: Added {total_loaded_aggregated} from existing files, Saved {total_saved_new_individual} new individual files.")
    logging.info(f"Saving {len(all_results_combined)} combined results to {final_output_path}...")
    if not all_results_combined:
        logging.warning("No results were collected in the combined list. Skipping final aggregated save.")
    else:
        try:
            with open(final_output_path, 'w', encoding='utf-8') as f_final:
                json.dump(all_results_combined, f_final, ensure_ascii=False, indent=4)
            logging.info(f"Successfully saved combined results to {final_output_path}")
        except Exception as e:
            logging.error(f"Failed to save final combined file {final_output_path}: {e}")

if __name__ == "__main__":
    main()
