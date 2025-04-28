import json
import os
import logging
import time
import re
import concurrent.futures

# --- Configuration ---
from pathlib import Path

SFT_OUTPUT_FILE = "/home/farias/tcc/qa_generation/fine_tuning/dpo/generate_data/output/sft_output_qa_val.json"
ANNOTATIONS_OUTPUT_DIR = Path("/home/farias/tcc/qa_generation/fine_tuning/dpo/generate_data/output/annotations_val")
ALL_ANNOTATIONS_FILENAME = "all_annotations_val.json"
DPO_DATASET_FILENAME = "dpo_dataset_val.json"
OUTPUT_DIR = Path("/home/farias/tcc/qa_generation/fine_tuning/dpo/generate_data/output/")
# ASSUMPTION: Ground truth data is in a file mapping hash to its full text content
GROUND_TRUTH_FILE = "/home/farias/tcc/qa_generation/fine_tuning/dpo/generate_data/ground_truth.json"
# INPUT_QA_FILE = "/home/farias/tcc/qa_generation/input_qa.jsonl"
INPUT_QA_FILE = "/home/farias/tcc/qa_generation/synthetic_qa/output/unb_training_data_val.jsonl"

# GEMINI_MODEL = "gemini-2.5-flash-preview-04-17"  # Or your preferred Gemini model
GEMINI_MODEL = "gemini-2.5-pro-preview-03-25"  # Or your preferred Gemini model
MAX_TOKENS_GENERATE = 65536
TEMPERATURE = 1.0

RPM = 10
DELAY_SECONDS = 60/RPM # Delay between API retries

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from google import genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    logger.error("genAI package not installed.")
    exit(1)

# The user's prompt template (escaped curly braces for format)
PROMPT_TEMPLATE = """Carefully read the following FAQ pairs as reference (SOLE GROUND TRUTH) below:

{FAQ_GROUND_TRUTH}

You will act as a 'Hallucination' annotator. I will provide you with a question, an answer to that question,. You need to determine whether the provided answer contains any hallucinatory content and annotate the type of hallucination.

'Hallucination' refers to content that contradicts or is unsupported the ground truth (the only source of true information here).
 ## Judgment Criteria:
 1. 'No Facts': The answer provides no factual claims — essentially content-free in terms of verifiable assertions.
 2. 'No Hallucination': The answer is **fully consistent** with the provided GROUND TRUTH and introduces strictly **no contradictory or unsupported** information in any way.
 3. 'Contradiction': The answer **contradicts or is not truthful** to the GROUND TRUTH information.
 4. 'Unverifiable': The answer contains information **not found in the GROUND TRUTH and cannot be confirmed or verified by them**.

 ## Task Process:
 1. Carefully read the question, which is as follows: "{QUESTION}"
 2. Carefully read the model answer, which is as follows: "{MODEL_ANSWER}"
 3. Conduct the analysis: Based on the above judgment criteria, separate the response into *full formed portuguese sentences* **without changing any character or formatting (including every new line \\n and spacing)**, and determine if the sentence contains hallucinations and output the type of hallucination FOR EACH SENTENCE
 4. If the sentence is a 'Contradiction' **provide the correct version of the sentence** in the **same style** and format of the model's answer. If 'No Hallucination' or 'No Facts', output null
 5. If the sentence is 'Unverifiable' **provide the correct version of the sentence** by either removing or replacing the unverifiable part with *verifiable useful* information.
 6. Provide the original qa hash for the source of the true information if the sentence is verifiable.
 7. Provide the preferred sentence = sentences + corrected_sentences (instead of hallucinations).
 8. Provide a short reason (summary) for the annotation.

It's important that, when possible, the 'corrected' sentences are not a copy of the ground truth text, but rather a corrected version of the model's answer that is consistent with the ground truth.

Return a json in this format:
{{
    "annotations": [
        {{
        "sentence": <sentence1>,
        "label": <label1>,
        "corrected": <corrected_sentence1>, // if contradictory or unverifiable
        "source": <source_qa_hash1> // if verifiable
        "reason": <reason_for_annotation>
        }}
        {{
        "sentence": <sentence2>,
        "label": <label2>,
        "corrected": <corrected_sentence2>, // if contradictory or unverifiable
        "source": <source_qa_hash2>  // if verifiable
        "reason": <reason_for_annotation>
        }}
        ...
    ],
    "preferred": <full_correct_sentence>
}}"""

# --- Helper Functions ---
def load_json(filepath):
    """Loads JSON data from a file."""
    if not os.path.exists(filepath):
        logger.error(f"File not found: {filepath}")
        return None
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from {filepath}: {e}")
        return None
    except Exception as e:
        logger.error(f"Error reading file {filepath}: {e}")
        return None

def sanitize_json_string(json_string: str) -> str:
    try:
        return json.dumps(json.loads(json_string), indent=2)
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to sanitize JSON string: {e}")
        pass

    # Handle the content directly as text
    try:
        # Find all annotation blocks with a more flexible pattern
        annotations = re.finditer(r'\{\s*"sentence"\s*:\s*"(.*?)"\s*,\s*"label"\s*:\s*".*?"\s*,\s*"corrected"\s*:\s*(.*?)\s*,\s*"source"\s*:.*?\}', json_string, re.DOTALL)
        
        fixed_annotations = []
        for match in re.finditer(r'\{\s*"sentence"\s*:\s*"(.*?)"\s*,\s*"label"\s*:\s*"(.*?)"\s*,\s*"corrected"\s*:\s*(.*?)\s*,\s*"source"\s*:\s*(.*?)\s*\}', json_string, re.DOTALL):
            s_text = match.group(1)
            label = match.group(2)
            c_text = match.group(3)
            source = match.group(4)

            s_text = s_text.strip('"').replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n')
            
            # Handle the case where corrected might be null or a string
            if c_text.strip().lower() == "null":
                corrected_part = '"corrected": null'
            else:
                c_text = c_text.strip('"').replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n')
                corrected_part = f'"corrected": "{c_text}"'
            
            # Handle source field
            if source.strip().lower() == "null":
                source_part = '"source": null'
            else:
                source = source.strip('"').replace('\\', '\\\\').replace('"', '\\"')
                source_part = f'"source": "{source}"'
            
            fixed_annotation = f'{{\n"sentence": "{s_text}",\n"label": "{label}",\n{corrected_part},\n{source_part}\n}}'
            fixed_annotations.append(fixed_annotation)
            
        # Rebuild the JSON with all annotations
        fixed_json = f'{{\n"annotations": [\n{",".join(fixed_annotations)}\n]\n}}'
        
        # Validate
        return json.dumps(json.loads(fixed_json), indent=2)
    except Exception:
        pass

    # If all else fails
    raise ValueError(f"JSON string could not be sanitized")
    

def json_if_valid(text: str):
    """
    Extract and parse JSON from text if it contains valid JSON.
    
    Args:
        text: Text that might contain JSON
        
    Returns:
        Parsed JSON object or None if no valid JSON found
    """
    if not text:
        return None
        
    # Find JSON-like pattern in the text
    json_pattern = r'(\{[\s\S]*\})'
    json_match = re.search(json_pattern, text)
    
    if not json_match:
        return None

    # Check if there's more than one JSON object in the text
    if len(re.findall(json_pattern, text)) > 1:
        return None  
    
    json_text = json_match.group(1)

    try:
        # Sanitize and parse the JSON
        sanitized = sanitize_json_string(json_text)
        data = json.loads(sanitized)
        return data
    except (json.JSONDecodeError, ValueError):
        return None


def call_gemini_api(prompt, api_key, model=GEMINI_MODEL):
    """Calls the Gemini API with a specific API key."""
    if not api_key:
        logger.error("API key not provided.")
        return None

    try:
        client = genai.Client(api_key=api_key)
        generation_config = {
            "max_output_tokens": MAX_TOKENS_GENERATE,
            "temperature": TEMPERATURE,
        }

        logger.info(f"Calling Gemini API with key ending in ...{api_key[-4:]}.")
        try:
            # Use generate_content directly on the model instance
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=generation_config
            )

            # Check for response and extract text
            if response and response.text:
                response_text = response.text
            else:
                logger.warning(f"Received empty response text from Gemini.")
                if response.candidates and response.candidates[0].finish_reason:
                    logger.warning(f"Finish reason: {response.candidates[0].finish_reason}")
                return None

            response_text = response_text.replace("```json", "").replace("```", "")
            json_response = json_if_valid(response_text)
            if json_response:
                logger.info(f"Successfully parsed JSON response.")
                return json_response
            else:
                logger.warning(f"Failed to parse JSON response. Response length: {len(response_text)}.")
                return None
                
        except Exception as e:
            logger.error(f"Gemini API call failed: {e}")
            return None

    except Exception as e:
        logger.error(f"Overall Gemini API setup error: {e}")
        return None

def process_item(item, ground_truth_data_str, api_key):
    """Process a single item with a specific API key"""
    question = item.get("question")
    temperature = item.get("parameters").get("temperature")
    answer = item.get("answer")
    qa_hash = item.get("qa_pair_hash")
    original_qa_hash = re.sub(r"_\d+$", "", qa_hash)  # Remove iteration number

    annotation_path = ANNOTATIONS_OUTPUT_DIR / f"annotation_{qa_hash}.json"

    # Load annotation if it exists
    if os.path.exists(annotation_path):
        with open(annotation_path, 'r', encoding='utf-8') as f:
            annotation_result = json.load(f)
            # ------- DELETE LATER ------
            annotation_result["temperature"] = temperature
            if True or 'hallucination_score' not in annotation_result:
                sentences_count = 0
                hallucination_count = 0
                for item in annotation_result["annotations"]:
                    sentences_count += 1 if item["label"] else 0
                    if item["label"] and item["label"].lower() in ["contradiction", "contradictory", "unverifiable"]:
                        hallucination_count += 1
                
                if sentences_count > 0:
                    hallucination_score = (hallucination_count / sentences_count)
                if hallucination_count:
                    hallucination_score += 0.1 * sentences_count
                annotation_result['hallucination_score'] = hallucination_score

                try:
                    with open(annotation_path, 'w', encoding='utf-8') as f:
                        json.dump(annotation_result, f, ensure_ascii=False, indent=2)
                        logger.info(f"Updated annotation result {annotation_path}")
                except Exception as e:
                    logger.error(f"Error saving annotation result for hash {qa_hash}: {e}")
            # ---------------------------

            logger.info(f"Annotation already exists for hash {qa_hash}. Skipping.")
            return qa_hash, annotation_result

    # Construct the full prompt
    full_prompt = PROMPT_TEMPLATE.format(
        FAQ_GROUND_TRUTH=ground_truth_data_str,
        QUESTION=question,
        MODEL_ANSWER=answer
    )

    logger.info(f"Processing hash {qa_hash}, Question: {question[:50]}...")
    
    start_time = time.time()

    # Call API with specific key
    annotation_result = None
    while annotation_result is None:
        annotation_result = call_gemini_api(full_prompt, api_key)

        if annotation_result is None:
            logger.error(f"Failed with API ending with {api_key[-4:]}....")
            logger.error(f"Failed to get annotation for hash: {qa_hash}. Retrying...")
            time.sleep(1)
    
    annotation_result["temperature"] = temperature

    # Calculate hallucination score
    hallucination_score = 0
    hallucination_count = 0
    sentences_count = 0
    for item in annotation_result["annotations"]:
        sentences_count += 1 if item["label"] else 0
        if item["label"] and item["label"].lower() in ["contradiction", "contradictory", "unverifiable"]:
            hallucination_count += 1
    
    if sentences_count > 0:
        hallucination_score = (hallucination_count / sentences_count)
    if hallucination_count:
        hallucination_score += 0.1 * sentences_count
    annotation_result['hallucination_score'] = hallucination_score

    # Save individual pair
    try:
        with open(annotation_path, 'w', encoding='utf-8') as f:
            json.dump(annotation_result, f, ensure_ascii=False, indent=2)
            logger.info(f"Saved annotation result {annotation_path}")
    except Exception as e:
        logger.error(f"Error saving annotation result for hash {qa_hash}: {e}")

    # Calculate elapsed time and sleep only if needed
    elapsed_time = time.time() - start_time
    if elapsed_time < DELAY_SECONDS+1:
        time.sleep(DELAY_SECONDS+1 - elapsed_time)

    return qa_hash, annotation_result

# --- Main Execution ---
if __name__ == "__main__":
    # Get API keys
    api_keys = [
        # os.environ.get("GEMINI_API_KEY"),
        # os.environ.get("GEMINI_API_KEY2"),
        # os.environ.get("GEMINI_API_KEY3"),
        # os.environ.get("GEMINI_API_KEY4"),
    ] + [os.environ.get("GEMINI_API_KEY2")] * 50

    valid_api_keys = [key for key in api_keys if key]
    if not valid_api_keys:
        logger.error("No valid Gemini API keys found.")
        exit(1)
    
    logger.info(f"Found {len(valid_api_keys)} valid API keys")

    # Load SFT outputs
    sft_data = load_json(SFT_OUTPUT_FILE)
    if sft_data is None:
        logger.error(f"Could not load SFT data from {SFT_OUTPUT_FILE}. Exiting.")
        exit(1)

    # Load Ground Truth data
    ground_truth_data = load_json(GROUND_TRUTH_FILE)
    if ground_truth_data is None:
        logger.warning(f"Could not load ground truth file: {GROUND_TRUTH_FILE}. Proceeding without detailed ground truth lookup.")
        ground_truth_data_str = "{}"  # Empty JSON object
    else:
        ground_truth_data_str = str(ground_truth_data)

    os.makedirs(ANNOTATIONS_OUTPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_annotations = {}

    # Process items using ThreadPoolExecutor
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(valid_api_keys)) as executor:
        futures = []
        
        # Submit tasks for each item, cycling through API keys
        for i, item in enumerate(sft_data):
            key_index = i % len(valid_api_keys)
            api_key = valid_api_keys[key_index]
            futures.append(
                executor.submit(process_item, item, ground_truth_data_str, api_key)
            )
            # time.sleep(0.05)
        
        # Process results as they complete
        for future in concurrent.futures.as_completed(futures):
            try:
                qa_hash, annotation_result = future.result()
                if annotation_result:
                    all_annotations[qa_hash] = annotation_result
            except Exception as e:
                logger.error(f"Error processing item: {e}")

    # Save all annotations to a single file
    all_annotations_path = OUTPUT_DIR / ALL_ANNOTATIONS_FILENAME
    try:
        with open(all_annotations_path, 'w', encoding='utf-8') as f:
            json.dump(all_annotations, f, ensure_ascii=False, indent=2)
            logger.info(f"Saved all annotations to {all_annotations_path}")
    except Exception as e:
        logger.error(f"Error saving all annotations: {e}")

    # Create a DPO dataset from the annotations
    dpo_dataset = []

    # Track the highest hallucination score for each original hash
    best_items = {}

    # First, find the items with highest hallucination scores for each original hash
    for item in sft_data:
        qa_hash = item.get("qa_pair_hash")
        original_qa_hash = re.sub(r"_\d+$", "", qa_hash)
        
        # Check if we have an annotation for this hash
        if qa_hash not in all_annotations:
            logger.warning(f"Annotation not found for hash {qa_hash}. Skipping.")
            continue
            
        # Get the annotation
        annotation = all_annotations[qa_hash]
        
        # Get hallucination score
        score = annotation.get('hallucination_score', 0)
        
        # Skip if score is too low
        if score < 0.1:
            logger.warning(f"Hallucination score too low for hash {qa_hash}. Skipping.")
            continue
        
        # Update if this is the best score for this hash
        if original_qa_hash not in best_items or score > best_items[original_qa_hash]['score']:
            best_items[original_qa_hash] = {
                'qa_hash': qa_hash,
                'score': score,
                'question': item.get("question"),
                'rejected': item.get("answer"),
                'temperature': item.get("parameters").get("temperature"),
                'annotation': annotation
            }


    input_data = []
    with open(INPUT_QA_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            input_data.append(json.loads(line))


    # Create DPO entries using the best items
    for original_qa_hash, item_data in best_items.items():

        question = item_data.get("question")
        rejected = item_data.get("rejected")
        temperature = item_data.get("temperature")
        # Find the preferred answer in ground_truth_data
        preferred = None
        for entry in input_data:
            if entry.get("qa_pair_hash") == original_qa_hash:
                # Get the assistant's message content as the preferred answer
                messages = entry.get("messages", [])
                for message in messages:
                    if message.get("role") == "assistant":
                        preferred = message.get("content")
                        break
                break
        # preferred = item_data.get("annotation").get("preferred")
        
        # Skip if we don't have all required fields
        if not (question and rejected and preferred):
            continue
            
        dpo_dataset.append({
            "question": question,
            "rejected": rejected,
            "preferred": preferred,
            "sft_temperature": temperature,
            "hallucination_score": round(item_data.get('score'), 2),
            "original_qa_hash": original_qa_hash,
            "item_hash": item_data.get('qa_hash')
        })

    # Save the DPO dataset
    dpo_dataset_path = OUTPUT_DIR / DPO_DATASET_FILENAME
    try:
        with open(dpo_dataset_path, 'w', encoding='utf-8') as f:
            json.dump(dpo_dataset, f, ensure_ascii=False, indent=2)
            logger.info(f"Saved DPO dataset with {len(dpo_dataset)} entries to {dpo_dataset_path}")
    except Exception as e:
        logger.error(f"Error saving DPO dataset: {e}")
