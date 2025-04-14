#!/usr/bin/env python3
"""
Convert synthetic QA data to the format required for fine-tuning with Unsloth.

This script takes the output from synthetic_qa_generator.py (JSON format) and 
converts it to the JSONL format required by the training script, with optional
validation set creation.
"""

import json
import argparse
from pathlib import Path
import random
import logging
import re
import unicodedata
import hashlib
from typing import List, Dict, Any

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

def normalize_text(text):
    """
    Normalize text to improve training quality by fixing common issues.
    
    Args:
        text: Text to normalize
        
    Returns:
        Normalized text
    """
    if not text:
        return ""
        
    # Fix common HTML entities that might remain
    text = text.replace('&nbsp;', ' ')
    text = text.replace('&amp;', '&')
    text = text.replace('&lt;', '<')
    text = text.replace('&gt;', '>')
    
    # Normalize to avoid inconsistent representation of accents
    text = unicodedata.normalize('NFC', text)
    
    return text.strip()

def filter_institutional_qa(qa_pairs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Filter QA pairs to keep only those with clear institutional references.
    
    Args:
        qa_pairs: List of QA pairs
        
    Returns:
        Filtered list of QA pairs
    """
    filtered_pairs = []
    
    for pair in qa_pairs:
        question = pair.get("question", "")
        answer = pair.get("answer", "")
        
        # Skip pairs without proper content
        # if not question or not answer:
            # continue

        # Normalize text to improve quality
        pair["question"] = normalize_text(question)
        pair["answer"] = normalize_text(answer)
        filtered_pairs.append(pair)
            
    logger.info(f"Kept {len(filtered_pairs)}/{len(qa_pairs)} QA pairs")
    return filtered_pairs


def convert_qa_to_training_format(qa_pairs: List[Dict[str, Any]], 
                                 output_file: Path,
                                 validation_split: float = 0.50,
                                 create_validation: bool = True) -> None:
    """
    Convert synthetic QA pairs to the format required for Unsloth training.
    Ensures each unique QA origin has the same proportion in training and validation sets.
    
    Args:
        qa_pairs: List of QA pairs with question/answer fields
        output_file: Path to save the converted output
        validation_split: Percentage of data to use for validation (0.0-1.0)
        create_validation: Whether to create a separate validation file
    """
    # Filter to keep only institutional QA pairs
    qa_pairs = filter_institutional_qa(qa_pairs)
    
    # Create training data in the required format
    all_examples = []
    
    # Extract and format all valid examples
    for pair in qa_pairs:
        # Extract question and answer from each pair
        question = pair.get("question", "")
        answer = pair.get("answer", "")
        url = pair.get("url", "")
        pair_hash = pair.get("qa_pair_hash", "")
        
        if not question or not answer:
            continue
        
        # Create a message object in the format expected by Unsloth
        message_obj = {
            "messages": [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer}
            ]
        }
        
        # Add url metadata if available (for debugging/tracing)
        if "url" in pair:
            message_obj["url"] = url
            
        message_obj["qa_pair_hash"] = pair_hash
        
        # Extract the origin hash (part before style and iteration)
        # Example: from "faq_bb7532e0fd99_Direct_0" extract "faq_bb7532e0fd99"
        origin_parts = pair_hash.split('_')
        if len(origin_parts) >= 2:
            origin_hash = f"{origin_parts[0]}_{origin_parts[1]}"  # e.g., "faq_bb7532e0fd99"
        else:
            origin_hash = pair_hash
            
        message_obj["origin_hash"] = origin_hash
        all_examples.append(message_obj)
    
    logger.info(f"Found {len(all_examples)} unique QA pairs after deduplication")
    
    # Check if we have any data to work with
    if not all_examples:
        logger.error("No valid training examples found after filtering!")
        return
    
    # Group all examples by origin hash
    examples_by_origin = {}
    for example in all_examples:
        origin_hash = example["origin_hash"]
        if origin_hash not in examples_by_origin:
            examples_by_origin[origin_hash] = []
        examples_by_origin[origin_hash].append(example)
    
    # Create training and validation sets with proportional distribution by origin
    train_data = []
    val_data = []
    
    logger.info(f"Distributing examples from {len(examples_by_origin)} distinct origins")

    for origin_hash, examples in examples_by_origin.items():
        # Shuffle examples for this origin
        random.shuffle(examples)

        if create_validation and validation_split > 0:
            num_examples = len(examples)
            if num_examples == 0:
                continue # Skip empty origins

            # Calculate the ideal floating-point number of validation examples
            ideal_val_count_float = num_examples * validation_split

            # Determine the base number of validation examples (integer part)
            base_val_count = int(ideal_val_count_float)

            # Determine the fractional part
            fractional_part = ideal_val_count_float - base_val_count

            # Probabilistically decide whether to add one more based on the fraction
            extra_val_count = 0
            if random.random() < fractional_part:
                extra_val_count = 1

            # Calculate the total desired validation count for this origin
            potential_val_count = base_val_count + extra_val_count

            # --- Apply Constraints ---
            # 1. Limit by available examples
            val_count = min(potential_val_count, num_examples)

            # 2. Ensure we don't put ALL examples in validation (if > 1 example exists)
            if val_count >= num_examples and num_examples > 1:
                val_count = num_examples - 1

            # 3. If we have only 1 example, it must go to training
            if num_examples == 1:
                val_count = 0
                
            # --- Final Calculation and Logging ---
            train_count = num_examples - val_count
            prob_text = f"Prob={fractional_part:.2f}" if fractional_part > 0 else ""
            logger.info(
                f"Origin {origin_hash}: Ideal={ideal_val_count_float:.2f} "
                f"(Base={base_val_count}, {prob_text}). "
                f"Rolled extra: {'Yes' if extra_val_count == 1 else 'No'}. "
                f"Taking {train_count} train, {val_count} val"
            )

            # --- Add to respective sets ---
            # Ensure slicing is correct even if val_count is 0
            train_data.extend(examples[:train_count]) # Take first train_count examples
            if val_count > 0:
                val_data.extend(examples[train_count:]) # Take remaining val_count examples

        else:
            # Add all to training if no validation
            train_data.extend(examples)
    
    # Remove the temporary origin_hash field we added
    for example in train_data + val_data:
        if "origin_hash" in example:
            del example["origin_hash"]
    
    # Shuffle the final datasets again for good measure
    random.shuffle(train_data)
    if val_data:
        random.shuffle(val_data)
    
    logger.info(f"Total examples: {len(train_data) + len(val_data)}")
    logger.info(f"Training examples: {len(train_data)}")
    if val_data:
        logger.info(f"Validation examples: {len(val_data)}")
    
    # Save training data
    train_output = output_file.with_suffix('.jsonl')
    with open(train_output, 'w', encoding='utf-8') as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    # Save validation data if we have any
    if val_data:
        val_output = output_file.with_name(f"{output_file.stem}_val.jsonl")
        with open(val_output, 'w', encoding='utf-8') as f:
            for item in val_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
                
        logger.info(f"Created training dataset with {len(train_data)} examples at {train_output}")
        logger.info(f"Created validation dataset with {len(val_data)} examples at {val_output}")
    else:
        logger.info(f"Created dataset with {len(train_data)} examples at {train_output}")


def main():
    parser = argparse.ArgumentParser(description="Convert synthetic QA data to training format")
    parser.add_argument("--input", required=True, help="Path to synthetic_qa_data.json")
    parser.add_argument("--output", default="training_data", help="Base name for output files")
    parser.add_argument("--validation-split", type=float, default=0.10, 
                        help="Percentage of data to use for validation (0.0-1.0)")
    parser.add_argument("--skip-validation", action="store_true", 
                        help="Skip creating a validation set")
    parser.add_argument("--seed", type=int, default=3407, help="Random seed for reproducibility")
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        return
    
    # Load the synthetic QA data
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            qa_data = json.load(f)
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing JSON file: {e}")
        return
    except UnicodeDecodeError as e:
        logger.error(f"Unicode error reading file (try different encoding): {e}")
        return
    
    logger.info(f"Loaded {len(qa_data)} QA pairs from {input_path}")
    

    # Convert and save in the required format
    output_path = Path(args.output)
    convert_qa_to_training_format(
        qa_data, 
        output_path, 
        validation_split=args.validation_split,
        create_validation=not args.skip_validation
    )


if __name__ == "__main__":
    main()