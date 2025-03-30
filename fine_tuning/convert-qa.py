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
from typing import List, Dict, Any

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

def normalize_text(text):
    if not text:
        return ""
        
    text = re.sub(r'\r\n', '\n', text)
    
    # Preserve newlines but normalize multiple consecutive spaces
    text = re.sub(r'[ \t]+', ' ', text)
    
    # Fix common HTML entities that might remain
    text = text.replace('&nbsp;', ' ')
    text = text.replace('&amp;', '&')
    text = text.replace('&lt;', '<')
    text = text.replace('&gt;', '>')
    
    # Normalize Unicode for consistent representation of accents
    text = unicodedata.normalize('NFC', text)
    
    return text.strip()


def convert_qa_to_training_format(qa_pairs: List[Dict[str, Any]], 
                                 output_file: Path,
                                 validation_split: float = 0.1,
                                 create_validation: bool = True) -> None:
    """
    Convert synthetic QA pairs to the format required for Unsloth training.
    
    Args:
        qa_pairs: List of QA pairs with question/answer fields
        output_file: Path to save the converted output
        validation_split: Percentage of data to use for validation (0.0-1.0)
        create_validation: Whether to create a separate validation file
    """
    # Create training data in the required format
    training_data = []
    
    # Keep track of unique question-answer combinations to avoid duplicates
    seen_pairs = set()
    
    for pair in qa_pairs:
        # Extract question and answer from each pair
        question = pair.get("question", "")
        answer = pair.get("answer", "")
        
        if not question or not answer:
            continue
        
        # Normalize text to improve quality
        question = normalize_text(question)
        answer = normalize_text(answer)
        
        # Create a simple hash to detect duplicates (using just the beginning of each)
        pair_hash = hash(f"{question[:50]}|{answer[:50]}")
        if pair_hash in seen_pairs:
            continue
        seen_pairs.add(pair_hash)
            
        # Create a message object in the format expected by Unsloth
        message_obj = {
            "messages": [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer}
            ]
        }
        
        # Add source metadata if available (for debugging/tracing)
        if "source" in pair or "url" in pair:
            source = pair.get("source", pair.get("url", ""))
            if source:
                message_obj["source"] = source
        
        training_data.append(message_obj)
    
    logger.info(f"Found {len(training_data)} unique QA pairs after deduplication")
    
    if not training_data:
        logger.error("No valid training examples!")
        return
    
    random.shuffle(training_data)
    
    # Split into training and validation if requested
    if create_validation and validation_split > 0:
        split_index = int(len(training_data) * (1 - validation_split))
        train_data = training_data[:split_index]
        val_data = training_data[split_index:]
        
        # Save training data
        train_output = output_file.with_suffix('.jsonl')
        with open(train_output, 'w', encoding='utf-8') as f:
            for item in train_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        # Save validation data
        val_output = output_file.with_name(f"{output_file.stem}_val.jsonl")
        with open(val_output, 'w', encoding='utf-8') as f:
            for item in val_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
                
        logger.info(f"Created training dataset with {len(train_data)} examples at {train_output}")
        logger.info(f"Created validation dataset with {len(val_data)} examples at {val_output}")
    else:
        # Save all data as training
        output = output_file.with_suffix('.jsonl')
        with open(output, 'w', encoding='utf-8') as f:
            for item in training_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        logger.info(f"Created dataset with {len(training_data)} examples at {output}")


def main():
    parser = argparse.ArgumentParser(description="Convert synthetic QA data to training format")
    parser.add_argument("--input", required=True, help="Path to synthetic_qa_data.json")
    parser.add_argument("--output", default="training_data", help="Base name for output files")
    parser.add_argument("--validation-split", type=float, default=0.1, 
                        help="Percentage of data to use for validation (0.0-1.0)")
    parser.add_argument("--skip-validation", action="store_true", 
                        help="Skip creating a validation set")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
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