#!/usr/bin/env python3
"""
Combines extracted FAQ JSON files and uploads the result to a Modal Volume.
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import List, Dict, Any

import modal

# --- Configuration ---
APP_NAME = "faq-data-uploader"
# Directory containing all outputs (faqs and chunks will be found in subfolders)
LOCAL_SOURCE_DIR = Path("./synthetic_qa/output/")
# Name for the combined file that will be created
COMBINED_FILENAME = "source_json_combined.json"
# Target directory WITHIN the Modal Volume where the combined file will be uploaded
TARGET_VOLUME_DIR = "/data" # Directory inside the volume
VOLUME_NAME = "faq-unb-chatbot-gemma-raft-data"

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# --- Modal Setup ---
app = modal.App(APP_NAME)

# Define the Volume (ensure it exists or set create_if_missing=True)
# It's better if the volume is already created by your training/inference setup
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

# Define a simple image, doesn't need GPU or many dependencies
image = modal.Image.debian_slim(python_version="3.10").pip_install("modal-client")

# --- Helper Function to Recursively Combine JSON Files from Multiple Sources ---
def combine_source_json_files(source_dirs: List[Path]) -> List[Dict[str, Any]]:
    """
    Recursively finds all .json files that are under a directory named 'extracted_faq' or 'extracted_chunks',
    regardless of nesting, and combines their contents (each file contains a list of dicts).
    """
    combined_data = []
    valid_dirnames = {"extracted_faq", "extracted_chunks"}
    for source_dir in source_dirs:
        if not source_dir.is_dir():
            logger.warning(f"Source directory not found: {source_dir}")
            continue
        # Recursively search for all subdirectories named extracted_faq or extracted_chunks
        for subdir in source_dir.rglob("*"):
            if subdir.is_dir() and subdir.name in valid_dirnames:
                json_files = list(subdir.rglob("*.json"))
                if not json_files:
                    logger.warning(f"No JSON files found in directory: {subdir}")
                    continue
                logger.info(f"Found {len(json_files)} JSON files to combine in {subdir} (recursively)...")
                for json_file in json_files:
                    logger.info(f"Found file: {json_file}")
                    try:
                        with open(json_file, "r", encoding="utf-8") as f:
                            data = json.load(f)
                            if isinstance(data, list):
                                valid_items = [item for item in data if isinstance(item, dict)]
                                combined_data.extend(valid_items)
                                logger.debug(f"Added {len(valid_items)} items from {json_file.relative_to(subdir)}")
                            else:
                                logger.warning(f"Skipping {json_file.name}: Content is not a list.")
                    except json.JSONDecodeError:
                        logger.error(f"Skipping {json_file.name}: Invalid JSON format.")
                    except Exception as e:
                        logger.error(f"Skipping {json_file.name}: Error reading file - {e}")
    logger.info(f"Total combined items (FAQ + chunks): {len(combined_data)}")
    return combined_data

# --- Modal Function to Upload Data ---
@app.function(image=image, volumes={TARGET_VOLUME_DIR: volume})
def upload_combined_data(combined_data: List[Dict[str, Any]], target_filename: str):
    """
    Writes the combined data list to a JSON file inside the specified
    directory within the Modal Volume.
    """
    target_path = Path(TARGET_VOLUME_DIR) / target_filename
    logger.info(f"Attempting to write {len(combined_data)} items to {target_path} in Modal Volume '{VOLUME_NAME}'...")

    try:
        # Ensure target directory exists within the volume mount path
        target_path.parent.mkdir(parents=True, exist_ok=True)

        with target_path.open("w", encoding="utf-8") as f:
            json.dump(combined_data, f, ensure_ascii=False, indent=2)

        # IMPORTANT: Commit changes to make them persistent in the volume
        volume.commit()
        logger.info(f"Successfully wrote combined data to {target_path} and committed to volume.")

    except Exception as e:
        logger.error(f"Failed to write or commit data to {target_path}: {e}", exc_info=True)
        # Depending on the error, you might want to raise it
        raise

# --- Local Entrypoint to Run the Process ---
@app.local_entrypoint()
def main(local_source_dir: str = str(LOCAL_SOURCE_DIR)):
    """
    Local entrypoint:
    1. Recursively finds all subfolders named 'extracted_faq' or 'extracted_chunks' under the source dir and combines their .json files.
    2. Calls the Modal function to upload the combined data.
    """
    logger.info(f"Starting FAQ/chunk data combination and upload process.")
    source_dir = Path(local_source_dir)
    logger.info(f"Combining JSON files from subfolders of: {source_dir}")
    combined_data = combine_source_json_files([source_dir])
    if not combined_data:
        logger.error("No data combined. Aborting upload.")
        sys.exit(1) # Exit with error code
    # 2. Upload to Modal Volume via remote function
    logger.info(f"Calling Modal function to upload combined data as '{COMBINED_FILENAME}'...")
    try:
        upload_combined_data.remote(combined_data, COMBINED_FILENAME)
        logger.info("Modal upload function executed.")
        logger.info(f"Verify the file exists in volume '{VOLUME_NAME}' at path '{TARGET_VOLUME_DIR}/{COMBINED_FILENAME}'")
        logger.info(f"Example verification command: modal volume ls {VOLUME_NAME} {TARGET_VOLUME_DIR}")
    except Exception as e:
        logger.error(f"An error occurred during the Modal function execution: {e}", exc_info=True)
        sys.exit(1) # Exit with error code
    logger.info("Process completed.")