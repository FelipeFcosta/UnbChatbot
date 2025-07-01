#!/usr/bin/env python3
"""
Synthetic QA Generator for Institutional Chatbots

This script processes institutional content (HTML, PDF, etc.) and generates
synthetic question-answer pairs for training chatbots.
"""

import os
import argparse
import logging
import json
from bs4 import BeautifulSoup
import yaml
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from slugify import slugify
import re
import fnmatch
import hashlib
import threading
import time

# Import modules
from modules.file_processor import FileProcessor
from modules.faq_processor import FAQProcessor
from modules.qa_processor_raft import QAProcessorRAFT
from modules.text_chunker import TextChunker
from modules.qa_generator import QAGenerator
from modules.utils import create_hash, FileType
from modules.component_processor import ComponentProcessor
from modules.offerings_processor import OfferingsProcessor

# Logging setup
with open(config_path := os.path.join(os.path.dirname(__file__), 'config.yaml'), 'r', encoding='utf-8') as f:
    _config = yaml.safe_load(f)
logging_level = getattr(logging, _config.get('global', {}).get('logging_level', 'INFO').upper(), logging.INFO)
logging.basicConfig(
    level=logging_level,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


class SyntheticQADataGenerator:
    """Main class for generating synthetic QA data from a corpus of documents."""
    
    def __init__(self, config_path: str):
        """Initialize with the given configuration file."""
        self.config_path = Path(config_path)
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
            
        # Load config
        with open(self.config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
            
        
        # Initialize components
        self.file_processor = FileProcessor()
        self.text_chunker = TextChunker(self.config)
        self.qa_generator = QAGenerator(self.config)
        self.processed_files_hashes = set()
        self._hash_lock = threading.Lock()
        
        logger.info(f"Initialized QA Generator in standard mode")


    @staticmethod
    def _calculate_file_hash(file_path: Path) -> str:
        """
        Calculate a SHA256 hash of a file's content using the first and last 4KB.
        """
        h = hashlib.sha256()
        chunk_size = 4096

        try:
            with file_path.open("rb") as f:
                h.update(f.read(chunk_size))
                if os.path.getsize(file_path) > chunk_size:
                    f.seek(-chunk_size, os.SEEK_END)
                    h.update(f.read(chunk_size))
        except (IOError, OSError) as e:
            logger.error(f"Error reading file {file_path} for hashing: {e}")
            return "" # Return empty hash on error

        return h.hexdigest()

    def _handle_ignored_folder_outputs(self, output_path: Path) -> None:
        """Rename residual extracted folders when an input folder is fully ignored. (primarily for RAG reasons)
        """
        for folder_name in ("extracted_chunks", "extracted_faq"):
            original_path = output_path / folder_name
            if original_path.exists() and original_path.is_dir():
                new_path = output_path / f"_{folder_name}"
                if new_path.exists():
                    continue
                try:
                    original_path.rename(new_path)
                    logger.info(
                        f"Renamed leftover '{original_path}' to '{new_path}' because its source folder is ignored."
                    )
                except Exception as e:
                    logger.error(f"Unable to rename {original_path} to {new_path}: {e}")

    def process_directory(self, input_dir: str, output_dir: str) -> None:
        """
        Process all files in a directory (non-recursively) to generate synthetic QA data.
        Args:
            input_dir: Directory to scan for files
            output_dir: Directory to save the generated QA data
        """
        input_path = Path(input_dir)
        base_dir = Path(self.config["base_dir"])

        output_path = Path(output_dir)

        # Get all supported files
        supported_extensions = self.config.get("file_processing", {}).get(
            "include_extensions", ['.html', '.htm', '.pdf', '.txt', '.md']
        )
        all_files_paths = []
        for ext in supported_extensions:
            all_files_paths.extend(list(input_path.glob(f"*{ext}")))

        # Get ignore patterns from config
        ignore_patterns = self.config.get("file_processing", {}).get("ignore", [])

        # Filter out ignored files by their relative path to base_dir, supporting wildcards
        filtered_files_paths = []
        for file_path in all_files_paths:
            rel_path = str(file_path.relative_to(base_dir))
            if any(fnmatch.fnmatch(rel_path, pattern) for pattern in ignore_patterns):
                logger.info(f"Skipping ignored file or folder: {rel_path}")
                continue

            try:
                if file_path.stat().st_size == 0:
                    logger.info(f"Skipping empty file: {rel_path}")
                    continue
            except (OSError, IOError) as e:
                continue

            # Check for duplicate file content
            file_hash = self._calculate_file_hash(file_path)
            with self._hash_lock: # lock to avoid race conditions
                if file_hash in self.processed_files_hashes:
                    logger.info(f"Skipping duplicate file: {rel_path} (content already processed)")
                    continue
                self.processed_files_hashes.add(file_hash)

            filtered_files_paths.append(file_path)
        all_files_paths = filtered_files_paths

        if not all_files_paths:
            self._handle_ignored_folder_outputs(output_path)
            logger.warning(f"No supported files found in {input_path}")
            return

        logger.info(f"Found {len(all_files_paths)} files to process in {input_path}")
        output_path.mkdir(parents=True, exist_ok=True)

        # First identify potential FAQ files (only HTML files can be FAQs)
        all_files = []
        
        for file_path in all_files_paths:
            if file_path.suffix.lower() in ['.html', '.htm']:
                try:
                    soup = FileProcessor.preprocess_html(file_path)
                    if FAQProcessor.detect_faq_document(soup, file_path.name):
                        logger.info(f"Detected FAQ document: {file_path.relative_to(base_dir)}")
                        all_files.append((FileType.FAQ, soup, file_path))
                    elif ComponentProcessor.detect_component_document(file_path, self.config):
                        logger.info(f"Detected Component document: {file_path.relative_to(base_dir)}")
                        all_files.append((FileType.COMPONENT, soup, file_path))
                    elif OfferingsProcessor.detect_offerings_document(file_path, self.config):
                        logger.info(f"Detected Offerings document: {file_path.relative_to(base_dir)}")
                        all_files.insert(0, (FileType.OFFERINGS, soup, file_path)) # process offerings first
                    else:
                        all_files.append((FileType.REGULAR, soup, file_path))
                except Exception as e:
                    logger.error(f"Error checking if {file_path} is FAQ: {e}")
                    all_files.append((FileType.REGULAR, None, file_path))
            else:
                all_files.append((FileType.REGULAR, None, file_path))

        faq_qa_pairs = []
        regular_qa_pairs = []
        component_qa_pairs = []
        if all_files:
            files_to_process = []
            
            # Process files sequentially in this directory
            for type, soup, file_path in tqdm(all_files, desc="Processing files: " + input_dir, leave=True):
                rel_path = file_path.relative_to(base_dir)

                try:
                    self.process_file(soup, file_path, base_dir, output_path, type)
                    if type != FileType.OFFERINGS:  # Skip offerings for RAFT processing
                        files_to_process.append((soup, file_path, rel_path, type))

                except Exception as e:
                    logger.error(f"Error processing file {file_path}: {e}")

            if files_to_process:
                # Process all files together in RAFT
                try:
                    raft_qa_pairs = QAProcessorRAFT.generate_raft_training_data(files_to_process, output_path, self.config)
                    if raft_qa_pairs:
                        logger.info(f"Generated {len(raft_qa_pairs)} QA pairs using QAProcessorRAFT")
                        # Separate by type for final output if needed
                        for qa_pair in raft_qa_pairs:
                            file_type = qa_pair.get('file_type', FileType.REGULAR)
                            if file_type == FileType.FAQ:
                                faq_qa_pairs.append(qa_pair)
                            elif file_type == FileType.COMPONENT:
                                component_qa_pairs.append(qa_pair)
                            else:
                                regular_qa_pairs.append(qa_pair)
                except Exception as e:
                    logger.error(f"Error processing files with QAProcessorRAFT: {e}")
        else:
            logger.info("No files can be processed!")
            return

        # Combine FAQ and non-FAQ QA pairs
        all_qa_pairs = faq_qa_pairs + regular_qa_pairs + component_qa_pairs
                    
        if not all_qa_pairs:
            logger.warning(f"No QA pairs generated for {input_dir}")




    def process_file(self, soup, file_path, base_dir, output_dir, file_type: FileType):
        """
        Process a single file to generate synthetic QA pairs.

        Extract text from the file, chunk it, and then generate QA pairs using the QA generator.
        If the file is not an HTML file, it attempts to find a related HTML file for additional context.

        Returns:
            list: A list of generated QA pairs. Each QA pair is a dictionary containing 'question' and 'answer' keys.
        """
        logger.debug(f"Processing file {file_path}")
        
        try:
            # Extract relative path for source info
            rel_path = file_path.relative_to(base_dir)
            file_title = file_path.stem
            if soup and soup.title:
                title_text = soup.title.get_text(strip=True)
                if (len(title_text) > 0):
                    file_title = title_text
            file_hash = create_hash(str(rel_path))
            safe_title_slug = slugify(file_title)
            
            # Extract text from file with appropriate settings
            extracted_text_dir = output_dir / "extracted_text"
            extracted_text_dir.mkdir(parents=True, exist_ok=True)
            extracted_text_path = extracted_text_dir / f"{safe_title_slug}_{file_hash}.txt"

            if os.path.exists(extracted_text_path):
                logger.debug(f"Structured text already exists for {rel_path}")
                with open(extracted_text_path, 'r', encoding='utf-8') as f:
                    text = f.read()
            else:
                logger.info(f"Extracting text from {file_path}")
                text = self.file_processor.extract_text_from_file(file_path, file_type, self.config)

                if file_type == FileType.COMPONENT:
                    try:
                        component_code = os.getxattr(str(file_path), b'user.component_code').decode('utf-8')
                    except Exception as e:
                        component_code = file_path.stem[:7]
                    # add course offerings to the text
                    text = ComponentProcessor.add_course_offerings_to_text(text, component_code, rel_path, output_dir, self.config)

                if text:
                    with open(extracted_text_path, 'w', encoding='utf-8') as f:
                        f.write(text)

            if not text:
                logger.warning(f"No text extracted from {rel_path}")
                return []

            # don't continue processing offerings since they will be attached to components
            if file_type == FileType.OFFERINGS:
                # check if already extracted
                extracted_offerings_dir = output_dir / "extracted_offerings"
                extracted_offerings_dir.mkdir(parents=True, exist_ok=True)
                extracted_offerings_path = extracted_offerings_dir / f"{safe_title_slug}_{file_hash}.json"

                if os.path.exists(extracted_offerings_path):
                    logger.debug(f"Offerings already extracted for {rel_path}")
                else:
                    logger.info(f"Extracting offerings from {file_path}")
                    original_offerings = OfferingsProcessor.extract_offerings_from_text(text, file_path, self.config)
                    with open(extracted_offerings_path, 'w', encoding='utf-8') as f:
                        json.dump(original_offerings, f, ensure_ascii=False, indent=2)

                logger.info(f"Skipping full processing for offerings document {file_path}")
                return []
                    
            # Handle chunking for both REGULAR and COMPONENT files (same logic)
            if file_type in [FileType.REGULAR, FileType.COMPONENT]:
                extracted_chunks_dir = output_dir / "extracted_chunks"
                extracted_chunks_dir.mkdir(parents=True, exist_ok=True)
                extracted_chunks_path = extracted_chunks_dir / f"{safe_title_slug}_{file_hash}.json"

                if os.path.exists(extracted_chunks_path):
                    logger.debug(f"Chunks already exist for {rel_path}")
                    with open(extracted_chunks_path, 'r', encoding='utf-8') as f:
                        chunks = json.load(f)
                    logger.debug(f"Loaded {len(chunks)} chunks from {rel_path}")
                    # add metadata retroactively
                    if self.text_chunker.add_metadata_to_items(chunks, file_path, file_title, file_type):
                        with open(extracted_chunks_path, 'w', encoding='utf-8') as f:
                            json.dump(chunks, f, ensure_ascii=False, indent=2)
                        logger.info(f"Added metadata and loaded {len(chunks)} chunks to {extracted_chunks_path}")
                else:
                    chunks = self.text_chunker.chunk_text(text, rel_path)
                    if chunks:
                        if self.text_chunker.add_metadata_to_items(chunks, file_path, file_title, file_type):
                            with open(extracted_chunks_path, 'w', encoding='utf-8') as f:
                                json.dump(chunks, f, ensure_ascii=False, indent=2)
                            logger.info(f"Saved {len(chunks)} chunks to {extracted_chunks_path}")
                
                if not chunks:
                    logger.debug(f"No chunks created from {rel_path}")
                    return []

            # Attempt to find and process a related HTML file for context based on 'user.source_html_path' metadata.
            context_html_text = None
            if file_path.suffix.lower() not in ['.html', '.htm']:
                logger.info(f"File {file_path.name} is not HTML. Attempting to find source HTML context.")
                
                source_html_path = None
                try:
                    source_html_path = os.getxattr(str(file_path), b'user.source_html_path').decode('utf-8')
                except Exception as e:
                    logger.error(f"Error retrieving 'user.source_html_path' for {file_path.name}: {e}", exc_info=True)

                if source_html_path:
                    try:
                        source_html_file_path = base_dir / source_html_path
                        
                        if source_html_file_path.exists() and source_html_file_path.is_file():
                            source_html_file_hash = create_hash(str(source_html_path))
                            soup_source_html = FileProcessor.get_soup(source_html_file_path)
                            safe_source_html_title_slug = slugify(soup_source_html.title.get_text(strip=True))

                            source_html_extracted_text_path = extracted_text_dir / f"{safe_source_html_title_slug}_{source_html_file_hash}.txt"

                            if os.path.exists(source_html_extracted_text_path):
                                logger.debug(f"Context HTML text already cached for {source_html_file_path.name} at {source_html_extracted_text_path}")
                                with open(source_html_extracted_text_path, 'r', encoding='utf-8') as f:
                                    context_html_text = f.read()
                            else:
                                logger.info(f"Extracting text from source HTML file: {source_html_file_path.name}")
                                extracted_context_text = self.file_processor.extract_text_from_file(source_html_file_path, FileType.REGULAR, self.config)
                                if extracted_context_text:
                                    context_html_text = extracted_context_text
                                    with open(source_html_extracted_text_path, 'w', encoding='utf-8') as f:
                                        f.write(context_html_text)
                                    logger.debug(f"Saved extracted text for source HTML {source_html_file_path.name} to {source_html_extracted_text_path}")
                                else:
                                    logger.warning(f"No text could be extracted from source HTML file: {source_html_file_path.name}")
                        else:
                            logger.error(
                                f"Source HTML file {source_html_file_path} (from 'user.source_html_path' value: '{source_html_path}') "
                                f"does not exist locally or is not a file. This contradicts the 'absolute correct' assumption."
                            )
                    
                    except Exception as e:
                        logger.error(f"Error processing source HTML context from path '{source_html_path}' for file {file_path.name}: {e}", exc_info=True)
            
            else: # file_path.suffix.lower() is '.html' or '.htm'
                logger.debug(f"File {file_path.name} is already HTML/HTM. No separate source HTML context needed.")

            # Generate QA pairs based on file type
            if file_type == FileType.FAQ:
                # check if already extracted
                extracted_faq_dir = output_dir / "extracted_faq"
                extracted_faq_dir.mkdir(parents=True, exist_ok=True)
                extracted_faq_path = extracted_faq_dir / f"{safe_title_slug}_{file_hash}.json"

                if os.path.exists(extracted_faq_path):
                    logger.debug(f"FAQ already extracted for {rel_path}")
                    with open(extracted_faq_path, 'r', encoding='utf-8') as f:
                        original_faq = json.load(f)
                    # add metadata retroactively
                    if self.text_chunker.add_metadata_to_items(original_faq, file_path, file_title, file_type):
                        with open(extracted_faq_path, 'w', encoding='utf-8') as f:
                            json.dump(original_faq, f, ensure_ascii=False, indent=2)
                        logger.info(f"Added metadata to FAQ items for {extracted_faq_path}")
                else:
                    original_faq = FAQProcessor.extract_faq_from_text(text, file_path, self.config)
                    # Add metadata fields to each FAQ item if missing
                    if self.text_chunker.add_metadata_to_items(original_faq, file_path, file_title, file_type):
                        with open(extracted_faq_path, 'w', encoding='utf-8') as f:
                            json.dump(original_faq, f, ensure_ascii=False, indent=2)
                        logger.info(f"Added metadata to FAQ items for {extracted_faq_path}")
                    with open(extracted_faq_path, 'w', encoding='utf-8') as f:
                        json.dump(original_faq, f, ensure_ascii=False, indent=2)

                try:
                    qa_pairs = self.qa_generator.generate_qa_pairs_from_faq(
                        original_faq=original_faq,
                        file_path=file_path,
                        file_title=file_title,
                        output_dir=output_dir,
                        batch_size=5
                    )
                except Exception as e:
                    logger.error(f"Error generating QA pairs from FAQ: {e}")
                    return []
            elif file_type in [FileType.REGULAR, FileType.COMPONENT]:
                batch_size = 10 if file_type == FileType.COMPONENT else 5
                qa_pairs = self.qa_generator.generate_qa_pairs(
                    chunks=chunks,
                    file_path=file_path,
                    file_title=file_title,
                    output_dir=output_dir,
                    full_document_text=text,
                    context_html_text=context_html_text,
                    batch_size=batch_size
                )

            return qa_pairs
            
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            return []

    def process_all_directories(self, root_input_dir: str, output_dir: str, max_workers: int = 4) -> None:
        """
        Recursively process all directories under root_input_dir, calling process_directory for each one.
        Args:
            root_input_dir: Root directory to start searching for subdirectories
            output_dir: Directory to save output files (will mirror input structure)
            max_workers: Maximum number of concurrent workers for processing directories
        """
        root_path = Path(root_input_dir)
        directories_to_process = []
        
        # Check if exclusive folder processing is enabled
        exclusive_folder = self.config.get("debugging", {}).get("exclusive_folder")
        if exclusive_folder:
            logger.info(f"Debugging mode: Processing only folder '{exclusive_folder}'")
            target_dir = root_path / exclusive_folder
            if target_dir.exists() and target_dir.is_dir():
                rel_dir = os.path.relpath(target_dir, root_path)
                out_dir = os.path.join(output_dir, rel_dir) if rel_dir != '.' else output_dir
                directories_to_process.append((str(target_dir), out_dir))
                logger.info(f"Target directory found: {target_dir}")
            else:
                logger.error(f"Exclusive folder '{exclusive_folder}' not found or is not a directory in {root_path}")
                return
        else:
            # Collect all directories that need processing
            for dirpath, dirnames, filenames in os.walk(root_path):
                rel_dir = os.path.relpath(dirpath, root_path)
                out_dir = os.path.join(output_dir, rel_dir) if rel_dir != '.' else output_dir
                directories_to_process.append((dirpath, out_dir))
        
        # Separate offerings directories from others
        offerings_dirs = []
        other_dirs = []
        
        for input_dir, sub_output_dir in directories_to_process:
            input_path = Path(input_dir)
            # check if there are any offerings documents in the directory
            has_offerings = any(
                OfferingsProcessor.detect_offerings_document(file_path, self.config)
                for file_path in list(input_path.glob("*.html")) + list(input_path.glob("*.htm"))
                if file_path.is_file()
            )
            
            if has_offerings:
                offerings_dirs.append((input_dir, sub_output_dir))
            else:
                other_dirs.append((input_dir, sub_output_dir))
        
        # Process offerings directories first
        if offerings_dirs:
            logger.info(f"Processing {len(offerings_dirs)} offerings directories first")
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_dir = {
                    executor.submit(self.process_directory, input_dir, sub_output_dir): (input_dir, sub_output_dir)
                    for input_dir, sub_output_dir in offerings_dirs
                }
                for future in tqdm(as_completed(future_to_dir), total=len(offerings_dirs), desc="Processing offerings directories"):
                    input_dir, sub_output_dir = future_to_dir[future]
                    try:
                        future.result()
                    except Exception as e:
                        logger.error(f"Error processing offerings directory {input_dir}: {e}")
        
        # Then process all other directories
        if other_dirs:
            logger.info(f"Processing {len(other_dirs)} other directories")
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_dir = {
                    executor.submit(self.process_directory, input_dir, sub_output_dir): (input_dir, sub_output_dir)
                    for input_dir, sub_output_dir in other_dirs
                }
                for future in tqdm(as_completed(future_to_dir), total=len(other_dirs), desc="Processing other directories"):
                    input_dir, sub_output_dir = future_to_dir[future]
                    try:
                        future.result()
                    except Exception as e:
                        logger.error(f"Error processing directory {input_dir}: {e}")

        # After all directories are processed, join all raft_training_data_*.jsonl files
        all_raft_files = list(Path(output_dir).rglob('raft_training_data_*.jsonl'))
        
        ignore_patterns = self.config.get("file_processing", {}).get("ignore", [])
        filtered_raft_files = []
        
        for raft_file in all_raft_files:
            try:
                rel_output_path = raft_file.relative_to(Path(output_dir))
                rel_dir_path = str(rel_output_path.parent) if rel_output_path.parent != Path('.') else ''
                if rel_dir_path and any(fnmatch.fnmatch(rel_dir_path, pattern) for pattern in ignore_patterns):
                    continue
                filtered_raft_files.append(raft_file)
            except Exception as e:
                filtered_raft_files.append(raft_file)
        
        all_qa_pairs = []
        for raft_file in filtered_raft_files:
            with open(raft_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        all_qa_pairs.append(json.loads(line))
                    except Exception as e:
                        logger.error(f"Error reading line in {raft_file}: {e}")
        if all_qa_pairs:
            final_output = Path(output_dir) / "synthetic_qa_data_raft.json"
            with open(final_output, 'w', encoding='utf-8') as f:
                json.dump(all_qa_pairs, f, ensure_ascii=False, indent=2)
            logger.info(f"Final synthetic_qa_data_raft.json saved with {len(all_qa_pairs)} QA pairs at {final_output}")
        else:
            logger.warning(f"No RAFT QA pairs found to join in {output_dir}")


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Generate synthetic QA data from institutional content.")
    parser.add_argument("--config", default="synthetic_qa/config.yaml", help="Path to configuration YAML file")
    parser.add_argument("--input", required=True, help="Directory containing input files")
    parser.add_argument("--output", default="./output", help="Directory to save output files")
    parser.add_argument("--threads", type=int, default=4, help="Maximum number of concurrent workers")
    args = parser.parse_args()

    # Update config file to include (or replace) base_dir
    with open(args.config, 'r', encoding='utf-8') as f:
        config_text = f.read()
    config_text = re.sub(r'\n?base_dir:.*\n?', '', config_text)
    config_text += f"\nbase_dir: {args.input}  # set by synthetic_qa_generator\n"
    with open(args.config, 'w', encoding='utf-8') as f:
        f.write(config_text)

    try:
        generator = SyntheticQADataGenerator(args.config)
        generator.process_all_directories(args.input, args.output, args.threads)
    except Exception as e:
        logger.error(f"Error during execution: {e}")
        raise


if __name__ == "__main__":
    main()