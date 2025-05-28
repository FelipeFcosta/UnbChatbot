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
        
        logger.info(f"Initialized QA Generator in standard mode")


    def process_directory(self, output_dir: str, max_workers: int = 4) -> None:
        """
        Process all files in a directory to generate synthetic QA data.
        
        Args:
            output_dir: Directory to save the generated QA data
            max_workers: Maximum number of concurrent workers for processing
        """
        input_path = Path(self.config["base_dir"])
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Get all supported files
        supported_extensions = self.config.get("file_processing", {}).get(
            "include_extensions", ['.html', '.htm', '.pdf', '.txt', '.md', '.docx', '.doc']
        )
        all_files_paths = []
        
        for ext in supported_extensions:
            all_files_paths.extend(list(input_path.glob(f"**/*{ext}")))
            
        if not all_files_paths:
            logger.warning(f"No supported files found in {self.config['base_dir']}")
            return
            
        logger.info(f"Found {len(all_files_paths)} files to process")
        
        # First identify potential FAQ files (only HTML files can be FAQs)
        all_files = []
        
        for file_path in all_files_paths:
            if file_path.suffix.lower() in ['.html', '.htm']:
                try:
                    soup = FileProcessor.preprocess_html(file_path)
                    if FAQProcessor.detect_faq_document(soup, file_path.name):
                        logger.info(f"Detected FAQ document: {file_path.relative_to(input_path)}")
                        all_files.append((FileType.FAQ, soup, file_path))
                    elif ComponentProcessor.detect_component_document(file_path, self.config):
                        logger.info(f"Detected Component document: {file_path.relative_to(input_path)}")
                        all_files.append((FileType.COMPONENT, soup, file_path))
                    elif OfferingsProcessor.detect_offerings_document(file_path, self.config):
                        logger.info(f"Detected Offerings document: {file_path.relative_to(input_path)}")
                        all_files.insert(0, (FileType.OFFERINGS, soup, file_path)) # process offerings first
                    else:
                        all_files.append((FileType.REGULAR, soup, file_path))
                except Exception as e:
                    logger.error(f"Error checking if {file_path} is FAQ: {e}")
                    all_files.append((FileType.REGULAR, None, file_path))
            else:
                all_files.append((FileType.REGULAR, None, file_path))


        files_process_batches = []

        faq_qa_pairs = []
        regular_qa_pairs = []

        if all_files:
            faq_files = []
            regular_files = []
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_file = {
                    executor.submit(self.process_file, soup, file_path, input_path, output_path, type): (type, soup, file_path)
                    for type, soup, file_path in all_files
                }
                for future in tqdm(as_completed(future_to_file), total=len(all_files), desc="Processing each file for default QA pairs"):
                    type, soup, file_path = future_to_file[future]
                    rel_path = file_path.relative_to(input_path)

                    try:
                        qa_pairs = future.result()
                        if qa_pairs:
                            if type == FileType.FAQ:
                                faq_files.append((soup, file_path, rel_path))
                            else:
                                regular_files.append((soup, file_path, rel_path))
                    except Exception as e:
                        logger.error(f"Error processing file {file_path}: {e}")
            if faq_files or regular_files:
                files_process_batches.append((FileType.FAQ, faq_files))
                files_process_batches.append((FileType.REGULAR, regular_files))
        else:
            logger.info("No files can be processed!")
            return

        for batch_type, files_to_process in files_process_batches:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future = executor.submit(QAProcessorRAFT.generate_raft_training_data, files_to_process, output_path, self.config)
                future_to_desc = {future: batch_type}
                for future in tqdm(as_completed(future_to_desc), total=1, desc=f"Processing {batch_type.name.lower()} files with QAProcessorRAFT"):
                    try:
                        raft_qa_pairs = future.result()
                        if raft_qa_pairs:
                            if batch_type == FileType.FAQ:
                                faq_qa_pairs.extend(raft_qa_pairs)
                                logger.info(f"Generated {len(raft_qa_pairs)} QA pairs from FAQ files using QAProcessorRAFT")
                            else:
                                regular_qa_pairs.extend(raft_qa_pairs)
                                logger.info(f"Generated {len(raft_qa_pairs)} QA pairs from regular files using QAProcessorRAFT")
                    except Exception as e:
                        logger.error(f"Error processing {batch_type.name.lower()} files with QAProcessorRAFT: {e}")


        # Combine FAQ and non-FAQ QA pairs
        all_qa_pairs = faq_qa_pairs + regular_qa_pairs
                    
        # Save all QA pairs to a final JSON file
        if all_qa_pairs:
            final_output = output_path / "synthetic_qa_data_raft.json"
            with open(final_output, 'w', encoding='utf-8') as f:
                json.dump(all_qa_pairs, f, ensure_ascii=False, indent=2)

            logger.info(f"Generated a total of {len(all_qa_pairs)} QA pairs ({len(faq_qa_pairs)} from FAQs, {len(regular_qa_pairs)} from regular documents)")
            logger.info(f"Final output saved to {final_output}")
        else:
            logger.warning("No QA pairs were generated")


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
                file_title = soup.title.get_text(strip=True)
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
                text = self.file_processor.extract_text_from_file(file_path, self.config)

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
                    with open(extracted_offerings_path, 'r', encoding='utf-8') as f:
                        original_offerings = json.load(f)
                else:
                    original_offerings = OfferingsProcessor.extract_offerings_from_text(text, file_path, self.config)
                    with open(extracted_offerings_path, 'w', encoding='utf-8') as f:
                        json.dump(original_offerings, f, ensure_ascii=False, indent=2)

                logger.info(f"Skipping offerings document {file_path} full processing")
                return []
            
            if file_type == FileType.COMPONENT:
                # add course offerings to the text
                acronym = file_path.stem[0:3]
                text = ComponentProcessor.add_course_offerings_to_text(text, acronym, self.config)
            
            if file_type is not FileType.FAQ:
                # Try to load chunks from file if it exists, otherwise generate and save
                extracted_chunks_dir = output_dir / "extracted_chunks"
                extracted_chunks_dir.mkdir(parents=True, exist_ok=True)
                extracted_chunks_path = extracted_chunks_dir / f"{safe_title_slug}_{file_hash}.json"

                if os.path.exists(extracted_chunks_path):
                    logger.debug(f"Chunks already exist for {rel_path}")
                    with open(extracted_chunks_path, 'r', encoding='utf-8') as f:
                        chunks = json.load(f)
                    logger.debug(f"Loaded {len(chunks)} chunks from {rel_path}")
                else:
                    chunks = self.text_chunker.chunk_text(text, rel_path)
                    with open(extracted_chunks_path, 'w', encoding='utf-8') as f:
                        json.dump(chunks, f, ensure_ascii=False, indent=2)
                    logger.debug(f"Created {len(chunks)} chunks from {rel_path}")
                
                if not chunks:
                    logger.debug(f"No chunks created from {rel_path}")
                    return []
                

            context_html_text = None
            # Attempt to find and process a related HTML file for context,
            # if the current file is not HTML itself, based on 'user.source_html_path' metadata.
            if file_path.suffix.lower() not in ['.html', '.htm']:
                logger.info(f"File {file_path.name} is not HTML. Attempting to find source HTML context.")
                
                source_html_path: str | None = None
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
                                extracted_context_text = self.file_processor.extract_text_from_file(source_html_file_path, self.config)
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

            if file_type == FileType.FAQ:
                # check if already extracted
                extracted_faq_dir = output_dir / "extracted_faq"
                extracted_faq_dir.mkdir(parents=True, exist_ok=True)
                extracted_faq_path = extracted_faq_dir / f"{safe_title_slug}_{file_hash}.json"

                if os.path.exists(extracted_faq_path):
                    logger.debug(f"FAQ already extracted for {rel_path}")
                    with open(extracted_faq_path, 'r', encoding='utf-8') as f:
                        original_faq = json.load(f)
                else:
                    original_faq = FAQProcessor.extract_faq_from_text(text, file_path, self.config)
                    with open(extracted_faq_path, 'w', encoding='utf-8') as f:
                        json.dump(original_faq, f, ensure_ascii=False, indent=2)

                qa_pairs = self.qa_generator.generate_qa_pairs_from_faq(
                    original_faq=original_faq,
                    file_path=file_path,
                    file_title=file_title,
                    output_dir=output_dir,
                    batch_size=5
                )
            else:
                qa_pairs = self.qa_generator.generate_qa_pairs(
                    chunks=chunks,
                    file_path=file_path,
                    file_title=file_title,
                    output_dir=output_dir,
                    full_document_text=text,
                    context_html_text=context_html_text,
                    batch_size=5
                )

            return qa_pairs
            
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            return []


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
        generator.process_directory(args.output, args.threads)
    except Exception as e:
        logger.error(f"Error during execution: {e}")
        raise


if __name__ == "__main__":
    main()