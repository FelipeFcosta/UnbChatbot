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

# Import modules
from modules.file_processor import FileProcessor
from modules.faq_processor import FAQProcessor
from modules.faq_processor_raft import FAQProcessorRAFT
from modules.text_chunker import TextChunker
from modules.qa_generator import QAGenerator
from modules.utils import get_hash

# Logging setup
logging.basicConfig(
    level=logging.INFO,
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


    def process_directory(self, input_dir: str, output_dir: str, max_workers: int = 4) -> None:
        """
        Process all files in a directory to generate synthetic QA data.
        
        Args:
            input_dir: Directory containing the files to process
            output_dir: Directory to save the generated QA data
            max_workers: Maximum number of concurrent workers for processing
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Get all supported files
        supported_extensions = self.config.get("file_processing", {}).get(
            "include_extensions", ['.html', '.htm', '.pdf', '.txt', '.md', '.docx', '.doc']
        )
        all_files = []
        
        for ext in supported_extensions:
            all_files.extend(list(input_path.glob(f"**/*{ext}")))
            
        if not all_files:
            logger.warning(f"No supported files found in {input_dir}")
            return
            
        logger.info(f"Found {len(all_files)} files to process")
        
        # First identify potential FAQ files (only HTML files can be FAQs)
        faq_files = []
        regular_files = []
        
        for file_path in all_files:
            rel_path = file_path.relative_to(input_path)
            if file_path.suffix.lower() in ['.html', '.htm']:
                try:
                    soup = FileProcessor.get_soup(file_path)
                    if FAQProcessor.detect_faq_document(soup, file_path.name):
                        logger.info(f"Detected FAQ document: {file_path.relative_to(input_path)}")
                        faq_files.append((soup, file_path, rel_path))
                    else:
                        regular_files.append((soup, file_path, rel_path))
                except Exception as e:
                    logger.error(f"Error checking if {file_path} is FAQ: {e}")
                    regular_files.append((None, file_path, rel_path))
            else:
                regular_files.append((None, file_path, rel_path))
        
        files_process_batches = []

        if len(faq_files) > 0:
            logger.info(f"Identified {len(faq_files)} FAQ files and {len(regular_files)} non-FAQ files")
            files_process_batches.append(("faq", faq_files))
        
        faq_qa_pairs = []
        regular_qa_pairs = []

        if regular_files:
            processed_regular_files = []
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_file = {
                    executor.submit(self.process_file, soup, file_path, input_path, output_path): (soup, file_path)
                    for soup, file_path, _ in regular_files
                }
                for future in tqdm(as_completed(future_to_file), total=len(regular_files), desc="Processing non-FAQ files"):
                    soup, file_path = future_to_file[future]
                    rel_path = file_path.relative_to(input_path)

                    try:
                        qa_pairs = future.result()
                        if qa_pairs:
                            processed_regular_files.append((soup, file_path, rel_path))
                            logger.info(f"Generated {len(qa_pairs)} QA pairs from {file_path.relative_to(input_path)}")
                    except Exception as e:
                        logger.error(f"Error processing file {file_path}: {e}")
            if processed_regular_files:
                files_process_batches.append(("regular", processed_regular_files))


        for batch_type, files_to_process in files_process_batches:
            is_faq = batch_type == "faq"
            # TODO: remove this
            if is_faq:
                continue 
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future = executor.submit(FAQProcessorRAFT.generate_raft_training_data, files_to_process, output_path, self.config, is_faq)
                future_to_desc = {future: batch_type}
                for future in tqdm(as_completed(future_to_desc), total=1, desc=f"Processing {batch_type} files with FAQProcessorRAFT"):
                    try:
                        raft_qa_pairs = future.result()
                        if raft_qa_pairs:
                            if batch_type == "faq":
                                faq_qa_pairs.extend(raft_qa_pairs)
                                logger.info(f"Generated {len(raft_qa_pairs)} QA pairs from FAQ files using FAQProcessorRAFT")
                            else:
                                regular_qa_pairs.extend(raft_qa_pairs)
                                logger.info(f"Generated {len(raft_qa_pairs)} QA pairs from regular files using FAQProcessorRAFT")
                    except Exception as e:
                        logger.error(f"Error processing {batch_type} files with FAQProcessorRAFT: {e}")


        # Combine FAQ and non-FAQ QA pairs
        all_qa_pairs = faq_qa_pairs + regular_qa_pairs
                    
        # Save all QA pairs to a final JSON file
        if all_qa_pairs:
            final_output = output_path / "synthetic_qa_data_raft_test.json"
            with open(final_output, 'w', encoding='utf-8') as f:
                json.dump(all_qa_pairs, f, ensure_ascii=False, indent=2)

            logger.info(f"Generated a total of {len(all_qa_pairs)} QA pairs ({len(faq_qa_pairs)} from FAQs, {len(regular_qa_pairs)} from regular documents)")
            logger.info(f"Final output saved to {final_output}")
        else:
            logger.warning("No QA pairs were generated")


    def process_file(self, soup, file_path, base_dir, output_dir):
        """
        Process a single file to generate synthetic QA pairs.

        Extract text from the file, chunk it, and then generate QA pairs using the QA generator.
        If the file is not an HTML file, it attempts to find a related HTML file for additional context.

        Returns:
            list: A list of generated QA pairs. Each QA pair is a dictionary containing 'question' and 'answer' keys.
        """
        try:
            # Extract relative path for source info
            rel_path = file_path.relative_to(base_dir)
            file_title = soup.title.get_text(strip=True) if soup else file_path.stem
            file_hash = get_hash(str(rel_path))
            safe_title_slug = slugify(file_title)
            
            # Extract text from file with appropriate settings
            extracted_text_dir = output_dir / "extracted_text"
            extracted_text_dir.mkdir(parents=True, exist_ok=True)
            extracted_text_path = extracted_text_dir / f"{safe_title_slug}_{file_hash}.txt"

            if os.path.exists(extracted_text_path):
                logger.info(f"Structured text already exists for {rel_path}")
                with open(extracted_text_path, 'r', encoding='utf-8') as f:
                    text = f.read()
            else:
                text = self.file_processor.extract_text_from_file(file_path, self.config)

                with open(extracted_text_path, 'w', encoding='utf-8') as f:
                    f.write(text)
                
            if not text:
                logger.warning(f"No text extracted from {rel_path}")
                return []
            
            # Try to load chunks from file if it exists, otherwise generate and save
            extracted_chunks_dir = output_dir / "extracted_chunks"
            extracted_chunks_dir.mkdir(parents=True, exist_ok=True)
            extracted_chunks_path = extracted_chunks_dir / f"{safe_title_slug}_{file_hash}.json"

            if os.path.exists(extracted_chunks_path):
                logger.info(f"Chunks already exist for {rel_path}")
                with open(extracted_chunks_path, 'r', encoding='utf-8') as f:
                    chunks = json.load(f)
            else:
                chunks = self.text_chunker.chunk_text(text, rel_path)
                with open(extracted_chunks_path, 'w', encoding='utf-8') as f:
                    json.dump(chunks, f, ensure_ascii=False, indent=2)
                
            if not chunks:
                logger.info(f"No chunks created from {rel_path}")
                return []
                
            logger.info(f"Created {len(chunks)} chunks from {rel_path}")

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
                            source_html_file_hash = get_hash(str(source_html_path))
                            soup_source_html = FileProcessor.get_soup(source_html_file_path)
                            safe_source_html_title_slug = slugify(soup_source_html.title.get_text(strip=True))

                            source_html_extracted_text_path = extracted_text_dir / f"{safe_source_html_title_slug}_{source_html_file_hash}.txt"

                            if os.path.exists(source_html_extracted_text_path):
                                logger.info(f"Context HTML text already cached for {source_html_file_path.name} at {source_html_extracted_text_path}")
                                with open(source_html_extracted_text_path, 'r', encoding='utf-8') as f:
                                    context_html_text = f.read()
                            else:
                                logger.info(f"Extracting text from source HTML file: {source_html_file_path.name}")
                                extracted_context_text = self.file_processor.extract_text_from_file(source_html_file_path, self.config)
                                if extracted_context_text:
                                    context_html_text = extracted_context_text
                                    with open(source_html_extracted_text_path, 'w', encoding='utf-8') as f:
                                        f.write(context_html_text)
                                    logger.info(f"Saved extracted text for source HTML {source_html_file_path.name} to {source_html_extracted_text_path}")
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
                logger.info(f"File {file_path.name} is already HTML/HTM. No separate source HTML context needed.")


            qa_pairs = self.qa_generator.generate_qa_pairs(
                chunks=chunks,
                source_path=rel_path,
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
    
    try:
        generator = SyntheticQADataGenerator(args.config)
        generator.process_directory(args.input, args.output, args.threads)
    except Exception as e:
        logger.error(f"Error during execution: {e}")
        raise


if __name__ == "__main__":
    main()