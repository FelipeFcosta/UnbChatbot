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
import yaml
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

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
            "include_extensions", ['.html', '.htm', '.pdf', '.txt', '.md']
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
        non_faq_files = []
        
        for file_path in all_files:
            if file_path.suffix.lower() in ['.html', '.htm']:
                try:
                    soup = FileProcessor.preprocess_html(file_path)
                    if FAQProcessor.detect_faq_document(soup, file_path.name):
                        logger.info(f"Detected FAQ document: {file_path.relative_to(input_path)}")
                        faq_files.append((soup, file_path))
                    else:
                        non_faq_files.append(file_path)
                except Exception as e:
                    logger.error(f"Error checking if {file_path} is FAQ: {e}")
                    non_faq_files.append(file_path)
            else:
                non_faq_files.append(file_path)
        
        logger.info(f"Identified {len(faq_files)} FAQ files and {len(non_faq_files)} non-FAQ files")
        
        # Process all FAQs individually (never group them with other files)
        faq_qa_pairs = []
        faq_files = [] # TODO: delete this 

        if faq_files:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Create future for each FAQ file
                # No need to loop through faq_files if generate_raft_training_data processes all files
                future = executor.submit(FAQProcessorRAFT.generate_raft_training_data, faq_files, output_path, self.config)
                future_to_faq = {future: "All FAQ files"}

                # Collect results as they complete
                for future in tqdm(as_completed(future_to_faq), total=len(faq_files), desc="Processing FAQ files"):
                    faq_path = future_to_faq[future]
                    try:
                        qa_pairs = future.result()
                        if qa_pairs:
                            faq_qa_pairs.extend(qa_pairs)
                            logger.info(f"Generated {len(qa_pairs)} QA pairs from FAQ {faq_path.relative_to(input_path)}")
                    except Exception as e:
                        logger.error(f"Error processing FAQ file {faq_path}: {e}")
        
        # Now process the remaining non-FAQ files
        non_faq_qa_pairs = []
        
        if non_faq_files:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Create future for each file
                future_to_file = {
                    executor.submit(self.process_file, file_path, input_path, output_path, self.config): file_path 
                    for file_path in non_faq_files
                }
                
                # Collect results as they complete
                for future in tqdm(as_completed(future_to_file), total=len(non_faq_files), desc="Processing non-FAQ files"):
                    file_path = future_to_file[future]
                    try:
                        qa_pairs = future.result()
                        if qa_pairs:
                            non_faq_qa_pairs.extend(qa_pairs)
                            logger.info(f"Generated {len(qa_pairs)} QA pairs from {file_path.relative_to(input_path)}")
                    except Exception as e:
                        logger.error(f"Error processing file {file_path}: {e}")
        
        # Combine FAQ and non-FAQ QA pairs
        all_qa_pairs = faq_qa_pairs + non_faq_qa_pairs
                    
        # Save all QA pairs to a final JSON file
        if all_qa_pairs:
            final_output = output_path / "synthetic_qa_data_raft.json"
            with open(final_output, 'w', encoding='utf-8') as f:
                json.dump(all_qa_pairs, f, ensure_ascii=False, indent=2)

            logger.info(f"Generated a total of {len(all_qa_pairs)} QA pairs ({len(faq_qa_pairs)} from FAQs, {len(non_faq_qa_pairs)} from regular documents)")
            logger.info(f"Final output saved to {final_output}")
        else:
            logger.warning("No QA pairs were generated")


    def process_file(self, file_path, base_dir, output_dir, config):
        """Process a single file to generate QA pairs."""
        try:
            # Extract relative path for source info
            rel_path = file_path.relative_to(base_dir)
            source_path = str(file_path)
            
            # Extract text from file with appropriate settings
            extracted_text_dir = output_dir / "extracted_text"
            file_hash = get_hash(str(file_path))
            extracted_text_dir.mkdir(parents=True, exist_ok=True)
            extracted_text_path = extracted_text_dir / f"{file_path.stem}_{file_hash}.txt"

            if os.path.exists(extracted_text_path):
                logger.info(f"Structured text already exists for {file_path}")
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
            extracted_chunks_path = extracted_chunks_dir / f"{file_path.stem}_{file_hash}.json"

            if os.path.exists(extracted_chunks_path):
                logger.info(f"Chunks already exist for {file_path}")
                with open(extracted_chunks_path, 'r', encoding='utf-8') as f:
                    chunks = json.load(f)
            else:
                chunks = self.text_chunker.chunk_text(text, file_path)
                with open(extracted_chunks_path, 'w', encoding='utf-8') as f:
                    json.dump(chunks, f, ensure_ascii=False, indent=2)
                
            if not chunks:
                logger.info(f"No chunks created from {rel_path}")
                return []
                
            logger.info(f"Created {len(chunks)} chunks from {rel_path}")
            
            qa_pairs = self.qa_generator.generate_qa_pairs(
                chunks=chunks,
                source_path=source_path,
                output_dir=output_dir,
                full_document_text=text,
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