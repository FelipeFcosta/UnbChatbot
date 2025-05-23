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
from modules.utils import group_related_files, get_hash

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
            
        # Set factual mode from config
        self.factual_mode = self.config.get("factual_mode", False)
        
        # Get HTML processing settings
        html_config = self.config.get("processing", {}).get("html_pages", {})
        self.html_full_doc_threshold = html_config.get("full_document_threshold", 20000)
        self.comprehensive_questions = html_config.get("comprehensive_questions", True)
        
        # Initialize components
        self.file_processor = FileProcessor()
        self.text_chunker = TextChunker()
        self.qa_generator = QAGenerator(self.config)
        
        logger.info(f"Initialized QA Generator in {'factual' if self.factual_mode else 'standard'} mode")
        
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
        
        if self.factual_mode and non_faq_files:
            # Group files by directory or relationship (but not FAQs)
            file_groups = group_related_files(non_faq_files, input_path)
            logger.info(f"Grouped non-FAQ files into {len(file_groups)} sets for processing")
            
            # Process each group
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_group = {
                    executor.submit(self.process_file_group, group, input_path, output_path): i 
                    for i, group in enumerate(file_groups)
                }
                
                for future in tqdm(as_completed(future_to_group), total=len(file_groups), desc="Processing file groups"):
                    group_idx = future_to_group[future]
                    try:
                        qa_pairs = future.result()
                        if qa_pairs:
                            non_faq_qa_pairs.extend(qa_pairs)
                            logger.info(f"Generated {len(qa_pairs)} QA pairs from group {group_idx}")
                    except Exception as e:
                        logger.error(f"Error processing group {group_idx}: {e}")
                        
        elif non_faq_files:  # Standard processing: files individually 
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Create future for each file
                future_to_file = {
                    executor.submit(self.process_file, file_path, input_path, output_path): file_path 
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
            
    def process_file_group(self, file_group, base_dir, output_dir):
        """Process a group of related files together."""
        try:
            combined_text = ""
            sources = []
            
            # Extract domain info from the first file in the group for consistency
            if file_group:
                first_file = file_group[0]
                domain, path, url = FileProcessor.extract_domain_and_path(first_file)
                institution = FileProcessor.get_institution_name(domain)
            else:
                domain, path, url = "", "", ""
                institution = ""
            
            # Combine all file contents with clear separation
            for file_path in file_group:
                rel_path = file_path.relative_to(base_dir)
                source_info = f"{rel_path}"
                sources.append(source_info)
                
                # Extract text from file
                text = self.file_processor.extract_text_from_file(file_path)
                if not text:
                    continue
                    
                # Add clear document boundary and source information
                _, file_path_part, file_url = FileProcessor.extract_domain_and_path(file_path)
                combined_text += f"\n\n--- DOCUMENT: {domain}/{file_path_part} ---\n\n{text}\n\n"
            
            if not combined_text:
                logger.warning(f"No text extracted from file group")
                return []
                
            # Create a combined source identifier
            if len(sources) == 1:
                source_info = f"{domain}/{path}"
            else:
                # Use the common directory or a concatenation of first 2 files
                source_info = f"Group of {len(sources)} related files from {institution}"
                if len(sources) > 2:
                    source_info += f" ({domain})"
            
            # Chunk the combined text, potentially keeping it all together in factual mode
            chunks = self.text_chunker.chunk_text(combined_text, factual_mode=self.factual_mode)
            if not chunks:
                logger.warning(f"No chunks created from file group")
                return []
                
            logger.info(f"Created {len(chunks)} chunks from file group with {len(file_group)} files")
            
            # Generate QA pairs for each chunk
            all_pairs = []
            for i, chunk in enumerate(chunks):
                # Generate a hash for this chunk
                chunk_hash = f"{get_hash(source_info)}_{i}"
                
                # Use the first file in the group as the source path
                source_path = str(file_group[0])
                
                # Generate QA pairs
                qa_pairs = self.qa_generator.generate_qa_pairs(
                    chunk=chunk,
                    source_path=source_path,
                    output_dir=output_dir,
                    chunk_hash=chunk_hash
                )
                
                all_pairs.extend(qa_pairs)
                
            return all_pairs
            
        except Exception as e:
            logger.error(f"Error processing file group: {e}")
            return []

    def process_file(self, file_path, base_dir, output_dir):
        """Process a single file to generate QA pairs."""
        try:
            # Extract relative path for source info
            rel_path = file_path.relative_to(base_dir)
            source_path = str(file_path)
            
            # Determine if this is an HTML file that might be an FAQ
            is_html = file_path.suffix.lower() in ['.html', '.htm']

            if is_html:
                content = FileProcessor.preprocess_html(file_path)
                soup = content  # Already a BeautifulSoup object
            else:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                soup = content

            # Check if this is an FAQ document and process accordingly
            if is_html and self.config.get("processing", {}).get("faq", {}).get("enabled", True):
                if FAQProcessor.detect_faq_document(soup, file_path.name):
                    logger.info(f"Detected FAQ document: {rel_path}")
                    return FAQProcessor.process_faq_document(soup, file_path, output_dir, self.config)
            
            # Standard processing if not a FAQ or FAQ processing is disabled
            
            # Extract text from file with appropriate settings
            if is_html:
                text = self.file_processor.extract_text_from_html(soup, file_path)
            else:
                text = self.file_processor.extract_text_from_file(file_path)
                
            if not text:
                logger.warning(f"No text extracted from {rel_path}")
                return []
            
            # Determine if this should be processed as a single document
            is_full_document = False
            if is_html and self.comprehensive_questions and len(text) <= self.html_full_doc_threshold:
                # Force HTML content below threshold to be a single chunk with comprehensive questions
                is_full_document = True
                chunks = self.text_chunker.chunk_text(
                    text, 
                    factual_mode=self.factual_mode,
                    force_single_chunk=True
                )
                logger.info(f"Processing HTML as a single document for comprehensive question coverage: {rel_path}")
            else:
                # Normal chunking
                chunks = self.text_chunker.chunk_text(
                    text, 
                    factual_mode=self.factual_mode
                )
                
            if not chunks:
                logger.warning(f"No chunks created from {rel_path}")
                return []
                
            logger.info(f"Created {len(chunks)} chunks from {rel_path}")
            
            # Generate QA pairs for each chunk
            all_pairs = []
            for i, chunk in enumerate(chunks):
                # Generate a hash for this chunk
                chunk_hash = f"{get_hash(str(rel_path))}_{i}"
                
                # Generate QA pairs, indicating if this is a full document for comprehensive questions
                qa_pairs = self.qa_generator.generate_qa_pairs(
                    chunk=chunk,
                    source_path=source_path,
                    output_dir=output_dir,
                    chunk_hash=chunk_hash,
                    is_full_document=is_full_document and i == 0,  # Only the first chunk should generate comprehensive questions
                    is_faq=False  # This is not an FAQ document (we handle FAQs separately)
                )
                
                all_pairs.extend(qa_pairs)
                
            return all_pairs
            
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
    parser.add_argument("--factual", action="store_true", help="Enable factual mode (optimized for factual accuracy)")
    args = parser.parse_args()
    
    try:
        generator = SyntheticQADataGenerator(args.config)
        
        # Override factual mode from command line if specified
        if args.factual:
            generator.factual_mode = True
            generator.qa_generator.factual_mode = True
            logger.info("Factual mode enabled from command line")
            
        generator.process_directory(args.input, args.output, args.threads)
    except Exception as e:
        logger.error(f"Error during execution: {e}")
        raise


if __name__ == "__main__":
    main()