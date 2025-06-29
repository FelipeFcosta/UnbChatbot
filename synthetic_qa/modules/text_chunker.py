"""
Text chunking module for the Synthetic QA Generator.

This module handles dividing text into semantically meaningful chunks for processing.
"""

import logging
import time
import yaml
import os
from typing import List, Dict, Any, Optional
from pathlib import Path

from .file_processor import FileProcessor
# Local imports
from .llm_client import LLMClient
from .utils import FileType, create_hash

with open(os.path.join(os.path.dirname(__file__), '../config.yaml'), 'r', encoding='utf-8') as f:
    _config = yaml.safe_load(f)
logging_level = getattr(logging, _config.get('global', {}).get('logging_level', 'INFO').upper(), logging.INFO)
logging.basicConfig(
    level=logging_level,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


class TextChunker:
    """LLM-powered text chunker.

    This new implementation delegates the segmentation task to a Large Language
    Model (LLM).  It builds an instruction-rich prompt (in *Portuguese*, as
    required by downstream components) describing exactly how the text should be
    split and expects a JSON array where each element has the shape

        {"chunk": "...", "topic": "..."}

    Only the *chunk* value is used by the rest of the pipeline at the moment.
    The *topic* is preserved in the raw LLM response so it can be surfaced later
    if needed.
    """

    DEFAULT_PROVIDER_CONFIG: Dict[str, Any] = {
        "provider": "genai",
        "model": "gemini-1.5-flash",
        "temperature": 0.2,
        "max_tokens": 8192,
        "rate_limit_rpm": 4,
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Create a new :class:`TextChunker`.

        Parameters
        ----------
        config : dict, optional
            The full application configuration.  The provider settings for the
            *text chunking* step are expected under

                config["providers"]["text_chunking"]

            If that section is missing, a sensible default pointing at Gemini
            1.5-flash will be used.
        """

        self.config = config or {}

        provider_cfg = (
            self.config.get("providers", {}).get("text_chunking")
            or self.DEFAULT_PROVIDER_CONFIG
        )

        # Merge defaults with explicit values (explicit wins)
        merged_cfg = {**self.DEFAULT_PROVIDER_CONFIG, **provider_cfg}
        self.max_chunking_length = merged_cfg.get("max_chunking_length", 40000)
        try:
            self.llm_client = LLMClient(merged_cfg)
        except Exception as e:
            logger.error(f"Falha ao inicializar LLMClient para chunking (usando fallback heurístico): {e}")
            self.llm_client = None


    def chunk_text(self, text: str, file_path: Path) -> List[dict]:
        """Split *text* into semantically coherent chunks using an LLM.

        The LLM is instructed to strictly return a JSON array of
        objects containing *chunk* and *topic* keys.  In case the LLM returns an
        invalid payload or the call fails.
        """

        if not text.strip():
            return []

        # If the LLM client is not available, we can't proceed.
        if not self.llm_client:
            logger.warning("LLMClient not available for chunking. Returning empty list.")
            return []

        # Split text if it's too long
        text_parts = self._split_text_if_needed(text)
        all_chunks = []

        for part_idx, part in enumerate(text_parts):
            response = None
            prompt = TextChunker._build_prompt(part)

            while not response:
                logger.info(f"Requesting LLM-based chunking for file {file_path.name} part {part_idx+1}/{len(text_parts)}...")
                try:
                    response = self.llm_client.generate_text(
                        prompt,
                        json_output=True,
                        temperature=self.llm_client.config.get("temperature", 0.3),
                    )
                except Exception as e:
                    logger.error(f"LLM chunking failed for a text part for file {file_path.name}: {e}")
                    pass
                if not response:
                    time.sleep(1)
                    logger.warning(f"LLM chunking failed for a text part for file {file_path.name}. Retrying...")

            if response:
                # get the "chunks" field which will contain the list
                response_chunks = response.get("chunks", [])
                if isinstance(response_chunks, list):
                    file_hash = create_hash(str(file_path))
                    part_chunks = []
                    for idx, item in enumerate(response_chunks):
                        if "chunk" in item and "topic" in item:
                            chunk_data = {
                                "chunk": item["chunk"].strip(),
                                "topic": item.get("topic", "").strip(),
                                "chunk_hash": f"chunk_{file_hash}_{idx + len(all_chunks)}"
                            }
                            # Add professor field if present
                            if "professor" in item and isinstance(item["professor"], str) and item["professor"].strip():
                                chunk_data["professor"] = item["professor"].strip()
                            part_chunks.append(chunk_data)
                    if part_chunks:
                        logger.debug(f"LLM returned {len(part_chunks)} chunks for a part for file {file_path.name}.")
                        all_chunks.extend(part_chunks)
                    else:
                        logger.warning(f"LLM returned valid JSON but no valid chunks for a part for file {file_path.name}.")
                        return []
                else:
                    logger.warning(f"Unexpected JSON structure returned by LLM for a part for file {file_path.name}")
                    return []
            else:
                logger.warning(f"No response or invalid JSON from LLM for a part for file {file_path.name}")
                return []
        if not all_chunks:
            logger.warning(f"LLM chunking resulted in no chunks for the entire document for file {file_path.name}.")
            return []

        return all_chunks


    def _split_text_if_needed(self, text: str) -> List[str]:
        """Splits text into smaller parts if it exceeds the maximum length."""
        if len(text) <= self.max_chunking_length:
            return [text]

        logger.info(f"Text length ({len(text)}) exceeds max_chunking_length ({self.max_chunking_length}). Splitting...")
        
        parts_to_process = [text]
        final_parts = []

        while parts_to_process:
            current_part = parts_to_process.pop(0)
            if len(current_part) <= self.max_chunking_length:
                final_parts.append(current_part)
                continue

            mid_point = len(current_part) // 2
            split_pos = current_part.rfind('.\n', 0, mid_point)
            if split_pos != -1:
                split_pos += 2
            else:
                split_pos = current_part.rfind('\n', 0, mid_point)
                if split_pos != -1:
                    split_pos += 1
                else:
                    # fallbacks
                    split_pos = current_part.rfind('. ', 0, mid_point)
                    if split_pos != -1:
                        split_pos += 2
                    else:
                        split_pos = mid_point
            
            part1 = current_part[:split_pos]
            part2 = current_part[split_pos:]

            # Add the new parts to the front of the queue to be processed
            parts_to_process.insert(0, part2)
            parts_to_process.insert(0, part1)

        logger.info(f"Split text into {len(final_parts)} parts.")
        return final_parts


    @staticmethod
    def _build_prompt(text: str) -> str:
        """Construct the instruction prompt (in Portuguese)."""

        instructions = (
            "You are an expert Text Analyst and Content Chunking specialist. Your primary goal is to segment the provided text into meaningful, self-contained informational units. These units, or 'chunks', will later qbe used as the basis for generating question-answer pairs and as context documents for a retrieval-augmented generation (RAG) system.\n\n"
            "Therefore, each chunk must adhere to the following critical criteria:\n"
            "1. Semantic Cohesion & Completeness: Each chunk should focus on a single, distinct topic, concept, rule, process, or piece of information. It must be internally complete, presenting a full idea without requiring immediate reference to another chunk. Don't split a sentence or an idea across chunks.\n"
            "2. Answerability: A chunk must contain sufficient information to comprehensively answer one or more specific, plausible questions about its content. Imagine someone reading only that chunk; they should be able to extract a clear answer to a relevant question concerning the chunk's main subject.\n"
            "3. Appropriate Length:\n"
            "   - **NO Short Chunks**: DO NOT CREATE create chunks that are just a few words (like just one line) or a single trivial sentence, unless the short segment is very relevant and very distinct from the rest of the text.\n"
            "        - *Joining small units into a single chunk than splitting them into multiple chunks.*\n"
            "   - Aim for Substantiality: Chunks should generally span one to several paragraphs if those paragraphs together cover a single, answerable topic.\n"
            "   - Avoid Overly Long Chunks: If a section of text is very long and covers multiple distinct answerable topics, break it down further.\n"
            "4. Context Preservation: While breaking down the text, strive to maintain the natural flow and relationships between ideas within each chunk. The chunk should make sense in isolation.\n"
            "5. **CONTENT-ONLY FOCUS**: Each chunk must contain actual content. Never create chunks that consist solely of metadata, UI elements, navigation text, headers without content, footers, copyright notices, or other non-informational elements.\n"
            "6. Preserve **all** original markdown formatting (headings, lists, bold/italic, links) inside each chunk exactly as it appears in the source text.\n"
            "7. If you encounter a segment that is not useful on its own (such as a title, metadata, date, or other fragment that does not provide answerable information), do not create a separate chunk for it. Instead, join it with the next (or previous) chunk to form a complete, answerable unit.\n"
            "8. PROFESSOR INFORMATION: If the text contains information about a professor (e.g., their biography, research areas, contact information, courses taught, academic background, etc.), you MUST identify the professor's full name and include it in the 'professor' field for all chunks related to that professor. If no professor or multiple professors are mentioned in a chunk, 'professor' field should be None.\n"
            "9. Use the topic field to describe the main subject of the chunk (without redundancy, repeating or being verbose), and also to add important context information that couldn't be present in the chunk.\n"
            "**10. Under no circumstances should you leave out any information (even a single character) from the text.**\n"
            "**11. Under no circumstances should you add any information (even a single character) to the chunk text.**\n"
            "The chunks will be used as the ORIGINAL GROUND TRUTH source documents for the RAG system, so they must be as complete and accurate as possible.\n\n"
            "Your task:\n"
            "Given the text below, divide it into such chunks. Focus on the quality and utility of each chunk for future Q&A and RAG purposes.\n\n"
            "Return ONLY a JSON object in the following format (IN PORTUGUESE):\n"
            "{\n  \"chunks\": [\n    {\"chunk\": \"...\", \"topic\": \"...\", \"professor\": \"Professor full name (only if the chunk contains professor information, otherwise omit this field)\"},\n    ...\n  ]\n}\n\n"
            "TEXT:\n" + text.strip()
        )
        return instructions

    @staticmethod
    def add_metadata_to_items(items, file_path, file_title, file_type):
        """
        Ensures each item (chunk or FAQ) has the required metadata fields. Returns True if any item was updated.
        """
        try:
            file_url = os.getxattr(str(file_path), b'user.original_url').decode('utf-8')
        except Exception:
            file_url = FileProcessor.extract_domain_and_path(file_path)[2]
        try:
            source_page_url = os.getxattr(str(file_path), b'user.source_page_url').decode('utf-8')
        except Exception:
            source_page_url = ""
        updated = False
        for item in items:
            if not item.get('file_title') and file_title:
                item['file_title'] = file_title
                updated = True
            if not item.get('file_name') and file_path:
                item['file_name'] = file_path.name
                updated = True
            if file_url:
                item['file_url'] = file_url
                updated = True
            if source_page_url:
                item['source_page_url'] = source_page_url
                updated = True
            if not item.get('file_type') and file_type:
                item['file_type'] = str(file_type)
                updated = True
        return updated
