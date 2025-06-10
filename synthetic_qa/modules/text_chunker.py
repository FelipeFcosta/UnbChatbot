"""
Text chunking module for the Synthetic QA Generator.

This module handles dividing text into semantically meaningful chunks for processing.
"""

import logging
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
        try:
            self.llm_client = LLMClient(merged_cfg)
        except Exception as e:
            logger.error(f"Falha ao inicializar LLMClient para chunking (usando fallback heurístico): {e}")
            self.llm_client = None


    def chunk_text(self, text: str, file_path: Path) -> List[str]:
        """Split *text* into semantically coherent chunks using an LLM.

        The LLM is instructed (in Portuguese) to strictly return a JSON array of
        objects containing *chunk* and *topic* keys.  In case the LLM returns an
        invalid payload or the call fails, the method falls back to a basic
        heuristic chunking strategy so that the overall pipeline can still
        proceed.
        """

        if not text or not text.strip():
            return []

        # If the LLM client is available attempt LLM-based chunking first
        response = None
        if self.llm_client:
            prompt = TextChunker._build_prompt(text)

            logger.debug("Requesting LLM-based chunking...")
            try:
                response = self.llm_client.generate_text(
                    prompt,
                    json_output=True,
                    temperature=self.llm_client.config.get("temperature", 0.2),
                )
            except Exception as e:
                logger.error(f"LLM chunking failed: {e}")

        if response:
            # get the "chunks" field which will contain the list
            response = response["chunks"]
            if isinstance(response, list):
                file_hash = create_hash(str(file_path))
                chunks = [
                    {
                        "chunk": item["chunk"].strip(),
                        "topic": item.get("topic", "").strip(),
                        "chunk_hash": f"chunk_{file_hash}_{idx}"
                    }
                    for idx, item in enumerate(response)
                    if "chunk" in item and "topic" in item
                ]
                if chunks:
                    logger.debug(f"LLM returned {len(chunks)} chunks.")
                    return chunks
            else:
                logger.warning("Unexpected JSON structure returned by LLM; falling back to heuristic chunking.")
        else:
            logger.warning("No response or invalid JSON from LLM; falling back to heuristic chunking.")

        # Invalid or empty response – return empty list so caller can decide.
        return []


    @staticmethod
    def _build_prompt(text: str) -> str:
        """Construct the instruction prompt (in Portuguese)."""

        instructions = (
            "You are an expert Text Analyst and Content Chunking specialist. Your primary goal is to segment the provided text into meaningful, self-contained informational units. These units, or 'chunks', will later qbe used as the basis for generating question-answer pairs and as context documents for a retrieval-augmented generation (RAG) system.\n\n"
            "Therefore, each chunk must adhere to the following critical criteria:\n"
            "1. Semantic Cohesion & Completeness: Each chunk should focus on a single, distinct topic, concept, rule, process, or piece of information. It must be internally complete, presenting a full idea without requiring immediate reference to another chunk. Avoid splitting sentences or tightly knit ideas across chunks.\n"
            "2. Answerability: A chunk must contain sufficient information to comprehensively answer one or more specific, plausible questions about its content. Imagine someone reading only that chunk; they should be able to extract a clear answer to a relevant question concerning the chunk's main subject.\n"
            "3. Appropriate Length:\n"
            "   - Avoid Overly Short Chunks: Do not create chunks that are just a few words or a single trivial sentence, UNLESS that very short segment represents a complete, atomic, and significant piece of information (e.g., a formal definition).\n"
            "   - Aim for Substantiality: Chunks should generally span one to several paragraphs if those paragraphs together cover a single, answerable topic.\n"
            "   - Avoid Overly Long Chunks: If a section of text covers multiple distinct answerable topics, break it down further.\n"
            "4. Context Preservation: While breaking down the text, strive to maintain the natural flow and relationships between ideas within each chunk. The chunk should make sense in isolation.\n"
            "5. Preserve **all** original markdown formatting (headings, lists, bold/italic, links) inside each chunk exactly as it appears in the source text.\n"
            "6. If you encounter a segment that is not useful on its own (such as a title, metadata, date, or other fragment that does not provide answerable information), do not create a separate chunk for it. Instead, join it with the next (or previous) chunk to form a complete, answerable unit.\n\n"
            "7. Under no circumstances should you leave out any information from the text.\n"
            "Your task:\n"
            "Given the text below, divide it into such chunks. Focus on the quality and utility of each chunk for future Q&A and RAG purposes.\n\n"
            "Return ONLY a JSON object in the following format (IN PORTUGUESE):\n"
            "{\n  \"chunks\": [\n    {\"chunk\": \"...\", \"topic\": \"...\"},\n    ...\n  ]\n}\n\n"
            "TEXT:\n" + text.strip()
        )
        return instructions

    @staticmethod
    def write_component_chunks_for_directory(input_dir: Path, base_dir: Path, component_files: list, output_path: Path) -> Path:
        """
        Write a single extracted_chunks/components_{dir_hash}.json file for all component files in a directory.
        Each chunk contains the whole text of a component file.
        """
        from slugify import slugify
        import json
        import os
        # Compute hash for the directory path relative to base_dir
        rel_dir_path = str(input_dir.relative_to(base_dir))
        dir_hash = create_hash(rel_dir_path)
        extracted_text_dir = output_path / "extracted_text"
        extracted_chunks_dir = output_path / "extracted_chunks"
        extracted_chunks_dir.mkdir(parents=True, exist_ok=True)
        components_chunks_path = extracted_chunks_dir / f"components_{dir_hash}.json"
        if os.path.exists(components_chunks_path):
            logger.info(f"Component chunks file already exists: {components_chunks_path}, skipping creation.")
            return components_chunks_path
        component_chunks = []
        for file_path in component_files:
            rel_path = file_path.relative_to(base_dir)
            # get soup
            soup = FileProcessor.get_soup(file_path)
            file_title = file_path.stem
            if soup and soup.title:
                file_title = soup.title.get_text(strip=True)
            file_hash = f"{create_hash(str(rel_path))}"
            safe_title_slug = slugify(file_title)
            extracted_text_path = extracted_text_dir / f"{safe_title_slug}_{file_hash}.txt"
            if os.path.exists(extracted_text_path):
                with open(extracted_text_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                chunk = {
                    "chunk": text.strip(),
                    "chunk_hash": f"chunk_comp_{file_hash}_0",
                    "file_name": file_path.name,
                }
                # Ensure all required metadata fields are present
                TextChunker.add_metadata_to_items(
                    [chunk], file_path, file_title, FileType.COMPONENT
                )
                component_chunks.append(chunk)
            else:
                logger.warning(f"Extracted text not found for component: {rel_path}")
        if component_chunks:
            with open(components_chunks_path, 'w', encoding='utf-8') as f:
                json.dump(component_chunks, f, ensure_ascii=False, indent=2)
            logger.info(f"Wrote {len(component_chunks)} component chunks to {components_chunks_path}")
        return components_chunks_path

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
            if not item.get('file_url') and file_url:
                item['file_url'] = file_url
                updated = True
            if not item.get('source_page_url') and source_page_url:
                item['source_page_url'] = source_page_url
                updated = True
            if not item.get('file_type') and file_type:
                item['file_type'] = str(file_type)
                updated = True
        return updated
