"""
Text chunking module for the Synthetic QA Generator.

This module handles dividing text into semantically meaningful chunks for processing.
"""

import re
import logging

logger = logging.getLogger(__name__)

# Constants for chunking
MAX_CHUNK_SIZE = 4000  # Characters per chunk for context window management
MIN_CHUNK_SIZE = 1000  # Minimum characters per chunk to ensure meaningful context
OVERLAP_SIZE = 200     # Characters of overlap between chunks for context continuity
MAX_CONTEXT_TOKENS = 7000  # Maximum tokens for a full context in factual mode


class TextChunker:
    """Handles chunking of text into manageable segments for LLM processing."""
    
    @staticmethod
    def chunk_text(text: str, factual_mode: bool = False, 
                  max_tokens: int = MAX_CONTEXT_TOKENS,
                  force_single_chunk: bool = False,
                  small_doc_threshold: int = 10000) -> list:
        """
        Split text into semantically meaningful chunks.
        
        Args:
            text: The text to chunk
            factual_mode: Whether to optimize for factual accuracy (prefers full context)
            max_tokens: Maximum number of tokens allowed for a full context (in factual mode)
            force_single_chunk: Whether to force processing as a single chunk
            small_doc_threshold: Character threshold below which document is always kept whole
            
        Returns:
            List of text chunks
        """
        if not text.strip():
            return []

        # Force single chunk if requested or document is small (regardless of mode)
        if force_single_chunk or len(text) <= small_doc_threshold:
            logger.info(f"Using single chunk mode (document size: {len(text)} chars)")
            return [text]

        # In factual mode, try to keep entire documents together if possible
        if factual_mode:
            # Estimate tokens (rough approximation: 4 chars ≈ 1 token)
            estimated_tokens = len(text) / 4
            
            # If text can fit within max_tokens, return it as a single chunk
            if estimated_tokens <= max_tokens:
                logger.info(f"Using full context mode (estimated {estimated_tokens:.0f} tokens)")
                return [text]
            
            logger.info(f"Document too large for full context ({estimated_tokens:.0f} estimated tokens). Using semantic chunking with larger chunks and overlaps.")
            # If too large, use larger chunks with more overlap
            max_chunk_size = int(max_tokens * 4 * 0.8)  # 80% of max tokens, converted to chars
            min_chunk_size = int(max_chunk_size * 0.5)  # Larger minimum chunk size
            overlap_size = int(max_chunk_size * 0.2)    # 20% overlap
            
        else:
            # Standard chunking parameters
            max_chunk_size = MAX_CHUNK_SIZE
            min_chunk_size = MIN_CHUNK_SIZE
            overlap_size = OVERLAP_SIZE
            
        # First split by sections (double newlines often indicate section breaks)
        sections = re.split(r'\n\s*\n', text)
        
        chunks = []
        current_chunk = ""
        
        for section in sections:
            # If section is very short, just add it to the current chunk
            if len(section) < 200:
                if current_chunk and len(current_chunk) + len(section) + 2 <= max_chunk_size:
                    current_chunk += "\n\n" + section
                else:
                    if current_chunk:
                        chunks.append(current_chunk)
                    current_chunk = section
                continue
                
            # Split larger sections by sentences
            sentences = re.split(r'(?<=[.!?])\s+', section)
            
            for sentence in sentences:
                if len(sentence.strip()) == 0:
                    continue
                    
                # If adding this sentence exceeds max chunk size, start a new chunk
                if current_chunk and len(current_chunk) + len(sentence) + 1 > max_chunk_size:
                    chunks.append(current_chunk)
                    current_chunk = sentence
                else:
                    if current_chunk:
                        current_chunk += " " + sentence
                    else:
                        current_chunk = sentence
        
        # Add the last chunk if it exists
        if current_chunk:
            chunks.append(current_chunk)
            
        # Filter out chunks that are too small to be meaningful
        chunks = [chunk for chunk in chunks if len(chunk) >= min_chunk_size]
        
        # Add overlap between chunks for context continuity
        overlapped_chunks = []
        for i, chunk in enumerate(chunks):
            if i > 0:
                # Add ending of previous chunk to the beginning of current
                prev_end = chunks[i-1][-overlap_size:] if len(chunks[i-1]) > overlap_size else chunks[i-1]
                chunk = prev_end + "\n...\n" + chunk
            
            overlapped_chunks.append(chunk)
            
        return overlapped_chunks