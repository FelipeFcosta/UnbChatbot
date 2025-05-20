import logging
import json
import re
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Optional

from slugify import slugify

from .llm_client import LLMClient
from .file_processor import FileProcessor
from modules.utils import get_hash

logger = logging.getLogger(__name__)


class QAGenerator:
    """Generate baseline (\"default\") question-answer pairs from text chunks.

    This class purposefully keeps things simple: it creates *only* the initial
    examples that will later be expanded by FAQProcessorRAFT (or other
    processors).  Each source document receives a single file named
    `<slug>_default_examples.json` containing the accumulated seed pairs.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config or {}

        providers_config = self.config.get("providers", {})

        # Robust fallback chain for question generation provider
        question_config = providers_config.get("default_question")

        answer_config = providers_config.get("default_answer", {})

        self.question_client = LLMClient(question_config)
        self.answer_client = LLMClient(answer_config)

        # Cache the 'Default' writing style details for prompt use
        ws = self.config.get("question_styles", {}).get("writing_styles", [])
        self.default_style = next((s for s in ws if s.get("name").lower() == "default"), None) 


    def _build_question_prompt(self, chunks_batch: List[str], full_document: str, source_info: Dict[str, str]) -> str:
        """Prompt for generating baseline questions from a batch of chunks.

        If *full_document* is provided it is supplied inside <FULL_DOCUMENT> tags so the
        model has additional context for phrasing while still being instructed to
        ask ONLY about content present in the chunks.
        """

        institution = source_info.get("institution", source_info.get("domain", ""))

        full_doc_prompt_section = f"\n<FULL_DOCUMENT>\n{full_document}\n</FULL_DOCUMENT>"

        style_section = self._default_style_section() if self.default_style else ""

        chunk_prompts = []
        for i, single_chunk in enumerate(chunks_batch):
            chunk_prompts.append(f"<CHUNK_{i+1}>\n{single_chunk}\n</CHUNK_{i+1}>\n")
        all_chunks_str = "\n".join(chunk_prompts)

        return (
            "You are an LLM Generator creating synthetic data for a university chatbot.\n"
            f"The chatbot serves the institution: {institution}.\n\n"
            f"You will receive a batch of {len(chunks_batch)} text excerpts from a larger document (provided as <FULL_DOCUMENT>). " \
            "For each excerpt <CHUNK_N>, your task is to craft ONE natural question in Portuguese that students, faculty, or staff might ask. " \
            "Each question MUST be answerable using ONLY the information contained in its corresponding <CHUNK_N>. " \
            "Use the full document only to understand context and improve wording; do NOT include information that appears exclusively outside the chunk.\n\n"
            f"Generate exactly {len(chunks_batch)} questions, one per line, in the same order as the input <CHUNK_N>s. " \
            "Return ONLY the questions, one per line, with no numbering or other text.\n\n"
            f"{style_section}"
            f"{all_chunks_str}"
            f"{full_doc_prompt_section}"
        )

    def _default_style_section(self) -> str:
        """Return formatted style instructions for the Default writing style."""
        style = self.default_style
        if not style:
            return ""
        name = style.get("name", "")
        desc = style.get("description", "")
        goal = style.get("goal", "")
        return (
            f"WRITING STYLE: {name}\n"
            f"Description: {desc}\n"
            f"Goal: {goal}\n\n"
        )

    def _build_answer_prompt(self, questions_batch: List[str], chunks_for_answers_batch: List[str], source_info: Dict[str, str]) -> str:
        """Prompt for generating answers for a given batch of questions."""

        institution = source_info.get("institution", source_info.get("domain", ""))
        style_section = self._default_style_section() if self.default_style else ""

        qa_prompts_parts = []
        for i, (question, chunk) in enumerate(zip(questions_batch, chunks_for_answers_batch)):
            qa_prompts_parts.append(
                f"<ITEM_{i+1}>\n"
                f"<QUESTION_{i+1}>\n{question}\n</QUESTION_{i+1}>\n"
                f"<CONTEXT_{i+1}>\n{chunk}\n</CONTEXT_{i+1}>\n"
                f"</ITEM_{i+1}>"
            )
        all_qa_items_str = "\n\n".join(qa_prompts_parts)

        return (
            "You are an assistant helping to create high-quality FAQ answers for a university chatbot.\n\n"
            f"INSTITUTION: {institution}\n\n"
            f"{style_section}"
            f"You will receive a batch of {len(questions_batch)} question-context items, each enclosed in <ITEM_N> tags. "
            "For each item, answer the <QUESTION_N> using ONLY the information in its corresponding <CONTEXT_N>. "
            "If a specific context does not provide enough information, reply for that item with \"Não possuo informações suficientes para responder a essa pergunta.\"\n\n"
            f"{all_qa_items_str}\n\n"
            f"Generate exactly {len(questions_batch)} answers, one per line, in the same order as the input items. "
            "Return ONLY the answers, one per line, with no numbering or other text."
        )

    
    def generate_qa_pairs(
        self,
        chunks: List[str],
        source_path: str,
        output_dir: Path,
        full_document_text: str,
        batch_size: int = 1,
    ) -> List[Dict[str, str]]:
        """Generate one baseline QA pair for each chunk provided, processing in batches."""

        file_path = Path(source_path)
        domain, rel_path, url = FileProcessor.extract_domain_and_path(file_path)
        institution = FileProcessor.get_institution_name(domain)
        source_info = { # Common source_info for all chunks from this file
            "domain": domain,
            "path": rel_path,
            "url": url,
            "institution": institution,
        }

        all_qa_pairs: List[Dict[str, str]] = []

        qa_dir = output_dir / "qa_pairs"
        qa_dir.mkdir(parents=True, exist_ok=True) # Ensure directory exists early
        default_file_name = f"{slugify(file_path.stem)}_default_examples.json"
        default_file_path = qa_dir / default_file_name

        try:
            logger.info(f"Attempting to load existing default examples from: {default_file_path}")
            with open(default_file_path, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
            loaded_count = len(existing_data)
            logger.info(f"Successfully loaded {loaded_count} items from {default_file_path}. Skipping generation.")
            return existing_data
        except Exception as e:
            logger.info(f"Could not load existing default examples from {default_file_path}. Regenerating.")

        for i in range(0, len(chunks), batch_size):
            current_batch_chunks = chunks[i:i + batch_size]
            if not current_batch_chunks:
                continue

            # 1. Generate questions for the current batch of chunks
            question_batch_prompt = self._build_question_prompt(current_batch_chunks, full_document_text, source_info)
            raw_questions_str = self.question_client.generate_text(question_batch_prompt, temperature=0.7)

            if not raw_questions_str:
                logger.warning(f"No questions generated for batch starting at chunk {i} of {file_path}")
                continue

            generated_questions = [q.strip() for q in raw_questions_str.strip().splitlines() if q.strip()]

            if len(generated_questions) != len(current_batch_chunks):
                logger.info(f"Question count mismatch: {len(generated_questions)} vs {len(current_batch_chunks)}. Skipping batch.")
                logger.info(f"LLM Raw questions: {raw_questions_str}")
                continue

            # 2. Generate answers for the batch of questions and their original chunks
            answer_batch_prompt = self._build_answer_prompt(generated_questions, current_batch_chunks, source_info)
            raw_answers_str = self.answer_client.generate_text(answer_batch_prompt, temperature=0.5)

            if not raw_answers_str:
                logger.warning(f"No answers generated for batch starting at chunk {i} of {file_path}")
                continue

            generated_answers = [a.strip() for a in raw_answers_str.strip().splitlines() if a.strip()]

            if len(generated_answers) != len(generated_questions):
                logger.info(f"Answer count mismatch: {len(generated_answers)} vs {len(generated_questions)}. Skipping batch.")
                logger.info(f"LLM Raw answers: {raw_answers_str}")
                continue

            # 3. Combine questions, original chunks, and answers into QA pairs
            for idx_in_batch in range(len(generated_questions)):
                question = generated_questions[idx_in_batch]
                answer = generated_answers[idx_in_batch]

                all_qa_pairs.append({
                    "question": question,
                    "answer": answer,
                    "url": url,
                    "qa_pair_hash": f"default_{get_hash(str(file_path) + question)}"
                })

        if not all_qa_pairs:
            return []

        try:
            with open(default_file_path, 'w', encoding='utf-8') as f:
                json.dump(all_qa_pairs, f, ensure_ascii=False, indent=2)
            logger.info(f"Created new default examples file with {len(all_qa_pairs)} QA pairs: {default_file_path}")
        except Exception as e:
            logger.error(f"Failed to write new default examples file {default_file_path}: {e}")

        return all_qa_pairs
