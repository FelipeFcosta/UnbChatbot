import logging
import json
import os
from pathlib import Path
from typing import Dict, Any, List, Callable

from slugify import slugify

from .llm_client import LLMClient
from .file_processor import FileProcessor
from modules.utils import create_hash

logger = logging.getLogger(__name__)

# Hardcoded Default writing style
DEFAULT_STYLE = {
    "name": "Default",
    "goal": "Establish a clean, well-formed baseline; represent clear, unambiguous queries.",
    "description": "Generate a clear, straightforward question using complete sentences. Avoid excessive formality or informality. Focus on accurately representing the core content of the original question/chunk. **Crucially: If the original answer/chunk was very short (like \"Sim.\", \"Não.\"), ensure this question naturally elicits the expanded, polite answer generated for the Default pair.** The question should sound like a typical, clear query a student might ask an assistant."
}

# Hardcoded Default style section
DEFAULT_STYLE_SECTION = (
    "WRITING STYLE: Default\n"
    "Description: Generate a clear, straightforward question using complete sentences. Avoid excessive formality or informality. Focus on accurately representing the core content of the original question/chunk. **Crucially: If the original answer/chunk was very short (like \"Sim.\", \"Não.\"), ensure this question naturally elicits the expanded, polite answer generated for the Default pair.** The question should sound like a typical, clear query a student might ask an assistant.\n"
    "Goal: Establish a clean, well-formed baseline; represent clear, unambiguous queries.\n"
)

class QAGenerator:
    """Generate baseline (\"default\") question-answer pairs from text chunks or original FAQ pairs.

    This class purposefully keeps things simple: it creates *only* the initial
    examples that will later be expanded by QAProcessorRAFT (or other
    processors).
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config or {}

        providers_config = self.config.get("providers", {})

        # Robust fallback chain for question generation provider
        question_config = providers_config.get("default_question")

        answer_config = providers_config.get("default_answer", {})

        self.question_client = LLMClient(question_config)
        self.answer_client = LLMClient(answer_config)

        # Use hardcoded Default style
        self.default_style = DEFAULT_STYLE

    @staticmethod
    def _build_question_prompt(chunks_batch: List[str], full_document: str, context_html_text: str, source_info: Dict[str, str]) -> str:
        """Prompt for generating baseline questions from a batch of chunks.

        If *full_document* is provided it is supplied inside <FULL_DOCUMENT> tags so the
        model has additional context for phrasing while still being instructed to
        ask ONLY about content present in the chunks.
        """

        institution = source_info.get("institution", source_info.get("domain", ""))

        full_doc_prompt_section = f"\n<FULL_DOCUMENT>\n{full_document}\n</FULL_DOCUMENT>"

        chunk_prompts = []
        for i, single_chunk in enumerate(chunks_batch):
            chunk_prompts.append(f"<CHUNK_{i+1}>\n{single_chunk['chunk']}\n</CHUNK_{i+1}>\n")
        all_chunks_str = "\n".join(chunk_prompts)

        context_html_section = (
            f"\n<CONTEXT_HTML>\n{context_html_text}\n</CONTEXT_HTML>"
            if context_html_text else ""
        )
        return (
            "You are an LLM Generator creating synthetic data for the University of Brasilia chatbot.\n"
            f"You will receive a batch of {len(chunks_batch)} text excerpts from a larger document (provided as <FULL_DOCUMENT>). "
            "For each excerpt <CHUNK_N>, your task is to craft ONE natural question in Portuguese that students, faculty, or staff might ask. "
            "Each question MUST be answerable using ONLY the information contained in its corresponding <CHUNK_N>. "
            "Use the full document only to understand context and improve wording; do NOT include information that appears exclusively outside the chunk.\n"
            "In addition, you may also receive extra context (provided as <CONTEXT_HTML>). This extra context *may* help clarify information if something in the chunk is ambiguous or unclear, but you must still ensure that each question is answerable using only its corresponding chunk. Use the extra context only to improve the naturalness or clarity of the question, not to introduce information that is not present in the chunk.\n"
            "Remember: the user (student) has no access to or awareness of these documents. They are simply seeking information or help, not referencing any source. Therefore, craft each question so that it is specific enough it naturally requires the information from the chunk to be answered without feeling forced or artificial. Make the question feel authentic and relevant, as if it could be asked in a real conversation.\n\n"
            "So, for this document information to be retrieved, the question will probably have to mention the document name/title/subject.\n\n"
            f"Generate exactly {len(chunks_batch)} questions, one per line, in the same order as the input <CHUNK_N>s. "
            "Return ONLY the questions, one per line, with no numbering or other text.\n\n"
            f"{DEFAULT_STYLE_SECTION}"
            f"{all_chunks_str}"
            f"{full_doc_prompt_section}"
            f"{context_html_section}"
        )

    @staticmethod
    def _build_answer_prompt(questions_batch: List[str], chunks_for_answers_batch: List[str], source_info: Dict[str, str]) -> str:
        """Prompt for generating answers for a given batch of questions."""

        institution = source_info.get("institution", source_info.get("domain", ""))

        qa_prompts_parts = []
        for i, (question, chunk) in enumerate(zip(questions_batch, chunks_for_answers_batch)):
            qa_prompts_parts.append(
                f"<ITEM_{i+1}>\n"
                f"<QUESTION_{i+1}>\n{question}\n</QUESTION_{i+1}>\n"
                f"<CONTEXT_{i+1}>\n{chunk['chunk']}\n</CONTEXT_{i+1}>\n"
                f"</ITEM_{i+1}>"
            )
        all_qa_items_str = "\n\n".join(qa_prompts_parts)

        return (
            "You are an assistant helping to create high-quality FAQ answers for the University of Brasilia chatbot.\n\n"
            f"{DEFAULT_STYLE_SECTION}"
            f"You will receive a batch of {len(questions_batch)} question-context items, each enclosed in <ITEM_N> tags. "
            "For each item, answer the <QUESTION_N> using ONLY the information in its corresponding <CONTEXT_N>.\n"
            "If the chunk contains an URL and it's relevant to the question, mention it in the answer.\n"
            "If a specific context does not provide enough information, reply for that item with \"Não possuo informações suficientes para responder a essa pergunta.\"\n\n"
            f"{all_qa_items_str}\n\n"
            f"Generate exactly {len(questions_batch)} answers, one per line, in the same order as the input items. "
            "Return ONLY the answers, one per line, with no numbering or other text."
        )

    @staticmethod
    def _build_faq_default_qa_pair_prompt(original_qa_batch: List[Dict[str, str]], source_info: Dict[str, str]) -> str:
        """Prompt for rephrasing a batch of original QA pairs (question and answer simultaneously)."""
        institution = source_info.get("institution", source_info.get("domain", ""))

        item_prompts_parts = []
        for i, qa_pair in enumerate(original_qa_batch):
            item_prompts_parts.append(
                f"<ITEM_{i+1}>\n"
                f"<ORIGINAL_QUESTION_{i+1}>\n{qa_pair['question']}\n</ORIGINAL_QUESTION_{i+1}>\n"
                f"<ORIGINAL_ANSWER_{i+1}>\n{qa_pair['answer']}\n</ORIGINAL_ANSWER_{i+1}>\n"
                f"</ITEM_{i+1}>"
            )
        all_original_items_str = "\n\n".join(item_prompts_parts)

        return (
            f"You are an LLM assistant refining FAQ content for the {institution} chatbot, guided by the WRITING STYLE section below.\n"
            f"{DEFAULT_STYLE_SECTION}"  # style_section includes a trailing \n if not empty, or is ""
            f"You will receive {len(original_qa_batch)} original Q&A pairs in <ITEM_N> tags.\n"
            "For each <ITEM_N>:\n"
            "1. Rephrase <ORIGINAL_QUESTION_N> if needed. If it's already good, you may keep it largely unchanged but ensure it fits a conversational context.\n"
            "2. Formulate an answer for your rephrased question using ONLY <ORIGINAL_ANSWER_N>. If the original is a good fit, it can be largely unchanged.\n"
            "3. If <ORIGINAL_ANSWER_N> is insufficient for the rephrased question, the rephrased_answer MUST BE: 'Não possuo informações suficientes para responder a essa pergunta.' Include relevant URLs from the original answer.\n\n"
            f"{all_original_items_str}\n\n"
            "Output: For each item, provide a JSON object on a new line: {'rephrased_question': 'Your rephrased Q', 'rephrased_answer': 'Your rephrased A'}.\n"
            f"Generate exactly {len(original_qa_batch)} JSON objects. ONLY JSON, one per line."
        )

    def _process_chunk_batch(
        self,
        current_batch_chunks: List[Dict[str, str]], # Each dict is {'chunk': text}
        source_info: Dict[str, str],
        full_document_text: str,
        context_html_text: str
    ) -> List[Dict[str, str]]:
        """Processes a batch of chunks to generate questions and then answers."""
        batch_results = []

        # 1. Generate questions
        question_batch_prompt = QAGenerator._build_question_prompt(current_batch_chunks, full_document_text, context_html_text, source_info)
        raw_questions_str = self.question_client.generate_text(question_batch_prompt, temperature=0.7)

        if not raw_questions_str:
            logger.warning(f"No questions generated for a batch of {len(current_batch_chunks)} chunks. Source: {source_info.get('path', 'N/A')}")
            return []

        generated_questions = [q.strip() for q in raw_questions_str.strip().splitlines() if q.strip()]

        if len(generated_questions) != len(current_batch_chunks):
            logger.warning(f"Question count mismatch: expected {len(current_batch_chunks)}, got {len(generated_questions)}. Source: {source_info.get('path', 'N/A')}. Raw questions: {raw_questions_str}")
            return []

        # 2. Generate answers for the batch of questions and their original chunks
        answer_batch_prompt = QAGenerator._build_answer_prompt(generated_questions, current_batch_chunks, source_info)
        raw_answers_str = self.answer_client.generate_text(answer_batch_prompt, temperature=0.5)

        if not raw_answers_str:
            logger.warning(f"No answers generated for {len(generated_questions)} questions. Source: {source_info.get('path', 'N/A')}")
            return []

        generated_answers = [a.strip() for a in raw_answers_str.strip().splitlines() if a.strip()]

        if len(generated_answers) != len(generated_questions):
            logger.warning(f"Answer count mismatch: expected {len(generated_questions)}, got {len(generated_answers)}. Source: {source_info.get('path', 'N/A')}. Raw answers: {raw_answers_str}")
            return []

        for q, a in zip(generated_questions, generated_answers):
            batch_results.append({"question": q, "answer": a})
        return batch_results

    def _process_faq_batch(
        self,
        current_batch_faqs: List[Dict[str, str]], # Each dict is {'question': ..., 'answer': ...}
        source_info: Dict[str, str]
    ) -> List[Dict[str, str]]:
        """Processes a batch of original FAQs to generate rephrased Q&A pairs."""
        batch_results = []

        rephrase_qa_prompt = QAGenerator._build_faq_default_qa_pair_prompt(current_batch_faqs, source_info)
        raw_rephrased_qa_str = self.question_client.generate_text(rephrase_qa_prompt, temperature=0.7)

        if not raw_rephrased_qa_str:
            logger.warning(f"No rephrased Q&A pairs generated for a batch of {len(current_batch_faqs)} FAQs. Source: {source_info.get('path', 'N/A')}")
            return []

        generated_rephrased_pairs_data = []
        parse_errors = 0
        for line_num, line in enumerate(raw_rephrased_qa_str.strip().splitlines()):
            if not line.strip():
                continue
            try:
                pair = json.loads(line.strip())
                if isinstance(pair, dict) and 'rephrased_question' in pair and 'rephrased_answer' in pair:
                    generated_rephrased_pairs_data.append(pair)
                else:
                    logger.warning(f"Invalid JSON object structure in line {line_num+1} for FAQ batch. Source: {source_info.get('path', 'N/A')}: {line}")
                    parse_errors += 1
            except json.JSONDecodeError as jde:
                logger.warning(f"JSONDecodeError in line {line_num+1} for FAQ batch. Source: {source_info.get('path', 'N/A')}: {line}. Error: {jde}")
                parse_errors += 1

        if len(generated_rephrased_pairs_data) != len(current_batch_faqs):
            logger.warning(f"Generated rephrased Q&A pairs length ({len(generated_rephrased_pairs_data)}) does not match original FAQ length ({len(current_batch_faqs)}). Source: {source_info.get('path', 'N/A')}")
            return []
        
        if parse_errors > 0 and not generated_rephrased_pairs_data:
             logger.warning(f"All {parse_errors} lines failed to parse as Q&A JSON for FAQ batch. Source: {source_info.get('path', 'N/A')}. Raw: {raw_rephrased_qa_str}")
             return []
        elif parse_errors > 0:
            logger.warning(f"{parse_errors} lines failed to parse as Q&A JSON for FAQ batch, processing successfully parsed pairs. Source: {source_info.get('path', 'N/A')}.")

        for pair_data in generated_rephrased_pairs_data:
            batch_results.append({
                "question": pair_data['rephrased_question'],
                "answer": pair_data['rephrased_answer']
            })
        return batch_results


    def _generate_and_save_qa(
        self,
        items_to_process: List[Any],
        batch_processing_func: Callable[..., List[Dict[str, str]]],
        source_path: str,
        file_title: str,
        output_dir: Path,
        batch_size: int,
        batch_func_kwargs: Dict[str, Any]
    ) -> List[Dict[str, str]]:
        """Generic worker to process items in batches, generate Q&A, and save them."""

        file_path = Path(source_path)
        domain, path, url = FileProcessor.extract_domain_and_path(file_path)
        rel_path = file_path.relative_to(Path(self.config["base_dir"]))
        institution = FileProcessor.get_institution_name(domain)
        source_info = {
            "domain": domain, "path": path, "url": url, "institution": institution,
        }

        default_qa_dir = output_dir / "default_qa"
        default_qa_dir.mkdir(parents=True, exist_ok=True)

        default_qa_filename = f"default_{slugify(file_title)}_{create_hash(str(rel_path))}.json"
        qa_output_file_path = default_qa_dir / default_qa_filename

        if os.path.exists(qa_output_file_path):
            try:
                with open(qa_output_file_path, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
                logger.info(f"Successfully loaded {len(existing_data)} items from {qa_output_file_path}. Skipping generation.")
                return existing_data
            except Exception as e:
                logger.warning(f"Could not load existing data from {qa_output_file_path}: {e}. Regenerating.")

        all_qa_pairs = []
        for i in range(0, len(items_to_process), batch_size):
            current_batch = items_to_process[i:i + batch_size]
            if not current_batch:
                continue

            processed_batch_results = batch_processing_func(
                current_batch, source_info, **batch_func_kwargs
            )

            for res in processed_batch_results:
                question = res["question"]
                answer = res["answer"]
                all_qa_pairs.append({
                    "question": question,
                    "answer": answer,
                    "url": url, 
                    "qa_pair_hash": f"default_{create_hash(str(rel_path) + question)}"
                })
        
        if not all_qa_pairs:
            logger.info(f"No QA pairs generated for {file_path}.")
            return []

        try:
            with open(qa_output_file_path, 'w', encoding='utf-8') as f:
                json.dump(all_qa_pairs, f, ensure_ascii=False, indent=2)
            logger.info(f"Created new QA file with {len(all_qa_pairs)} pairs: {qa_output_file_path}.")
        except Exception as e:
            logger.error(f"Failed to write new QA file {qa_output_file_path}: {e}")

        return all_qa_pairs


    def generate_qa_pairs(
        self,
        chunks: List[Dict[str, str]],
        file_path: str,
        file_title: str,
        output_dir: Path,
        full_document_text: str,
        context_html_text: str,
        batch_size: int = 1,
    ) -> List[Dict[str, str]]:
        """Generate one baseline QA pair for each chunk provided, processing in batches."""
        
        batch_func_kwargs = {
            "full_document_text": full_document_text,
            "context_html_text": context_html_text,
        }
        
        return self._generate_and_save_qa(
            items_to_process=chunks,
            batch_processing_func=self._process_chunk_batch,
            source_path=file_path,
            file_title=file_title,
            output_dir=output_dir,
            batch_size=batch_size,
            batch_func_kwargs=batch_func_kwargs
        )


    def generate_qa_pairs_from_faq(
        self,
        original_faq: List[Dict[str, Any]], 
        file_path: str,
        file_title: str,
        output_dir: Path,
        batch_size: int = 1,
    ) -> List[Dict[str, str]]:
        """Generate rephrased QA pairs from original FAQ pairs, processing in batches."""

        return self._generate_and_save_qa(
            items_to_process=original_faq,
            batch_processing_func=self._process_faq_batch,
            source_path=file_path,
            file_title=file_title,
            output_dir=output_dir,
            batch_size=batch_size,
            batch_func_kwargs={} 
        )
