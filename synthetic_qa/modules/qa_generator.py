import logging
import json
import os
from pathlib import Path
import time
from typing import Dict, Any, List, Callable

from slugify import slugify

from .llm_client import LLMClient
from .file_processor import FileProcessor
from modules.utils import create_hash

logger = logging.getLogger(__name__)

DEFAULT_STYLE_SECTION_QUESTION = (
    "WRITING STYLE: Default Question\n"
    "Description: Generate a clear, straightforward question using complete sentences. Avoid excessive formality or informality. Focus on accurately representing the core content of the original question/chunk. **Crucially: If the original answer/chunk was very short (like \"Sim.\", \"Não.\"), ensure this question naturally elicits the expanded, polite answer generated for the Default pair.** The question should sound like a typical, clear query a student might ask an assistant.\n"
    "Goal: Establish a clean, well-formed baseline; represent clear, unambiguous queries.\n"
)

DEFAULT_STYLE_SECTION_ANSWER = (
    "WRITING STYLE: Default Answer\n"
    "Description: Generate a clear, direct, and polite answer using complete sentences. The answer should be helpful and directly address the user's question based *only* on the provided context. Format the answer using markdown for better readability in a chatbot interface (e.g., use bullet points for lists, bold for emphasis on key terms, etc). If the original context was very brief (e.g., just \"Yes\" or \"No\"), expand it into a full, polite sentence (e.g., \"Sim, é possível fazer isso.\"). Maintain a neutral and informative tone.\n"
    "Goal: Provide a clear, self-contained, and polite answer to the user's question, formatted for a chatbot.\n"
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


    @staticmethod
    def _build_question_prompt(chunks_batch: List[str], document_context: str, context_html_text: str, file_title: str = "", file_name: str = "") -> str:
        """Prompt for generating baseline questions from a batch of chunks.

        If *document_context* is provided it is supplied inside <DOCUMENT_CONTEXT> tags so the
        model has additional context for phrasing while still being instructed to
        ask ONLY about content present in the chunks.
        """
        file_metadata_section = ""
        if file_title or file_name:
            metadata_parts = []
            if file_title:
                metadata_parts.append(f"Document Title: '{file_title}'")
            if file_name:
                metadata_parts.append(f"File Name: '{file_name}'")
            file_metadata_section = f"You are working with excerpts from: {', '.join(metadata_parts)}.\n"

        document_context_prompt_section = f"\n<DOCUMENT_CONTEXT>\n{document_context}\n</DOCUMENT_CONTEXT>"

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
            f"{file_metadata_section}"
            f"You will receive a batch of {len(chunks_batch)} text excerpts (<CHUNK_N>). You will also be given the surrounding text from the original document where these excerpts appeared (provided as <DOCUMENT_CONTEXT>). "
            "For each excerpt <CHUNK_N>, your task is to craft ONE natural question in Portuguese that students, faculty, or staff might ask. "
            "Each question MUST be answerable using ONLY the information contained in its corresponding <CHUNK_N>. "
            "Use the document context only to understand the surrounding text and improve wording; do NOT include information that appears exclusively outside the chunk.\n"
            "In addition, you may also receive extra context (provided as <CONTEXT_HTML>). This extra context *may* help clarify information if something in the chunk is ambiguous or unclear, but you must still ensure that each question is answerable using only its corresponding chunk. Use the extra context only to improve the naturalness or clarity of the question, not to introduce information that is not present in the chunk.\n"
            "Remember: the user (student) has no access to or awareness of these documents and it not very knowledgeable about the specific chunk content. They are simply seeking information or help, not referencing any source. Therefore, **craft each question so that it is specific enough it naturally requires the information from the chunk to be answered without feeling forced or artificial**. Make the question feel authentic and relevant, as if it could be asked in a real conversation.\n\n"
            "So, for this document information to be retrieved, the question will probably have to mention the document name/title/subject (but not always about specific contents).\n\n"
            "Output: Return a JSON object with a 'questions' key containing a list of questions, following the template below:\n\n"
            "{\n  \"questions\": [\n    \"...\",\n    ...\n  ]\n}\n\n"
            f"Generate a single, valid JSON object containing exactly {len(chunks_batch)} questions. Respond with ONLY the JSON object.\n\n"
            f"{DEFAULT_STYLE_SECTION_QUESTION}"
            f"{all_chunks_str}"
            f"{document_context_prompt_section}"
            f"{context_html_section}"
        )

    @staticmethod
    def _build_answer_prompt(questions_batch: List[str], chunks_for_answers_batch: List[str], file_title: str = "", file_name: str = "") -> str:
        """Prompt for generating answers for a given batch of questions."""

        file_metadata_section = ""
        if file_title or file_name:
            metadata_parts = []
            if file_title:
                metadata_parts.append(f"Document Title: '{file_title}'")
            if file_name:
                metadata_parts.append(f"File Name: '{file_name}'")
            file_metadata_section = f"You are generating answers based on content from: {', '.join(metadata_parts)}.\n\n"

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
            f"{file_metadata_section}"
            f"{DEFAULT_STYLE_SECTION_ANSWER}"
            f"You will receive a batch of {len(questions_batch)} question-context items, each enclosed in <ITEM_N> tags. "
            "For each item, answer the <QUESTION_N> using ONLY the information in its corresponding <CONTEXT_N>.\n"
            "If the chunk contains an URL and it's relevant to the question, mention it in the answer.\n"
            "If a specific context does not provide enough information, reply for that item with something like \"Não possuo informações suficientes para responder a essa pergunta.\"\n\n"
            f"{all_qa_items_str}\n\n"
            "Output: Return a JSON object with an 'answers' key containing a list of answers, following the template below:\n\n"
            "{\n  \"answers\": [\n    \"...\",\n    ...\n  ]\n}\n\n"
            f"Generate a single, valid JSON object containing exactly {len(questions_batch)} answers. Respond with ONLY the JSON object."
        )

    @staticmethod
    def _build_faq_default_qa_pair_prompt(original_qa_batch: List[Dict[str, str]], file_title: str = "", file_name: str = "") -> str:
        """Prompt for rephrasing a batch of original QA pairs (question and answer simultaneously)."""

        file_metadata_section = ""
        if file_title or file_name:
            metadata_parts = []
            if file_title:
                metadata_parts.append(f"Document Title: '{file_title}'")
            if file_name:
                metadata_parts.append(f"File Name: '{file_name}'")
            file_metadata_section = f"You are refining Q&A content from: {', '.join(metadata_parts)}.\n\n"

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
            f"You are an LLM assistant refining FAQ content for the University of Brasilia chatbot, guided by the WRITING STYLE section below.\n"
            f"{file_metadata_section}"
            f"{DEFAULT_STYLE_SECTION_QUESTION}\n{DEFAULT_STYLE_SECTION_ANSWER}"
            f"You will receive {len(original_qa_batch)} original Q&A pairs in <ITEM_N> tags.\n"
            "For each <ITEM_N>:\n"
            "1. Rephrase <ORIGINAL_QUESTION_N> if needed. If it's already good, you may keep it largely unchanged but ensure it fits a conversational context.\n"
            "2. Formulate an answer for your rephrased question using ONLY <ORIGINAL_ANSWER_N>. If the original is already a good fit, it can be mostly unchanged, but ensure it fits the conversational context (in the default style).\n"
            "3. If <ORIGINAL_ANSWER_N> is insufficient for the rephrased question, the rephrased_answer should be something like: 'Não possuo informações suficientes para responder a essa pergunta.' Include relevant URLs from the original answer.\n\n"
            "4. Preserve all the original question and answer information, including URLs.\n"
            f"{all_original_items_str}\n\n"
            "Output: Return a JSON object with a 'qa_pairs' key containing a list of question-answer pairs, following the template below:\n\n"
            "{\n  \"qa_pairs\": [\n    {\"rephrased_question\": \"...\", \"rephrased_answer\": \"...\"},\n    ...\n  ]\n}\n\n"
            f"Generate a single, valid JSON object containing exactly {len(original_qa_batch)} pairs. Respond with ONLY the JSON object."
        )

    def _process_chunk_batch(
        self,
        current_batch_chunks: List[Dict[str, Any]],
        rel_path: Path,
        document_context_text: str,
        context_html_text: str,
        file_title: str = ""
    ) -> List[Dict[str, Any]]:
        """Processes a batch of chunks to generate questions and then answers."""
        
        # Extract file metadata
        file_name = rel_path.name
        
        # 1. Generate questions
        question_batch_prompt = QAGenerator._build_question_prompt(
            current_batch_chunks, document_context_text, context_html_text, file_title, file_name
        )
        question_response = None
        logger.info(f"Requesting LLM-based question generation for batch of {len(current_batch_chunks)} chunks from {str(rel_path)}...")
        question_response = self.question_client.generate_text(
            question_batch_prompt,
            temperature=0.7,
            json_output=True
        )
        if not question_response or not isinstance(question_response, dict):
            logger.warning(f"Invalid question response format for batch of {len(current_batch_chunks)} chunks from {str(rel_path)}")
            return []
        
        generated_questions = question_response.get("questions", [])
        if not isinstance(generated_questions, list):
            logger.warning(f"Could not extract questions list from response for {str(rel_path)}")
            return []

        # 2. Generate answers for the batch of questions and their original chunks
        answer_batch_prompt = QAGenerator._build_answer_prompt(generated_questions, current_batch_chunks, file_title, file_name)
        logger.info(f"Generating answers for {len(generated_questions)} questions from {str(rel_path)}")

        answer_response = self.answer_client.generate_text(
            answer_batch_prompt,
            temperature=0.5,
            json_output=True
        )

        if not answer_response or not isinstance(answer_response, dict):
            logger.warning(f"Invalid answer response format for batch of {len(generated_questions)} questions from {str(rel_path)}")
            return []

        generated_answers = answer_response.get("answers", [])
        if not isinstance(generated_answers, list):
            logger.warning(f"Could not extract answers list from response for {str(rel_path)}")
            return []

        # 3. Combine results
        batch_results = []
        if len(generated_questions) == len(current_batch_chunks) and len(generated_answers) == len(current_batch_chunks):
            for i, (q, a) in enumerate(zip(generated_questions, generated_answers)):
                updated_chunk = current_batch_chunks[i].copy()
                updated_chunk["question"] = q
                updated_chunk["answer"] = a
                updated_chunk["qa_pair_hash"] = f"default_{create_hash(str(rel_path) + q)}"
                updated_chunk.pop("chunk", None)
                updated_chunk.pop("chunk_hash", None)

                batch_results.append(updated_chunk)
        else:
            logger.warning(
                f"Mismatch in generated items for {str(rel_path)}. "
                f"Expected: {len(current_batch_chunks)}, "
                f"Got: {len(generated_questions)} questions, {len(generated_answers)} answers."
            )
            return []
        
        return batch_results

    def _process_faq_batch(
        self,
        current_batch_faqs: List[Any],
        rel_path: Path,
        file_title: str = ""
    ) -> List[Any]:
        """Processes a batch of original FAQs to generate rephrased Q&A pairs."""
        batch_results = []

        # Extract file metadata
        file_name = rel_path.name

        rephrase_qa_batch_prompt = QAGenerator._build_faq_default_qa_pair_prompt(current_batch_faqs, file_title, file_name)
        
        # LLMClient with json_output=True is expected to return a dict.
        logger.info(f"Requesting LLM-based rephrasing for batch of {len(current_batch_faqs)} FAQs from {str(rel_path)}...")
        response = None
        response = self.question_client.generate_text(
            rephrase_qa_batch_prompt,
            temperature=0.7,
            json_output=True
        )

        if not response or not isinstance(response, dict):
            logger.warning(f"Invalid response format for batch of {len(current_batch_faqs)} FAQs for {str(rel_path)}")
            return []
                
        generated_rephrased_pairs_data = response.get("qa_pairs", [])
        if not isinstance(generated_rephrased_pairs_data, list):
            return []
        
        if len(generated_rephrased_pairs_data) != len(current_batch_faqs):
            logger.warning(f"Mismatch in generated items for {str(rel_path)}. Expected: {len(current_batch_faqs)}, Got: {len(generated_rephrased_pairs_data)}")
            return []

        for i, pair_data in enumerate(generated_rephrased_pairs_data):
            if isinstance(pair_data, dict) and all(key in pair_data for key in ['rephrased_question', 'rephrased_answer']):
                updated_pair = current_batch_faqs[i].copy()
                updated_pair["question"] = pair_data["rephrased_question"]
                updated_pair["answer"] = pair_data["rephrased_answer"]
                updated_pair["qa_pair_hash"] = f"default_{updated_pair['qa_pair_hash']}"
                
                batch_results.append(updated_pair)
        
        return batch_results

    def _create_contextual_window(self, full_document_text: str, current_batch: List[Dict[str, Any]], rel_path: Path) -> str:
        """
        Creates a contextual window of text around a batch of chunks.
        """
        if not full_document_text or not current_batch or 'chunk' not in current_batch[0]:
            return full_document_text

        first_chunk_text = current_batch[0].get('chunk')[-200:]
        second_chunk_text = current_batch[1].get('chunk')[-200:] if len(current_batch) > 1 else None
        last_chunk_text = current_batch[-1].get('chunk')[-200:]

        if not first_chunk_text or not last_chunk_text:
            return full_document_text

        try:
            start_index = full_document_text.find(first_chunk_text)
            
            # Search for last chunk after the first one to avoid incorrect matches
            end_index_search_start = start_index + len(first_chunk_text) if start_index != -1 else 0
            last_chunk_found_at = full_document_text.find(last_chunk_text, end_index_search_start)

            if start_index == -1 and last_chunk_found_at == -1 and second_chunk_text:
                start_index = full_document_text.find(second_chunk_text)
            
            if start_index != -1 or last_chunk_found_at != -1:
                start_index = last_chunk_found_at if start_index == -1 else start_index
                last_chunk_found_at = start_index+1 if last_chunk_found_at == -1 else last_chunk_found_at

                end_index = last_chunk_found_at + len(last_chunk_text)
                
                # Define the window
                window_start = max(0, start_index - 4000)
                window_end = min(len(full_document_text), end_index + 4000)
                
                contextual_text = full_document_text[window_start:window_end]
                logger.debug(f"Using contextual window for {str(rel_path)} ({window_start}-{window_end})")
                return contextual_text
            else:
                logger.warning(f"Could not locate chunk batch in document for {str(rel_path)}. Using full document as context.")
                return full_document_text

        except Exception as e:
            logger.error(f"Error creating contextual window for {str(rel_path)}: {e}. Using full document.")
            return full_document_text

    def _generate_and_save_qa(
        self,
        items_to_process: List[Any],
        processing_func: Callable[..., List[Dict[str, str]]],
        source_path: str,
        file_title: str,
        output_dir: Path,
        batch_size: int,
        batch_func_kwargs: Dict[str, Any]
    ) -> List[Dict[str, str]]:
        """Generic worker to process items in batches, generate Q&A, and save them."""

        file_path = Path(source_path)
        rel_path = file_path.relative_to(Path(self.config["base_dir"]))
        
        default_qa_dir = output_dir / "default_qa"
        default_qa_dir.mkdir(parents=True, exist_ok=True)

        default_qa_filename = f"default_{slugify(file_title)}_{create_hash(str(rel_path))}.json"
        qa_output_file_path = default_qa_dir / default_qa_filename

        if os.path.exists(qa_output_file_path):
            try:
                with open(qa_output_file_path, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)

                if len(existing_data) != len(items_to_process):
                    logger.warning(
                        "Cached default QA file (%s) has %d items but the current input has %d. Regenerating to keep lists aligned.",
                        qa_output_file_path, len(existing_data), len(items_to_process)
                    )
                    os.remove(qa_output_file_path)
                else:
                    logger.info(
                        "Successfully loaded %d items from %s. Skipping generation.",
                        len(existing_data), qa_output_file_path
                    )
                    return existing_data
            except Exception as e:
                logger.warning(f"Could not load existing data from {qa_output_file_path}: {e}. Regenerating.")

        all_qa_pairs = []
        for i in range(0, len(items_to_process), batch_size):
            current_batch = items_to_process[i:i + batch_size]
            if not current_batch:
                continue
            
            # Create a mutable copy for this iteration
            current_batch_kwargs = batch_func_kwargs.copy()
            
            # If we're processing chunks, try to create a contextual window
            if processing_func.__name__ == '_process_chunk_batch' and 'full_document_text' in current_batch_kwargs:
                current_batch_kwargs['document_context_text'] = self._create_contextual_window(
                    full_document_text=current_batch_kwargs.pop("full_document_text"),
                    current_batch=current_batch,
                    rel_path=rel_path
                )

            processed_batch_results = []
            while len(processed_batch_results) == 0:
                processed_batch_results = processing_func(
                    current_batch, rel_path, **current_batch_kwargs
                )
                time.sleep(1)
                logger.info(f"Retrying {processing_func.__name__} for batch of {len(current_batch)} items from {str(rel_path)}...")


            for res in processed_batch_results:
                all_qa_pairs.append(res)

        if len(all_qa_pairs) != len(items_to_process):
            logger.error(
                "Generated %d QA pairs but expected %d – aborting write for %s",
                len(all_qa_pairs), len(items_to_process), qa_output_file_path
            )
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
            "file_title": file_title,
        }
        
        return self._generate_and_save_qa(
            items_to_process=chunks,
            processing_func=self._process_chunk_batch,
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
            processing_func=self._process_faq_batch,
            source_path=file_path,
            file_title=file_title,
            output_dir=output_dir,
            batch_size=batch_size,
            batch_func_kwargs={"file_title": file_title} 
        )

    def _process_component_text(
        self,
        items_to_process: List[str],  # Always a list of one component text
        rel_path: Path,
        file_title: str = ""
    ) -> List[Dict[str, str]]:
        """Processes a single component markdown text to generate QA pairs."""
        component_text = items_to_process[0]
        
        # Extract file metadata
        file_name = rel_path.name
        
        file_metadata_section = ""
        if file_title or file_name:
            metadata_parts = []
            if file_title:
                metadata_parts.append(f"Document Title: '{file_title}'")
            if file_name:
                metadata_parts.append(f"File Name: '{file_name}'")
            file_metadata_section = f"You are generating Q&A content from: {', '.join(metadata_parts)}.\n\n"
        
        prompt = (
            "You are an LLM Generator creating synthetic data for the University of Brasilia chatbot.\n\n"
            f"{file_metadata_section}"
            "The chatbot can answer any question about the university, but the following Q&A pairs are examples where the student's question required information from this component document (retrieved by a RAG system).\n\n"
            "The user is a student who does not know about the present document, so the question must be specific (but not contrived) enough for the document to be retrieved.\n\n"
            "You will receive the Markdown text of a university component, including code, name, syllabus (ementa), objectives, bibliography, and offerings (teachers, schedules, vacancies, location, etc). Generate the minimum number of natural, realistic, relevant, and useful question-answer pairs IN PORTUGUESE that a brazilian student might ask about this document, using only the information present.\n"
            "Do NOT reference the document. Do NOT generate questions about missing information.\n"
            "Your response should be in styled (designed for easy student understanding) well-formatted markdown.\n"
            "Return a JSON object with a 'qa_pairs' key containing a list of question-answer pairs, following the template below:\n\n"
            "{\n  qa_pairs: [\n    {\"question\": \"...\", \"answer\": \"...\"},\n    ...\n  ]\n}\n\n"
            "Component text:\n\n"
            f"{component_text}"
        )
        response = None
        logger.info(f"Requesting LLM-based QA generation for component from {str(rel_path)}...")
        response = self.question_client.generate_text(
            prompt,
            json_output=True,
            temperature=0.4
        )

        qa_pairs = response.get('qa_pairs', [])
        if not qa_pairs:
            logger.warning(f"No QA pairs generated for {str(rel_path)}")
            return []

        return qa_pairs

    def generate_qa_pairs_from_component(
        self,
        component_text: str,
        file_path: str,
        file_title: str,
        output_dir: Path,
    ) -> List[Dict[str, str]]:
        """Generate QA pairs from a component's full markdown text (not chunked)."""
        return self._generate_and_save_qa(
            items_to_process=[component_text],
            processing_func=self._process_component_text,
            source_path=file_path,
            file_title=file_title,
            output_dir=output_dir,
            batch_size=1,
            batch_func_kwargs={"file_title": file_title}
        )
