# qa_processor_raft.py

import hashlib
import os
import logging
import json
from pathlib import Path
from typing import List, Tuple, Dict, Any
from bs4 import BeautifulSoup
from tqdm import tqdm
import textwrap
import random
import time
from slugify import slugify

from modules.utils import create_hash

# Assuming llm_client and file_processor are in the same directory structure
from .llm_client import LLMClient
from .file_processor import FileProcessor

logger = logging.getLogger(__name__)

# --- Constants for RAFT ---
# Prompt for generating the CoT Answer (A*) based ONLY on the Golden Answer (D*)
# NOTE: This prompt specifically asks for a FORMAL style as requested.
ANSWER_TAG = "<ANSWER>"
COT_ANSWER_GENERATION_PROMPT = f"""
You are an expert assistant creating training data for a university chatbot that will chat with the students.
Given an original Question and its corresponding original Answer (which serves as the definitive context), generate a detailed chat response. This response has two parts: internal reasoning in english (which won't be shown to the user) and the final answer in portuguese.

**Instructions:**

1.  **Analyze:** Understand the user's implicit need based on the 'Original Question'.
2.  **Internal Reasoning `<REASON>:`:** Provide a short and clear, objective reasoning based *exclusively* on the information within the '{{context_source_name}} (Context)'. Do NOT add outside information. This reasoning is for internal understanding and training purposes aimed at FACTUALITY.
3.  **Reasoning Citations:** **Within the Internal Reasoning section only**, when referencing specific parts of the '{{context_source_name}} (Context)', enclose the verbatim text within `##begin_quote##` and `##end_quote##`. Text outside these tags should be your own reasoning/connecting words.
4.  **Style:** Maintain a **friendly, formal, modern, effective, polite, expert chatbot assistant persona** suitable for the University of Brasília (UnB).
5.  **Final User-Facing Answer Tag:** Conclude the *entire* response with the final answer intended **directly for the end-user**, prefixed EXACTLY with `{ANSWER_TAG}:`.
6.  **User-Facing Answer Content:**
    *   **Visibility:** Understand that **only the text following the `{ANSWER_TAG}:` tag will be shown to the end-user.** The user will *not* see the '{{context_source_name}} (Context)' you were given, nor the 'Internal Reasoning' section.
    *   **Self-Contained:** The final answer must be **self-contained and directly usable**. It should not refer back to "the context" ambiguously.
    *   **Include Links/Sources:** If the answer relies on specific web pages or documents mentioned in the '{{context_source_name}} (Context)', **include the necessary Markdown links (e.g., `[Link Text](URL)`) directly within this final answer section.** If specific documents are sources, reference them clearly (e.g., "according to the Course Regulation document available at [link]").
    *   **Completeness:** Ensure all relevant information from the '{{context_source_name}} (Context)' needed to address the 'Original Question' is summarized or directly included in this final answer.
    * **Always answer the question directly first** (*no greetings* unless the user greets you).
7.  **If Unanswerable:** If the '{{context_source_name}} (Context)' genuinely doesn't contain the information to answer the 'Original Question', state that clearly in the Internal Reasoning and use `{ANSWER_TAG}: Lamento, mas não possuo informações suficientes para responder à sua pergunta sobre este tópico específico.`.

**Input:**
- Original Question: '{{original_question}}'
- {{context_source_name}} (Context): '{{context_content}}'
"""


# Prompt for generating a styled question (Q) based on an original pair and a style
# (Adapted from your original generate_styled_qa)
STYLED_QUESTION_GENERATION_PROMPT_TEMPLATE = """
You are an LLM Generator creating synthetic data for a university chatbot.
Create ONE alternative FAQ question based on the Original Pair provided below.

**WRITING STYLE**: {style_name}
- Description: {style_description}
- Goal: {style_goal}

**Instructions:**
- Rewrite *only the question*, preserving the **exact original meaning and intent**.
- **DO NOT ADD ANY NEW INFORMATION**.
- Follow the specified writing style closely.
- The user knows they are talking to an assistant chatbot.
- Output must be IN PORTUGUESE.
{previous_questions_prompt}

**Original Pair:**
- Original Question: '{original_question}'
- Original Answer: '{original_answer}'

**Previous Styled Questions for this item:**
{previous_questions_str}

**Output Format:**
Return ONLY the single generated alternative question IN PORTUGUESE. Do not include ANY other text, numbering, or explanations.
"""

class QAProcessorRAFT:
    """
    Generates RAFT-formatted training data from FAQ documents.
    Each original FAQ Answer serves as a Golden Document (D*).
    Other original FAQ Answers serve as Distractors (Dk).
    """

    # Keep detection logic if needed upstream
    @staticmethod
    def detect_faq_document(soup: BeautifulSoup, filename: str) -> bool:
        """(Copied from original) Determine if a document is an FAQ."""

        faq_indicators = ['faq', 'perguntas', 'frequentes', 'duvidas', 'q&a']
        if any(indicator in filename.lower() for indicator in faq_indicators): return True
        title = soup.find('title')
        if title and any(indicator in title.text.lower() for indicator in faq_indicators): return True
        if len(soup.find_all('details')) > 2 and len(soup.find_all('summary')) > 2: return True
        questions_count = 0
        for tag in soup.find_all(['b', 'strong']):
            text = tag.get_text().strip()
            if text.endswith('?') or any(text.lower().startswith(word) for word in ['como', 'existe', 'existem', 'qual', 'quais', 'o que', 'onde', 'quando', 'por que', 'posso']): questions_count += 1
        if questions_count > 3: return True
        return False


    @staticmethod
    def generate_styled_question_raft(
        original_question: str,
        original_answer: str,
        writing_style: Dict[str, str],
        previous_styled_questions: List[str],
        llm_client: LLMClient
    ) -> str | None:
        """Generates a single styled question based on the original pair and style."""
        style_name = writing_style.get("name", "Unknown Style")
        style_desc = writing_style.get("description", "")
        style_goal = writing_style.get("goal", "")

        previous_questions_prompt = ""
        previous_questions_str = ""
        if previous_styled_questions:
            previous_questions_prompt = "- The new question should be distinct from the previous styled questions (do different phrasings, coherent reorderings, or focus on a different part of the answer)."
            previous_questions_str = "\n".join([f"- {pq}" for pq in previous_styled_questions])

        prompt = STYLED_QUESTION_GENERATION_PROMPT_TEMPLATE.format(
            style_name=style_name,
            style_description=style_desc,
            style_goal=style_goal,
            original_question=original_question,
            original_answer=original_answer,
            previous_questions_prompt=previous_questions_prompt,
            previous_questions_str=previous_questions_str
        )

        try:
            response = llm_client.generate_text(prompt.lstrip(), temperature=0.7)
            if response:
                # Basic cleaning, remove potential numbering/bullets if LLM adds them
                styled_q = response.strip().lstrip('*- ').splitlines()[0].strip()
                return styled_q
            else:
                logger.warning(f"LLM returned empty response for styled question generation (Style: {style_name})")
                return None
        except Exception as e:
            logger.error(f"Error generating styled question (Style: {style_name}): {e}", exc_info=True)
            return None

    @staticmethod
    def generate_cot_answer_raft(
        original_question: str,
        original_answer: str,
        chunk: Dict[str, Any],
        llm_client: LLMClient,
    ) -> str | None:
        """Generates the CoT Answer (A*) based only on the original Q/A pair."""

        try:
            context_source_name = "Chunk" if chunk else "Original Answer"
            context_content = chunk['chunk'] if chunk else original_answer
            prompt = COT_ANSWER_GENERATION_PROMPT.format(
                original_question=original_question,
                context_source_name=context_source_name,
                context_content=context_content
            )

            response = llm_client.generate_text(prompt.lstrip(), temperature=0.5)
            if response and f"{ANSWER_TAG}:" in response:
                return response.strip()
            else:
                logger.warning("LLM returned empty response for CoT answer generation.")
                return None
        except Exception as e:
            logger.error(f"Error generating CoT answer: {e}", exc_info=True)
            return None

    @staticmethod
    def generate_raft_training_data(
        files: List[Tuple[BeautifulSoup, Path, Path]],
        output_dir: Path,
        config: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Generates RAFT training examples from default QA pairs, using extracted_faq or extracted_chunks as answer/distractor sources.
        Args:
            files: List of tuples containing (soup, file_path, rel_path)
            output_dir: Directory to save intermediate generated files
            config: Configuration dictionary
        Returns:
            List of dictionaries, each formatted as a RAFT training example
            (including the 'messages' key for fine-tuning).
        """
        # Create directories for output
        default_qa_dir = output_dir / "default_qa"
        extracted_faq_dir = output_dir / "extracted_faq"
        extracted_chunks_dir = output_dir / "extracted_chunks"

        default_qa_dir.mkdir(parents=True, exist_ok=True)
        extracted_faq_dir.mkdir(parents=True, exist_ok=True)
        extracted_chunks_dir.mkdir(parents=True, exist_ok=True)

        debug_dir = output_dir / "debug" / "qa_pairs"
        debug_dir.mkdir(parents=True, exist_ok=True)
        raft_qa_dir = output_dir / "qa_pairs_raft"
        raft_qa_dir.mkdir(parents=True, exist_ok=True)

        final_default_qa = []
        final_extracted_faq = []
        final_extracted_chunks = []

        for soup, file_path, rel_path in files:
            file_title = file_path.stem
            if soup and soup.title:
                file_title = soup.title.get_text(strip=True)
            safe_title_slug = slugify(file_title)

            file_hash = create_hash(str(rel_path))
            default_qa_path = default_qa_dir / f"default_{safe_title_slug}_{file_hash}.json"
            extracted_faq_path = extracted_faq_dir / f"{safe_title_slug}_{file_hash}.json"
            extracted_chunks_path = extracted_chunks_dir / f"{safe_title_slug}_{file_hash}.json"

            # Load default QA (baseline)
            if not os.path.exists(default_qa_path):
                raise FileNotFoundError(f"Default QA file not found: {default_qa_path}")
            with open(default_qa_path, 'r', encoding='utf-8') as f:
                default_qa = json.load(f)

            try:
                file_url = os.getxattr(str(file_path), b'user.original_url').decode('utf-8')
            except Exception as e:
                file_url = FileProcessor.extract_domain_and_path(file_path)[2]

            for qa in default_qa:
                qa['file_title'] = file_title
                qa['file_name'] = file_path.name
                qa['file_url'] = file_url
            final_default_qa.extend(default_qa)

            # load extracted_faq (for FAQ files)
            extracted_faq = []
            if os.path.exists(extracted_faq_path):
                with open(extracted_faq_path, 'r', encoding='utf-8') as f:
                    extracted_faq = json.load(f)

                for faq in extracted_faq:
                    faq['file_title'] = file_title
                    faq['file_name'] = file_path.name
                    faq['file_url'] = file_url
                final_extracted_faq.extend(extracted_faq)
            # load extracted_chunks (for non-FAQ files)
            extracted_chunks = []
            if os.path.exists(extracted_chunks_path):
                with open(extracted_chunks_path, 'r', encoding='utf-8') as f:
                    extracted_chunks = json.load(f)
                for chunk in extracted_chunks:
                    chunk['file_title'] = file_title
                    chunk['file_name'] = file_path.name
                    chunk['file_url'] = file_url
                final_extracted_chunks.extend(extracted_chunks)

        # Remove 'default' style from writing_styles
        writing_styles = config.get("question_styles", {}).get("writing_styles", [])

        raft_config = config.get("processing", {}).get("raft", {})
        num_distract = raft_config.get("num_distractors", 3)
        p_golden = raft_config.get("p_golden_include", 0.8) # probability of including golden document

        llm_config_styled_q_provider = config.get("providers", {}).get("styled_question", {})
        llm_config_cot_a_provider = config.get("providers", {}).get("cot_answer", {})

        if not llm_config_cot_a_provider:
            llm_config_cot_a_provider = llm_config_styled_q_provider
        llm_client_styled_q = LLMClient(llm_config_styled_q_provider)
        llm_client_cot_a = LLMClient(llm_config_cot_a_provider)

        is_faq = len(final_extracted_faq) > 0
        # Prepare answer/distractor source
        if is_faq:
            all_original_qas = []
            for faq in final_extracted_faq:
                topics_str = f', Topics: "{faq.get("topics", "")}"' if faq.get("topics") else ''
                course_str = f', Course: "{faq.get("course", "")}"' if faq.get("course") else ''
                formatted_qa = (
                    f'Q: "{faq["question"]}"'
                    f', A: "{faq["answer"]}"'
                    f'{topics_str}'
                    f'{course_str}'
                )
                all_original_qas.append(formatted_qa)
        else:
            all_original_chunks = []
            for content in final_extracted_chunks:
                topics_str = f', Topic: "{content.get("topic")}"' if content.get("topic") else ''
                course_str = f', Course: "{content.get("course")}"' if content.get("course") else ''
                filename_str = f', File: "{content["file_name"]}"'
                url_str = (
                    f', URL: "[{content["file_title"]}]({content["file_url"]})"' if filename_str.endswith((".html\"", ".htm\""))
                    else f', URLs: "[{content["file_title"]}]({content["file_url"]})"'
                )
                formatted_chunk = f'Chunk: "{content["chunk"]}"{topics_str}{course_str}{filename_str}{url_str}'
                all_original_chunks.append(formatted_chunk)

        final_qa_pairs = final_extracted_faq + final_extracted_chunks # one or the other!

        previous_questions_cache = {qa["chunk_hash"] if "chunk_hash" in qa else qa["qa_pair_hash"]: [] for qa in final_qa_pairs}
        max_iterations_overall = max((style.get('iterations', 1) for style in writing_styles), default=1)
        all_training_examples = []
        file_generation_count = 0
        prev_filename = None
        # Main loop: iterate through each default QA pair
        for i, default_qa in enumerate(tqdm(final_default_qa, desc="Generating RAFT Examples")):
            file_title = default_qa["file_title"]
            file_name = default_qa["file_name"]
            file_url = default_qa["file_url"]

            if prev_filename and prev_filename != file_name:
                logger.info(f"Generated {file_generation_count} training examples for {prev_filename}")
                file_generation_count = 0
            prev_filename = file_name

            qa_hash = final_qa_pairs[i]["chunk_hash"] if "chunk_hash" in final_qa_pairs[i] else final_qa_pairs[i]["qa_pair_hash"]
            original_answer = final_qa_pairs[i]["answer"] if "answer" in final_qa_pairs[i] else None

            # FAQ: answer/distractor source is extracted_faq; non-FAQ: extracted_chunks
            if is_faq:
                golden_document = all_original_qas[i] if i < len(all_original_qas) else ''
                available_distractors = [qas for idx, qas in enumerate(all_original_qas) if idx != i]
                current_chunk = None
            else:
                golden_document = all_original_chunks[i] if i < len(all_original_chunks) else ''
                available_distractors = [chunk for idx, chunk in enumerate(all_original_chunks) if idx != i]
                current_chunk = final_extracted_chunks[i] if i < len(final_extracted_chunks) else None

            for iteration in range(max_iterations_overall):
                for style in writing_styles:
                    style_name = style.get("name")
                    safe_style_name = slugify(style_name.lower())
                    style_iterations = style.get("iterations", 1)
                    if iteration >= style_iterations:
                        continue
                    styled_question_path = raft_qa_dir / f"styled_q_{qa_hash}_{safe_style_name}_{iteration}.txt"
                    styled_q = None
                    while not styled_q:
                        if styled_question_path.exists():
                            with open(styled_question_path, 'r', encoding='utf-8') as f:
                                styled_q = f.read().strip()
                                logger.debug(f"Loaded styled {file_name} question from {styled_question_path}")
                        else:
                            logger.debug(f"Generating styled question for {file_name} ({qa_hash})")
                            previous_qs_for_pair = previous_questions_cache[qa_hash]
                            styled_q = QAProcessorRAFT.generate_styled_question_raft(
                                default_qa["question"], default_qa["answer"], style, previous_qs_for_pair, llm_client_styled_q
                            )
                            if styled_q:
                                previous_questions_cache[qa_hash].append(styled_q)
                                with open(styled_question_path, 'w', encoding='utf-8') as f:
                                    f.write(styled_q)
                                logger.info(f"Saved styled {file_name} question to {styled_question_path}")
                            else:
                                time.sleep(1)

                    cot_answer_path = raft_qa_dir / f"cot_a_{qa_hash}_{safe_style_name}_{iteration}.txt"
                    cot_answer_str = None
                    while not cot_answer_str:
                        if cot_answer_path.exists():
                            with open(cot_answer_path, 'r', encoding='utf-8') as f:
                                cot_answer_str = f.read()
                            logger.info(f"Loaded CoT {file_name} answer from {cot_answer_path}")
                        else:
                            logger.debug(f"Generating CoT answer for {file_name} ({qa_hash})")
                            cot_answer_str = QAProcessorRAFT.generate_cot_answer_raft(
                                styled_q, original_answer, current_chunk, llm_client_cot_a
                            )
                            if cot_answer_str:
                                cot_answer_str += f"\n> Fonte: {file_url}"
                                with open(cot_answer_path, 'w', encoding='utf-8') as f:
                                    f.write(cot_answer_str)
                                logger.info(f"Saved CoT {file_name} answer to {cot_answer_path}")
                            else:
                                time.sleep(1)
                    actual_num_distract = min(num_distract, len(available_distractors))
                    context_docs = []
                    golden_present_flag = False
                    golden_idx = -1
                    if random.uniform(0, 1) < p_golden and actual_num_distract >= 0:
                        golden_present_flag = True
                        context_docs.append(golden_document)
                        if actual_num_distract > 0:
                            distractors_dk = random.sample(available_distractors, actual_num_distract)
                            context_docs.extend(distractors_dk)
                        random.shuffle(context_docs)
                        try:
                            golden_idx = context_docs.index(golden_document)
                        except ValueError:
                            golden_idx = -1
                            golden_present_flag = False
                    else:
                        golden_present_flag = False
                        golden_idx = -1
                        num_needed = min(actual_num_distract + 1, len(available_distractors))
                        if num_needed > 0:
                            context_docs = random.sample(available_distractors, num_needed)
                            random.shuffle(context_docs)
                    assembled_context_str = ""
                    for doc_content in context_docs:
                        assembled_context_str += f"<DOCUMENT>{doc_content}</DOCUMENT>\n"
                    user_content = assembled_context_str + "\n" + styled_q
                    assistant_content = cot_answer_str
                    training_example = {
                        "question": user_content,
                        "answer": assistant_content,
                        "original_qa_pair_hash": qa_hash,
                        "style_name": style_name,
                        "styled_question": styled_q,
                        "raft_qa_pair_hash": f"raft_{qa_hash}_{safe_style_name}_{iteration}",
                        "golden_present": golden_present_flag,
                        "golden_index": golden_idx,
                        "num_distractors_in_context": len(context_docs) - (1 if golden_present_flag else 0),
                        "file_url": file_url,
                        "file_name": file_name,
                    }
                    all_training_examples.append(training_example)
                    file_generation_count += 1

        logger.info(f"Finished RAFT data generation. Generated {len(all_training_examples)} total training examples.")
        try:
            # Create a hash of the batch of files for uniqueness, using relative paths
            batch_rel_paths = sorted([str(rel_path) for _, _, rel_path in files])
            batch_hash = create_hash("::".join(batch_rel_paths))
    
            raft_batches_dir = output_dir / "raft_datasets"
            raft_batches_dir.mkdir(parents=True, exist_ok=True)
            raft_dataset_file = raft_batches_dir / f"raft_training_data_{batch_hash}.jsonl"
            with open(raft_dataset_file, 'w', encoding='utf-8') as f:
                for example in all_training_examples:
                    f.write(json.dumps(example, ensure_ascii=False) + '\n')
            logger.info(f"Saved complete RAFT training data to {raft_dataset_file}")
        except Exception as e:
            logger.error(f"Failed to save complete RAFT dataset: {e}")
        return all_training_examples

