# faq_processor_raft.py

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

from modules.utils import get_hash

# Assuming llm_client and file_processor are in the same directory structure
from .llm_client import LLMClient
from .file_processor import FileProcessor
from .faq_processor import FAQProcessor
import numpy as np

logger = logging.getLogger(__name__)

# --- Constants for RAFT ---
# Prompt for generating the CoT Answer (A*) based ONLY on the Golden Answer (D*)
# NOTE: This prompt specifically asks for a FORMAL style as requested.
ANSWER_TAG = "<ANSWER>"
COT_ANSWER_GENERATION_PROMPT = f"""
You are an expert assistant creating training data for a university chatbot that will chat with the students.
Given an original Question and its corresponding original Answer (which serves as the definitive context), generate a detailed chat response. This response has two parts: internal reasoning (which won't be shown to the user) and the final answer in portuguese.

**Instructions:**

1.  **Analyze:** Understand the user's implicit need based on the 'Original Question'.
2.  **Internal Reasoning `<REASON>:`:** Provide a short and clear, objective reasoning based *exclusively* on the information within the 'Original Answer (Context)'. Do NOT add outside information. This reasoning is for internal understanding and training purposes aimed at FACTUALITY.
3.  **Reasoning Citations:** **Within the Internal Reasoning section only**, when referencing specific parts of the 'Original Answer (Context)', enclose the verbatim text within `##begin_quote##` and `##end_quote##`. Text outside these tags should be your own reasoning/connecting words.
4.  **Style:** Maintain a **friendly, formal, modern, effective, polite, expert chatbot assistant persona** suitable for the University of Brasília (UnB).
5.  **Final User-Facing Answer Tag:** Conclude the *entire* response with the final answer intended **directly for the end-user**, prefixed EXACTLY with `{ANSWER_TAG}:`.
6.  **User-Facing Answer Content:**
    *   **Visibility:** Understand that **only the text following the `{ANSWER_TAG}:` tag will be shown to the end-user.** The user will *not* see the 'Original Answer (Context)' you were given, nor the 'Internal Reasoning' section.
    *   **Self-Contained:** The final answer must be **self-contained and directly usable**. It should not refer back to "the context" ambiguously.
    *   **Include Links/Sources:** If the answer relies on specific web pages or documents mentioned in the 'Original Answer (Context)', **include the necessary Markdown links (e.g., `[Link Text](URL)`) directly within this final answer section.** If specific documents are sources, reference them clearly (e.g., "according to the Course Regulation document available at [link]").
    *   **Completeness:** Ensure all relevant information from the 'Original Answer (Context)' needed to address the 'Original Question' is summarized or directly included in this final answer.
    * **Always answer the question directly first** (no greetings unless the user greets you).
7.  **If Unanswerable:** If the 'Original Answer (Context)' genuinely doesn't contain the information to answer the 'Original Question', state that clearly in the Internal Reasoning and use `{ANSWER_TAG}: Lamento, mas não possuo informações suficientes para responder à sua pergunta sobre este tópico específico.`.

**Input:**
Original Question: {{original_question}}
Original Answer (Context): {{original_answer}}
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
Original Question: {original_question}
Original Answer: {original_answer}

**Previous Styled Questions for this item (Avoid these exact phrasings):**
{previous_questions_str}

**Output Format:**
Return ONLY the single generated alternative question IN PORTUGUESE. Do not include ANY other text, numbering, or explanations.
"""

class FAQProcessorRAFT:
    """
    Generates RAFT-formatted training data from FAQ documents.
    Each original FAQ Answer serves as a Golden Document (D*).
    Other original FAQ Answers serve as Distractors (Dk).
    """

    # Keep detection logic if needed upstream
    @staticmethod
    def detect_faq_document(soup: BeautifulSoup, filename: str) -> bool:
        """(Copied from original) Determine if a document is an FAQ."""
        # (Implementation is the same as your provided FAQProcessor)
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
            previous_questions_prompt = "- The new question should be distinct from the previous styled questions (do different phrasings, coherent reorderings, etc)."
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
        original_answer: str, # This is D*
        llm_client: LLMClient
    ) -> str | None:
        """Generates the CoT Answer (A*) based only on the original Q/A pair."""
        try:
            prompt = COT_ANSWER_GENERATION_PROMPT.format(
                original_question=original_question,
                original_answer=original_answer
            )

            response = llm_client.generate_text(prompt.lstrip(), temperature=0.5)
            if response and f"{ANSWER_TAG}:" in response:
                return response.strip()
            elif response:
                logger.warning(f"Generated CoT answer lacked the required '{ANSWER_TAG}:' tag. Raw response: {response[:200]}...")
                 # Attempt to append tag if missing but content exists
                return response.strip() + f"\n{ANSWER_TAG}: (Tag ausente na geração original)"
            else:
                logger.warning("LLM returned empty response for CoT answer generation.")
                return None
        except Exception as e:
            logger.error(f"Error generating CoT answer: {e}", exc_info=True)
            return None

    @staticmethod
    def generate_raft_training_data(
        faq_files: List, # Original source file path for context/URL
        output_dir: Path,
        config: Dict[str, Any],
        is_faq: bool
    ) -> List[Dict[str, Any]]:
        """
        Generates RAFT training examples from extracted FAQ pairs.

        Args:
            soup: BeautifulSoup object of the document
            extracted_faqs: List of dictionaries, each with 'question', 'answer', 'qa_pair_hash'.
            file_path: Path object for the original source file.
            output_dir: Directory to save intermediate generated files.
            config: Configuration dictionary.

        Returns:
            List of dictionaries, each formatted as a RAFT training example
            (including the 'messages' key for fine-tuning).
        """

        # Create directories for output
        extracted_faq_dir = output_dir / "extracted_faq"
        extracted_faq_dir.mkdir(parents=True, exist_ok=True)
        extracted_chunks_dir = output_dir / "extracted_chunks"
        extracted_chunks_dir.mkdir(parents=True, exist_ok=True)

        debug_dir = output_dir / "debug" / "qa_pairs"
        debug_dir.mkdir(parents=True, exist_ok=True)
        raft_qa_dir = output_dir / "qa_pairs_raft"
        raft_qa_dir.mkdir(parents=True, exist_ok=True)

        final_extracted_faq = []
        final_extracted_chunks = []
        extracted_faq_paths = []
        extracted_chunks_paths = []
        for soup, file_path in faq_files:
            faq_title = file_path.stem
            if soup and soup.title:
                faq_title = soup.title.get_text(strip=True)
            safe_title_slug = slugify(faq_title)

            file_path_hash = get_hash(str(file_path))
            extracted_faq_path = extracted_faq_dir / f"{safe_title_slug}_{file_path_hash}.json"
            extracted_chunks_path = extracted_chunks_dir / f"{safe_title_slug}_{file_path_hash}.json"

            extracted_faq_paths.append(extracted_faq_path)
            extracted_chunks_paths.append(extracted_chunks_path)
            # Load or extract FAQ data
            extracted_faq = []
            if os.path.exists(extracted_faq_path):
                try:
                    with open(extracted_faq_path, 'r', encoding='utf-8') as f:
                        extracted_faq = json.load(f)
                    logger.info(f"Loaded {len(extracted_faq)} existing extracted QA pairs for {file_path}")
                except Exception as e:
                    logger.error(f"Error loading existing extracted FAQ from {extracted_faq_path}: {e}")
                    extracted_faq = []

            if not extracted_faq:
                logger.info(f"Extracting FAQ pairs for {file_path} using LLM.")
                extracted_faq = FAQProcessor.extract_faq(soup, file_path, output_dir, config)

                if extracted_faq:
                    try:
                        with open(extracted_faq_path, 'w', encoding='utf-8') as f:
                            json.dump(extracted_faq, f, ensure_ascii=False, indent=2)
                        logger.info(f"Saved {len(extracted_faq)} extracted FAQ pairs to {extracted_faq_path}")
                    except Exception as e:
                        logger.error(f"Error saving extracted FAQ to {extracted_faq_path}: {e}")
                else:
                    logger.warning(f"FAQ extraction yielded no results for {file_path}. Cannot proceed.")
                    return []
            
            # get chunks from "extracted_chunks"
            extracted_chunks = []
            if not is_faq:
                if os.path.exists(extracted_chunks_path):
                    try:
                        with open(extracted_chunks_path, 'r', encoding='utf-8') as f:
                            extracted_chunks = json.load(f)
                    except Exception as e:
                        logger.error(f"Error loading existing extracted chunks from {extracted_chunks_path}: {e}")
                        extracted_chunks = []
                else:
                    raise Exception(f"Extracted chunks file {extracted_chunks_path} does not exist.")

            final_extracted_faq.extend(extracted_faq)
            final_extracted_chunks.extend(extracted_chunks)
                

        # --- RAFT Configuration ---
        raft_config = config.get("processing", {}).get("raft", {})
        num_distract = raft_config.get("num_distractors", 3)
        p_golden = raft_config.get("p_golden_include", 0.8) # Probability to include D*
        writing_styles = config.get("question_styles").get("writing_styles", None)

        # Add original question as a "style"
        if is_faq:
            writing_styles.append({"name": "Verbatim", "description": "Original question", "goal": "Use original question verbatim", "iterations": 1})
        else: # no default style for non-FAQ files
            [style.update({"iterations": 0}) for style in writing_styles if "default" in style.get("name", "").lower()]

        llm_config_styled_q_provider = config.get("providers", {}).get("styled_question", {})
        llm_config_cot_a_provider = config.get("providers", {}).get("cot_answer", {}) # Config for CoT answer generation
        if not llm_config_cot_a_provider: # Fallback if not specified
            llm_config_cot_a_provider = llm_config_styled_q_provider
        llm_client_styled_q = LLMClient(llm_config_styled_q_provider)
        llm_client_cot_a = LLMClient(llm_config_cot_a_provider)

        # Extract URL for metadata
        domain, _, url = FileProcessor.extract_domain_and_path(file_path)

        logger.info(f"Starting RAFT data generation for {len(final_extracted_faq)} original FAQ pairs from {file_path}...")

        # Prepare data pools

        # Create formatted strings of all original Q&A pairs for use as potential documents
        all_original_qas = []
        all_original_chunks = []
        if is_faq:
            for faq in final_extracted_faq:
                # Format each QA pair with consistent structure
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
            for content in final_extracted_chunks:
                topics_str = f', Topic: "{content.get("topic")}"' if content.get("topic") else ''
                course_str = f', Course: "{content.get("course")}"' if content.get("course") else ''
                formatted_chunk = f'Chunk: "{content["chunk"]}"{topics_str}{course_str}'
                all_original_chunks.append(formatted_chunk)

        all_training_examples = []
        previous_questions_cache = {faq["qa_pair_hash"]: [] for faq in final_extracted_faq}

        max_iterations_overall = 0
        if writing_styles:
            max_iterations_overall = max(style.get('iterations', 1) for style in writing_styles)

        # --- Main Loop: Iterate through each original FAQ pair ---
        for i, original_faq in enumerate(tqdm(final_extracted_faq, desc="Generating RAFT Examples")):
            original_q = original_faq["question"]
            original_a = original_faq["answer"] # Golden document (D*)
            qa_hash = original_faq["qa_pair_hash"]
            if not is_faq and len(final_extracted_faq) == len(final_extracted_chunks):
                content = f'Chunk: "{final_extracted_chunks[i]["chunk"]}"'
                topics_str = f', Topic: "{final_extracted_chunks[i].get("topic")}"' if final_extracted_chunks[i].get("topic") else ''
                course_str = f', Course: "{final_extracted_chunks[i].get("course")}"' if final_extracted_chunks[i].get("course") else ''
            else:
                content = f'Q: "{original_q}", A: "{original_a}"'
                topics_str = f', Topics: "{original_faq.get("topics")}"' if original_faq.get("topics") else ''
                course_str = f', Course: "{original_faq.get("course")}"' if original_faq.get("course") else ''

            golden_document = f'{content}{topics_str}{course_str}'
            
            for iteration in range(max_iterations_overall):
                # --- Loop through Writing Styles to generate Questions (Q) and Answers (A*) ---
                for style in writing_styles:
                    style_name = style.get("name")
                    safe_style_name = slugify(style_name.lower())
                    style_iterations = style.get("iterations", 1)

                    if iteration >= style_iterations:
                        logger.info(f"Skipping style '{style_name}' for iteration {iteration + 1} (max: {style_iterations})")
                        continue

                    # --- Generate/Load Styled Question (Q) ---
                    styled_question_path = raft_qa_dir / f"styled_q_{qa_hash}_{safe_style_name}_{iteration}.txt"
                    styled_q = None

                    if "verbatim" in style_name.lower():
                        styled_q = original_q # Use original question directly
                    else:
                        while not styled_q:
                            if styled_question_path.exists():
                                try:
                                    with open(styled_question_path, 'r', encoding='utf-8') as f:
                                        styled_q = f.read().strip()
                                        logger.info(f"Loaded existing styled Q for {qa_hash}_{safe_style_name}_{iteration}")
                                except Exception as e:
                                    logger.error(f"Error loading styled question {styled_question_path}: {e}. Regenerating.")
                                    styled_q = None


                            if not styled_q:
                                logger.info(f"Generating styled Q for {qa_hash}_{safe_style_name}_{iteration}...")
                                # Pass currently known styled questions for this original pair to avoid repeats
                                previous_qs_for_this_pair = previous_questions_cache[qa_hash]
                                styled_q = FAQProcessorRAFT.generate_styled_question_raft(
                                    original_q, original_a, style, previous_qs_for_this_pair, llm_client_styled_q
                                )
                                if styled_q:
                                    previous_questions_cache[qa_hash].append(styled_q) # Update cache
                                    try:
                                        with open(styled_question_path, 'w', encoding='utf-8') as f:
                                            f.write(styled_q)
                                        logger.info(f"Saved generated styled Q to {styled_question_path}")
                                    except Exception as e:
                                        logger.error(f"Error saving styled question file {styled_question_path}: {e}")
                                else:
                                    logger.warning(f"Failed to generate styled question for {qa_hash}_{safe_style_name}_{iteration}. Retrying...")
                                    time.sleep(1)

                    # --- Generate/Load the CoT Answer (A*) for each styled question ---
                    cot_answer_path = raft_qa_dir / f"cot_a_{qa_hash}_{safe_style_name}_{iteration}.txt"
                    cot_answer_str = None
                    while not cot_answer_str:
                        if cot_answer_path.exists():
                            try:
                                with open(cot_answer_path, 'r', encoding='utf-8') as f:
                                    cot_answer_str = f.read()
                                    logger.info(f"Loaded existing CoT Answer for {qa_hash}_{safe_style_name}_{iteration}")
                            except Exception as e:
                                logger.error(f"Error loading CoT answer file {cot_answer_path}: {e}. Regenerating.")
                                cot_answer_str = None

                        if not cot_answer_str:
                            logger.info(f"Generating CoT Answer for {qa_hash}_{safe_style_name}_{iteration}...")

                            cot_answer_str = FAQProcessorRAFT.generate_cot_answer_raft(
                                styled_q, original_a, llm_client_cot_a
                            )

                            if cot_answer_str:
                                # get <ANSWER>: part
                                try:
                                    with open(cot_answer_path, 'w', encoding='utf-8') as f:
                                        f.write(cot_answer_str)
                                    logger.info(f"Saved generated CoT Answer to {cot_answer_path}")
                                except Exception as e:
                                    logger.error(f"Error saving CoT answer file {cot_answer_path}: {e}")
                            else:
                                logger.error(f"Failed to generate CoT Answer for {qa_hash}_{safe_style_name}_{iteration}. Retrying...")
                                time.sleep(1)

                    # --- Assemble Context (D* + Dk or just Dk) ---
                    # Pool of potential distractors: all original answers EXCEPT the current one
                    # TODO: ONLY CHANGE HERE!! perform actual retrieval as a RAG system would do in inference time
                    if is_faq:
                        available_distractors = [qas for idx, qas in enumerate(all_original_qas) if idx != i]
                    else:
                        available_distractors = [chunk for idx, chunk in enumerate(all_original_chunks) if idx != i]
                    
                    # Ensure we don't request more distractors than available
                    actual_num_distract = min(num_distract, len(available_distractors))
                    context_docs = []
                    golden_present_flag = False
                    golden_idx = -1

                    if random.uniform(0, 1) < p_golden and actual_num_distract >= 0: # Include D*
                        golden_present_flag = True
                        context_docs.append(golden_document) # Add D*
                        if actual_num_distract > 0:
                            distractors_dk = random.sample(available_distractors, actual_num_distract)
                            context_docs.extend(distractors_dk)
                        random.shuffle(context_docs)
                        try:
                            golden_idx = context_docs.index(golden_document)
                        except ValueError:
                            logger.error(f"CRITICAL: Oracle document lost after shuffle for {qa_hash}. This shouldn't happen.")
                            golden_idx = -1 # Fallback, though indicates error
                            golden_present_flag = False # Mark as potentially failed

                    else: # Exclude D*
                        golden_present_flag = False
                        golden_idx = -1
                        # Need num_distract + 1 distractors if possible
                        num_needed = min(actual_num_distract + 1, len(available_distractors))
                        if num_needed > 0:
                            context_docs = random.sample(available_distractors, num_needed)
                            random.shuffle(context_docs) # Shuffle just distractors
                        # else: context_docs remains empty if no distractors available

                    # Format the assembled context
                    assembled_context_str = ""
                    for doc_content in context_docs:
                        assembled_context_str += f"<DOCUMENT>{doc_content}</DOCUMENT>\n"

                    logger.info(f"Context assembled for {qa_hash}: {len(context_docs)} documents (Golden present: {golden_present_flag})")

                    # --- Create the final training example ---
                    user_content = assembled_context_str + "\n" + styled_q
                    assistant_content = cot_answer_str # Use the style-specific CoT answer

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
                        "url": url, # Add original URL
                        "file_name": file_path.name,
                        # Optional: Add original Q/A for reference/debugging if needed
                        # "original_question": original_q,
                        # "original_answer": original_a,
                    }
                    all_training_examples.append(training_example)

                # End style loop
        # End main FAQ loop

        logger.info(f"Finished RAFT data generation for {file_path}. Generated {len(all_training_examples)} total training examples.")

        # --- Optional: Save the full RAFT dataset ---
        try:
            raft_dataset_path = output_dir / f"raft_training_data_{file_path.stem}.jsonl"
            with open(raft_dataset_path, 'w', encoding='utf-8') as f:
                for example in all_training_examples:
                    f.write(json.dumps(example, ensure_ascii=False) + '\n')
            logger.info(f"Saved complete RAFT training data to {raft_dataset_path}")
        except Exception as e:
            logger.error(f"Failed to save complete RAFT dataset: {e}")

        return all_training_examples

