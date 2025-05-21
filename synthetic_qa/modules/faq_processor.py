"""
FAQ processing module for the Synthetic QA Generator.

This module handles detection and processing of FAQ documents.
"""

import hashlib
import os
import logging
import json
from pathlib import Path
from typing import List, Tuple, Dict, Any
from bs4 import BeautifulSoup
import textwrap

from slugify import slugify

from .llm_client import LLMClient
from .file_processor import FileProcessor

logger = logging.getLogger(__name__)


class FAQProcessor:
    STYLE_DEFAULT_FAQS = True

    """Handles specialized processing for FAQ documents."""
    
    @staticmethod
    def detect_faq_document(soup: BeautifulSoup, filename: str) -> bool:
        """
        Determine if a document is an FAQ using multiple signals.
        
        Args:
            soup: BeautifulSoup object of the document
            filename: Name of the file
            
        Returns:
            Boolean indicating if the document is an FAQ
        """
        # Check filename for indicators
        faq_indicators = ['faq', 'perguntas', 'frequentes', 'duvidas', 'q&a']
        if any(indicator in filename.lower() for indicator in faq_indicators):
            return True
        
        # Check title or headings
        title = soup.find('title')
        if title and any(indicator in title.text.lower() for indicator in faq_indicators):
            return True
        
        # Check for structured Q&A patterns
        if len(soup.find_all('details')) > 2 and len(soup.find_all('summary')) > 2:
            return True
            
        # Check for other Q&A patterns (bold text followed by paragraphs)
        questions_count = 0
        for tag in soup.find_all(['b', 'strong']):
            text = tag.get_text().strip()
            if text.endswith('?') or any(text.lower().startswith(word) for word in 
                                   ['como', 'existe', 'existem', 'qual', 'quais', 'o que', 'onde', 'quando', 'por que', 'posso']):
                questions_count += 1
        
        if questions_count > 3:
            return True
            
        return False


    @staticmethod
    def create_qa_hash(file_path: Path, question: str):
        return f"faq_{hashlib.sha256((str(file_path) + question).encode()).hexdigest()[:12]}"


    @staticmethod
    def extract_faq(soup: BeautifulSoup, file_path: Path, output_dir: Path, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract question-answer pairs from an FAQ document using LLM processing.
        
        Args:
            soup: BeautifulSoup object of the document
            file_path: Path to the HTML file
            output_dir: Path to the output directory
            config: Configuration dictionary
            
        Returns:
            List of dictionaries containing extracted FAQ data
        """
        from .file_processor import FileProcessor
        
        try:
            structured_text_dir = output_dir / "extracted_text"

            llm_client = LLMClient(config.get("providers", {}).get("faq_extraction", {}))

            # check if already present in file
            if structured_text_dir.exists():
                extracted_faq_hash = hashlib.sha256(f"{file_path}".encode()).hexdigest()[:12]
                structured_text_path = structured_text_dir / f"{file_path.stem}_{extracted_faq_hash}.txt"

                if structured_text_path.exists():
                    logger.info(f"Structured text already exists for {file_path}")
                    with open(structured_text_path, 'r', encoding='utf-8') as f:
                        structured_text = f.read()
            else:
                # Extract text with preserved structure
                structured_text = FileProcessor.extract_text_from_html(soup, file_path, llm_client)
                # Save it (ensure directory exists first)
                structured_text_dir.mkdir(parents=True, exist_ok=True)
                extracted_faq_hash = hashlib.sha256(f"{file_path}".encode()).hexdigest()[:12]
                structured_text_path = structured_text_dir / f"{slugify(file_path.stem)}_{extracted_faq_hash}.txt"
                with open(structured_text_path, 'w', encoding='utf-8') as f:
                    f.write(structured_text)
            

            domain, _, _ = FileProcessor.extract_domain_and_path(file_path)

            if not structured_text:
                logger.warning("No text content extracted from HTML")
                return []
            
            # Construct the prompt for LLM extraction of Q&A pairs
            prompt = f"""
            Extract EVERY question and answer from this university markdown FAQ file and convert them to a structured JSON format. Follow these requirements exactly:

            1. Output a JSON string with this structure:
            {{
            "qa_pairs": [
                {{
                "question": FAQ question,
                "answer": FAQ answer,
                "topics": ["Topic1, Topic2, ..."],
                "course": "specific course"
                }},
                // More QA pairs...
            ]
            }}

            2. **Critical requirements**:
            - Both "question" and "answer" fields **can never be null**
            - **"topics" are EXPLICITLY PRESENT in the FAQ AS titles/headers above a QA pair or a number of QA pairs** (can be null if no topics).
            - **include all nested topics as just a flat array**
            - "course" can be null or be a specific course

            3. Content processing:
            - Preserve all markdown formatting and links.
            - Detect question-answer pairs intelligently (tip: QA are in different hierarchical markdown levels)
            - **DO NOT ALTER OR ADD CONTENT** except to fix clear formatting issues
            - Every text should be verbatim to the original FAQ
            
            Available courses: {FileProcessor.INSTITUTION_COURSES.get(domain, "All Courses")}

            **IMPORTANT**: You MUST specify a particular course (from the available courses list) if ANY of these conditions is met:
            1. The original question explicitly mentions that specific course
            2. The **question topic explicitly mentions a specific course** (pay careful attention to the topics field)
            3. The document structure as a whole is about a specific course (in which case all QA pairs will have this course)

            After the approach explanation, return the **complete**, well-formed JSON with all extracted QA pairs from the university FAQ content below:
            {structured_text}
            """

            # Call the LLM to extract QA pairs
            response = llm_client.generate_text(
                textwrap.dedent(prompt).lstrip(),
                json_output=True,
                temperature=0.4
            )

            if not response:
                logger.warning("No response from LLM")
                return []

            correcting_prompt = (
                prompt
                + "\nYour output:\n"
                + json.dumps(response, ensure_ascii=False, indent=2)
                + "\nRewrite your output (json only). Do any of these corrections for every pair where mistakes were made:\n"
                "- Remove any questions in the 'topics' field.\n"
                "- Remove any answer in the 'question' field.\n"
                "- Remove any question in the 'answer' field.\n"
                "- Add/remove a course if it was forgotten/misplaced.\n"
                "- Remove any text that is not verbatim to the original FAQ.\n"
                "- Delete any QA pair that doesn't make sense.\n"
                "\n**CHANGE NOTHING IF NO MISTAKE WAS MADE**"
            )

            new_response = llm_client.generate_text(
                textwrap.dedent(correcting_prompt).lstrip(),
                json_output=True,
                temperature=0.4
            )

            if new_response:
                response = new_response

            # Extract JSON from the response
            try:
                qa_pairs = []

                for item in response.get("qa_pairs", []):
                    question = item.get("question", "")
                    answer = item.get("answer", "")
                    topics = item.get("topics", [])
                    course = item.get("course", "")

                    if question and answer:
                        qa_pairs.append({
                            "question": question.strip(),
                            "answer": answer.strip(),
                            "topics": topics if topics else None,
                            "course": course.strip() if course else None,
                            "qa_pair_hash": FAQProcessor.create_qa_hash(file_path, question)
                        })

                logger.info(f"Successfully extracted {len(qa_pairs)} QA pairs using LLM")
                return qa_pairs

            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse LLM response as JSON: {e}")
                logger.debug(f"Raw response: {response}")
                return []
                
        except Exception as e:
            logger.error(f"Error extracting QA pairs from HTML: {e}")
            return []


    @staticmethod
    def generate_styled_qa(
        questions: List[str],
        answers: List[str],
        previous_answers_batch: List[List[str]],
        previous_questions_batch: List[List[str]],
        writing_style: Dict[str, str],
        llm_client: LLMClient
    ) -> List[Dict[str, str]]:

        """
        Generate styled versions of question and answer pairs in batch based on a specified writing style.

        Args:
            questions: List of original questions.
            answers: List of original answers, corresponding to the questions.
            previous_answers_batch: List of lists of previous answers to avoid repeating answers for that specific question.
            previous_questions_batch: List of lists of previous questions.
            writing_style: Dictionary with writing style name and description.
            llm_client: LLM client for generation.

        Returns:
            List of dictionaries, each containing a styled question and answer pair.
            Items where generation failed for either question or answer are omitted.
        """

        batch_size = len(questions)
        if batch_size == 0:
            return []
      
        # Get writing style information
        writing_style_name = writing_style.get("name", "")
        writing_style_goal = writing_style.get("goal", "")
        writing_style_desc = writing_style.get("description", "")

        # --- 1. Generate ALL Styled Questions in One Call ---
        logger.info(f"Generating {batch_size} styled questions in a single batch prompt...")

        previous_questions_prompt_template = ""
        # check if any pair has a previous question
        if any(previous_questions_batch):
            previous_questions_prompt_template = "The new question should be distinct from the previous styled questions (do different phrasings, coherent reorderings, etc)."

        question_prompt_parts = [
            f"""You are an LLM Generator that will create synthetic data from original FAQ pairs to fine tune a specialist chatbot model.

Create an alternative FAQ question for EACH of the original pairs listed below.
**WRITING STYLE**: {writing_style_name}
- Description: {writing_style_desc}
- Goal: {writing_style_goal}

Instructions for EACH question:
- Rewrite *only the question*, while **preserving all the original meaning and intent**.
- **DO NOT ADD ANY NEW INFORMATION**.
- **Follow the specified writing style closely.**
- The user knows they are talking to an assistant chatbot.
- Output must be IN PORTUGUESE.
- Avoid repeated patterns between generated questions.

{previous_questions_prompt_template}

Original Pairs:"""
        ]

        for i, (q, a) in enumerate(zip(questions, answers)):
            # previous questions list for this item
            previous_questions_str = ""
            if previous_questions_batch[i]:
                formatted_prev_questions = "\n".join([f"- {pq}" for pq in previous_questions_batch[i]])
                previous_questions_str = f"Previous Styled Questions for this item:\n{formatted_prev_questions}"
            question_prompt_parts.append(
                f"Item #{i+1}:\n"
                f"Original Question: {q}\n"
                f"Original Answer: {a}\n"
                f"{previous_questions_prompt_template}\n"
                f"{previous_questions_str}"
            )

        question_prompt_parts.append(f"""
**Output Format:**
Return ONLY the {batch_size} generated alternative questions **in order**, IN PORTUGUESE.
Each generated question should be on a new line.
[Generated Question for Pair #1]
[Generated Question for Pair #2]
...
Do not include ANY other text, numbering, or explanations before or after the generated questions.""")

        styled_question_prompt = "\n".join(question_prompt_parts)
        llm_question_response = llm_client.generate_text(styled_question_prompt.lstrip(), temperature=0.7)

        # ** MODIFIED PARSING LOGIC **
        styled_questions = []
        if llm_question_response:
            lines = [line.strip() for line in llm_question_response.splitlines() if line.strip()]
            if len(lines) == batch_size:
                styled_questions = lines
            else:
                logger.error(f"Question generation parsing error: Expected {batch_size} lines, but received {len(lines)} non-empty lines.")
                logger.debug(f"LLM Response (Questions):\n{llm_question_response}")
                return []
        else:
            if not styled_questions:
                # This condition is technically covered above, but kept for clarity
                logger.error("Failed to generate or parse styled questions for the batch. Aborting.")
                return []

        # --- 2. Generate ALL Styled Answers in One Call ---
        logger.info(f"Generating {batch_size} styled answers...")

        previous_answers_prompt_template = ""
        # check if any pair has a previous answer
        if any(previous_answers_batch):
            previous_answers_prompt_template = "The new answer should be distinct from the previous answers for a specific answer (different phrasings, structures and coherent reorderings)."


        answer_prompt_parts = [
             f"""You are an LLM Generator that will create synthetic data from original FAQ pairs to fine tune a specialist chatbot model.

You are creating alternative FAQ answers IN PORTUGUESE for the items listed below.
General Instructions for EACH answer:
- Rewrite the 'Original Answer' provided for the item.
- Use the corresponding 'Styled Question' as context for the rewrite.
- **Preserve ALL the exact original information** from the 'Original Answer'.
- Format the answer in a clear, helpful way. Preserve markdown/links but restructure if helpful.
- **CRITICAL: NEVER ADD INFORMATION THAT IS NOT IN THE ORIGINAL ANSWER.** Do not infer, assume, or add confirmations unless explicitly present. Exact same meaning is required.
- Try to understand and somewhat match the question's writing style ("{writing_style_name}"), BUT always maintain the persona of a formal, modern, serious, polite, expert assistant for a brazilian university (UnB - Universidade de Brasília).
- Always answer the question directly first.

{previous_answers_prompt_template}.

- Every factual information should be preserved, none added.
Original pairs:"""
        ]

        for i, (styled_q, orig_a, prev_ans_list) in enumerate(zip(styled_questions, answers, previous_answers_batch)):
            previous_answers_str = ""
            if prev_ans_list:
                formatted_prev_answers = "\n".join([f"- {pa}" for pa in prev_ans_list])
                previous_answers_str = f"Previous Answers for this item:\n{formatted_prev_answers}"

            answer_prompt_parts.append(
                f"Item #{i+1}:\n"
                f"Styled Question: {styled_q}\n"
                f"Original Answer: {orig_a}\n"
                f"{previous_answers_str}"
            )

        answer_prompt_parts.append(f"""
**Output Format:**
Return ONLY the {batch_size} generated alternative answers, IN PORTUGUESE.
The strings can contain newline characters (\\n) for multi-line answers. Ensure correct JSON string escaping.
{{
  "styled_answers": [
    {{"a": "Generated Answer for Item #1."}},
    {{"a": "Generated Answer for Item #2."}},
    ...
  ]
}}

Ensure the order of your generated answers matches the order of the Input Items above.
The entire response should be just the JSON array.
**DON'T MAKE ASSUMPTIONS** if original answer isn't clear (always preserve meaning).""")


        styled_answer_prompt = "\n".join(answer_prompt_parts)
        llm_answer_response = llm_client.generate_text(styled_answer_prompt.lstrip(), temperature=1.0, json_output=True)
        if not llm_answer_response:
            return []
        
        try:
            # Extract the QA pairs
            styled_answers = []
            for pair in llm_answer_response.get("styled_answers", []):
                a = pair["a"].strip()
                styled_answers.append(a)
            logger.info(f"Generated {len(styled_answers)} styled answers")
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse rephrased QA response as JSON: {e}")
            return []

        # styled_answers = None
        # if isinstance(llm_answer_response, list) and len(llm_answer_response) == batch_size:
        #      # Optional: Add check if all items are strings if needed
        #      styled_answers = llm_answer_response

        if not styled_answers:
            logger.error("Aborting batch due to failure in generating/receiving styled answers.")
            return []
        logger.info(f"Successfully received {len(styled_answers)} styled answers.")


        # --- 3. Combine Results ---
        generated_pairs = []
        # This check should always pass if we reached here due to prior checks, but good practice
        if len(styled_questions) == batch_size and len(styled_answers) == batch_size:
             for q, a in zip(styled_questions, styled_answers):
                 generated_pairs.append({"question": q, "answer": a})
        else:
            # Should not happen if logic above is correct
            return []

        return generated_pairs


    @staticmethod
    def process_faq_document(soup: BeautifulSoup, file_path: Path, output_dir: Path, config: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        Process an HTML document as an FAQ to generate comprehensive training data.
        (Includes indefinite retries for generation)

        Args:
            soup: BeautifulSoup object of the document
            file_path: Path to the HTML file
            output_dir: Directory to save output files
            config: Configuration dictionary

        Returns:
            List of QA pairs with variations
        """
        # --- Imports ---
        import json
        import os
        import hashlib
        import time # <--- Import time module for sleep
        from pathlib import Path
        from typing import List, Dict, Any
        from bs4 import BeautifulSoup
        from slugify import slugify
        # Assuming these imports are relative to the file containing this method
        from .qa_generator import QAGenerator
        from .file_processor import FileProcessor # Assuming FileProcessor is available
        from .faq_processor import FAQProcessor # Assuming FAQProcessor is available
        from .llm_client import LLMClient # Assuming LLMClient is available

        # Setup logger if not already configured at module level
        import logging
        logger = logging.getLogger(__name__)
        # --- End Imports ---

        try:
            # Detect if this is an FAQ document
            is_faq = FAQProcessor.detect_faq_document(soup, file_path.name)
            if not is_faq:
                logger.info(f"{file_path} does not appear to be a FAQ document")
                return []

            logger.info(f"Processing {file_path} as a FAQ document")

            # Extract domain and path information
            domain, path, url = FileProcessor.extract_domain_and_path(file_path)
            institution = FileProcessor.get_institution_name(domain)
            courses = FileProcessor.get_institution_courses(domain) # Assuming this method exists

            faq_config_provider = config.get("providers", {}).get("faq_extraction", {})
            faq_title = soup.title.get_text(strip=True) if soup.title else institution
            safe_title_slug = slugify(faq_title) if faq_title else "untitled-faq"

            # Create directories for output
            extracted_faq_dir = output_dir / "extracted_faq"
            extracted_faq_dir.mkdir(parents=True, exist_ok=True)
            debug_dir = output_dir / "debug" / "qa_pairs"
            debug_dir.mkdir(parents=True, exist_ok=True)
            qa_dir = output_dir / "qa_pairs"
            qa_dir.mkdir(parents=True, exist_ok=True)


            extracted_faq_hash = hashlib.sha256(f"{file_path}_{faq_config_provider.get('model')}".encode()).hexdigest()[:12]
            extracted_faq_path = extracted_faq_dir / f"{safe_title_slug}_{extracted_faq_hash}.json"

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

            # --- Prepare for processing ---
            qa_generator = QAGenerator(config)
            writing_styles = qa_generator.writing_styles
            if not writing_styles:
                 writing_styles = [{"name": "Default", "iterations": 1, "description": ""}]

            # Calculate the maximum number of iterations needed based on style definitions
            max_iterations_overall = 0
            if writing_styles:
                 max_iterations_overall = max(style.get('iterations', 1) for style in writing_styles)
            logger.info(f"Determined max iterations needed across all styles: {max_iterations_overall}")

            batch_size = config.get("processing").get("faq").get("batch_size", 1)
            styled_q_provider_config = config.get("providers", {}).get("styled_question", {})
            rate_limit_rpm = styled_q_provider_config.get("rate_limit_rpm", 4)
            retry_delay_seconds = 60 / rate_limit_rpm if rate_limit_rpm > 0 else 15

            all_training_examples = []

            # Add original FAQ to list first
            for i, faq_item in enumerate(extracted_faq):
                if not isinstance(faq_item, dict) or "question" not in faq_item or "answer" not in faq_item or "qa_pair_hash" not in faq_item:
                    logger.warning(f"Skipping invalid extracted FAQ item at index {i} in {file_path}: {faq_item}")
                    continue
                original_example = {
                    "question": faq_item["question"],
                    "answer": faq_item["answer"],
                    "url": url,
                    "qa_pair_hash": faq_item["qa_pair_hash"],
                    "type": "original_verbatim"
                }
                all_training_examples.append(original_example)

            # --- Check for existing style files and identify styles needing processing ---
            # Store styles to process along with their safe names
            styles_to_process_map = {} # Maps safe_name -> style_dict
            style_to_examples = {} # Maps safe_name -> list of examples

            default_faq = []

            for writing_style in writing_styles:
                writing_style_name = writing_style.get("name")
                if not writing_style_name: continue # Skip styles without names

                # Generate safe name once
                safe_style_name = slugify(writing_style_name.lower())
                if not safe_style_name:
                    safe_style_name = f"style_{hashlib.md5(writing_style_name.encode()).hexdigest()[:6]}"

                style_file_path = qa_dir / f"{safe_title_slug}_{safe_style_name}_examples.json"

                if style_file_path.exists():
                    logger.info(f"File for style '{writing_style_name}' exists: {style_file_path}. Loading existing examples.")
                    try:
                        with open(style_file_path, 'r', encoding='utf-8') as f:
                            existing_examples_for_style = json.load(f)
                            all_training_examples.extend(existing_examples_for_style)
                            style_to_examples[safe_style_name] = existing_examples_for_style
                            if 'default' in writing_style_name.lower():
                                default_faq = existing_examples_for_style

                        logger.info(f"Loaded {len(existing_examples_for_style)} examples for style '{writing_style_name}'")
                    except Exception as e:
                        logger.error(f"Error loading style file {style_file_path}: {e}")
                        # If loading fails, we might still want to process it
                        styles_to_process_map[safe_style_name] = writing_style
                        style_to_examples[safe_style_name] = [] # Initialize empty if loading failed
                else:
                    # Needs processing
                    styles_to_process_map[safe_style_name] = writing_style
                    style_to_examples[safe_style_name] = []

            if not styles_to_process_map:
                logger.info(f"All styles already processed or no styles defined/left for {file_path}.")
                return all_training_examples

            faq_to_process = extracted_faq

            if FAQProcessor.STYLE_DEFAULT_FAQS and default_faq:
                faq_to_process = default_faq

            # --- Batch Processing Loop ---
            for i in range(0, len(faq_to_process), batch_size):
                batch_items = faq_to_process[i:i+batch_size]
                batch_items = [item for item in batch_items if isinstance(item, dict) and "question" in item and "answer" in item and "qa_pair_hash" in item]
                if not batch_items: continue

                batch_questions = [item["question"] for item in batch_items]
                batch_answers = [item["answer"] for item in batch_items]
                batch_hashes = [item["qa_pair_hash"] for item in batch_items]
                previous_answers_batches = [[] for _ in range(len(batch_items))]
                previous_questions_batches = [{} for _ in range(len(batch_items))]

                # add default answers to previous answers
                for idx, styled_pair in enumerate(batch_items):
                    if 'answer' in styled_pair and styled_pair['answer'] not in previous_answers_batches[idx]:
                        previous_answers_batches[idx].append(styled_pair['answer'])
                
                # --- Iteration Loop (up to max needed) ---
                for iteration in range(max_iterations_overall):
                    logger.info(f"Current total examples collected: {len(all_training_examples)}")
                    logger.info(f"Starting Iteration {iteration + 1}/{max_iterations_overall} for batch {i//batch_size + 1}, processing pairs {i} to {min(i + batch_size, len(faq_to_process))} of {len(faq_to_process)} total.")

                    # --- Style Loop (check styles needing processing) ---
                    for safe_style_name, writing_style in styles_to_process_map.items():
                        writing_style_name = writing_style.get("name") # Get original name for logging etc.

                        # *** Check if this style needs this iteration ***
                        style_iterations_defined = writing_style.get('iterations', 1)
                        if iteration >= style_iterations_defined: # iteration is 0-indexed
                            # logger.debug(f"Skipping style '{writing_style_name}' for iteration {iteration + 1} (max: {style_iterations_defined})")
                            continue

                        # Style needs processing for this iteration
                        logger.info(f"Processing style '{writing_style_name}' for iteration {iteration + 1}")
                        style_iteration_hash_suffix = f"{safe_style_name}_{iteration}" # Use 0-based index

                        styled_qa_hashes = [f"{qa_hash}_{style_iteration_hash_suffix}" for qa_hash in batch_hashes]
                        styled_paths = [qa_dir / f"styled_{styled_qa_hash}.txt" for styled_qa_hash in styled_qa_hashes]
                        styled_debug_paths = [debug_dir / f"styled_debug_{styled_qa_hash}.txt" for styled_qa_hash in styled_qa_hashes]

                        existing_pairs_for_style_iteration = [None] * len(batch_items)
                        indices_to_generate = []
                        for idx, styled_path in enumerate(styled_paths):
                            if styled_path.exists():
                                previous_questions_batches[idx][writing_style_name] = []
                                try:
                                    with open(styled_path, 'r', encoding='utf-8') as f: styled_pair = json.load(f)
                                    if isinstance(styled_pair, dict) and 'question' in styled_pair and 'answer' in styled_pair:
                                        existing_pairs_for_style_iteration[idx] = styled_pair
                                        logger.info(f"Loaded existing styled pair for {styled_path}")
                                        if styled_pair.get('answer') not in previous_answers_batches[idx]:
                                            previous_answers_batches[idx].append(styled_pair['answer'])
                                        if styled_pair.get('question') not in previous_questions_batches[idx].get(writing_style_name):
                                            previous_questions_batches[idx][writing_style_name].append(styled_pair['question'])
                                    else: indices_to_generate.append(idx)
                                except Exception: indices_to_generate.append(idx)
                            else: indices_to_generate.append(idx)

                        if indices_to_generate:
                            logger.info(f"Attempting generation for {len(indices_to_generate)} pairs (Style: '{writing_style_name}', Iteration: {iteration+1}).")
                            batch_questions_gen = [batch_questions[idx] for idx in indices_to_generate]
                            batch_answers_gen = [batch_answers[idx] for idx in indices_to_generate]
                            batch_previous_gen = [previous_answers_batches[idx] for idx in indices_to_generate]
                            batch_previous_q_gen = [previous_questions_batches[idx].get(writing_style_name) for idx in indices_to_generate]
                            llm_client_styled = LLMClient(styled_q_provider_config)
                            generated_pairs = None
                            attempt_count = 0

                            # --- Indefinite Retry Loop ---
                            while True:
                                attempt_count += 1
                                try:
                                    current_attempt_pairs = FAQProcessor.generate_styled_qa(
                                        batch_questions_gen, batch_answers_gen, batch_previous_gen,
                                        batch_previous_q_gen, writing_style, llm_client_styled
                                    )
                                    if current_attempt_pairs and isinstance(current_attempt_pairs, list) and len(current_attempt_pairs) == len(indices_to_generate):
                                         all_valid = all(isinstance(p, dict) and 'question' in p and 'answer' in p for p in current_attempt_pairs)
                                         if all_valid:
                                            generated_pairs = current_attempt_pairs
                                            logger.info(f"Generation successful on attempt {attempt_count}.")
                                            break # Exit retry loop
                                         else: logger.warning(f"Attempt {attempt_count} returned pairs with invalid format.")
                                    else: logger.warning(f"Attempt {attempt_count} failed or returned unexpected result (Expected {len(indices_to_generate)}, Got: {len(current_attempt_pairs) if isinstance(current_attempt_pairs, list) else 'None/Invalid'}).")
                                except Exception as e: logger.error(f"Exception on attempt {attempt_count} (Style: '{writing_style_name}'): {type(e).__name__}", exc_info=False)

                                logger.info(f"Waiting {retry_delay_seconds:.2f}s before next attempt...")
                                time.sleep(retry_delay_seconds)
                            # --- End Indefinite Retry Loop ---

                            # Process successful generation
                            for gen_idx, orig_idx in enumerate(indices_to_generate):
                                styled_pair = generated_pairs[gen_idx]
                                existing_pairs_for_style_iteration[orig_idx] = styled_pair
                                if styled_pair.get('answer') and styled_pair.get('answer') not in previous_answers_batches[orig_idx]:
                                    previous_answers_batches[orig_idx].append(styled_pair['answer'])
                                if styled_pair.get('question') and styled_pair.get('question') not in previous_questions_batches[orig_idx].get(writing_style_name):
                                    previous_questions_batches[orig_idx][writing_style_name].append(styled_pair['question'])
                                # Save individual pair
                                try:
                                    with open(styled_paths[orig_idx], 'w', encoding='utf-8') as f: json.dump(styled_pair, f, ensure_ascii=False, indent=2)
                                except Exception as e: logger.error(f"Error saving pair {styled_paths[orig_idx]}: {e}")
                                # Save debug info
                                try:
                                  with open(styled_debug_paths[orig_idx], 'w', encoding='utf-8') as f:
                                      f.write(f"Orig Q: {batch_questions[orig_idx]}\nOrig A: {batch_answers[orig_idx]}\n---\n")
                                      f.write(f"Gen for Style: {writing_style_name}, Iter: {iteration+1}, Attempt: {attempt_count}\n")
                                      f.write(f"Style Conf: {json.dumps(writing_style, indent=2, ensure_ascii=False)}\n")
                                      f.write(f"Prev Ans: {json.dumps(batch_previous_gen[gen_idx], indent=2, ensure_ascii=False)}\n---\n")
                                      f.write(f"Styled Q: {styled_pair.get('question', 'N/A')}\nStyled A: {styled_pair.get('answer', 'N/A')}\n")
                                except Exception as e: logger.error(f"Error saving debug {styled_debug_paths[orig_idx]}: {e}")
                        # End if indices_to_generate

                        # Add successful pairs (loaded or generated) to lists
                        for idx, styled_pair in enumerate(existing_pairs_for_style_iteration):
                            if styled_pair:
                                styled_example = {
                                    "question": styled_pair['question'],
                                    "answer": styled_pair['answer'],
                                    "url": url,
                                    "qa_pair_hash": styled_qa_hashes[idx],
                                    "original_qa_pair_hash": batch_hashes[idx],
                                    "type": f"styled_{safe_style_name}",
                                    "iteration": iteration # 0-based iteration index
                                }
                                # Append to the list for this specific style (for final saving)
                                style_to_examples[safe_style_name].append(styled_example)
                                # Append to the overall list being returned
                                # Avoid duplicates if reprocessing occurred due to load errors
                                if styled_example not in all_training_examples:
                                     all_training_examples.append(styled_example)

                    # End Style Loop for this iteration
                # End Iteration Loop for this batch
            # End Batch Processing Loop

            # --- Save examples grouped by writing style ---
            for safe_style_name, examples in style_to_examples.items():
                 if safe_style_name in styles_to_process_map: # Only save styles processed in this run
                     if examples:
                        style_file_path = qa_dir / f"{safe_title_slug}_{safe_style_name}_examples.json"
                        # Ensure we only save examples matching this type from the collected list
                        # (Mitigates potential issues if examples were mixed up, though unlikely with current logic)
                        filtered_examples = [ex for ex in examples if ex.get("type") == f"styled_{safe_style_name}"]
                        if filtered_examples:
                             try:
                                 with open(style_file_path, 'w', encoding='utf-8') as f:
                                     json.dump(filtered_examples, f, ensure_ascii=False, indent=2)
                                 logger.info(f"Saved {len(filtered_examples)} examples for style '{styles_to_process_map[safe_style_name].get('name')}' to {style_file_path}")
                             except Exception as e:
                                 logger.error(f"Error saving final examples for style '{safe_style_name}': {e}")
                        else:
                            logger.info(f"No examples of type 'styled_{safe_style_name}' collected for saving.")
                     else:
                         logger.info(f"No examples generated/collected for style '{styles_to_process_map[safe_style_name].get('name')}' during this run.")


            logger.info(f"Finished processing {file_path}. Total training examples generated/loaded: {len(all_training_examples)}")
            # Deduplicate final list based on the unique styled hash
            final_examples_dict = {ex['qa_pair_hash']: ex for ex in all_training_examples}
            final_examples = list(final_examples_dict.values())
            if len(final_examples) < len(all_training_examples):
                logger.info(f"Deduplicated final examples from {len(all_training_examples)} to {len(final_examples)}.")

            return final_examples

        except Exception as e:
            logger.exception(f"Critical error processing FAQ document {file_path}: {e}", exc_info=True)
            return []