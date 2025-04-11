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
    def extract_faq(soup: BeautifulSoup, file_path: Path, llm_client: LLMClient) -> List[Dict[str, Any]]:
        """
        Extract question-answer pairs from an FAQ document using LLM processing.
        
        Args:
            soup: BeautifulSoup object of the document
            file_path: Path to the HTML file
            llm_client: LLM client for generation
            
        Returns:
            List of dictionaries containing extracted FAQ data
        """
        from .file_processor import FileProcessor
        
        try:
            structured_text_dir = file_path.parent / "structured_text"

            # check if already present in file
            if structured_text_dir.exists():
                structured_text_path = structured_text_dir / f"{file_path.stem}_structured.txt"
                if structured_text_path.exists():
                    logger.info(f"Structured text already exists for {file_path}")
                    with open(structured_text_path, 'r', encoding='utf-8') as f:
                        structured_text = f.read()
            else:
                # Extract text with preserved structure
                structured_text = FileProcessor.extract_text_from_html(soup, file_path, llm_client)
                # save it
                structured_text_dir.mkdir(parents=True, exist_ok=True)
                structured_text_path = structured_text_dir / f"{file_path.stem}_structured.txt"
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
        writing_style: Dict[str, str],
        llm_client: LLMClient
    ) -> List[Dict[str, str]]:

        """
        Generate styled versions of question and answer pairs in batch based on a specified writing style.

        Args:
            questions: List of original questions.
            answers: List of original answers, corresponding to the questions.
            previous_answers_batch: List of lists of previous answers to avoid repeating answers for that specific question.
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

Original Pairs:"""
        ]

        for i, (q, a) in enumerate(zip(questions, answers)):
            question_prompt_parts.append(f"Q{i+1}: {q}\nA{i+1}: {a}\n")

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
        
        Args:
            soup: BeautifulSoup object of the document
            file_path: Path to the HTML file
            output_dir: Directory to save output files
            config: Configuration dictionary
            
        Returns:
            List of QA pairs with variations
        """
        from .llm_client import LLMClient
        from .qa_generator import QAGenerator
        import json
        from slugify import slugify

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
            courses = FileProcessor.get_institution_courses(domain)
            
            # Construct the base URL for link resolution
            base_url = f"https://{domain}"
            if path:
                # Get directory part of the path for base URL
                path_parts = path.split('/')
                if '.' in path_parts[-1]:  # If last part looks like a file
                    path_parts = path_parts[:-1]
                if path_parts:
                    base_url = f"{base_url}/{'/'.join(path_parts)}"
            
            # Store in source info dictionary used throughout the processing
            faq_config = config.get("providers", {}).get("faq_extraction", {})
            faq_title = soup.title.get_text() if soup.title else institution

            source_info = {
                "domain": domain,
                "path": path,
                "url": url,
                "institution": institution,
                "courses": courses,
                "file_path": str(file_path),
                "base_url": base_url,
                "faq_title": faq_title
            }

            # Create directories for output
            extracted_faq_dir = output_dir / "extracted_faq"
            extracted_faq_dir.mkdir(parents=True, exist_ok=True)
            debug_dir = output_dir / "debug" / "qa_pairs"
            debug_dir.mkdir(parents=True, exist_ok=True)

            extracted_faq_hash = hashlib.sha256(f"{file_path}_{faq_config.get('model')}".encode()).hexdigest()[:12]
            extracted_faq_path = extracted_faq_dir / f"{slugify(faq_title)}_{extracted_faq_hash}.json"

            # Load or extract FAQ data
            if os.path.exists(extracted_faq_path):
                try:
                    with open(extracted_faq_path, 'r', encoding='utf-8') as f:
                        extracted_faq = json.load(f)
                    logger.info(f"Loaded {len(extracted_faq)} existing extracted QA pairs for {file_path}")
                except Exception as e:
                    logger.error(f"Error loading existing extracted FAQ: {e}")
                    extracted_faq = []
            else:
                extract_faq_llm_client = LLMClient(faq_config)
                extracted_faq = FAQProcessor.extract_faq(soup, file_path, extract_faq_llm_client)

                try:
                    with open(extracted_faq_path, 'w', encoding='utf-8') as f:
                        json.dump(extracted_faq, f, ensure_ascii=False, indent=2)
                except Exception as e:
                    logger.error(f"Error saving extracted FAQ: {e}")

            # Get FAQ specific configuration
            faq_config = config.get("processing", {}).get("faq", {})
            
            # Get number of iterations for FAQ documents
            total_iterations = config.get("iterations", {}).get("faq_document", 1)
            batch_size = config.get("iterations", {}).get("batch_size", 1)
            
            # Get writing styles
            qa_generator = QAGenerator(config)
            writing_styles = qa_generator.writing_styles if qa_generator.writing_styles else [{"name": "Default", "description": ""}]
                            
            # Create directories for output
            qa_dir = output_dir / "qa_pairs"
            qa_dir.mkdir(parents=True, exist_ok=True)
            debug_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate training examples with variations and related questions
            all_training_examples = []

            # Add original FAQ to database
            for i, faq_item in enumerate(extracted_faq):
                batch_questions = faq_item["question"]
                batch_answers = faq_item["answer"]
                batch_hashes = faq_item["qa_pair_hash"]
                # Keep original verbatim but add domain attribution
                verbatim_question = batch_questions
                verbatim_answer = batch_answers

                original_example = {
                    "question": verbatim_question,
                    "answer": verbatim_answer,
                    "url": url,
                    "qa_pair_hash": batch_hashes,
                    "type": "original_verbatim"
                }
                all_training_examples.append(original_example)

            # Initialize a dictionary to store examples by writing style
            style_to_examples = {}
            existing_examples = []
            # Check which writing styles already have files and remove them from processing
            styles_to_process = []
            for writing_style in writing_styles:
                writing_style_name = writing_style.get("name")
                style_file_path = qa_dir / f"{slugify(faq_title)}_{slugify(writing_style_name.lower())}_examples.json"
                
                if style_file_path.exists():
                    logger.info(f"File for style '{writing_style_name}' already exists. Skipping this style.")
                    # Load existing examples for this style to include in the final return
                    try:
                        with open(style_file_path, 'r', encoding='utf-8') as f:
                            existing_examples = json.load(f)
                            all_training_examples.extend(existing_examples)
                        logger.info(f"Loaded {len(existing_examples)} existing examples for style '{writing_style_name}'")
                    except Exception as e:
                        logger.error(f"Error loading existing style file: {e}")
                else:
                    styles_to_process.append(writing_style)

            faq_to_process = extracted_faq
            
            if 'default' not in [style.get("name").lower() for style in styles_to_process] and FAQProcessor.STYLE_DEFAULT_FAQS:
                if existing_examples:
                    faq_to_process = existing_examples
                    logger.info("Using existing examples for default style processing")


            for i in range(0, len(faq_to_process), batch_size):
                # Create a batch of FAQ items
                batch_items = faq_to_process[i:i+batch_size]
                
                batch_questions = []
                batch_answers = []
                batch_hashes = []
                
                for item in batch_items:
                    batch_questions.append(item["question"])
                    batch_answers.append(item["answer"])
                    batch_hashes.append(item["qa_pair_hash"])


                logger.info(f"{len(all_training_examples)} training examples generated so far")
                logger.info(f"Processing batch of {len(batch_items)} QA pairs: {i+1} to {min(i+batch_size, len(faq_to_process))} of {len(faq_to_process)}")

                # For each iteration, generate all alternate writing styles
                for iteration in range(total_iterations):
                    writing_style = None
                    previous_answers_batches = [[] for _ in range(len(batch_items))]

                    for writing_style in styles_to_process:
                        writing_style_name = slugify(writing_style.get("name"))
                        style_hash = f"{writing_style_name}_{iteration}"

                        styled_qa_hashes = [f"{qa_hash}_{style_hash}" for qa_hash in batch_hashes]
                        styled_paths = [qa_dir / f"styled_{styled_qa_hash}.txt" for styled_qa_hash in styled_qa_hashes]
                        styled_debug_paths = [debug_dir / f"styled_debug_{styled_qa_hash}.txt" for styled_qa_hash in styled_qa_hashes]

                        # Check which files already exist
                        existing_pairs = []
                        need_generation = False

                        for idx, styled_path in enumerate(styled_paths):
                            # Generate or load styled QA pair
                            if styled_path.exists():
                                try:
                                    with open(styled_path, 'r', encoding='utf-8') as f:
                                        styled_pair = json.load(f)
                                        previous_answers_batches[idx].append(styled_pair['answer'])
                                    logger.info(f"Loaded existing styled QA pair for {styled_qa_hashes[idx]}")
                                    existing_pairs.append(styled_pair)
                                except Exception as e:
                                    logger.error(f"Error loading existing styled QA pair: {e}")
                                    existing_pairs.append(None)
                                    need_generation = True
                            else:
                                existing_pairs.append(None)
                                need_generation = True
                            
                        generated_pairs = None
                        while not generated_pairs and need_generation:
                            if need_generation:
                                llm_client = LLMClient(config.get("providers", {}).get("styled_question", {}))

                                # Identify which indices need generation
                                indices_to_generate = [idx for idx, pair in enumerate(existing_pairs) if pair is None]
                                batch_questions_gen = [batch_questions[idx] for idx in indices_to_generate]
                                batch_answers_gen = [batch_answers[idx] for idx in indices_to_generate]
                                batch_previous_gen = [previous_answers_batches[idx] for idx in indices_to_generate]
                            
                                # Generate styled QA pairs
                                generated_pairs = FAQProcessor.generate_styled_qa(
                                    batch_questions_gen, 
                                    batch_answers_gen, 
                                    batch_previous_gen, 
                                    writing_style, 
                                    llm_client
                                )

                                # Merge generated pairs back into the complete list
                                for gen_idx, orig_idx in enumerate(indices_to_generate):
                                    if generated_pairs:
                                        existing_pairs[orig_idx] = generated_pairs[gen_idx]

                        # Process each styled pair in the batch
                        for idx, styled_pair in enumerate(existing_pairs):
                            if styled_pair:
                                # Update previous answers for this item
                                previous_answers_batches[idx].append(styled_pair['answer'])
                                
                                # Create example and add to collections
                                styled_example = {
                                    "question": styled_pair['question'],
                                    "answer": styled_pair['answer'],
                                    "url": url,
                                    "qa_pair_hash": styled_qa_hashes[idx],
                                    "type": "styled", 
                                }
                                all_training_examples.append(styled_example)
                                
                                if writing_style_name not in style_to_examples:
                                    style_to_examples[writing_style_name] = []
                                style_to_examples[writing_style_name].append(styled_example)
                                
                                # Save the styled pair
                                try:
                                    with open(styled_paths[idx], 'w', encoding='utf-8') as f:
                                        json.dump(styled_pair, f, ensure_ascii=False, indent=2)
                                except Exception as e:
                                    logger.error(f"Error saving style QA pair: {e}")
                                
                                # Save debug info
                                with open(styled_debug_paths[idx], 'w', encoding='utf-8') as f:
                                    f.write(f"Original Question: {batch_questions[idx]}\nAnswer: {batch_answers[idx]}\n")
                            else:
                                logger.error(f"Failed to generate styled QA pair for {styled_qa_hashes[idx]}.")


            # Save examples grouped by writing style
            for style_name, examples in style_to_examples.items():
                style_file_path = qa_dir / slugify(f"{(faq_title)}_{style_name.lower()}_examples.json")
                try:
                    with open(style_file_path, 'w', encoding='utf-8') as f:
                        json.dump(examples, f, ensure_ascii=False, indent=2)
                    logger.info(f"Saved {len(examples)} examples for writing style {style_name}")
                except Exception as e:
                    logger.error(f"Error saving examples for writing style {style_name}: {e}")
                

            
            return all_training_examples
        
        except Exception as e:
            logger.error(f"Error processing FAQ document {file_path}: {e}")
            return []