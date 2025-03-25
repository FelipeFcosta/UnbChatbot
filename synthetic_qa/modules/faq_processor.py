"""
FAQ processing module for the Synthetic QA Generator.

This module handles detection and processing of FAQ documents.
"""

import hashlib
import os
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Any
from bs4 import BeautifulSoup
import textwrap

from .llm_client import LLMClient
from .file_processor import FileProcessor

logger = logging.getLogger(__name__)


class FAQProcessor:
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
    def extract_faq(soup: BeautifulSoup, file_path: Path, llm_client: LLMClient) -> List[Dict[str, Any]]:
        """
        Extract question-answer pairs from an FAQ document using LLM processing.
        
        Args:
            soup: BeautifulSoup object of the document
            file_path: Path to the HTML file
            llm_client: LLM client for generation
            
        Returns:
            List of (question, answer) tuples
        """
        from .file_processor import FileProcessor
        import json
        
        try:
            # Extract text with preserved structure
            structured_text = FileProcessor.extract_text_from_html(soup, file_path, llm_client)

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
                "faq_q": "FAQ question",
                "faq_a": "FAQ answer",
                "topics": ["Topic1, Topic2, ..."],
                "course": "specific course"
                }},
                // More QA pairs...
            ]
            }}

            2. **Critical requirements**:
            - Both "faq_q" and "faq_a" fields **can never be null**
            - **"topics" are EXPLICITLY PRESENT in the FAQ AS titles/headers above a QA pair or a number of QA pairs** (can be null if no topics).
            - **include all nested topics as just a flat array**
            - "course" can be null or be a specific course

            3. Content processing:
            - Preserve all markdown formatting and links
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
                + "\nRewrite your output. Do any of these corrections for every pair where mistakes were made:\n"
                "- Remove any questions in the 'topics' field.\n"
                "- Remove any answer in the 'faq_q' field.\n"
                "- Remove any question in the 'faq_a' field.\n"
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
                    question = item.get("faq_q", "")
                    answer = item.get("faq_a", "")
                    topics = item.get("topics", [])
                    course = item.get("course", "")

                    if question and answer:
                        qa_pairs.append({
                            "faq_q": question.strip(),
                            "faq_a": answer.strip(),
                            "topics": topics,
                            "course": course.strip()
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
    def create_context_chunks(faq: List[Dict[str, Any]], context_size: int = 2) -> List[Tuple[str, str, str, str]]:
        """
        Create chunks with surrounding questions for context.
        
        Args:
            qa_pairs: List of (question, answer, topics) tuples
            context_size: Number of neighboring QA pairs to include as context
            
        Returns:
            List of (question, answer, topics, context) tuples
        """
        contextual_chunks = []
        
        for i, qa_pair in enumerate(faq):
            question = qa_pair['faq_q']
            answer = qa_pair['faq_a']
            topics = qa_pair['topics']
            course = qa_pair['course']

            # Get neighboring questions for context
            start_idx = max(0, i - context_size)
            end_idx = min(len(faq), i + context_size + 1)

            has_added_context_header = False
            context = ""
            for j in range(start_idx, end_idx):
                if j != i:
                    q = faq[j]['faq_q']
                    a = faq[j]['faq_a']
                    ts = faq[j]['topics']

                    # Only include context questions with matching topics
                    if set(t.lower() for t in topics) == set(t.lower() for t in ts):
                        if not has_added_context_header:
                            context += f"Context Questions:\n"
                        context += f"q: {q}\na: {a}\n\n"
            
            contextual_chunks.append((question, answer, topics, course, context))
        
        return contextual_chunks


def determine_generation_parameters(question: str, answer: str, available_styles: List[Dict[str, str]]) -> Dict[str, Any]:
    """
    Determine how many styles, rephrases, and related questions to generate
    based on the length of the QA pair.
    
    Args:
        question: The question text
        answer: The answer text
        available_styles: List of all available writing styles
        
    Returns:
        Dictionary with generation parameters
    """
    combined_words = len(question.split()) + len(answer.split())
    
    if combined_words <= 100:
        return {
            "rephrased_count": 1,
            "related_count": 0
        }
    elif combined_words <= 200:
        return {
            "rephrased_count": 2,
            "related_count": 1
        }
    elif combined_words <= 300:
        return {
            "rephrased_count": 3,
            "related_count": 2
        }
    else:
        return {
            "rephrased_count": 3,
            "related_count": 3
        }

def generate_styled_qa(original_qa: Tuple[str, str, str], context: str, writing_style: Dict[str, str], llm_client: LLMClient, source_info: Dict[str, str]) -> Tuple[str, str]:
    """
    Generate a styled version of a question and answer pair based on a specified writing style.
    
    Args:
        original_qa: The original question/answer/topics tuple
        context: The context including neighboring QA pairs
        writing_style: Dictionary with writing style name and description
        llm_client: LLM client for generation
        source_info: Dictionary with domain, institution, url info
        
    Returns:
        Tuple of (styled_question, styled_answer)
    """
    # Extract domain and institution info for better context
    domain = source_info.get("domain", "")
    url = source_info.get("url", "")
    institution = source_info.get("institution", domain)

    faq_title = source_info['faq_title']
    
    # Get writing style information
    writing_style_name = writing_style.get("name", "Default")
    writing_style_desc = writing_style.get("description", "")
    
    # Generate alternative question
    course_prompt = ""
    if original_qa['course']:
        course_prompt = f"\nIMPORTANT: You MUST specify the {original_qa['course']} course in the question."

    styled_question_prompt = f"""
    You are creating an alternative FAQ question from {institution}. The original question is:

    "{original_qa['faq_q']}"

    Additional Context:
    * FAQ title: {faq_title}
    * FAQ url: {url}
    * Question Topic: "{' | '.join(original_qa['topics'])}"

    **WRITING STYLE**: {writing_style_desc}

    Rewrite only the question, while **preserving all the original meaning and intent**. **DO NOT ADD ANY NEW INFORMATION**
    {course_prompt}

    **Follow the specified writing style and question type closely.**
    Return ONLY the alternative question IN PORTUGUESE with no additional text.
    """                        

    styled_question = llm_client.generate_text(textwrap.dedent(styled_question_prompt).lstrip(), temperature=0.4)

    # Generate alternative answer based on the alternative question
    styled_answer_prompt = f"""
    You are creating an alternative FAQ answer. The original question is:
    
    "{styled_question}"

    and the original (canonical) answer is:

    "{original_qa['faq_a']}"
    
    Additional Context:
    * Question writing style: "{writing_style_name}"
    * Questions Topic: "{' | '.join(original_qa['topics'])}"
    * "{context}"

    Rewrite the original answer, while **preserving all the exact original information in a clear, helpful format**.
    
    You should try to understand and somewhat match the original question writing style, but always be a formal, modern, serious, polite, expert assistant for a brazilian university (UnB - Universidade de Brasília).
    
    Preserve all markdown formatting and links

    **CRITICAL: NEVER ADD INFORMATION THAT IS NOT IN THE ORIGINAL ANSWER. DO NOT OVER INTERPRET QUESTIONS OR MAKE ASSUMPTIONS ABOUT THE ANSWER.**
    **Your rewrite must have the exact same meaning as the original - use only information explicitly present in the original answer or context questions.**
    **DO NOT add affirmative/negative statements, confirmations, or interpretations if they're not explicitly in the original.**

    Context QA rules:
    - Use the context QA to write a more thorough answer, summarizing other responses that potentially address other students' questions proactively.
    - No need to summarize every other context QA, the most generic/basic context answers should be chosen.
    - Separate the main answer from the additional answer
    - The transition between the main and additional answer should be smooth, logical and explained

    Return ONLY the alternative answer IN PORTUGUESE with no additional text.

    IMPORTANT: If the question mentions a specific university course, include that course in the answer.
    **DON'T MAKE ASSUMPTIONS** if original answer isn't clear (always preserve meaning)**

    Keep the source (FONTE) at the end of the new answer
    """

    styled_answer = llm_client.generate_text(textwrap.dedent(styled_answer_prompt).lstrip(), temperature=0.4)
    
    return {"question": styled_question, "answer": styled_answer}


def generate_rephrased_qa(qa: Tuple[str, str], num_pairs: int, llm_client: LLMClient, writing_style: Dict[str, str], source_info: Dict[str, str]) -> List[Dict[str, str]]:
    """
    Generate multiple rephrased versions of the original question and answer.
    
    Args:
        qa: The original question and answer
        num_pairs: number of rephrased pairs to generate
        llm_client: LLM client for generation
        writing_style: writing style
        source_info: Dictionary with domain, institution, url info
        
    Returns:
        List of dictionaries with rephrased questions and answers
    """
    import json
    import re
    
    # Extract domain and institution info for better context
    domain = source_info.get("domain", "")
    url = source_info.get("url", "")
    institution = source_info.get("institution", domain)
    faq_title = source_info['faq_title']

    writing_style_name = writing_style.get("name", "Default")
    writing_style_desc = writing_style.get("description", "")

    prompt = f"""
    Please help me rephrase this question/answer from a FAQ ({faq_title}) in {num_pairs} different ways. 
    
    ORIGINAL QUESTION:
    "{qa['question']}"
    
    ORIGINAL ANSWER:
    "{qa['answer']}"

    The versions must be distinct from each other (if more than one) and they should:
    1. Keep same or slightly less information (in questions and answers)
    2. Vary in LENGTH, STRUCTURE, and WORDING (with same meaning! **absolute synonyms**)
    3. Include both direct and indirect question forms
    4. Use different vocabulary while maintaining the SAME MEANING AND INTENT
    5. Preserve all markdown formatting and links

    The question was written in a **{writing_style_name}** style.
    - Described as {writing_style_desc}.
    **Be sure to keep this same style while rephrasing**.
    The answer was slighted adjusted for that, but it's always from a formal, modern, serious, polite, expert assistant for a brazilian university (UnB - Universidade de Brasília).
    
    Both question and answer should follow the same above rules one at a time.

    **CRITICAL: NEVER ADD INFORMATION THAT IS NOT IN THE ORIGINAL ANSWER. DO NOT OVER INTERPRET QUESTIONS OR MAKE ASSUMPTIONS ABOUT THE ANSWER.**
    **DO NOT add affirmative/negative statements, confirmations, or interpretations if they're not explicitly in the original.**

    Generate exactly {num_pairs} coherent and distinct QA variations (NO TWO QUESTIONS OR ANSWERS SHOULD BE SIMILAR) IN PORTUGUESE.
    FOLLOW THIS STRICTLY: Return ONLY the rephrased questions and answers in json.

    {{
      "qa_pairs": [
        {{"q": "...", "a": "..."}},
        ...
      ]
    }}

    Keep the source (FONTE) at the final (isolated) line of the new answer.
    """

    response = llm_client.generate_text(textwrap.dedent(prompt).lstrip(), json_output=True, temperature=0.5)
    if not response:
        return []
    
    try:
        # Extract the QA pairs
        rephrased_pairs = []
        for pair in response.get("qa_pairs", []):
            q = pair.get("q", "").strip()
            a = pair.get("a", "").strip()
            
            if q and a:  # Ensure both question and answer exist
                 rephrased_pairs.append({"question": q, "answer": a})
        
        logger.info(f"Generated {len(rephrased_pairs)} rephrased QA pairs")
        return rephrased_pairs
        
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse rephrased QA response as JSON: {e}")
        return []


def generate_related_questions(question: str, answer: str, num_pairs: int, context: str, llm_client: LLMClient, source_info: Dict[str, str]) -> List[str]:
    """
    Generate additional questions that can be answered by the same content.
    
    Args:
        question: The original question
        question: The original answer
        num_pairs: number of related pairs to generate
        context: The context including neighboring QA pairs
        llm_client: LLM client for generation
        source_info: Dictionary with domain, institution, url info
        
    Returns:
        List of related questions
    """
    import json
    import re

    # Extract domain and institution info for better context
    domain = source_info.get("domain", "")
    url = source_info.get("url", "")
    institution = source_info.get("institution", domain)

    if not num_pairs:
        logger.info(f"Generated 0 related QA pairs")
        return []

    prompt = f"""
    I'm creating training data for an institutional chatbot for {institution}. Based on this question and answer pair:
    
    ORIGINAL QUESTION:
    "{question}"
    
    ORIGINAL ANSWER:
    "{answer}"

    "{context}"

    Generate {num_pairs} additional questions and answers that could also be answered by this same answer. Consider:
    1. Questions about specific aspects mentioned in the answer but not directly addressed in the original question
    2. More specific questions about details in the *original answer*
    3. Questions from different perspectives or use cases
    4. Questions seeking the same information but focusing on different parts of the *original answer*
    5. New answers should have the exact same meaning as before (but with added context).
    6. *Reorganize each new answer* so that it reflects the *new focus of the question*.
    7. The generated related questions should be **different** from each other (if more than one).
    8. Preserve all markdown formatting and links

    **DO NOT ADD ANY NEW INFORMATION**

    IMPORTANT:
    - Use the context QA *only* to write a more thorough answer, summarizing other responses to potentially address other students' questions proactively.
    
    These should be NEW questions, not rephrases of the original question.
    They must be fully answerable with ONLY the information in the provided *original answer*.
    
    Generate exactly {num_pairs} new questions based on the original answer content.
    
    FOLLOW THIS STRICTLY: Return ONLY the related questions and answers in json.

    {{
      "qa_pairs": [
        {{"q": "...", "a": "..."}},
        ...
      ]
    }}

    Keep the source (FONTE) at the final (isolated) line of the new answer.
    """
    
    response = llm_client.generate_text(textwrap.dedent(prompt).lstrip(), json_output=True, temperature=0.5)
    if not response:
        logger.warning(f"Failed to generate related QA response.")
        return []

    try:
        # Extract the QA pairs
        related_pairs = []
        for pair in response.get("qa_pairs", []):
            q = pair.get("q", "").strip()
            a = pair.get("a", "").strip()
            
            if q and a:
                related_pairs.append({"question": q, "answer": a})
        
        logger.info(f"Generated {len(related_pairs)} related QA pairs")
        return related_pairs
        
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse related QA response as JSON: {e}")
        return []


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

        if os.path.exists(extracted_faq_path):
            try:
                with open(extracted_faq_path, 'r', encoding='utf-8') as f:
                    extracted_faq = json.load(f)
                logger.info(f"Loaded {len(extracted_faq)} existing extracted QA pairs for {file_path}")
            except Exception as e:
                logger.error(f"Error loading existing extracted FAQ: {e}")
                rephrased_qa_pairs = []
        else:
            faq_client = LLMClient(faq_config)
            
            extracted_faq = FAQProcessor.extract_faq(soup, file_path, faq_client)

            try:
                with open(extracted_faq_path, 'w', encoding='utf-8') as f:
                    json.dump(extracted_faq, f, ensure_ascii=False, indent=2)
            except Exception as e:
                logger.error(f"Error saving extracted FAQ: {e}")

        # Get FAQ specific configuration
        faq_config = config.get("processing", {}).get("faq", {})
        context_size = faq_config.get("context_size", 2)
        
        # Get number of iterations for FAQ documents
        total_iterations = config.get("iterations", {}).get("faq_document", 1)
        
        # Get writing styles
        qa_generator = QAGenerator(config)
        writing_styles = qa_generator.writing_styles if qa_generator.writing_styles else [{"name": "Default", "description": ""}]
        
        # Create context chunks with neighboring questions
        contextual_chunks = FAQProcessor.create_context_chunks(extracted_faq, context_size)
                
        # Create directories for output
        qa_dir = output_dir / "qa_pairs"
        qa_dir.mkdir(parents=True, exist_ok=True)
        debug_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate training examples with variations and related questions
        all_training_examples = []

        for i, (question, answer, topics, course, context) in enumerate(contextual_chunks):
            logger.info(f"{len(all_training_examples)} training examples generated so far")
            logger.info(f"Processing QA pair {i + 1} of {len(contextual_chunks)}")

            generation_params = determine_generation_parameters(question, answer, writing_styles)

            # Create a unique identifier for this QA pair
            qa_pair_hash = f"faq_{hashlib.sha256((str(file_path) + question).encode()).hexdigest()[:12]}"
            
            # Keep original verbatim but add domain attribution
            verbatim_question = question
            verbatim_answer = answer + "\n" + f"*FONTE: [{faq_title}]({url})*"

            original_example = {
                "question": verbatim_question,
                "answer": verbatim_answer,
                "source": str(file_path),
                "url": url,
                "domain": domain,
                "institution": institution,
                "qa_pair_hash": qa_pair_hash,
                "type": "original_verbatim"
            }
            all_training_examples.append(original_example)
                        
            # For each iteration, generate all alternate writing styles
            for iteration in range(total_iterations):
                for writing_style in writing_styles:
                    writing_style_name = writing_style.get("name")
                    style_hash = f"{writing_style_name}_{iteration}"

                    faq_config = config.get("providers", {}).get("question", {})
                    faq_client = LLMClient(faq_config)

                    original_qa = {
                        "faq_q": verbatim_question,
                        "faq_a": verbatim_answer,
                        "topics": topics,
                        "course": course
                    }

                    styled_qa_hash = f"{qa_pair_hash}_{style_hash}"

                    styled_path = qa_dir / f"styled_{styled_qa_hash}.txt"
                    styled_debug_path = debug_dir / f"styled_debug_{styled_qa_hash}.txt"

                    if styled_path.exists():
                        # Load existing rephrased questions
                        try:
                            with open(styled_path, 'r', encoding='utf-8') as f:
                                styled_pair = json.load(f)
                            logger.info(f"Loaded existing styled QA pair for {styled_qa_hash}")
                        except Exception as e:
                            logger.error(f"Error loading existing styled QA pair: {e}")
                            styled_pair = []
                    else:
                        styled_pair = generate_styled_qa(original_qa, context, writing_style, faq_client, source_info)

                    if styled_pair['question'] and styled_pair['answer']:
                        styled_example = {
                            "question": styled_pair['question'],
                            "answer": styled_pair['answer'],
                            "source": str(file_path),
                            "url": url,
                            "domain": domain,
                            "institution": institution,
                            "qa_pair_hash": styled_qa_hash,
                            "type": "styled",
                            "writing_style": writing_style_name,
                            "iteration": iteration
                        }
                        all_training_examples.append(styled_example)

                        try:
                            with open(styled_path, 'w', encoding='utf-8') as f:
                                json.dump(styled_pair, f, ensure_ascii=False, indent=2)
                        except Exception as e:
                            logger.error(f"Error saving style QA pair: {e}")
                        
                        with open(styled_debug_path, 'w', encoding='utf-8') as f:
                            f.write(f"Original Question: {question}\nAnswer: {answer}\n")

                    rephrased_path = qa_dir / f"rephrased_{styled_qa_hash}.txt"
                    rephrased_debug_path = debug_dir / f"rephrased_debug_{styled_qa_hash}.txt"
                    
                    if rephrased_path.exists():
                        # Load existing rephrased questions
                        try:
                            with open(rephrased_path, 'r', encoding='utf-8') as f:
                                rephrased_qa_pairs = json.load(f)
                            logger.info(f"Loaded {len(rephrased_qa_pairs)} existing rephrased QA pairs for {styled_qa_hash}")
                        except Exception as e:
                            logger.error(f"Error loading existing rephrased QA pairs: {e}")
                            rephrased_qa_pairs = []
                    else:
                        # Generate rephrased QA pairs
                        rephrased_qa_pairs = generate_rephrased_qa(styled_pair, generation_params['rephrased_count'], faq_client, writing_style, source_info)

                        # Save rephrased QA pairs and debug info
                        if rephrased_qa_pairs:
                            try:
                                with open(rephrased_path, 'w', encoding='utf-8') as f:
                                    json.dump(rephrased_qa_pairs, f, ensure_ascii=False, indent=2)
                            except Exception as e:
                                logger.error(f"Error saving rephrased QA pairs: {e}")
                            
                            with open(rephrased_debug_path, 'w', encoding='utf-8') as f:
                                f.write(f"Original Question: {styled_pair['question']}\nAnswer: {styled_pair['answer']}")

                
                    # Add rephrased qa to the training examples
                    for j, rephrased_pair in enumerate(rephrased_qa_pairs):

                        all_training_examples.append({
                            "question": rephrased_pair['question'],
                            "answer": rephrased_pair['answer'],
                            "source": str(file_path),
                            "url": url,
                            "domain": domain,
                            "institution": institution,
                            "qa_pair_hash": f"{styled_qa_hash}_rephrased_{j}",
                            "type": "rephrased"
                        })
            
                    related_path = qa_dir / f"related_{styled_qa_hash}.txt"
                    related_debug_path = debug_dir / f"related_debug_{styled_qa_hash}.txt"
                
                    if related_path.exists():
                        # Load existing related questions
                        try:
                            with open(related_path, 'r', encoding='utf-8') as f:
                                related_qa_pairs = json.load(f)
                            logger.info(f"Loaded {len(related_qa_pairs)} existing related questions for {styled_qa_hash}")
                        except Exception as e:
                            logger.error(f"Error loading existing related QA pairs: {e}")
                            related_qa_pairs = []
                    else:
                        # Generate related questions
                        related_qa_pairs = generate_related_questions(styled_pair['question'], answer, generation_params['related_count'], context, faq_client, source_info)
                    
                        # Save related QA pairs and debug info
                        if related_qa_pairs:
                            try:
                                with open(related_path, 'w', encoding='utf-8') as f:
                                    json.dump(related_qa_pairs, f, ensure_ascii=False, indent=2)
                            except Exception as e:
                                logger.error(f"Error saving related QA pairs: {e}")
                            
                            with open(related_debug_path, 'w', encoding='utf-8') as f:
                                f.write(f"Original Question: {styled_pair['question']}\nAnswer: {styled_pair['answer']}")
                
                    # Add related questions to the training examples
                    for k, related_pair in enumerate(related_qa_pairs):
                        all_training_examples.append({
                            "question": related_pair['question'],
                            "answer": related_pair['answer'],
                            "source": str(file_path),
                            "url": url,
                            "domain": domain,
                            "institution": institution,
                            "qa_pair_hash": f"{styled_qa_hash}_related_{k}",
                            "type": "related"
                        })
        
        return all_training_examples
    
    except Exception as e:
        logger.error(f"Error processing FAQ document {file_path}: {e}")
        return []
