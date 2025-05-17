import re
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

from .llm_client import LLMClient
from .file_processor import FileProcessor

logger = logging.getLogger(__name__)


class QAGenerator:
    """Handles generation of questions and answers from text chunks."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize with provided configuration."""
        self.config = config
        self.question_client = LLMClient(config.get("providers", {}).get("question", {}))
        self.answer_client = LLMClient(config.get("providers", {}).get("answer", {}))

        # Load question style variations from config
        # Ensure default structure if keys are missing
        question_styles_config = config.get("question_styles", {})
        self.writing_styles = question_styles_config.get("writing_styles", [])
        self.question_types = question_styles_config.get("question_types", [])

        # Add default styles if none are provided to ensure at least one run
        if not self.writing_styles:
             self.writing_styles = [{"name": "Default", "description": "", "iterations": 1}]
        if not self.question_types:
             self.question_types = [{"name": "Default", "description": ""}] # Iterations typically driven by writing style


    def _get_style_prompt(self, style_type: str = None, style_name: str = None) -> str:
        """
        Get a prompt fragment for a specific writing style or question type.

        Args:
            style_type: 'writing_styles' or 'question_types'
            style_name: Name of the specific style or type

        Returns:
            Style instruction string or empty string if not found
        """
        if not style_type or not style_name or style_name == "Default": # Don't add prompt for default
            return ""

        styles = self.config.get("question_styles", {}).get(style_type, [])

        for style in styles:
            if style.get("name") == style_name:
                # Return description if available, otherwise empty string
                return style.get("description", "")

        logger.warning(f"Style '{style_name}' of type '{style_type}' not found in config.")
        return ""

    def get_question_prompt(self,
                           chunk: str,
                           source_info: Dict[str, str],
                           is_full_document: bool = False,
                           writing_style: Optional[str] = None,
                           question_type: Optional[str] = None) -> str:
        """
        Create a prompt for generating questions from a text chunk.

        Args:
            chunk: The text chunk to generate questions from
            source_info: Dictionary with domain, path, url, institution info
            is_full_document: Whether this chunk represents a complete document with multiple sections
            writing_style: Optional writing style name
            question_type: Optional question type name

        Returns:
            A formatted prompt string
        """
        # Get style-specific instructions
        writing_style_prompt = self._get_style_prompt("writing_styles", writing_style)
        question_type_prompt = self._get_style_prompt("question_types", question_type)

        # Combine style prompts
        style_instructions = ""
        if writing_style_prompt:
            # Use the 'goal' field if present and description is missing/empty
            # Find the style dict again to get the goal
            style_dict = next((s for s in self.writing_styles if s.get("name") == writing_style), None)
            style_desc = writing_style_prompt # Already fetched description
            style_goal = style_dict.get("goal", "") if style_dict else ""
            # Prioritize description, fallback to goal
            prompt_content = style_desc if style_desc else style_goal
            if prompt_content:
                 style_instructions += f"WRITING STYLE ({writing_style}): {prompt_content}\n\n"

        if question_type_prompt:
            style_dict = next((s for s in self.question_types if s.get("name") == question_type), None)
            style_desc = question_type_prompt # Already fetched description
            style_goal = style_dict.get("goal", "") if style_dict else ""
            # Prioritize description, fallback to goal
            prompt_content = style_desc if style_desc else style_goal
            if prompt_content:
                style_instructions += f"QUESTION TYPE ({question_type}): {prompt_content}\n\n"

        # Construct domain and URL information
        domain = source_info.get("domain", "")
        # path = source_info.get("path", "") # Path not directly used in prompt
        url = source_info.get("url", "")
        institution = source_info.get("institution", domain)

        # Construct URL reference for prompts
        url_reference = f"{url}" if url else "[Source Document]" # Fallback if URL is missing

        if is_full_document:
            # Special prompt for full documents with multiple sections
            return f"""
You are generating training examples for a university chatbot for {institution}. The questions will be used to train an institutional AI assistant.

INSTRUCTION:
1. Generate questions ONLY about information explicitly mentioned in the provided document
2. Create diverse questions covering DIFFERENT SECTIONS of the document
3. Include specific details mentioned in various sections
4. Include at least one question about each major section or topic
5. Generate questions that will be useful for university students, faculty, and staff
6. Focus on questions that have clear, factual answers based on the text
7. IMPORTANT: ALWAYS explicitly mention the URL "{url_reference}" in each question, if available. If no URL, mention "{institution}".
8. Make it clear the question is about information from {institution}

{style_instructions}
INFORMATION SOURCE: {institution} ({url_reference})

INFORMATION:
{chunk}

Generate 6-10 diverse questions that:
- Can be answered DIRECTLY from the provided content
- Represent how real humans would naturally ask about this information
- Cover different sections and aspects throughout the ENTIRE document
- Would be relevant to an institutional chatbot
- ALWAYS reference the source: Use the URL "{url_reference}" if available, otherwise mention {institution}.

Return ONLY the questions, one per line, with no numbering or formatting.
"""
        else:
            return f"""
You are generating training examples for a university chatbot for {institution}. The questions will be used to train an institutional AI assistant.

INSTRUCTION:
1. Generate questions about the following information from {institution}
2. Ensure the questions are relevant to students, faculty, or visitors
3. Generate questions that someone might actually ask about this content
4. IMPORTANT: ALWAYS explicitly mention the URL "{url_reference}" in each question, if available. If no URL, mention "{institution}".
5. Make it clear the question is about information from {institution}

{style_instructions}
INFORMATION SOURCE: {institution} ({url_reference})

INFORMATION:
{chunk}

Generate 3-5 diverse questions that:
- Cover different aspects of the information provided
- Range from factual to conceptual understanding
- Represent how real humans would naturally ask about this information
- Are specific enough to be answerable from the provided information
- ALWAYS reference the source: Use the URL "{url_reference}" if available, otherwise mention {institution}.

Return ONLY the questions, one per line, with no numbering or formatting.
"""

    def get_answer_prompt(self, question: str, chunk: str, source_info: Dict[str, str]) -> str:
        """
        Create a prompt for generating an answer to a question from a text chunk.

        Args:
            question: The question to answer
            chunk: The text chunk containing information to answer the question
            source_info: Dictionary with domain, path, url, institution info

        Returns:
            A formatted prompt string
        """
        # Extract domain and institution info
        domain = source_info.get("domain", "")
        # path = source_info.get("path", "") # Path not directly used in prompt
        url = source_info.get("url", "")
        institution = source_info.get("institution", domain)
        url_reference = f"{url}" if url else "[Source Document]" # Fallback if URL is missing
        attribution_intro = f"De acordo com as informações de {institution} ({url_reference})," if url else f"De acordo com as informações de {institution},"

        return f"""
You are an expert institutional assistant for {institution}. You're answering a question based on information from the source: {url_reference}.

IMPORTANT:
- Start your answer by referring to "{institution}" and the source "{url_reference}" using the phrase "{attribution_intro}" or similar.
- Base your answer ONLY on the information provided in the context below
- If the information doesn't contain enough details to fully answer the question, acknowledge the limitations

INFORMATION:
{chunk}

QUESTION: {question}

Provide a helpful, accurate, and concise answer. Begin your response with "{attribution_intro}" or a similar attribution phrase that mentions both the institution and the source reference.
"""

    def parse_questions(self, question_text: str) -> List[str]:
        """
        Parse generated questions text into a list of individual questions.

        Args:
            question_text: The text containing multiple questions

        Returns:
            A list of individual questions
        """
        if not question_text:
            return []

        questions = []
        for line in question_text.splitlines():
            line = line.strip()
            if not line:
                continue

            # Remove any leading numbering/bullet points/markdown
            line = re.sub(r'^[\s]*[\d\.\-\*]+\s*', '', line)
            line = re.sub(r'\*+', '', line) # Remove markdown bold/italics etc.

            # Basic check for question format (ends with ?, contains words)
            if line.endswith('?') and len(line.split()) > 2:
                questions.append(line)
            elif len(line) > 15: # Keep longer lines even without '?' just in case
                 logger.debug(f"Keeping line without '?' due to length: {line}")
                 # Append '?' if missing and seems like a question based on common starts
                 if line.lower().startswith(('qual', 'como', 'onde', 'quando', 'quem', 'por que', 'o que', 'será que', 'existe', 'é possível', 'poderia', 'gostaria de saber')):
                      if not line.endswith('?'): line += '?'
                 questions.append(line)


        return questions

    def generate_qa_pairs(self,
                         chunk: str,
                         source_path: str,
                         output_dir: Path,
                         chunk_hash: str,
                         is_full_document: bool = False,
                         is_faq: bool = False) -> List[Dict[str, str]]:
        """
        Generate question-answer pairs from a text chunk, applying style iterations.

        Args:
            chunk: The text chunk to generate QA pairs from
            source_path: File path of the source content
            output_dir: Directory to save generated questions and answers
            chunk_hash: A hash of the chunk for file naming
            is_full_document: Whether this chunk represents a complete document
            is_faq: Whether this chunk is from an FAQ document (might influence LLM behavior indirectly via prompts)

        Returns:
            A list of dictionaries containing questions and answers
        """
        # Extract domain and path information
        file_path = Path(source_path)
        domain, path, url = FileProcessor.extract_domain_and_path(file_path)
        institution = FileProcessor.get_institution_name(domain)

        source_info = {
            "domain": domain,
            "path": path,
            "url": url,
            "institution": institution,
            "file_path": str(file_path) # Ensure string for dict
        }

        # Create output directories if they don't exist
        qa_dir = output_dir / "qa_pairs"
        debug_dir = output_dir / "debug"
        qa_dir.mkdir(parents=True, exist_ok=True)
        debug_dir.mkdir(parents=True, exist_ok=True)

        all_qa_pairs = []

        # Use default styles if config is empty
        writing_styles_to_run = self.writing_styles if self.writing_styles else [{"name": "Default", "description": "", "iterations": 1}]
        question_types_to_run = self.question_types if self.question_types else [{"name": "Default", "description": ""}]

        # Iterate through each writing style
        for writing_style in writing_styles_to_run:
            writing_style_name = writing_style.get("name", "UnknownStyle")
            # Get the number of iterations for this specific style, default to 1
            num_iterations = writing_style.get("iterations", 1)

            logger.info(f"Processing Writing Style: '{writing_style_name}' for {num_iterations} iterations.")

            # Run the specified number of iterations for this writing style
            for iteration in range(num_iterations):
                # Iterate through each question type for the current writing style iteration
                for question_type in question_types_to_run:
                    question_type_name = question_type.get("name", "UnknownType")

                    # Unique identifier for this specific combination and iteration
                    style_iteration_hash = f"{writing_style_name}_{question_type_name}_iter{iteration}"
                    base_filename = f"{chunk_hash}_{style_iteration_hash}"

                    questions_path = qa_dir / f"questions_{base_filename}.txt"
                    q_debug_path = debug_dir / f"q_debug_{base_filename}.txt"

                    logger.debug(f"Generating for: Chunk={chunk_hash}, Style={writing_style_name}, Type={question_type_name}, Iteration={iteration}")

                    # Generate or load questions for this style combination and iteration
                    if questions_path.exists():
                        logger.info(f"Using existing questions file: {questions_path}")
                        question_list_text = questions_path.read_text(encoding="utf-8")
                    else:
                        # Generate questions with this style
                        question_prompt = self.get_question_prompt(
                            chunk,
                            source_info,
                            is_full_document=is_full_document,
                            writing_style=writing_style_name,
                            question_type=question_type_name
                        )

                        # Use moderate temperature
                        temperature = 0.7
                        question_list_text = self.question_client.generate_text(
                            question_prompt,
                            temperature=temperature
                        )

                        if not question_list_text:
                            logger.error(f"Failed to generate questions for chunk {chunk_hash} with style combo {style_iteration_hash}")
                            continue # Skip to next style/iteration

                        # Save outputs
                        try:
                            questions_path.write_text(question_list_text, encoding="utf-8")
                            q_debug_path.write_text(question_prompt, encoding="utf-8")
                        except IOError as e:
                             logger.error(f"Error writing question files for {base_filename}: {e}")
                             continue # Skip if cannot write files


                    # Parse questions
                    questions = self.parse_questions(question_list_text)
                    if not questions:
                        logger.warning(f"No valid questions parsed from output for chunk {chunk_hash} with style combo {style_iteration_hash}. Raw text:\n{question_list_text[:500]}...")
                        # Optionally save the raw text that failed parsing for debugging
                        failed_q_parse_path = debug_dir / f"failed_q_parse_{base_filename}.txt"
                        try:
                           failed_q_parse_path.write_text(f"---PROMPT---\n{question_prompt}\n\n---RESPONSE---\n{question_list_text}", encoding='utf-8')
                        except IOError as e:
                           logger.error(f"Could not write failed parse debug file {failed_q_parse_path}: {e}")
                        continue # Skip to next style/iteration

                    # Generate answers for each question
                    for i, question in enumerate(questions, 1):
                        q_index = f"q{i}"
                        answer_path = qa_dir / f"answer_{base_filename}_{q_index}.txt"
                        a_debug_path = debug_dir / f"a_debug_{base_filename}_{q_index}.txt"

                        if answer_path.exists():
                            logger.info(f"Using existing answer file: {answer_path}")
                            answer = answer_path.read_text(encoding="utf-8")
                        else:
                            # Generate answer
                            answer_prompt = self.get_answer_prompt(question, chunk, source_info)
                            # Use moderate temperature
                            temperature = 0.5
                            answer = self.answer_client.generate_text(
                                answer_prompt,
                                temperature=temperature
                            )

                            if not answer:
                                logger.error(f"Failed to generate answer for question {i} ({question[:50]}...) in chunk {chunk_hash} with style combo {style_iteration_hash}")
                                continue # Skip this question

                            # Save outputs
                            try:
                                answer_path.write_text(answer, encoding="utf-8")
                                a_debug_path.write_text(answer_prompt, encoding="utf-8")
                            except IOError as e:
                                logger.error(f"Error writing answer files for {base_filename}_{q_index}: {e}")
                                continue # Skip this QA pair if cannot write files

                        # Add to QA pairs list
                        qa_pair = {
                            "question": question,
                            "answer": answer,
                            "source": source_path,
                            "url": url,
                            "domain": domain,
                            "institution": institution,
                            "chunk_hash": f"{chunk_hash}_{style_iteration_hash}_{q_index}", # More specific hash
                            "writing_style": writing_style_name,
                            "question_type": question_type_name,
                            "iteration": iteration # Record the iteration number for this style
                        }
                        all_qa_pairs.append(qa_pair)

        logger.info(f"Generated {len(all_qa_pairs)} QA pairs for chunk {chunk_hash} across all styles and iterations.")
        return all_qa_pairs