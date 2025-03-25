"""
QA Generator module for the Synthetic QA Generator.

This module handles generating questions and answers from text chunks with domain attribution.
"""

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
        self.factual_mode = config.get("factual_mode", False)
        
        # Load question style variations from config
        self.writing_styles = config.get("question_styles", {}).get("writing_styles", [])
        self.question_types = config.get("question_styles", {}).get("question_types", [])

    def _get_style_prompt(self, style_type: str = None, style_name: str = None) -> str:
        """
        Get a prompt fragment for a specific writing style or question type.
        
        Args:
            style_type: 'writing_styles' or 'question_types'
            style_name: Name of the specific style or type
            
        Returns:
            Style instruction string or empty string if not found
        """
        if not style_type or not style_name:
            return ""
            
        styles = self.config.get("question_styles", {}).get(style_type, [])
        
        for style in styles:
            if style.get("name") == style_name:
                return style.get("description", "")
                
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
        writing_style_prompt = self._get_style_prompt("writing_styles", writing_style) if writing_style else ""
        question_type_prompt = self._get_style_prompt("question_types", question_type) if question_type else ""
        
        # Combine style prompts
        style_instructions = ""
        if writing_style_prompt:
            style_instructions += f"WRITING STYLE: {writing_style_prompt}\n\n"
        if question_type_prompt:
            style_instructions += f"QUESTION TYPE: {question_type_prompt}\n\n"
        
        # Construct domain and URL information
        domain = source_info.get("domain", "")
        path = source_info.get("path", "")
        url = source_info.get("url", "")
        institution = source_info.get("institution", domain)
        
        # Construct URL reference for prompts
        url_reference = f"{url}" if url else ""
        
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
7. IMPORTANT: ALWAYS explicitly mention the URL "{url_reference}" in each question
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
- ALWAYS reference the URL "{url_reference}" and the {institution} directly in the question text

Return ONLY the questions, one per line, with no numbering or formatting.
"""
        elif self.factual_mode:
            return f"""
You are generating training examples for a university chatbot for {institution}. The questions will be used to train an institutional AI assistant.

INSTRUCTION:
1. Generate questions ONLY about information explicitly mentioned in the provided document
2. Do not create questions that require inference beyond what is directly stated
3. Include specific details like dates, requirements, and rule numbers in your questions
4. Generate questions that will be useful for university students, faculty, and staff
5. Focus on questions that have clear, factual answers based on the text
6. IMPORTANT: ALWAYS explicitly mention the URL "{url_reference}" in each question
7. Make it clear the question is about information from {institution}

{style_instructions}
INFORMATION SOURCE: {institution} ({url_reference})

INFORMATION:
{chunk}

Generate 3-5 diverse questions that:
- Can be answered DIRECTLY from the provided content
- Represent how real humans would naturally ask about this information
- Cover different important aspects of the information provided
- Would be relevant to an institutional chatbot
- ALWAYS reference the URL "{url_reference}" and the {institution} directly in the question text

Return ONLY the questions, one per line, with no numbering or formatting.
"""
        else:
            return f"""
You are generating training examples for a university chatbot for {institution}. The questions will be used to train an institutional AI assistant.

INSTRUCTION:
1. Generate questions about the following information from {institution}
2. Ensure the questions are relevant to students, faculty, or visitors
3. Generate questions that someone might actually ask about this content
4. IMPORTANT: ALWAYS explicitly mention the URL "{url_reference}" in each question
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
- ALWAYS reference the URL "{url_reference}" and the {institution} directly in the question text

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
        path = source_info.get("path", "")
        url = source_info.get("url", "")
        institution = source_info.get("institution", domain)
        
        if self.factual_mode:
            return f"""
You are an expert institutional assistant for {institution}. You're answering a question based on information from {url}.

IMPORTANT: Your response must be factually accurate and based ONLY on the information provided.
- Start your answer by referring to "{institution}" and the URL "{url}"
- If the information doesn't contain enough details to fully answer the question, acknowledge the limitations
- Include specific details, numbers, and requirements exactly as stated in the document
- Do not make assumptions or inferences beyond what is directly stated
- If different parts of the document appear to conflict, acknowledge the ambiguity in your answer
- If referring to specific sections, rules, or article numbers, cite them explicitly

INFORMATION:
{chunk}

QUESTION: {question}

Provide a helpful, accurate, and strictly factual answer. Begin your response with "De acordo com as informações de {institution} ({url})," or a similar attribution phrase that mentions both the institution and URL.
"""
        else:
            return f"""
You are an expert institutional assistant for {institution}. You're answering a question based on information from {url}.

IMPORTANT:
- Start your answer by referring to "{institution}" and the URL "{url}"
- Base your answer ONLY on the information provided in the context below
- If the information doesn't contain enough details to fully answer the question, acknowledge the limitations

INFORMATION:
{chunk}

QUESTION: {question}

Provide a helpful, accurate, and concise answer. Begin your response with "De acordo com as informações de {institution} ({url})," or a similar attribution phrase that mentions both the institution and URL.
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
            
        # Split by newlines and filter out empty lines or non-question lines
        questions = []
        for line in question_text.splitlines():
            line = line.strip()
            if not line:
                continue
                
            # Remove any numbering or bullet points
            line = re.sub(r'^[\d\.\-\*]+\s*', '', line)
            # Remove markdown formatting if present
            line = re.sub(r'\*+', '', line)
            
            # Ensure it's a question (has a question mark)
            if '?' in line:
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
        Generate question-answer pairs from a text chunk.
        
        Args:
            chunk: The text chunk to generate QA pairs from
            source_path: File path of the source content
            output_dir: Directory to save generated questions and answers
            chunk_hash: A hash of the chunk for file naming
            is_full_document: Whether this chunk represents a complete document with multiple sections
            is_faq: Whether this chunk is from an FAQ document
            
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
            "file_path": source_path
        }
        
        # Create output directories if they don't exist
        qa_dir = output_dir / "qa_pairs"
        debug_dir = output_dir / "debug"
        qa_dir.mkdir(parents=True, exist_ok=True)
        debug_dir.mkdir(parents=True, exist_ok=True)
        
        # Get number of iterations based on document type
        if is_faq:
            iterations = self.config.get("iterations", {}).get("faq_document", 2)
        else:
            iterations = self.config.get("iterations", {}).get("regular_document", 3)
            
        logger.info(f"Generating with {iterations} iterations for {'FAQ' if is_faq else 'regular'} document")
        
        # Generate questions with different styles
        all_qa_pairs = []
        
        # Get style variations from config
        writing_styles = self.writing_styles if self.writing_styles else [{"name": "Default", "description": ""}]
        question_types = self.question_types if self.question_types else [{"name": "Default", "description": ""}]
        
        # For each iteration, generate all combinations of writing styles and question types
        for iteration in range(iterations):
            for writing_style in writing_styles:
                writing_style_name = writing_style.get("name")
                
                for question_type in question_types:
                    question_type_name = question_type.get("name")
                    
                    style_hash = f"{writing_style_name}_{question_type_name}_{iteration}"
                    questions_path = qa_dir / f"questions_{chunk_hash}_{style_hash}.txt"
                    q_debug_path = debug_dir / f"q_debug_{chunk_hash}_{style_hash}.txt"
                    
                    # Generate or load questions for this style combination
                    if questions_path.exists():
                        logger.info(f"Using existing questions file: {questions_path}")
                        question_list_text = questions_path.read_text(encoding="utf-8")
                    else:
                        # Generate questions with this style
                        question_prompt = self.get_question_prompt(
                            chunk, 
                            source_info, 
                            is_full_document=is_full_document,
                            writing_style=writing_style_name if writing_style_name != "Default" else None,
                            question_type=question_type_name if question_type_name != "Default" else None
                        )
                        
                        # Use lower temperature for factual mode
                        temperature = 0.3 if self.factual_mode else 0.7
                        question_list_text = self.question_client.generate_text(
                            question_prompt, 
                            temperature=temperature
                        )
                        
                        if not question_list_text:
                            logger.error(f"Failed to generate questions for chunk {chunk_hash} with style {style_hash}")
                            continue
                            
                        # Save outputs
                        questions_path.write_text(question_list_text, encoding="utf-8")
                        q_debug_path.write_text(question_prompt, encoding="utf-8")
                        
                    # Parse questions
                    questions = self.parse_questions(question_list_text)
                    if not questions:
                        logger.warning(f"No valid questions parsed from output for chunk {chunk_hash} with style {style_hash}")
                        continue
                        
                    # Generate answers for each question
                    for i, question in enumerate(questions, 1):
                        answer_path = qa_dir / f"answer_{chunk_hash}_{style_hash}_q{i}.txt"
                        a_debug_path = debug_dir / f"a_debug_{chunk_hash}_{style_hash}_q{i}.txt"
                        
                        if answer_path.exists():
                            logger.info(f"Using existing answer file: {answer_path}")
                            answer = answer_path.read_text(encoding="utf-8")
                        else:
                            # Generate answer
                            answer_prompt = self.get_answer_prompt(question, chunk, source_info)
                            # Use lower temperature for factual mode
                            temperature = 0.2 if self.factual_mode else 0.5
                            answer = self.answer_client.generate_text(
                                answer_prompt,
                                temperature=temperature
                            )
                            
                            if not answer:
                                logger.error(f"Failed to generate answer for question {i} in chunk {chunk_hash} with style {style_hash}")
                                continue
                                
                            # Save outputs
                            answer_path.write_text(answer, encoding="utf-8")
                            a_debug_path.write_text(answer_prompt, encoding="utf-8")
                            
                        # Add to QA pairs
                        qa_pair = {
                            "question": question,
                            "answer": answer,
                            "source": source_path,
                            "url": url,
                            "domain": domain,
                            "institution": institution,
                            "chunk_hash": f"{chunk_hash}_{style_hash}_{i}",
                            "writing_style": writing_style_name,
                            "question_type": question_type_name,
                            "iteration": iteration
                        }
                        all_qa_pairs.append(qa_pair)
        
        return all_qa_pairs