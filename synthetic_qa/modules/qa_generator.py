"""
QA Generator module for the Synthetic QA Generator.

This module handles generating questions and answers from text chunks.
"""

import re
import logging
from pathlib import Path
from typing import Dict, Any, List

from .llm_client import LLMClient

logger = logging.getLogger(__name__)


class QAGenerator:
    """Handles generation of questions and answers from text chunks."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize with provided configuration."""
        self.config = config
        self.question_client = LLMClient(config.get("providers", {}).get("question", {}))
        self.answer_client = LLMClient(config.get("providers", {}).get("answer", {}))
        self.factual_mode = config.get("factual_mode", False)
        
    def get_question_prompt(self, chunk: str, source_info: str, is_full_document: bool = False) -> str:
        """
        Create a prompt for generating questions from a text chunk.
        
        Args:
            chunk: The text chunk to generate questions from
            source_info: Information about the source of the text
            is_full_document: Whether this chunk represents a complete document with multiple sections
            
        Returns:
            A formatted prompt string
        """
        if is_full_document:
            # Special prompt for full documents with multiple sections
            return f"""
                You are generating training examples for a university chatbot that MUST provide factually accurate information.

                INSTRUCTION:
                1. Generate questions ONLY about information explicitly mentioned in the provided document
                2. Create diverse questions covering DIFFERENT SECTIONS of the document
                3. Include specific details mentioned in various sections
                4. Include at least one question about each major section or topic
                5. Generate questions that will be useful for university students, faculty, and staff
                6. Focus on questions that have clear, factual answers based on the text

                INFORMATION SOURCE: {source_info}

                INFORMATION:
                {chunk}

                Generate 6-10 diverse questions that:
                - Can be answered DIRECTLY from the provided content
                - Represent how real humans would naturally ask about this information
                - Cover different sections and aspects throughout the ENTIRE document
                - Would be relevant to an institutional chatbot

                Return ONLY the questions, one per line, with no numbering or formatting.
                """
        elif self.factual_mode:
            return f"""
                You are generating training examples for a university chatbot that MUST provide factually accurate information.

                INSTRUCTION:
                1. Generate questions ONLY about information explicitly mentioned in the provided document
                2. Do not create questions that require inference beyond what is directly stated
                3. Include specific details like dates, requirements, and rule numbers in your questions
                4. Generate questions that will be useful for university students, faculty, and staff
                5. Focus on questions that have clear, factual answers based on the text

                INFORMATION SOURCE: {source_info}

                INFORMATION:
                {chunk}

                Generate 3-5 diverse questions that:
                - Can be answered DIRECTLY from the provided content
                - Represent how real humans would naturally ask about this information
                - Cover different important aspects of the information provided
                - Would be relevant to an institutional chatbot

                Return ONLY the questions, one per line, with no numbering or formatting.
                """
        else:
            return f"""
                You are an expert at creating training data for language models. Your task is to generate diverse and natural questions 
                that a student, faculty member, or visitor might ask about the following information from {source_info}.

                INFORMATION:
                {chunk}

                Generate 3-5 diverse questions that:
                1. Cover different aspects of the information provided
                2. Range from factual to conceptual understanding
                3. Represent how real humans would naturally ask about this information
                4. Are specific enough to be answerable from the provided information

                Return ONLY the questions, one per line, with no numbering or formatting.
                """

    def get_answer_prompt(self, question: str, chunk: str, source_info: str) -> str:
        """
        Create a prompt for generating an answer to a question from a text chunk.
        
        Args:
            question: The question to answer
            chunk: The text chunk containing information to answer the question
            source_info: Information about the source of the text
            
        Returns:
            A formatted prompt string
        """
        if self.factual_mode:
            return f"""
                You are an expert institutional assistant for the University of Brasília. You're answering a question based on 
                information from {source_info}.

                IMPORTANT: Your response must be factually accurate and based ONLY on the information provided.
                - If the information doesn't contain enough details to fully answer the question, acknowledge the limitations
                - Include specific details, numbers, and requirements exactly as stated in the document
                - Do not make assumptions or inferences beyond what is directly stated
                - If different parts of the document appear to conflict, acknowledge the ambiguity in your answer
                - If referring to specific sections, rules, or article numbers, cite them explicitly

                INFORMATION:
                {chunk}

                QUESTION: {question}

                Provide a helpful, accurate, and strictly factual answer based ONLY on the information provided above.
                """
        else:
            return f"""
                You are an expert institutional assistant for the University of Brasília. You're answering a question based on 
                information from {source_info}. 

                INFORMATION:
                {chunk}

                QUESTION: {question}

                Provide a helpful, accurate, and concise answer based ONLY on the information provided above. 
                If the information doesn't contain enough details to fully answer the question, acknowledge the limitations 
                of what you know from the provided context.
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
                          source_info: str, 
                          output_dir: Path,
                          chunk_hash: str,
                          is_full_document: bool = False) -> List[Dict[str, str]]:
        """
        Generate question-answer pairs from a text chunk.
        
        Args:
            chunk: The text chunk to generate QA pairs from
            source_info: Information about the source of the text
            output_dir: Directory to save generated questions and answers
            chunk_hash: A hash of the chunk for file naming
            is_full_document: Whether this chunk represents a complete document with multiple sections
            
        Returns:
            A list of dictionaries containing questions and answers
        """
        # Create output directories if they don't exist
        qa_dir = output_dir / "qa_pairs"
        debug_dir = output_dir / "debug"
        qa_dir.mkdir(parents=True, exist_ok=True)
        debug_dir.mkdir(parents=True, exist_ok=True)
        
        # File paths for caching and debug
        questions_path = qa_dir / f"questions_{chunk_hash}.txt"
        q_debug_path = debug_dir / f"q_debug_{chunk_hash}.txt"
        
        # Generate or load questions
        if questions_path.exists():
            logger.info(f"Using existing questions file: {questions_path}")
            question_list_text = questions_path.read_text(encoding="utf-8")
        else:
            # Generate questions
            question_prompt = self.get_question_prompt(chunk, source_info, is_full_document=is_full_document)
            # Use lower temperature for factual mode
            temperature = 0.3 if self.factual_mode else None
            question_list_text = self.question_client.generate_text(
                question_prompt, 
                temperature=temperature
            )
            # TODO: remove this
            question_list_text = "Quais são os requisitos para participar do Programa ANDIFES de Mobilidade Acadêmica?\nComo posso obter mais informações sobre o Programa de Mobilidade Acadêmica Internacional da UnB?\nO que é o Programa CAPES/Brafitec e como ele funciona para alunos de Engenharia?\nQuais são as condições para um estudante realizar a mobilidade acadêmica por até 2 semestres em outra IFES do Brasil?\nOnde posso encontrar a lista das universidades participantes do Programa ANDIFES de Mobilidade Acadêmica?"

            if not question_list_text:
                logger.error(f"Failed to generate questions for chunk {chunk_hash}")
                return []
                
            # Save outputs
            questions_path.write_text(question_list_text, encoding="utf-8")
            q_debug_path.write_text(question_prompt, encoding="utf-8")
            
        # Parse questions
        questions = self.parse_questions(question_list_text)
        if not questions:
            logger.warning(f"No valid questions parsed from output for chunk {chunk_hash}")
            return []
            
        # Generate answers for each question
        qa_pairs = []
        for i, question in enumerate(questions, 1):
            answer_path = qa_dir / f"answer_{chunk_hash}_q{i}.txt"
            a_debug_path = debug_dir / f"a_debug_{chunk_hash}_q{i}.txt"
            
            if answer_path.exists():
                logger.info(f"Using existing answer file: {answer_path}")
                answer = answer_path.read_text(encoding="utf-8")
            else:
                # Generate answer
                answer_prompt = self.get_answer_prompt(question, chunk, source_info)
                # Use lower temperature for factual mode
                temperature = 0.2 if self.factual_mode else None
                answer = self.answer_client.generate_text(
                    answer_prompt,
                    temperature=temperature
                )
                # TODO: remove this
                answer = "## Requisitos para participar do Programa ANDIFES de Mobilidade Acadêmica\n1. Estar regularmente matriculado em um curso de graduação de uma Universidade Federal.\n2. Ter concluído pelo menos 20% da carga horária de integralização do curso de origem.\n3. Ter no máximo 2 reprovações acumuladas nos dois períodos letivos que antecedem o pedido de mobilidade."

                if not answer:
                    logger.error(f"Failed to generate answer for question {i} in chunk {chunk_hash}")
                    continue
                    
                # Save outputs
                answer_path.write_text(answer, encoding="utf-8")
                a_debug_path.write_text(answer_prompt, encoding="utf-8")
                
            # Add to QA pairs
            qa_pairs.append({
                "question": question,
                "answer": answer,
                "source": source_info,
                "chunk_hash": chunk_hash
            })
            
            break   # TODO: remove this
            
        return qa_pairs