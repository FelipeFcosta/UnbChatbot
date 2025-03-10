"""
FAQ processing module for the Synthetic QA Generator.

This module handles detection and processing of FAQ documents.
"""

import hashlib
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Any
from bs4 import BeautifulSoup

from .llm_client import LLMClient

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
                                   ['como', 'qual', 'quais', 'o que', 'onde', 'quando', 'por que']):
                questions_count += 1
        
        if questions_count > 3:
            return True
            
        return False
    
    @staticmethod
    def extract_qa_pairs(soup: BeautifulSoup) -> List[Tuple[str, str]]:
        """
        Extract question-answer pairs based on document structure.
        
        Args:
            soup: BeautifulSoup object of the document
            
        Returns:
            List of (question, answer) tuples
        """
        qa_pairs = []
        
        # Try details/summary structure
        if soup.find_all('details') and soup.find_all('summary'):
            for details in soup.find_all('details'):
                summary = details.find('summary')
                if summary:
                    question = summary.get_text().strip()
                    # Clone details and remove summary to get just answer text
                    details_copy = BeautifulSoup(str(details), 'html.parser')
                    if details_copy.summary:
                        details_copy.summary.decompose()
                    answer = details_copy.get_text().strip()
                    if question and answer:
                        qa_pairs.append((question, answer))
        
        # If no pairs found, try bold/strong text pattern
        if not qa_pairs:
            bold_elements = soup.find_all(['b', 'strong'])
            for bold in bold_elements:
                # Check if this looks like a question
                question_text = bold.get_text().strip()
                if (question_text.endswith('?') or 
                    any(question_text.lower().startswith(word) for word in 
                       ['como', 'qual', 'quais', 'o que', 'onde', 'quando', 'por que'])):
                    # Look for answer text
                    answer_elements = []
                    next_elem = bold.find_parent().find_next_sibling()
                    while next_elem and not next_elem.find(['b', 'strong']):
                        answer_elements.append(next_elem.get_text().strip())
                        next_elem = next_elem.find_next_sibling()
                    
                    if answer_elements:
                        answer = ' '.join(answer_elements)
                        qa_pairs.append((question_text, answer))
        
        # Try dt/dd pattern (definition lists)
        if not qa_pairs:
            dts = soup.find_all('dt')
            for dt in dts:
                question_text = dt.get_text().strip()
                # Check next element is a dd
                dd = dt.find_next_sibling('dd')
                if dd:
                    answer_text = dd.get_text().strip()
                    if question_text and answer_text:
                        qa_pairs.append((question_text, answer_text))
        
        # Try h3/p pattern (common in manually formatted FAQs)
        if not qa_pairs:
            for heading in soup.find_all(['h3', 'h4']):
                question_text = heading.get_text().strip()
                if question_text.endswith('?'):
                    answer_elements = []
                    next_elem = heading.find_next_sibling()
                    while next_elem and next_elem.name not in ['h3', 'h4']:
                        if next_elem.name == 'p':
                            answer_elements.append(next_elem.get_text().strip())
                        next_elem = next_elem.find_next_sibling()
                    
                    if answer_elements:
                        answer = ' '.join(answer_elements)
                        qa_pairs.append((question_text, answer))
        
        return qa_pairs
    
    @staticmethod
    def create_context_chunks(qa_pairs: List[Tuple[str, str]], context_size: int = 2) -> List[Tuple[str, str, str]]:
        """
        Create chunks with surrounding questions for context.
        
        Args:
            qa_pairs: List of (question, answer) tuples
            context_size: Number of neighboring QA pairs to include as context
            
        Returns:
            List of (question, answer, context) tuples
        """
        contextual_chunks = []
        
        for i, (question, answer) in enumerate(qa_pairs):
            # Get neighboring questions for context
            start_idx = max(0, i - context_size)
            end_idx = min(len(qa_pairs), i + context_size + 1)
            
            context = "FAQ Section:\n\n"
            for j in range(start_idx, end_idx):
                q, a = qa_pairs[j]
                if j == i:
                    context += f"FOCUS QUESTION:\n{q}\n\nFOCUS ANSWER:\n{a}\n\n"
                else:
                    context += f"Context Question:\n{q}\n\nContext Answer:\n{a}\n\n"
            
            contextual_chunks.append((question, answer, context))
        
        return contextual_chunks


def generate_rephrased_questions(question: str, answer: str, context: str, llm_client: LLMClient) -> List[str]:
    """
    Generate multiple rephrased versions of the original question.
    
    Args:
        question: The original question
        answer: The answer to the question
        context: The context including neighboring QA pairs
        llm_client: LLM client for generation
        
    Returns:
        List of rephrased questions
    """
    prompt = f"""
    I'm creating training data for an institutional chatbot for a university. Please help me rephrase this question in many different ways:
    
    ORIGINAL QUESTION: {question}
    
    ANSWER: {answer}
    
    Generate as many rephrased versions of this question as you can think of. They should:
    1. Keep the exact same meaning
    2. Vary in formality, structure, and wording
    3. Include both direct and indirect question forms
    4. Use different vocabulary while maintaining the intent
    5. Be natural expressions someone might actually use
    
    Don't limit yourself to a specific number. Generate as many coherent, distinct variations as possible.
    
    Return ONLY the rephrased questions, one per line, with no numbering, explanations, or other text.
    """
    
    response = llm_client.generate_text(prompt, temperature=0.7)
    if not response:
        return []
    
    # Parse response to get just the questions
    rephrased = []
    for line in response.split('\n'):
        line = line.strip()
        if line and ('?' in line or any(line.lower().startswith(word) for word in 
                    ['como', 'qual', 'quais', 'o que', 'onde', 'quando', 'por que'])):
            rephrased.append(line)
    
    return rephrased


def generate_related_questions(question: str, answer: str, context: str, llm_client: LLMClient) -> List[str]:
    """
    Generate additional questions that can be answered by the same content.
    
    Args:
        question: The original question
        answer: The answer to the question
        context: The context including neighboring QA pairs
        llm_client: LLM client for generation
        
    Returns:
        List of related questions
    """
    prompt = f"""
    I'm creating training data for an institutional chatbot for a university. Based on this question and answer pair:
    
    ORIGINAL QUESTION: {question}
    
    ANSWER: {answer}
    
    Generate additional questions that would also be answered by this exact same answer. Consider:
    1. Different aspects mentioned in the answer that weren't explicitly asked about
    2. More specific questions about details in the answer
    3. Questions from different perspectives or use cases
    4. Questions seeking the same information but focusing on different parts
    
    These should be NEW questions, not rephrases of the original question. They must be fully answerable with ONLY the information in the provided answer.
    
    Generate as many relevant questions as possible based on the content.
    
    Return ONLY the questions, one per line, with no numbering, explanations, or other text.
    """
    
    response = llm_client.generate_text(prompt, temperature=0.7)
    if not response:
        return []
    
    # Parse response to get just the questions
    related = []
    for line in response.split('\n'):
        line = line.strip()
        if line and ('?' in line or any(line.lower().startswith(word) for word in 
                    ['como', 'qual', 'quais', 'o que', 'onde', 'quando', 'por que'])):
            related.append(line)
    
    return related


def process_faq_document(file_path: Path, output_dir: Path, config: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Process an HTML document as an FAQ to generate comprehensive training data.
    
    Args:
        file_path: Path to the HTML file
        output_dir: Directory to save output files
        config: Configuration dictionary
        
    Returns:
        List of QA pairs with variations
    """
    from .llm_client import LLMClient
    
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        soup = BeautifulSoup(content, 'html.parser')
        
        # Detect if this is an FAQ document
        is_faq = FAQProcessor.detect_faq_document(soup, file_path.name)
        if not is_faq:
            logger.info(f"{file_path} does not appear to be an FAQ document")
            return []
        
        logger.info(f"Processing {file_path} as an FAQ document")
        
        # Extract the QA pairs
        qa_pairs = FAQProcessor.extract_qa_pairs(soup)
        if not qa_pairs:
            logger.warning(f"No QA pairs extracted from {file_path}")
            return []
        
        logger.info(f"Extracted {len(qa_pairs)} QA pairs from {file_path}")
        
        # Get FAQ specific configuration
        faq_config = config.get("processing", {}).get("faq", {})
        context_size = faq_config.get("context_size", 2)
        generate_rephrased = faq_config.get("generate_rephrased_questions", True)
        generate_related = faq_config.get("generate_related_questions", True)
        max_rephrased = faq_config.get("max_rephrased_questions", 5)
        max_related = faq_config.get("max_related_questions", 3)
        
        # Create context chunks with neighboring questions
        contextual_chunks = FAQProcessor.create_context_chunks(qa_pairs, context_size)
        
        # Initialize LLM clients
        question_client = LLMClient(config.get("providers", {}).get("question", {}))
        
        # Create directories for output
        qa_dir = output_dir / "qa_pairs"
        debug_dir = output_dir / "debug"
        qa_dir.mkdir(parents=True, exist_ok=True)
        debug_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate training examples with variations and related questions
        all_training_examples = []
        
        for i, (question, answer, context) in enumerate(contextual_chunks):
            # Create a unique identifier for this QA pair
            chunk_hash = f"faq_{hashlib.sha256((str(file_path) + question).encode()).hexdigest()[:12]}"
            
            # Save original QA pair
            original_example = {
                "question": question,
                "answer": answer,
                "source": str(file_path),
                "chunk_hash": chunk_hash,
                "type": "original"
            }
            all_training_examples.append(original_example)
            
            # Check if we should generate rephrased questions
            if generate_rephrased:
                rephrased_path = qa_dir / f"rephrased_{chunk_hash}.txt"
                rephrased_debug_path = debug_dir / f"rephrased_debug_{chunk_hash}.txt"
                
                if rephrased_path.exists():
                    # Load existing rephrased questions
                    with open(rephrased_path, 'r', encoding='utf-8') as f:
                        rephrased_questions = [line.strip() for line in f if line.strip()]
                    logger.info(f"Loaded {len(rephrased_questions)} existing rephrased questions for {chunk_hash}")
                else:
                    # Generate rephrased questions
                    rephrased_questions = generate_rephrased_questions(question, answer, context, question_client)
                    
                    # Save rephrased questions and debug info
                    if rephrased_questions:
                        with open(rephrased_path, 'w', encoding='utf-8') as f:
                            f.write('\n'.join(rephrased_questions))
                        
                        with open(rephrased_debug_path, 'w', encoding='utf-8') as f:
                            f.write(f"Original Question: {question}\nAnswer: {answer}\nContext: {context}")
                
                # Limit number of rephrased questions if needed
                if max_rephrased > 0 and len(rephrased_questions) > max_rephrased:
                    rephrased_questions = rephrased_questions[:max_rephrased]
                
                # Add rephrased questions to the training examples
                for rephrased in rephrased_questions:
                    all_training_examples.append({
                        "question": rephrased,
                        "answer": answer,
                        "source": str(file_path),
                        "chunk_hash": chunk_hash,
                        "type": "rephrased"
                    })
            
            # Check if we should generate related questions
            if generate_related:
                related_path = qa_dir / f"related_{chunk_hash}.txt"
                related_debug_path = debug_dir / f"related_debug_{chunk_hash}.txt"
                
                if related_path.exists():
                    # Load existing related questions
                    with open(related_path, 'r', encoding='utf-8') as f:
                        related_questions = [line.strip() for line in f if line.strip()]
                    logger.info(f"Loaded {len(related_questions)} existing related questions for {chunk_hash}")
                else:
                    # Generate related questions
                    related_questions = generate_related_questions(question, answer, context, question_client)
                    
                    # Save related questions and debug info
                    if related_questions:
                        with open(related_path, 'w', encoding='utf-8') as f:
                            f.write('\n'.join(related_questions))
                        
                        with open(related_debug_path, 'w', encoding='utf-8') as f:
                            f.write(f"Original Question: {question}\nAnswer: {answer}\nContext: {context}")
                
                # Limit number of related questions if needed
                if max_related > 0 and len(related_questions) > max_related:
                    related_questions = related_questions[:max_related]
                
                # Add related questions to the training examples
                for related in related_questions:
                    all_training_examples.append({
                        "question": related,
                        "answer": answer,
                        "source": str(file_path),
                        "chunk_hash": chunk_hash,
                        "type": "related"
                    })
        
        return all_training_examples
    
    except Exception as e:
        logger.error(f"Error processing FAQ document {file_path}: {e}")
        return []