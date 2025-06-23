"""
FAQ processing module for the Synthetic QA Generator.

This module handles detection and processing of FAQ documents.
"""

import hashlib
import logging
import json
from pathlib import Path
from typing import List, Dict, Any
from bs4 import BeautifulSoup
import textwrap

from modules.utils import create_hash

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
    def extract_faq_from_text(text: str, file_path: Path, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract question-answer pairs from an FAQ document using LLM processing.
        
        Args:
            text: Text of the FAQ
            file_path: Path to the file
            config: Configuration dictionary
            
        Returns:
            List of dictionaries containing extracted FAQ data
        """
        domain, _, _ = FileProcessor.extract_domain_and_path(file_path)
        rel_path = file_path.relative_to(Path(config["base_dir"]))

        
        try:
            llm_client = LLMClient(config.get("providers", {}).get("faq_extraction", {}))

            prompt = (
                "Extract EVERY question and answer from this university markdown FAQ file and convert them to a structured JSON format. Follow these requirements exactly:\n"
                "\n"
                "1. Output a JSON string with this structure:\n"
                "{\n"
                '"qa_pairs": [\n'
                "    {\n"
                '    "question": FAQ question,\n'
                '    "answer": FAQ answer,\n'
                '    "topics": ["Topic1, Topic2, ..."],\n'
                '    "course": "specific course"\n'
                "    },\n"
                "    // More QA pairs...\n"
                "]\n"
                "}\n"
                "\n"
                "2. **Critical requirements**:\n"
                '- Both "question" and "answer" fields **can never be null**\n'
                '- **"topics" are EXPLICITLY PRESENT in the FAQ AS titles/headers above a QA pair or a number of QA pairs** (can be null if no topics).\n'
                '- **include all nested topics as just a flat array**\n'
                '- If the question lacks an explicit subject or referent (e.g. "O que é?" or "Como funciona?"), rewrite the question to include the correct subject or referent in the question based on the context.\n'
                '- "course" can be null or be a specific course\n'
                "\n"
                "3. Content processing:\n"
                "- Preserve all markdown formatting and links.\n"
                "- Detect question-answer pairs intelligently (tip: QA are in different hierarchical markdown levels)\n"
                "- **DO NOT ALTER OR ADD CONTENT** except to fix clear formatting issues\n"
                "- Every text should be verbatim to the original FAQ\n"
                "\n"
                f"Available courses: {FileProcessor.INSTITUTION_COURSES.get(domain, 'All Courses')}\n"
                "\n"
                "**IMPORTANT**: You MUST specify a particular course (from the available courses list) if ANY of these conditions is met:\n"
                "1. The original question explicitly mentions that specific course\n"
                "2. The **question topic explicitly mentions a specific course** (pay careful attention to the topics field)\n"
                "3. The document structure as a whole is about a specific course (in which case all QA pairs will have this course)\n"
                "\n"
                "After the approach explanation, return the **complete**, well-formed JSON with all extracted QA pairs from the university FAQ content below:\n"
                f"{text}"
            )

            # Call the LLM to extract QA pairs
            logger.info(f"Requesting LLM-based QA extraction for file {file_path.name}...")
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
                + "\n\n YOUR TASK:"
                + "\nRewrite your output (json only). Do any of these corrections for every pair where mistakes were made:\n"
                "- Remove any questions in the 'topics' field.\n"
                "- Remove any answer in the 'question' field.\n"
                "- Remove any question in the 'answer' field.\n"
                "- Add/remove a course if it was forgotten/misplaced (available: " + FileProcessor.INSTITUTION_COURSES.get(domain, 'All Courses') + ").\n"
                "- Remove any unwanted out of place newlines (\\n) in the middle of a text if present.\n"
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
                            "qa_pair_hash": create_hash(str(rel_path) + question)
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

