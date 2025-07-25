# qa_processor_raft.py

import os
import logging
import json
from pathlib import Path
from typing import List, Tuple, Dict, Any
from bs4 import BeautifulSoup
from tqdm import tqdm
import random
import time
from slugify import slugify
import requests

from modules.utils import create_hash, FileType

from .llm_client import LLMClient
from .file_processor import FileProcessor
from .multi_hop_qa_generator import MultiHopQAGenerator

logger = logging.getLogger(__name__)

METADATA_TAG = "<doc_metadata>"
METADATA_TAG_END = "</doc_metadata>"

# --- Constants for RAFT ---
# Prompt for generating the CoT Answer (A*) based ONLY on the Golden Answer (D*)
CITATION_SYSTEM_PROMPT = """---
### **Citation System**

Citations add credibility to the answer.

**Citation Format (Critical):**
You may optionally include a short quote from the context as a blockquote to support your statement. The source link *[{file_title}]({file_url})* must appear only once in the entire answer.

> *"...Relevant excerpt from context..."*
> *[{file_title}]({file_url})*
> *Rest of the answer...*

**Citation Rules:**
*   **CRITICAL:** A quote **must never** be placed at the very end of the answer. It should be placed where it supports a specific statement.
*   **CRITICAL:** The quote must contribute meaningfully to the answer and not be redundant with information you've already stated.
*   The quote must be relevant, add value, and be as short as possible. Use [...] to omit irrelevant parts.
*   If a quote does not add value, do not include it.
*   If a blockquote is **NOT** present, you must still include the source link. Place the *italicized* citation *[{file_title}]({file_url})* after the phrase that most heavily relies on the source information. The source link itself should be the only italicized link in the answer.

---"""


COT_ANSWER_GENERATION_PROMPT = f"""You are an expert assistant creating training data for a university chatbot at the University of Brasília (UnB).

### Your Task

Given a question and its corresponding context (from the {{context_source_name}}), your primary task is to generate a detailed chat response. This response is composed of **TWO mandatory parts**:

1.  **Internal Reasoning (in English)** - An internal analysis that will not be shown to the user.
2.  **Final Answer (in Portuguese)** - The public-facing response the student will see.

---

### **Instructions**

#### **Step 1: Understand the User's Need**

First, you must understand the user's implicit need based on the question.

#### **Step 2: Construct the Internal Reasoning**

This reasoning is for training purposes and must be focused on achieving factual accuracy. It must be written in English and enclosed within <REASON>...</REASON> tags.

#### **Rules for Reasoning:**
*   **A critical requirement:** Your reasoning must be based **exclusively** on the provided context. Do not add any outside information.
*   **Keep your reasoning SHORT and CONCISE/OBJECTIVE.**
*   Provide a logical explanation of how you will arrive at the answer using the given {{context_source_name}}. Analyze how to address the question using the provided text.
*   You **MUST** reference the relevant context text during your reasoning. When you do, you **must always** enclose the text in <quote>verbatim text</quote> tags.
*   This reasoning section is **purely** about how to answer the question factually. It is critical that you **do not** include anything about formatting rules, or any other instructions in this prompt.

##### **IMPLICIT Identification of the entity/subject(s) of the question**
*   While reasoning, you MUST quote EITHER a chunk text OR a specific metadata field naturally to confirm that the question's subject(s) or entity(s) exactly matches the one(s) in the context.
*   The implicit verification should be included naturally as the evidence appears in the context while you are reasoning and quoting the document.
*   **Do not** include multiple sources of evidence for confirmation of the correct entity/subject(s).

#### **Step 3: Write the Final Answer**

This is the student-facing answer. It must be written in Portuguese and enclosed within <ANSWER>...</ANSWER> tags.

**Rules for the Answer:**
*   You must answer the question directly first.
*   The answer must be **self-contained**. The student cannot see the context document or your reasoning, so the answer must not refer back to "the context" or "the document."
*   Ensure all relevant information from the context needed to address the Question is included. You must never invent or extrapolate information not found in the context.
*   Use a friendly, formal, and modern tone suitable for UnB students.
*   Format the answer with clear markdown for easy reading.
*   Do not include greetings (intros or outros) unless the user greets you first.
*   Include any necessary URLs from the context as markdown links: [Link Text](URL).
*   **DO NOT** acknowledge the existence of the context (document) to the user in any way (they don't know about the provided context).

{{citation_system_prompt}}

### **Edge Cases**
*   If the question is about pre-requisites, co-requisites, or equivalences of a discipline, consider the correct logic of OR (OU) and AND (E) operators when presenting the information.

#### **Unanswerable or missing Information**
*   If the context lacks the information to answer the question: State this in your <REASON> section.
*   In the <ANSWER> section, politely state that you do not have the information to properly answer the question (or correct the user's mistake without acknowledging the context existence) and **NOTHING MORE**.
*   **CRITICAL**: End the answer immediately after stating that you do not have the information to properly answer the question. DO NOT answer about what information you do have or give a source, unless for correcting/clarifying the user's obvious mistakes (then cite the source).

---

### **CRITICAL: Pre-Output Validation**

1.  First, make a draft (in your thinking process) of your complete <REASON> and <ANSWER> sections.
2.  Review your draft against **ALL** instructions and rules (especially syntactical ones) before outputting.
3.  Verify: Does this follow every rule? Are there any violations?
4.  Only output the final <REASON> and <ANSWER> after confirming full compliance internally.

---

### **Input**

*   **Question:** "{{original_question}}"
*   **{{context_source_name}}:** "{{context_content}}"
{METADATA_TAG}
*   **URL:** "[{{file_title}}]({{file_url}})"
*   **File Name:** "{{file_name}}"
*   **Topic:** "{{topic}}"
{{professor_str}}
{{course_str}}
{METADATA_TAG_END}

---

### **Required Output Format**


<REASON>your reasoning <quote>verbatim texts if needed</quote> your reasoning</REASON>
<ANSWER>resposta em português</ANSWER>"""

# Prompt for generating component styled questions directly from component text
COMPONENT_STYLED_QUESTION_GENERATION_PROMPT = """
You are an LLM Generator creating synthetic data for a university chatbot.

The chatbot can answer any question about the university, but the following questions are examples where the student's question required information from this component document (retrieved by a RAG system).

The user is a **student who does not know about the present document**, so the question must be specific enough (but not contrived) for the document to be retrieved.

You will receive the Markdown text of a university component, including code, name, syllabus (ementa), objectives, bibliography, and offerings (teachers, schedules, vacancies, location, etc).

**WRITING STYLE**: {style_name}
- Description: {style_description}
- Goal: {style_goal}

**Instructions:**
- Generate ONE natural, realistic, relevant, and useful question IN PORTUGUESE that a brazilian student might ask about this component.
- Use ONLY the information present in the component text.
- DO NOT reference the document directly.
- DO NOT generate a question about missing information.
- Follow the specified writing style closely.
- The question should be specific enough to retrieve this document but natural for a student to ask.
- The student probably does not know the component code, and if they mention it, it will most likely be by the component's name.
- IMPORTANT: Make the question different from previous questions to cover a different aspect of the component.

**Previous questions generated for this component:**
{previous_questions_str}

**Component text:**
{component_text}

**Output Format:**
Return ONLY the question in Portuguese, no other text, numbering, or explanations.
"""

UNANSWERABLE_QUESTION_GENERATION_PROMPT = """
You are an LLM Generator creating synthetic data for the University of Brasilia chatbot.

**GOAL:** Train the LLM to recognize when it lacks sufficient information to answer a question and respond appropriately with "I don't know" or "I can't answer that based on the information I have" instead of hallucinating.

**The new question should resemble the original question provided.**

**METADATA INFORMATION:**
- File Title: {file_title}
- File Name: {file_name}
- Original Question: {original_question}

**Instructions:**
- Rewrite the original question so that it contains one (or more) replaced section (replacing entities/subjects in the original question with wrong/misleading information).
- The replaced section should be plausible but obviously **wrong** (can't be answered by any other context).
- The new question's length should be more or less the same as the original question.
- The new question's format should be the similar to the original question but use different wording (and make it natural for a student to ask).
- The unanswerable question should be natural (not contrived just to be wrong).
- **The question should not be complex or too specific!**
- The question should be a single sentence, with no statements. Only a simple question with an honest mistake.
- Use natural, conversational Portuguese as a Brazilian student might ask.

The user is a student who does not know about the present document, so the question must be specific enough (but not contrived) for the document to be retrieved (by the RAG system)

**Context/Chunk:**
{context_content}

**Output Format:**
Return ONLY the single unanswerable question IN PORTUGUESE.
"""

# Prompt for generating a styled question (Q) based on an original pair and a style
# (Adapted from your original generate_styled_qa)
STYLED_QUESTION_GENERATION_PROMPT_TEMPLATE = """
You are an LLM Generator creating synthetic data for a university chatbot.
Create ONE alternative FAQ question based on the Original Pair provided below.

**WRITING STYLE**: {style_name}
- Description: {style_description}
- Goal: {style_goal}

**METADATA INFORMATION:**
- File Title: {file_title}
- File Name: {file_name}

**Instructions:**
- Rewrite *only the question*, preserving the **exact original meaning and intent**.
- The new question should be a natural question that a brazilian student might ask about this component.
- **DO NOT ADD ANY NEW INFORMATION** that is not present in the answer.
- Follow the specified writing style closely.
- Do not add any intro or greetings to the question.
- DO NOT reference the document or chunk directly (because when the user is asking a question it doesn't know about it).
- The user knows they are talking to an assistant chatbot.
- Do not write a question about any UI element (like navigation, footer, header, image captions, etc).
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
        file_title: str,
        file_name: str,
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
            previous_questions_prompt = "- The new question should be distinct from the previous styled questions (do different phrasings, coherent reorderings, *or focus on a different part of the answer!*)."
            previous_questions_str = "\n".join([f"- {pq}" for pq in previous_styled_questions])

        prompt = STYLED_QUESTION_GENERATION_PROMPT_TEMPLATE.format(
            style_name=style_name,
            style_description=style_desc,
            style_goal=style_goal,
            file_title=file_title,
            file_name=file_name,
            original_question=original_question,
            original_answer=original_answer,
            previous_questions_prompt=previous_questions_prompt,
            previous_questions_str=previous_questions_str,
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
        is_unanswerable: bool,
        llm_client: LLMClient,
        file_type: FileType = FileType.REGULAR,
        file_title: str = "",
        file_name: str = "",
        file_url: str = ""
    ) -> str | None:
        """Generates the Chain of Thought Answer based only on the original Q/A pair or component text."""

        try:
            # Use file_type to determine context source name
            context_source_name = "chunk" if chunk else "Original Answer"
            context_content = chunk['chunk'] if chunk else original_answer
            topic = chunk.get("topic", "") if chunk else ""
            professor_str = f'*   **Professor:** "{chunk.get("professor", "")}"' if chunk and chunk.get("professor") else ""
            course_str = f'*   **Course:** "{chunk.get("course", "")}"' if chunk and chunk.get("course") else ""

            citation_system_prompt = CITATION_SYSTEM_PROMPT.format(file_title=file_title, file_url=file_url) if not is_unanswerable else ""

            prompt = COT_ANSWER_GENERATION_PROMPT.format(
                original_question=original_question,
                context_source_name=context_source_name,
                context_content=context_content,
                topic=topic,
                professor_str=professor_str,
                course_str=course_str,
                file_title=file_title,
                file_name=file_name,
                file_url=file_url,
                citation_system_prompt=citation_system_prompt
            )

            response = llm_client.generate_text(prompt.lstrip(), temperature=0.7)
            if response and "<ANSWER>" in response and "<REASON>" in response:
                # check if </quote> is not present in the reasoning
                if "</quote>" not in response and not is_unanswerable:
                    logger.warning("LLM returned response without </quote> in the reasoning. This is not allowed.")
                    return None
                return response.strip() 
            else:
                logger.warning("LLM returned empty or invalid response for CoT answer generation (missing <ANSWER> and <REASON> tags).")
                return None
        except Exception as e:
            logger.error(f"Error generating CoT answer: {e}", exc_info=True)
            return None

    @staticmethod
    def generate_unanswerable_question_raft(
        chunk: Dict[str, Any],
        original_question: str,
        original_answer: str,
        file_title: str,
        file_name: str,
        llm_client: LLMClient
    ) -> str | None:
        """Generates an unanswerable question based on the chunk/context only."""

        if chunk:
            topic_str = f'Topic: "{chunk.get("topic")}"' if chunk.get("topic") else ''
            file_title_str = f', File Title: "{file_title}"' if file_title else ''
            professor_str = f', Professor: "{chunk.get("professor")}"' if chunk.get("professor") else ''
            course_str = f', Course: "{chunk.get("course")}"' if chunk.get("course") else ''
            context_content = chunk['chunk'] + '\n' + topic_str + file_title_str + professor_str + course_str
        else:
            file_title_str = f'File Title: "{file_title}", ' if file_title else ''
            context_content = original_answer + '\n' + file_title_str

        prompt = UNANSWERABLE_QUESTION_GENERATION_PROMPT.format(
            file_title=file_title,
            file_name=file_name,
            original_question=original_question,
            context_content=context_content
        )

        try:
            response = llm_client.generate_text(prompt.lstrip(), temperature=1.0)
            if response:
                unanswerable_q = response.strip().lstrip('*- ').splitlines()[0].strip()
                return unanswerable_q
            else:
                logger.warning("LLM returned empty response for unanswerable question generation")
                return None
        except Exception as e:
            logger.error(f"Error generating unanswerable question: {e}", exc_info=True)
            return None

    @staticmethod
    def format_doc(doc: Dict[str, Any]) -> str:
        """Convert a chunk/FAQ dict into the RAFT string representation used in <DOCUMENT> blocks.
        This centralises what was previously duplicated in several branches.
        """
        # Determine if we have a chunk or a FAQ pair
        if "chunk" in doc:  # regular/chunk document
            topic_str = f'Topic: "{doc.get("topic")}", ' if doc.get("topic") else ''
            professor_str = f'Professor: "{doc.get("professor")}", ' if doc.get("professor") else ''
            course_str = f'Course: "{doc.get("course")}", ' if doc.get("course") else ''
            file_type = doc.get("file_type")
            filename_str = '' if str(file_type) == str(FileType.COMPONENT) else f'File: "{doc.get("file_name", "")}", '
            is_html = str(doc.get("file_name", "")).lower().endswith((".html", ".htm"))
            if is_html:
                url_str = f'URL: "[{doc.get("file_title")}]({doc.get("file_url")})"'
            else:
                url_str = f'URLs: "{doc.get("source_page_url")} [{doc.get("file_title")}]({doc.get("file_url")})"'
            return f'"{doc.get("chunk")}"{METADATA_TAG}{topic_str}{professor_str}{course_str}{filename_str}{url_str}{METADATA_TAG_END}'
        else:  # FAQ document
            topic_list = doc.get("topic", []) or doc.get("topics", [])
            if isinstance(topic_list, str):
                topic_list = [topic_list]
            topic_str = f'Topic: "{", ".join(topic_list)}", ' if topic_list else ''
            course_str = f'Course: "{doc.get("course", "")}", ' if doc.get("course") else ''
            filename_str = f'File: "{doc.get("file_name", "")}", '
            url_str = f'URL: "[{doc.get("file_title")}]({doc.get("file_url")})"'
            return f'Q: "{doc.get("question")}", A: "{doc.get("answer")}"{METADATA_TAG}{topic_str}{course_str}{filename_str}{url_str}{METADATA_TAG_END}'

    @staticmethod
    def generate_raft_training_data(
        files: List[Tuple[BeautifulSoup, Path, Path, FileType]],
        output_dir: Path,
        config: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Generates RAFT training examples from default QA pairs, using extracted_faq or extracted_chunks as answer/distractor sources.
        Args:
            files: List of tuples containing (soup, file_path, rel_path, file_type)
            output_dir: Directory to save intermediate generated files
            config: Configuration dictionary
        Returns:
            List of dictionaries, each formatted as a RAFT training example
            (including the 'messages' key for fine-tuning).
        """

        all_training_examples = []
        
        if files:
            logger.info(f"Processing {len(files)} files for RAFT training.")
            
            # Create directories for output
            default_qa_dir = output_dir / "default_qa"
            extracted_faq_dir = output_dir / "extracted_faq"
            extracted_chunks_dir = output_dir / "extracted_chunks"

            default_qa_dir.mkdir(parents=True, exist_ok=True)
            extracted_faq_dir.mkdir(parents=True, exist_ok=True)
            extracted_chunks_dir.mkdir(parents=True, exist_ok=True)

            debug_dir = output_dir / "debug" / "qa_pairs"
            debug_dir.mkdir(parents=True, exist_ok=True)
            multihop_debug_dir = output_dir / "debug" / "multi_hop"
            multihop_debug_dir.mkdir(parents=True, exist_ok=True) 
            
            raft_qa_dir = output_dir / "qa_pairs_raft"
            raft_qa_dir.mkdir(parents=True, exist_ok=True)

            # Directory dedicated to cached distractor lists
            distractors_dir = output_dir / "distractors"
            distractors_dir.mkdir(parents=True, exist_ok=True)

            final_default_qa = []
            final_extracted_chunks = []
            contexts = []
            formatted_contexts = []

            for soup, file_path, rel_path, file_type in tqdm(files, desc="Loading file data for RAFT processing"):
                file_title = file_path.stem
                if soup and soup.title:
                    title_text = soup.title.get_text(strip=True)
                    if (len(title_text) > 0):
                        file_title = title_text
                safe_title_slug = slugify(file_title)

                file_hash = create_hash(str(rel_path))
                default_qa_path = default_qa_dir / f"default_{safe_title_slug}_{file_hash}.json"
                extracted_faq_path = extracted_faq_dir / f"{safe_title_slug}_{file_hash}.json"
                extracted_chunks_path = extracted_chunks_dir / f"{safe_title_slug}_{file_hash}.json"

                # Load default QA (baseline)
                if not os.path.exists(default_qa_path):
                    logger.warning(f"Default QA file not found: {default_qa_path}")
                    continue
                with open(default_qa_path, 'r', encoding='utf-8') as f:
                    default_qa = json.load(f)

                try:
                    file_url = os.getxattr(str(file_path), b'user.original_url').decode('utf-8')
                except Exception as e:
                    file_url = FileProcessor.extract_domain_and_path(file_path)[2]

                # Simple: retrieve source_page_url if available
                try:
                    source_page_url = os.getxattr(str(file_path), b'user.source_page_url').decode('utf-8')
                except Exception:
                    source_page_url = ""

                for qa in default_qa:
                    qa['file_title'] = file_title
                    qa['file_name'] = file_path.name
                    qa['file_url'] = file_url
                    qa['source_page_url'] = source_page_url
                    qa['file_type'] = file_type
                final_default_qa.extend(default_qa)

                # load extracted_faq (for FAQ files)
                extracted_faq = []
                if file_type == FileType.FAQ:
                    if os.path.exists(extracted_faq_path):
                        with open(extracted_faq_path, 'r', encoding='utf-8') as f:
                            extracted_faq = json.load(f)
                        # Sanity-check alignment count
                        if len(extracted_faq) != len(default_qa):
                            logger.warning(
                                "Mismatch between default_qa (%d) and extracted_faq (%d) for %s — rolling back to keep lists in sync.",
                                len(default_qa), len(extracted_faq), safe_title_slug
                            )
                            # Roll back and skip this file entirely
                            del final_default_qa[-len(default_qa):]
                            continue
                    else:
                        logger.warning(
                            f"Extracted FAQ file not found: {extracted_faq_path} — rolling back default QA additions to keep lists in sync.")
                        del final_default_qa[-len(default_qa):]
                        continue

                    for faq in extracted_faq:
                        faq['file_title'] = file_title
                        faq['file_name'] = file_path.name
                        faq['file_url'] = file_url
                        faq['file_type'] = file_type

                        # format faq for original RAFT context
                        
                        formatted_contexts.append(QAProcessorRAFT.format_doc(faq))
                    contexts.extend(extracted_faq)
                    
                # load extracted_chunks (for non-FAQ files)
                extracted_chunks = []
                if file_type != FileType.FAQ and os.path.exists(extracted_chunks_path):
                    with open(extracted_chunks_path, 'r', encoding='utf-8') as f:
                        extracted_chunks = json.load(f)
                    # Ensure alignment with `default_qa`
                    if len(extracted_chunks) != len(default_qa):
                        logger.warning(
                            "Mismatch between default_qa (%d) and extracted_chunks (%d) for %s — rolling back to keep lists in sync.",
                            len(default_qa), len(extracted_chunks), safe_title_slug
                        )
                        del final_default_qa[-len(default_qa):]
                        continue
                    for chunk in extracted_chunks:
                        # format chunk for original RAFT context
                        formatted_contexts.append(QAProcessorRAFT.format_doc(chunk))

                    final_extracted_chunks.extend(extracted_chunks)
                    contexts.extend(extracted_chunks)
                else:
                    if file_type != FileType.FAQ:
                        # skipping the file.
                        logger.warning(f"Extracted chunks file not found: {extracted_chunks_path} — rolling back default QA additions to keep lists in sync.")
                        del final_default_qa[-len(default_qa):]
                        continue

            writing_styles = config.get("question_styles", {}).get("writing_styles", [])

            raft_config = config.get("processing", {}).get("raft", {})
            num_distract = raft_config.get("num_distractors", 4)
            p_golden = raft_config.get("p_golden_include", 0.8) # probability of including golden document

            # --- New: Retrieval endpoint configuration (optional) ---
            retrieval_cfg = config.get("retrieval_endpoint", {})
            retrieval_url = retrieval_cfg.get("url")
            retrieval_timeout = retrieval_cfg.get("timeout", 10)

            def _get_semantic_distractors(styled_q: str, styled_hash: str, golden_hashes: set, k: int) -> List[str]:
                """Return *k* formatted distractor strings using the retrieval endpoint if available.
                """
                cache_path = distractors_dir / f"distractors_{styled_hash}.json"

                doc_dicts: List[Dict[str, Any]] = []

                # Try cache first
                if cache_path.exists():
                    try:
                        with open(cache_path, 'r', encoding='utf-8') as f_cache:
                            doc_dicts = json.load(f_cache)
                    except Exception as e:
                        logger.warning(f"Failed to load distractor cache {cache_path}: {e}. Ignoring cache.")

                # call remote endpoint
                if (not doc_dicts or len(doc_dicts) < k) and retrieval_url:
                    top_k = max(k + len(golden_hashes), 10)  # may include golden doc, and obey lower cap 10
                    try:
                        resp = requests.post(retrieval_url, json={"query": styled_q, "k": top_k}, timeout=retrieval_timeout)
                        resp.raise_for_status()
                        doc_dicts = resp.json().get("docs", [])
                        # Save to cache for future runs
                        try:
                            with open(cache_path, 'w', encoding='utf-8') as f_cache:
                                json.dump(doc_dicts, f_cache, ensure_ascii=False, indent=2)
                            logger.info(f"Saved distractor cache {cache_path}")
                        except Exception as e:
                            logger.warning(f"Could not write distractor cache {cache_path}: {e}")
                    except Exception as e:
                        logger.warning(f"Retrieval endpoint failed ({e}), falling back to empty result.")

                # Filter out golden document based on hash comparison
                filtered = [d for d in doc_dicts if d.get("chunk_hash") not in golden_hashes and d.get("qa_pair_hash") not in golden_hashes]

                if len(filtered) > k:
                    if "unanswerable" in styled_hash:
                        filtered = filtered[-k:] # don't risk having the correct answer
                    else:
                        filtered = filtered[:k]

                return [QAProcessorRAFT.format_doc(d) for d in filtered]

            llm_config_styled_q_provider = config.get("providers", {}).get("styled_question", {})
            llm_config_cot_a_provider = config.get("providers", {}).get("cot_answer", {})

            if not llm_config_cot_a_provider:
                llm_config_cot_a_provider = llm_config_styled_q_provider
            llm_client_styled_q = LLMClient(llm_config_styled_q_provider)
            llm_client_cot_a = LLMClient(llm_config_cot_a_provider)

            multi_hop_generator = MultiHopQAGenerator(config)

            previous_questions_cache = {qa["chunk_hash"] if "chunk_hash" in qa else qa["qa_pair_hash"]: [] for qa in contexts}
            max_iterations_overall = max((style.get('iterations', 1) for style in writing_styles), default=1)
            file_generation_count = 0
            prev_filename = None

            # log size of items to process
            logger.info(f"Processing {len(final_default_qa)} default QA pairs for directory {output_dir}")

            # Main loop: iterate through each default QA pair
            for i in tqdm(range(len(final_default_qa)), desc="Generating RAFT Examples"):
                default_qa = final_default_qa[i]
                file_title = default_qa["file_title"]
                file_name = default_qa["file_name"]
                file_url = default_qa["file_url"]
                source_page_url = default_qa["source_page_url"]
                file_type = default_qa["file_type"]

                if prev_filename and prev_filename != file_name:
                    logger.info(f"Generated {file_generation_count} training examples for {prev_filename}")
                    file_generation_count = 0
                prev_filename = file_name

                if contexts[i]["file_title"] != file_title:
                    logger.error(f"RUNTIME SYNCHRONIZATION BUG: default_qa[{i}] is from {file_title} but contexts[{i}] is from {contexts[i]['file_title']}")
                    logger.error(f"Skipping this item to prevent incorrect data association")
                    continue

                qa_hash = contexts[i]["chunk_hash"] if "chunk_hash" in contexts[i] else contexts[i]["qa_pair_hash"]
                original_answer = contexts[i]["answer"] if "answer" in contexts[i] else None
                current_chunk = contexts[i] if "chunk" in contexts[i] else None

                golden_document = formatted_contexts[i]
                available_distractors = [context for idx, context in enumerate(formatted_contexts) if idx != i]

                # determine how many iterations to run (components: always 1)
                iterations_to_run = 1 if file_type == FileType.COMPONENT else max_iterations_overall

                # Add unanswerable as a dynamic style
                styles_to_process = writing_styles.copy()
                should_add_unanswerable = False
                should_remove_naturalistic = False
                should_remove_casual = False
                should_remove_formal = False
                should_add_multihop = False
                
                if file_type == FileType.COMPONENT:
                    should_add_unanswerable = (i % 6 == 0)
                    should_add_multihop = (i % 3 == 0)
                else:
                    should_add_unanswerable = (i % 2 == 0)
                    should_remove_naturalistic = (i % 4 == 0)
                    should_remove_casual = ((i+1) % 4 == 0)
                    should_remove_formal = ((i+2) % 4 == 0)
                    should_add_multihop = True
                    
                if should_add_unanswerable:
                    unanswerable_style = {
                        "name": "unanswerable",
                        "iterations": 1
                    }
                    styles_to_process.append(unanswerable_style)

                if should_add_multihop:
                    multihop_style = {
                        "name": "multi-hop",
                        "iterations": 1
                    }
                    styles_to_process.append(multihop_style)

                if should_remove_naturalistic:
                    styles_to_process = [style for style in styles_to_process if "naturalistic" not in style.get("name").lower()]
                if should_remove_casual:
                    styles_to_process = [style for style in styles_to_process if "casual" not in style.get("name").lower()]
                if should_remove_formal:
                    styles_to_process = [style for style in styles_to_process if "formal" not in style.get("name").lower()]


                component_selected_style = None
                if file_type == FileType.COMPONENT and writing_styles:
                    # choose exactly one style per chunk (to reduce number of generations)
                    for style in writing_styles:
                        safe_style_name_tmp = slugify(style.get("name", "").lower())
                        styled_hash_tmp = f"{qa_hash}_{safe_style_name_tmp}_0"
                        styled_question_path_tmp = raft_qa_dir / f"styled_q_{styled_hash_tmp}.txt"
                        if styled_question_path_tmp.exists():
                            component_selected_style = style
                            break
                    if should_add_multihop:
                        component_selected_style = multihop_style
                    # random selection if no style is found
                    if component_selected_style is None:
                        component_selected_style = random.choice(writing_styles)


                for iteration in range(iterations_to_run):
                    for style in styles_to_process:
                        style_name = style.get("name")
                        safe_style_name = slugify(style_name.lower())

                        # for components, allow both the selected style and unanswerable/multihop style
                        skip_style = (
                            file_type == FileType.COMPONENT
                            and style.get("name") not in {"unanswerable", "multi-hop"}
                            and style is not component_selected_style
                        )
                        if skip_style:
                            continue
                        
                        if iteration >= style.get("iterations", 1):
                            continue

                        if style_name == "multi-hop":
                            if not current_chunk:
                                logger.warning(f"FAQ file")

                            styled_hash = f"{qa_hash}_multi-hop_{iteration}"
                            styled_question_path = raft_qa_dir / f"styled_q_{styled_hash}.txt"
                            cot_answer_path = raft_qa_dir / f"cot_a_{styled_hash}.txt"

                            if styled_question_path.exists() and cot_answer_path.exists():
                                with open(styled_question_path, 'r', encoding='utf-8') as f:
                                    styled_q = f.read()
                                logger.debug(f"Loaded styled question for multi-hop QA {qa_hash} from {styled_question_path}")

                                with open(cot_answer_path, 'r', encoding='utf-8') as f:
                                    cot_answer_str = f.read()
                                logger.debug(f"Loaded CoT answer for multi-hop QA {qa_hash} from {cot_answer_path}")
                            else:
                                logger.debug(f"Processing 'multi-hop' style for seed chunk: {qa_hash}")

                                retries = 3
                                multi_hop_qa_pair = None
                                while not multi_hop_qa_pair:
                                    if current_chunk:
                                        multi_hop_qa_pair = multi_hop_generator.generate_multi_hop_qa_pair(current_chunk, output_dir, file_type==FileType.COMPONENT)
                                    else:
                                        default_qa["chunk_hash"] = qa_hash
                                        multi_hop_qa_pair = multi_hop_generator.generate_multi_hop_qa_pair(default_qa, output_dir, file_type==FileType.COMPONENT)

                                    if not multi_hop_qa_pair:
                                        logger.warning(f"Failed to generate multi-hop QA for seed {qa_hash}. Retrying...")
                                        time.sleep(1)

                                    retries -= 1
                                    if retries <= 0:
                                        break

                                if not multi_hop_qa_pair or not all(multi_hop_qa_pair.get(key) for key in ["question", "reasoning", "answer"]):
                                    logger.warning(f"Failed to generate multi-hop QA for seed {qa_hash}. Skipping style.")
                                    continue


                                styled_q = multi_hop_qa_pair["question"]
                                with open(styled_question_path, 'w', encoding='utf-8') as f:
                                    f.write(styled_q)

                                cot_answer_str = f"{multi_hop_qa_pair['reasoning']}\n{multi_hop_qa_pair['answer']}"
                                with open(cot_answer_path, 'w', encoding='utf-8') as f:
                                    f.write(cot_answer_str)

                            multihop_golden_documents = multi_hop_generator.get_golden_documents(qa_hash, output_dir)
                            if not multihop_golden_documents:
                                logger.warning(f"Could not retrieve golden documents for multi-hop QA {qa_hash}. Skipping style.")
                                continue

                            golden_document_list = [QAProcessorRAFT.format_doc(doc) for doc in multihop_golden_documents]
                            
                            # All other chunks in the dataset are potential distractors
                            golden_hashes = {doc["chunk_hash"] if "chunk_hash" in doc else doc["qa_pair_hash"] for doc in multihop_golden_documents}

                            num_distract_multihop = num_distract+1 - len(golden_document_list)

                            distractors_dk = _get_semantic_distractors(styled_q, styled_hash, golden_hashes, num_distract_multihop)
                            context_docs = golden_document_list + distractors_dk
                            random.shuffle(context_docs)

                            # Assemble the training example (similar to the single-hop path)
                            assembled_context_str = "".join([f"<DOCUMENT>{doc}</DOCUMENT>\n\n" for doc in context_docs])
                            user_content = assembled_context_str + "\n" + styled_q

                            default_qa["chunk_hash"] = qa_hash
                            entity_name = multi_hop_generator.get_entity_name(current_chunk if current_chunk else default_qa, output_dir)

                            debug_data = {
                                "entity": entity_name,
                                "question": styled_q,
                                "cot_answer_str": cot_answer_str,
                                "chunk": current_chunk["chunk"] if current_chunk else None,
                                "faq": default_qa["question"] + '\n' + default_qa["answer"] if not default_qa else None,
                                "golden_documents": golden_document_list,
                                "distractors": distractors_dk,
                                "file_name": file_name,
                                "file_url": file_url,
                                "file_title": file_title,
                                "original_qa_pair_hash": qa_hash,
                                "styled_hash": styled_hash
                            }

                            debug_path = multihop_debug_dir / f"debug_{styled_hash}.json"
                            if not debug_path.exists():
                                with open(debug_path, 'w', encoding='utf-8') as f:
                                    json.dump(debug_data, f, ensure_ascii=False, indent=2)                            
                                logger.info(f"Saved debug for multi-hop QA {qa_hash} to {debug_path}")


                            training_example = {
                                "question": user_content,
                                "answer": cot_answer_str,
                                "original_qa_pair_hash": qa_hash,
                                "style_name": "multi-hop",
                                "styled_question": styled_q,
                                "raft_qa_pair_hash": f"raft_{styled_hash}",
                                "golden_present": True,
                                "golden_index": -1, # Index is not applicable for multi-golden path
                                "num_distractors_in_context": len(distractors_dk),
                                "file_url": file_url,
                                "file_name": file_name,
                                "file_type": str(file_type)
                            }
                            all_training_examples.append(training_example)
                            file_generation_count += 1

                            continue

                        styled_hash = f"{qa_hash}_{safe_style_name}_{iteration}"
                        styled_question_path = raft_qa_dir / f"styled_q_{styled_hash}.txt"
                        styled_q = None
                        question_loaded_from_cache = False

                        while not styled_q:
                            if styled_question_path.exists():
                                with open(styled_question_path, 'r', encoding='utf-8') as f:
                                    styled_q = f.read().strip()
                                    question_loaded_from_cache = True
                                    
                                if styled_q:
                                    logger.debug(f"Loaded styled {file_name} question from {styled_question_path}")
                                else:
                                    # File exists but is empty, delete it and regenerate
                                    logger.warning(f"Found empty styled question file, deleting: {styled_question_path}")
                                    styled_question_path.unlink()
                            else:
                                logger.debug(f"Generating styled question for {file_name} ({qa_hash})")
                                previous_qs_for_pair = previous_questions_cache[qa_hash]
                                
                                # Use different generation function for unanswerable style
                                if style_name == "unanswerable":
                                    styled_q = QAProcessorRAFT.generate_unanswerable_question_raft(
                                        current_chunk, default_qa["question"], original_answer, file_title, file_name, llm_client_styled_q
                                    )
                                else:
                                    styled_q = QAProcessorRAFT.generate_styled_question_raft(
                                        default_qa["question"], default_qa["answer"], style, file_title, file_name, previous_qs_for_pair, llm_client_styled_q
                                    )
                                    
                                if styled_q:
                                    with open(styled_question_path, 'w', encoding='utf-8') as f:
                                        f.write(styled_q)
                                    logger.info(f"Saved styled {file_name} question to {styled_question_path}")
                                else:
                                    time.sleep(1)

                        cot_answer_path = raft_qa_dir / f"cot_a_{styled_hash}.txt"
                        cot_answer_str = None
                        previous_questions_cache[qa_hash].append(styled_q)
                        while not cot_answer_str:
                            if cot_answer_path.exists() and question_loaded_from_cache:
                                with open(cot_answer_path, 'r', encoding='utf-8') as f:
                                    cot_answer_str = f.read()
                                    
                                if cot_answer_str:
                                    logger.debug(f"Loaded CoT {file_name} answer from {cot_answer_path}.")
                                else:
                                    # File exists but is empty, delete it and regenerate
                                    logger.warning(f"Found empty CoT answer file, deleting: {cot_answer_path}")
                                    cot_answer_path.unlink()
                            else:
                                is_unanswerable = style_name == "unanswerable"
                                cot_answer_str = QAProcessorRAFT.generate_cot_answer_raft(
                                    styled_q, original_answer, current_chunk, is_unanswerable, llm_client_cot_a, file_type, file_title, file_name, file_url
                                )
                                if cot_answer_str:
                                    with open(cot_answer_path, 'w', encoding='utf-8') as f:
                                        f.write(cot_answer_str)
                                    logger.info(f"Saved CoT answer for {file_name} to {cot_answer_path} ({i+1}/{len(final_default_qa)})")
                                else:
                                    time.sleep(1)

                        debug_data = {
                            "styled_q": styled_q,
                            "cot_answer_str": cot_answer_str,
                            "original_question": default_qa["question"],
                            "original_answer": default_qa["answer"],
                            "chunk": current_chunk["chunk"] if current_chunk else None,
                            "chunk_hash": current_chunk["chunk_hash"] if current_chunk else None,
                            "file_name": file_name,
                            "file_url": file_url,
                            "file_title": file_title,
                            "style_name": style_name,
                            "iteration": iteration,
                            "qa_hash": qa_hash,
                            "styled_hash": styled_hash
                        }
                        debug_path = debug_dir / f"debug_{styled_hash}.json"
                        if not debug_path.exists() or not question_loaded_from_cache:
                            with open(debug_path, 'w', encoding='utf-8') as f:
                                json.dump(debug_data, f, ensure_ascii=False, indent=2)

                        actual_num_distract = min(num_distract, len(available_distractors))
                        context_docs = []
                        golden_present_flag = False
                        golden_idx = -1
                        if actual_num_distract == 0:
                            p_golden = 1.0
                        
                        if random.uniform(0, 1) < p_golden and actual_num_distract >= 0:
                            golden_present_flag = True
                            context_docs.append(golden_document)
                            distractors_dk = _get_semantic_distractors(styled_q, styled_hash, {qa_hash}, num_distract)
                            if actual_num_distract > 0:
                                # If semantic retrieval returned fewer than needed, pad with random
                                if len(distractors_dk) < actual_num_distract:
                                    missing = actual_num_distract - len(distractors_dk)
                                    distractors_dk.extend(random.sample(available_distractors, missing))
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
                                context_docs = _get_semantic_distractors(styled_q, styled_hash, {qa_hash}, num_distract)
                                if len(context_docs) < num_needed: # If semantic retrieval returned fewer than needed, pad with random
                                    missing = num_needed - len(context_docs)
                                    context_docs.extend(random.sample(available_distractors, missing))
                                random.shuffle(context_docs)
                        
                        assembled_context_str = ""
                        for doc_content in context_docs:
                            assembled_context_str += f"<DOCUMENT>{doc_content}</DOCUMENT>\n\n"
                        if assembled_context_str == "":
                            logger.error(f"Assembled context string is empty for {file_name} ({qa_hash})")
                            continue
                        user_content = assembled_context_str + "\n" + styled_q
                        assistant_content = cot_answer_str
                        training_example = {
                            "question": user_content,
                            "answer": assistant_content,
                            "original_qa_pair_hash": qa_hash,
                            "style_name": style_name,
                            "styled_question": styled_q,
                            "raft_qa_pair_hash": f"raft_{styled_hash}",
                            "golden_present": golden_present_flag,
                            "golden_index": golden_idx,
                            "num_distractors_in_context": len(context_docs) - (1 if golden_present_flag else 0),
                            "file_url": file_url,
                            "file_name": file_name,
                            "file_type": str(file_type)
                        }
                        all_training_examples.append(training_example)
                        file_generation_count += 1

            logger.info(f"Finished RAFT data generation. Generated {len(all_training_examples)} total training examples.")
        
        try:
            # Create a hash of the batch of files for uniqueness, using relative paths
            all_rel_paths = sorted([str(rel_path) for _, _, rel_path, _ in files])
            raft_hash = create_hash("::".join(all_rel_paths))
    
            raft_dataset_file = output_dir / f"raft_training_data_{raft_hash}.jsonl"
            with open(raft_dataset_file, 'w', encoding='utf-8') as f:
                for example in all_training_examples:
                    f.write(json.dumps(example, ensure_ascii=False) + '\n')
            logger.info(f"Saved complete RAFT training data to {raft_dataset_file}")
        except Exception as e:
            logger.error(f"Failed to save complete RAFT dataset: {e}")
        return all_training_examples

