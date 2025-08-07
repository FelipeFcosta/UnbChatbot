import logging
import json
import requests
from pathlib import Path
from typing import Dict, Any, List
from .llm_client import LLMClient
import time

logger = logging.getLogger(__name__)

UNANSWERABLE_PROMPT = """
### **CRITICAL: Unanswerable or missing Information**
*   In <REASON>, **list** each source document to verify if the answer really doesn't exist in the sources.
*   If the quesiton is not answerable given the UnB sources, follow the rules below:
    *   In the <ANSWER> section, politely state that you do not have the information to properly answer the question (**DO NOT ACKNOWLEDGE the existence of the provided information beyond the question**) and NOTHING MORE.
    *   DO NOT mention the "provided context", the user is unaware of it.
    *   Do not infer information that is not present in the sources that is not obvious, only answer if it the sources exactly address the question.
    *   DO NOT mention the information you do have if it doesn't directly answer the question.

### **CRITICAL: Pre-Output Validation**

1.  First, make a draft (in your thinking process) of your complete <REASON> and <ANSWER> sections.
2.  Review your draft against **ALL** instructions and rules before outputting.
3.  Verify: Does this follow every rule? Are there any violations?
4.  Only output the final <REASON> and <ANSWER> after confirming full compliance internally.
    """

CITATION_PROMPT = """
### **Citation System**

Citations add credibility to the answer.
---

You may optionally include a short quote from the context as a blockquote to support your statement. The source link *[title](url)* must appear only once in the entire answer.

> *"...Relevant excerpt from a chunk..."*
> *[title](url)*
> *Rest of the answer...*

**Citation Rules:**
*   It should be placed where it supports a specific statement.
*   **CRITICAL:** The quote must contribute meaningfully to the answer and not be redundant with information you've already stated.
*   The quote must be relevant, add value, and be as short as possible. Use [...] to omit irrelevant parts.
*   If a quote does not add value, do not include it.
*   If a blockquote is **NOT** present, you must still include the source link. Place the *italicized* citation *[title](url)* after the phrase that most heavily relies on the source information as a *reference* (not part of the sentence).
*   In the <ANSWER> cite each source chunk you used. Place a *italicized* source URL *[title](url)* (as per the metadata of the document) after the phrase that most heavily relies on the source information. The source link.

---
"""

# This prompt is designed to extract one entity from one chunk.
ENTITY_EXTRACTION_PROMPT_TEMPLATE = """
You are a highly efficient text analysis AI. Your task is to generate a contextual topic label from the provided text chunk.

The label should not be general. It should represent a specific concept or cluster of concepts that appears in the text and could group together related chunks.

- The label will be used to retrieve other chunks (in a large database) that are related by this label. So it shouldn't be generic.
- The label should be **self-contained and unambiguous**, meaning **it should make sense WITHOUT relying on external context**.
- If no meaningful label can be found, respond with "N/A".
- Use "Topic" metadata only as an additional source of information, do not just mirror the "Topic" metadata.
{component_str}

- Respond with ONLY the name of the created label. Do not add any explanation or preamble..

**Text Chunk:**
---
{chunk_text}
---

In your thinking process, revise every rule and instruction again, but only output the label (in portuguese) in one line.
"""

# This prompt asks the LLM to generate a multi-hop QUESTION and select source chunks.
MULTI_HOP_QUESTION_GENERATION_PROMPT_TEMPLATE = """
You are an expert assistant creating training questions for the University of Brasília (UnB) chatbot.

Your task is to analyze a pool of related text chunks (from unb websites) and generate a single, high-quality question that requires information from them. (All chunks relate to the same primary subject: "{entity_name}")

The user is a **student who does not know about the present documents**, so the question must be specific enough (but not contrived or too long) for the documents to be retrieved.

### Instructions:

1.  **Analyze the Pool:** Read all {num_chunks} provided context chunks carefully.
2.  **Select Chunks:** From this pool, you MUST **select at least two** chunks that can be combined to answer a meaningful question.
3.  **Generate a Multi-Hop Question (in Portuguese):**
    - The question MUST NOT be answerable by any single chunk alone. It should *implicitly* require information from more than one chunk.
    - **SIMPLICITY**: The question should be useful, natural and something a student might realistically ask. Not too complex, long or contrived (like containing more than one question).
    - The question should contain no sentences/affirmation. Just a straight forward question that **implicitly** requires information from multiple sources (multi-hop).
    - The question **should not be multiparted** or contain any statements/assertions!
    - DO NOT reference the documents directly.
    - The question should be specific enough to retrieve these documents but natural for a student to ask (but NOT too long and NOT too complex).
    - If possible, the question should be related to the subject: "{entity_name}"
4. **Do not invent information**: Do not create or infer information that is not present in the chunks (like expanding an acronym you don't know).
5.  **Report the Sources:** You MUST report the 1-based indices of the chunks you selected in the `selected_chunk_indices` field of your JSON response.

---

### Context Chunks Pool for label "{entity_name}":

{context_chunks_str}

---

### If the chunks are not enough to generate a coherent multi-hop question:
- If you cannot find a question that requires information from multiple chunks and fits the instructions, **every json field should be null**.
- Otherwise, respond with a valid JSON object following the required output format.

### Required Output Format:

Return a single, valid JSON object following this exact structure. The `selected_chunk_indices` must be a list of integers corresponding to the <DOCUMENT_N> tags you used.

{{
  "question": "The generated multi-hop question in Portuguese.",
  "selected_chunk_indices": [...]
}}

Respond with ONLY the JSON object.
"""

# This prompt asks the LLM to generate an UNANSWERABLE QUESTION based on available chunks.
UNANSWERABLE_QUESTION_GENERATION_PROMPT = """You are an expert assistant creating training questions for the University of Brasília (UnB) chatbot.

**GOAL:** Train the LLM to recognize when it lacks sufficient information to answer a question and respond appropriately with "I don't know" or "I can't answer that based on the information I have" instead of hallucinating.

Your task is to analyze a pool of related text chunks (from unb websites) and generate a single, UNANSWERABLE question that contains wrong/misleading information. (All chunks relate to the same primary subject: "{entity_name}")

The user is a **student who does not know about the present documents**, so the question must be specific enough (but not contrived or too long) for the documents to be retrieved.

### Instructions:

1.  **Analyze the Pool:** Read all {num_chunks} provided context chunks carefully to understand what information IS available.
2.  **Generate an UNANSWERABLE Question (in Portuguese):**
    - Create a question that **cannot** be answered based on the provided source documents.
    - The question should sound natural and plausible, but include honest mistake(s) or false assumption that makes it **unanswerable**.
    - **SIMPLICITY**: The question should be straightforward and something a user might ask. Avoid overly complex or lengthy phrasing.
    - The question should contain no sentences or assertions - just a clear, direct question.
    - The question **should not be multiparted** or contain any statements/assertions!
    - Use natural, conversational Portuguese as a Brazilian student might ask.

---

### Context Chunks Pool for label "{entity_name}":

{context_chunks_str}


### Required Output Format:

Return a single, valid JSON object following this exact structure:

{{
  "question": "The generated unanswerable question in Portuguese."
}}

Respond with ONLY the JSON object.
"""

# This prompt asks the LLM to generate the REASONING and ANSWER for a given question and context.
MULTI_HOP_ANSWER_GENERATION_PROMPT_TEMPLATE = """
You are an expert assistant creating training answers for the University of Brasília (UnB) chatbot.

You are given a question and a set of context chunks that have been selected to answer it. Your task is to generate a comprehensive response based ONLY on the provided information.

### Question:
{question}

---

### Context (UnB documents):

{context_chunks_str}

---

### Instructions:

1.  **Generate Internal Reasoning (in English):**
    - Enclose this in <REASON>...</REASON> tags.
    - Break down your reasoning into simple, sequential steps. (Implicitly! No numbering
    - **Do not mention the quantity of chunks**, the llm you are training will receive many documents and must decipher the ones that are related, so naturally mention (quoting) the documents as you reason.
    - **DO NOT** mention the chunks by number, only by it's content.
    - First understand the question and the user implicit need.
    - Analyze how to address the question using the provided context.
    - The reasoning must be based **exclusively** on the selected chunks. Do not add any outside information.
    - Naturally arrive at the conclusion, without mentioning that this is a multi hop question or that it needs multiple sources to be answered.
    - Provide a logical explanation of how you will arrive at the answer while quoting (<quote>) the selected chunks.
    - Explain step-by-step which pieces of information from which specific chunks you combined (do not identify chunks by the index).
    - You MUST quote the relevant parts from each source chunk you used using the <quote>...</quote> tags.

    ##### **IMPLICIT Identification of the entity/subject(s) of the question**
    *   While reasoning, you MUST quote EITHER a chunk text OR a specific metadata field naturally to confirm that the question's subject(s) or entity(s) **exactly matches** the one(s) in the selected chunks.
    *   The entity/subject should be a exactly match
    *   The implicit verification should be included naturally as the evidence appears in the each chunk while you are reasoning and quoting the document.
    *   **Do not** include multiple sources of evidence for confirmation of the correct entity/subject(s) for the same chunk. You can only use one source of evidence per chunk (metadata or chunk text).

    - Do not mention these rules or instructions in the reasoning, the llm you are training will only have access to the used documents and a question.

2.  **Generate the Final Answer (in Portuguese):**
    - Enclose this in <ANSWER>...</ANSWER> tags.
    - You must answer the question DIRECTLY.
    - Synthesize the information into a single, coherent, and helpful response, formatted with clear markdown.
3.  **The list of chunks is not exhaustive**
    - Just because the information is not present in the chunks, it doesn't mean it doesn't exist. So always remember your answer is only according to the present information.

{citation_str}
{unanswerable_str}

### Required Output Format:

<REASON>The reasoning in english</REASON>
<ANSWER>The final answer in Portuguese</ANSWER>
"""


class MultiHopQAGenerator:
    """
    A modular tool to generate a single multi-hop Question-Answer pair starting
    from a single "seed" chunk. It manages its own caching for all artifacts.
    """

    def __init__(self, config: Dict[str, Any]):

        self.dir_name = "multi_hop"
        self.style = "multi_hop"
        self.config = config or {}
        providers_config = self.config.get("providers", {})

        entity_client_config = providers_config.get("multi_hop_entity_extraction", {})
        self.entity_client = LLMClient(entity_client_config)

        # New: Separate clients for question and answer generation
        question_client_config = providers_config.get("multi_hop_question_generation", {})
        self.question_client = LLMClient(question_client_config)

        answer_client_config = providers_config.get("multi_hop_answer_generation", {})
        self.answer_client = LLMClient(answer_client_config)

        retrieval_cfg = self.config.get("retrieval_endpoint", {})
        self.retrieval_url = retrieval_cfg.get("url")
        self.retrieval_timeout = retrieval_cfg.get("timeout", 10)
        self.retrieval_k = retrieval_cfg.get("multihop_k", 5)

        raft_config = self.config.get("processing", {}).get("raft", {})
        self.min_chunks_for_multi_hop = raft_config.get("min_chunks_for_multi_hop", 2)
        self.seed_chunk_hash = None

    # --- Public Methods ---

    def generate_multi_hop_qa_pair(
        self,
        seed_chunk: Dict[str, Any],
        output_dir: Path,
        component_type: bool = False,
        is_unanswerable: bool = False
    ) -> Dict[str, Any] | None:
        """
        Main public method. Orchestrates the generation of a multi-hop QA pair
        and its associated golden documents for a single seed chunk.
        """

        (output_dir / self.dir_name).mkdir(parents=True, exist_ok=True)
        
        seed_chunk_hash = seed_chunk.get("chunk_hash")
        if is_unanswerable:
            seed_chunk_hash = seed_chunk_hash + "_unanswerable"

        if not seed_chunk_hash:
            logger.error("Seed chunk is missing 'chunk_hash'. Cannot process.")
            return None

        self.seed_chunk_hash = seed_chunk_hash

        entity_name = None
        retries = 3
        while not entity_name:
            entity_name = self._extract_entity(seed_chunk, output_dir, component_type)
            if not entity_name:
                logger.warning(f"Could not determine a valid entity for seed chunk {self.seed_chunk_hash}. Retrying...")
                time.sleep(1)
                retries -= 1
                if retries <= 0:
                    logger.error(f"Could not determine a valid entity for seed chunk {self.seed_chunk_hash}. Skipping.")
                    return None

        cluster_docs = self._create_entity_docs(entity_name, seed_chunk, output_dir)
        if not cluster_docs or len(cluster_docs) < self.min_chunks_for_multi_hop:
            logger.debug(f"Cluster for seed {self.seed_chunk_hash} is too small. Skipping.")
            return None

        qa_pair = self._create_qa_pair(
            entity_name, cluster_docs, seed_chunk, output_dir
        )

        return qa_pair

    def get_entity_name(self, seed_chunk: Dict[str, Any], output_dir: Path) -> str | None:
        """
        Get the entity name from the seed chunk.
        """
        entity_cache_path = output_dir / self.dir_name / f"entity_{seed_chunk['chunk_hash']}.txt"

        if not entity_cache_path.exists():
            return None
        try:
            with open(entity_cache_path, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except Exception as e:
            logger.error(f"Failed to read cached entity for {seed_chunk['chunk_hash']}: {e}", exc_info=True)
            return None

    def get_entity_docs(
        self,
        seed_chunk_hash: str,
        output_dir: Path
    ) -> List[Dict[str, Any]] | None:
        """
        Retrieves the cached RAG cluster documents for a given entity name.
        This method is now fully self-contained.
        """
        docs_cache_path = output_dir / self.dir_name / f"entity_docs_{seed_chunk_hash}.json"

        if not docs_cache_path.exists():
            return None
        try:
            with open(docs_cache_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to read cached entity docs for {seed_chunk_hash}: {e}", exc_info=True)
            return None


    def get_golden_documents(
        self,
        seed_chunk_hash: str,
        output_dir: Path
    ) -> List[Dict[str, Any]] | None:
        """
        Retrieves the cached golden documents for a given seed_chunk_hash.
        This method is now fully self-contained.
        """
        golden_docs_path = output_dir / self.dir_name / f"golden_docs_{seed_chunk_hash}.json"

        if not golden_docs_path.exists():
            return None
        try:
            with open(golden_docs_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to read cached golden documents for {seed_chunk_hash}: {e}", exc_info=True)
            return None

    def _generate_unanswerable_question(
        self,
        entity_name: str,
        cluster_docs: List[Dict[str, Any]],
        output_dir: Path
    ) -> str | None:
        """
        Generates an unanswerable question based on the cluster docs.
        """
        from .qa_processor_raft import QAProcessorRAFT

        unanswerable_question_cache_path = output_dir / self.dir_name / f"question_{self.seed_chunk_hash}.txt"
        if unanswerable_question_cache_path.exists():
            with open(unanswerable_question_cache_path, 'r', encoding='utf-8') as f:
                return f.read().strip()

        context_chunks_for_question = "\n\n".join(
            f"<DOCUMENT_{i + 1}>{QAProcessorRAFT.format_doc(chunk)}</DOCUMENT_{i + 1}>"
            for i, chunk in enumerate(cluster_docs)
        )

        question_prompt = UNANSWERABLE_QUESTION_GENERATION_PROMPT.format(
            entity_name=entity_name,
            num_chunks=len(cluster_docs),
            context_chunks_str=context_chunks_for_question
        )

        try:
            question_data = None
            retries = 3
            while not question_data and retries > 0:
                response = self.question_client.generate_text(question_prompt, temperature=1.0, json_output=True)
                if response:
                    if response.get("question"):
                        question_data = response
                        break
                    logger.warning(f"Invalid unanswerable question data for seed {self.seed_chunk_hash}. Retrying...")
                    time.sleep(1)
                    retries -= 1
            
            if not question_data:
                logger.error(f"Failed to generate a valid unanswerable question for seed {self.seed_chunk_hash}. Skipping.")
                return None

            unanswerable_question = question_data["question"]
            
            # Cache the unanswerable question
            with open(unanswerable_question_cache_path, 'w', encoding='utf-8') as f:
                f.write(unanswerable_question)
            
            return unanswerable_question

        except Exception as e:
            logger.error(f"Failed during unanswerable question generation for seed {self.seed_chunk_hash}: {e}", exc_info=True)
            return None

    # --- Private Helper Methods ---

    def _extract_entity(self, seed_chunk: Dict[str, Any], output_dir: Path, component_type: bool = False) -> str | None:
        """Manages caching for entity extraction to avoid redundant LLM calls."""
        from .qa_processor_raft import QAProcessorRAFT
        
        entity_cache_path = output_dir / self.dir_name / f"entity_{seed_chunk['chunk_hash']}.txt"

        if entity_cache_path.exists():
            entity = entity_cache_path.read_text(encoding='utf-8').strip()
            logger.debug(f"Loading cached entity for {seed_chunk['chunk_hash']} from {entity_cache_path}")
            return entity

        chunk_text = QAProcessorRAFT.format_doc(seed_chunk)
        component_str = f"- Never use a component code (e.g. CIC0097) in the label." if component_type else ""

        prompt = ENTITY_EXTRACTION_PROMPT_TEMPLATE.format(chunk_text=chunk_text, component_str=component_str)
        try:
            response = self.entity_client.generate_text(prompt, temperature=1.0).strip()
            if response and "N/A" not in response:
                with open(entity_cache_path, 'w', encoding='utf-8') as f:
                    f.write(response)
                return response
            else:
                logger.warning(f"LLM returned empty or N/A for entity extraction (Seed chunk: {self.seed_chunk_hash})")
                return None
        except Exception as e:
            logger.error(f"Error generating entity for seed {self.seed_chunk_hash}: {e}", exc_info=True)
            return None

    def _create_entity_docs(self, entity_name: str, seed_chunk: Dict[str, Any], output_dir: Path) -> List[Dict[str, Any]] | None:
        """Manages caching and retrieval of a RAG cluster for a seed chunk."""
        docs_cache_path = output_dir / self.dir_name / f"entity_docs_{seed_chunk['chunk_hash']}.json"

        if docs_cache_path.exists():
            with open(docs_cache_path, 'r', encoding='utf-8') as f:
                return json.load(f)

        if not self.retrieval_url:
            return None

        try:
            resp = requests.post(self.retrieval_url, json={"query": entity_name, "k": self.retrieval_k}, timeout=self.retrieval_timeout)
            resp.raise_for_status()
            cluster_docs = resp.json().get("docs", [])
            with open(docs_cache_path, 'w', encoding='utf-8') as f:
                json.dump(cluster_docs, f, ensure_ascii=False, indent=2)
            return cluster_docs
        except Exception:
            return None

    def _create_qa_pair(
        self,
        entity_name: str,
        cluster_docs: List[Dict[str, Any]],
        seed_chunk: Dict[str, Any],
        output_dir: Path
    ) -> Dict[str, Any] | None:
        """
        Manages the two-step creation and caching of a QA pair.
        """
        from .qa_processor_raft import QAProcessorRAFT

        qa_pair_cache_path = output_dir / self.dir_name / f"multihop_{self.seed_chunk_hash}.json"
        if qa_pair_cache_path.exists():
            with open(qa_pair_cache_path, 'r', encoding='utf-8') as f:
                return json.load(f)

        index_to_chunk_map = {i + 1: chunk for i, chunk in enumerate(cluster_docs)}
        
        # Determine if this is for unanswerable question generation
        is_unanswerable = "unanswerable" in self.seed_chunk_hash
        
        if is_unanswerable:
            # Use the unanswerable question generation method
            question = self._generate_unanswerable_question(entity_name, cluster_docs, output_dir)
            if not question:
                logger.error(f"Failed to generate unanswerable question for seed {self.seed_chunk_hash}")
                return None
            # For unanswerable questions, we still use all chunks as golden docs for context
            golden_docs = cluster_docs
        else:
            # Regular multi-hop question generation
            context_chunks_for_question = "\n\n".join(
                f"<DOCUMENT_{i + 1}>{QAProcessorRAFT.format_doc(chunk)}</DOCUMENT_{i + 1}>"
                for i, chunk in enumerate(cluster_docs)
            )

            question_prompt = MULTI_HOP_QUESTION_GENERATION_PROMPT_TEMPLATE.format(
                entity_name=entity_name,
                num_chunks=len(cluster_docs),
                context_chunks_str=context_chunks_for_question
            )

            try:
                question_data = None
                retries = 3
                while not question_data and retries > 0:
                    response = self.question_client.generate_text(question_prompt, temperature=1.0, json_output=True)
                    if response:
                        if response.get("question") and response.get("selected_chunk_indices"):
                            if len(response["selected_chunk_indices"]) >= self.min_chunks_for_multi_hop:
                                question_data = response
                                break
                        else:
                            if "question" in response and "selected_chunk_indices" in response:
                                logger.warning(f"Question can't be created for seed {self.seed_chunk_hash}. Skipping.")
                                generated_pair = {
                                    "question": None,
                                    "response": None,
                                    "entity": entity_name,
                                    "seed_chunk_hash": self.seed_chunk_hash
                                }
                                with open(qa_pair_cache_path, 'w', encoding='utf-8') as f:
                                    json.dump(generated_pair, f, ensure_ascii=False, indent=2)
                                return generated_pair
                        logger.warning(f"Invalid question data for seed {self.seed_chunk_hash}. Retrying...")
                        time.sleep(1)
                        retries -= 1
                
                if not question_data:
                    logger.error(f"Failed to generate a valid multi-hop question for seed {self.seed_chunk_hash}. Skipping.")
                    return None

                question = question_data["question"]
                selected_indices = question_data["selected_chunk_indices"]
                golden_docs = [index_to_chunk_map[idx] for idx in selected_indices if idx in index_to_chunk_map]
                
                # Additional check in case LLM hallucinates indices
                if len(golden_docs) < self.min_chunks_for_multi_hop:
                    logger.error(f"LLM selected insufficient valid documents for seed {self.seed_chunk_hash}. Skipping.")
                    return None

            except Exception as e:
                logger.error(f"Failed during question generation for seed {self.seed_chunk_hash}: {e}", exc_info=True)
                return None

        if not is_unanswerable:
            self._save_golden_docs(golden_docs, seed_chunk['chunk_hash'], output_dir)

        # --- Step 2: Generate the Answer and Reasoning ---
        context_chunks_for_answer = "\n...\n\n".join(
            f"<DOCUMENT>{QAProcessorRAFT.format_doc(chunk)}</DOCUMENT>"
            for _, chunk in enumerate(golden_docs)
        )
        context_chunks_for_answer += "\n...\n"

        citation_str = CITATION_PROMPT
        unanswerable_str = ""
        
        if "unanswerable" in self.seed_chunk_hash:
            unanswerable_str = UNANSWERABLE_PROMPT
        
        answer_prompt = MULTI_HOP_ANSWER_GENERATION_PROMPT_TEMPLATE.format(
            question=question,
            context_chunks_str=context_chunks_for_answer,
            citation_str=citation_str,
            unanswerable_str=unanswerable_str
        )

        try:
            retries = 3
            response_str = None
            while not response_str and retries > 0:
                response_str = self.answer_client.generate_text(answer_prompt, temperature=0.7).strip()
                if "<REASON>" in response_str and "<ANSWER>" in response_str:
                    break
                logger.warning(f"Invalid answer data for seed {self.seed_chunk_hash}. Retrying...")
                time.sleep(1)
                retries -= 1

            if not response_str or "<REASON>" not in response_str or "<ANSWER>" not in response_str:
                logger.error(f"Answer generation for seed {self.seed_chunk_hash} produced invalid format. Skipping.")
                return None

            generated_pair = {
                "question": question,
                "response": response_str,
                "entity": entity_name,
                "seed_chunk_hash": self.seed_chunk_hash
            }

            with open(qa_pair_cache_path, 'w', encoding='utf-8') as f:
                json.dump(generated_pair, f, ensure_ascii=False, indent=2)
            
            return generated_pair

        except Exception as e:
            logger.error(f"Failed during answer generation for seed {self.seed_chunk_hash}: {e}", exc_info=True)
            return None

    def _save_golden_docs(self, golden_docs: List[Dict[str, Any]], seed_chunk_hash: str, output_dir: Path):
        """Saves the list of full golden document objects to a dedicated cache."""
        golden_docs_path = output_dir / self.dir_name / f"golden_docs_{seed_chunk_hash}.json"
        try:
            with open(golden_docs_path, 'w', encoding='utf-8') as f:
                json.dump(golden_docs, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Failed to write golden documents cache for {self.seed_chunk_hash}: {e}")