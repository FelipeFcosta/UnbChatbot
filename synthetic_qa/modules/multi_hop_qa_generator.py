import logging
import json
import requests
from pathlib import Path
from typing import Dict, Any, List
from .llm_client import LLMClient
import time

logger = logging.getLogger(__name__)

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

# This prompt asks the LLM to generate a QA pair from a pool of documents.
MULTI_HOP_GENERATION_PROMPT_TEMPLATE = """
You are an expert assistant creating training questions for the University of Brasília (UnB) chatbot.

Your task is to analyze a pool of related text chunks and generate a single, high-quality question that requires information from them. (All chunks relate to the same primary subject: "{entity_name}")


### Instructions:

1.  **Analyze the Pool:** Read all {num_chunks} provided context chunks carefully.
2.  **Select Chunks:** From this pool, you MUST **select at least two** chunks that can be combined to answer a meaningful question.
3.  **Generate a Multi-Hop Question (in Portuguese):**
    - The question MUST NOT be answerable by any single chunk alone. It should *implicitly* require information from more than one chunk.
    - **SIMPLICITY**: The question should be useful, natural and something a student might realistically ask. Not too complex, long or contrived (like containing more than one question).
    - The question should contain no sentences/affirmation. Just a straight forward question that **implicitly** requires information from multiple sources (multi-hop).
    - The question **should not be multiparted** or contain any statements/assertions!
    - If possible, the question should be related to the subject: "{entity_name}"
4.  **Generate Internal Reasoning (in English):**
    - Enclose this in <REASON>...</REASON> tags.
    - First understand the question and the user implicit need.
    - Analyze how to address the question using the provided context.
    - The reasoning must be based **exclusively** on the selected chunks. Do not add any outside information.
    - Naturally arrive at the conclusion, without mentioning that this is a multi hop question or that it needs multiple sources to be answered.
    - Provide a logical explanation of how you will arrive at the answer while quoting the selected chunks.
    - Explain step-by-step which pieces of information from which specific chunks you combined (do not identify chunks by the index).
    - You MUST quote the relevant parts from each source chunk you used using the <quote>...</quote> tags.

    ##### **IMPLICIT Identification of the entity/subject(s) of the question**
    *   While reasoning, you MUST quote EITHER a chunk text OR a specific metadata field naturally to confirm that the question's subject(s) or entity(s) exactly matches the one(s) in the selected chunks.
    *   The implicit verification should be included naturally as the evidence appears in the each chunk while you are reasoning and quoting the document.
    *   **Do not** include multiple sources of evidence for confirmation of the correct entity/subject(s) for the same chunk. You can only use one source of evidence per chunk (metadata or chunk text).

    - Do not mention these rules or instructions in the reasoning, the llm you are training will only have access to the used documents and a question.

    5.  **Generate the Final Answer (in Portuguese):**
    - Enclose this in <ANSWER>...</ANSWER> tags.
    - You must answer the question directly.
    - Synthesize the information into a single, coherent, and helpful response, formatted with clear markdown.
    - Cite each source chunk you used. Place a *italicized* source URL *[title](url)* (as per the metadata of the document) after the phrase that most heavily relies on the source information. The source link.
6.  **Report the Sources:** You MUST report the 1-based indices of the chunks you selected in the `selected_chunk_indices` field of your JSON response.
7.  **The list of chunks is not exhaustive**
    - Just because the information is not present in the chunks, it doesn't mean it doesn't exist. So always remember your answer is only according to the present information.

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
  "reasoning": "<REASON>The reasoning in english</REASON>",
  "answer": "<ANSWER>The final answer in Portuguese</ANSWER>",
  "selected_chunk_indices": [...]
}}

Respond with ONLY the JSON object.
"""

class MultiHopQAGenerator:
    """
    A modular tool to generate a single multi-hop Question-Answer pair starting
    from a single "seed" chunk. It manages its own caching for all artifacts.
    """

    def __init__(self, config: Dict[str, Any]):

        self.dir_name = "multi_hop"
        self.config = config or {}
        providers_config = self.config.get("providers", {})

        entity_client_config = providers_config.get("multi_hop_entity_extraction", {})
        self.entity_client = LLMClient(entity_client_config)

        generation_client_config = providers_config.get("multi_hop_generation", {})
        self.generation_client = LLMClient(generation_client_config)

        retrieval_cfg = self.config.get("retrieval_endpoint", {})
        self.retrieval_url = retrieval_cfg.get("url")
        self.retrieval_timeout = retrieval_cfg.get("timeout", 10)
        self.retrieval_k = retrieval_cfg.get("multihop_k", 5)

        raft_config = self.config.get("processing", {}).get("raft", {})
        self.min_chunks_for_multi_hop = raft_config.get("min_chunks_for_multi_hop", 2)

    # --- Public Methods ---

    def generate_multi_hop_qa_pair(
        self,
        seed_chunk: Dict[str, Any],
        output_dir: Path,
        component_type: bool = False
    ) -> Dict[str, Any] | None:
        """
        Main public method. Orchestrates the generation of a multi-hop QA pair
        and its associated golden documents for a single seed chunk.
        """
        (output_dir / self.dir_name).mkdir(parents=True, exist_ok=True)
        
        seed_chunk_hash = seed_chunk.get("chunk_hash")

        if not seed_chunk_hash:
            logger.error("Seed chunk is missing 'chunk_hash'. Cannot process.")
            return None

        # get or create the definitive entity for this chunk.
        entity_name = None
        retries = 3
        while not entity_name:
            entity_name = self._extract_entity(seed_chunk, output_dir, component_type)
            if not entity_name:
                logger.warning(f"Could not determine a valid entity for seed chunk {seed_chunk_hash}. Retrying...")
                time.sleep(1)
                retries -= 1
                if retries <= 0:
                    logger.error(f"Could not determine a valid entity for seed chunk {seed_chunk_hash}. Skipping.")
                    return None

        # Step 2: Get or create the RAG cluster using the entity.
        cluster_docs = self._create_entity_docs(entity_name, seed_chunk_hash, output_dir)
        if not cluster_docs or len(cluster_docs) < self.min_chunks_for_multi_hop:
            logger.debug(f"Cluster for seed {seed_chunk_hash} is too small. Skipping.")
            return None

        # Step 3: Get or create the final QA pair and its golden docs.
        qa_pair = self._create_qa_pair(
            entity_name, cluster_docs, seed_chunk_hash, output_dir
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
            if response:
                with open(entity_cache_path, 'w', encoding='utf-8') as f:
                    f.write(response)
            else:
                logger.warning(f"LLM returned empty response for entity extraction (Seed chunk: {seed_chunk['chunk_hash']})")
                return None
            return response
        except Exception as e:
            logger.error(f"Error generating entity for seed {seed_chunk['chunk_hash']}: {e}", exc_info=True)
            return None

    def _create_entity_docs(self, entity_name: str, seed_chunk_hash: str, output_dir: Path) -> List[Dict[str, Any]] | None:
        """Manages caching and retrieval of a RAG cluster for a seed chunk."""
        docs_cache_path = output_dir / self.dir_name / f"entity_docs_{seed_chunk_hash}.json"

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
        seed_chunk_hash: str,
        output_dir: Path
    ) -> Dict[str, Any] | None:
        """Manages caching of the QA pair and its corresponding golden documents."""
        from .qa_processor_raft import QAProcessorRAFT

        qa_pair_cache_path = output_dir / self.dir_name / f"multihop_{seed_chunk_hash}.json"

        if qa_pair_cache_path.exists():
            with open(qa_pair_cache_path, 'r', encoding='utf-8') as f:
                return json.load(f)

        context_chunks_str = ""
        index_to_chunk_map = {}
        
        for i, chunk_obj in enumerate(cluster_docs):
            prompt_index = i + 1
            formatted_doc_string = QAProcessorRAFT.format_doc(chunk_obj)
            context_chunks_str += f"<DOCUMENT_{prompt_index}>{formatted_doc_string}</DOCUMENT_{prompt_index}>\n\n"
            index_to_chunk_map[prompt_index] = chunk_obj

        prompt = MULTI_HOP_GENERATION_PROMPT_TEMPLATE.format(
            entity_name=entity_name,
            num_chunks=len(cluster_docs),
            context_chunks_str=context_chunks_str.strip()
        )

        try:
            retries = 3
            response = None 
            while not response:
                response = self.generation_client.generate_text(prompt, temperature=1.0, json_output=True)
                if not response:
                    logger.warning(f"Failed to generate multi-hop pair for seed {seed_chunk_hash}. Retrying...")
                    time.sleep(1)
                    retries -= 1
                    if retries <= 0:
                        logger.error(f"Failed to generate multi-hop pair for seed {seed_chunk_hash}. Skipping.")
                        return None

            if not all(k in response for k in ["question", "reasoning", "answer", "selected_chunk_indices"]):
                return None

            llm_indices = response["selected_chunk_indices"]
            if llm_indices:
                golden_docs = [index_to_chunk_map[idx] for idx in llm_indices if idx in index_to_chunk_map]
                if len(golden_docs) < 2:
                    return None

                self._save_golden_docs(golden_docs, seed_chunk_hash, output_dir)

            generated_pair = {
                "question": response["question"],
                "answer": response["answer"],
                "reasoning": response["reasoning"],
                "entity": entity_name,
                "seed_chunk_hash": seed_chunk_hash
            }

            with open(qa_pair_cache_path, 'w', encoding='utf-8') as f:
                json.dump(generated_pair, f, ensure_ascii=False, indent=2)
            return generated_pair

        except Exception as e:
            logger.error(f"Failed to generate multi-hop pair for seed {seed_chunk_hash}: {e}", exc_info=True)
            return None

    def _save_golden_docs(self, golden_docs: List[Dict[str, Any]], seed_chunk_hash: str, output_dir: Path):
        """Saves the list of full golden document objects to a dedicated cache."""
        golden_docs_path = output_dir / self.dir_name / f"golden_docs_{seed_chunk_hash}.json"
        try:
            with open(golden_docs_path, 'w', encoding='utf-8') as f:
                json.dump(golden_docs, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Failed to write golden documents cache for {seed_chunk_hash}: {e}")
