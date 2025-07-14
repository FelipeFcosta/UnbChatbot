#!/usr/bin/env python3
"""
Run RAFT inference using a GGUF LLM model stored in a Modal Volume,
exposed as a persistent web endpoint using llama-cpp-python with CUDA support.
Includes Retrieval-Augmented Generation (RAG) logic.
"""

import modal
import os
import logging
import json # Added for loading data
from pathlib import Path
from typing import List, Dict, Optional

# Package context (python -m fine_tuning.inference.raft_inference)
from . import config  # type: ignore
from .config import *  # noqa F403  (re-export)

# All configuration constants have been moved to `config.py`.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

app = modal.App(APP_NAME)

# IMPORTANT: Add FAISS and Sentence Transformers to the image build
cuda_version = "12.1.0"
flavor = "devel"
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.10")
    .apt_install(
        "git", "build-essential", "cmake", "curl",
        "libcurl4-openssl-dev", "libopenblas-dev", "libomp-dev", "clang",
        "gcc-11", "g++-11"
    )
    .pip_install(
        "modal-client",
        "fastapi[standard]",
        "huggingface_hub",
        "ninja",
        "wheel",
        "packaging",
        # --- Added for RAG ---
        "sentence-transformers",
        "faiss-cpu", # Use faiss-cpu for simplicity, or faiss-gpu if needed and GPU compatible
        "numpy",     # Dependency for FAISS/embeddings
        # ---------------------
    )
    .run_commands(
        'CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=75" pip install --no-cache-dir llama-cpp-python',
        gpu=GPU_CONFIG
    )
    .entrypoint([])
)

model_volume = modal.Volume.from_name(VOLUME_NAME)
document_volume = modal.Volume.from_name(DATA_VOLUME_NAME)
helper_llm_volume = modal.Volume.from_name(HELPER_LLM_VOLUME_NAME)

@app.cls(
    image=image,
    gpu=GPU_CONFIG,
    volumes={
        MODEL_MOUNT_PATH: model_volume,
        DATA_MOUNT_PATH: document_volume, # where documents source is mounted
        HELPER_LLM_MODEL_MOUNT_PATH: helper_llm_volume # where helper llm model is mounted
    },
    timeout=60*6,
    min_containers=1
)
@modal.concurrent(max_inputs=3)
class ModelEndpoint:
    def __init__(self):
        self.gguf_file_path_in_container = str(
            Path(MODEL_MOUNT_PATH) / MODEL_DIR_IN_VOLUME / CHECKPOINT_FOLDER / GGUF_FILENAME
        )
        self.helper_llm_gguf_file_path_in_container = str(
            Path(HELPER_LLM_MODEL_MOUNT_PATH) / HELPER_LLM_MODEL_DIR_IN_VOLUME / GGUF_FILENAME
        )
        self.llm = None
        self.helper_llm = None
        self.retriever = None
        self.regulars: List[Dict] = []
        self.pairs: List[Dict] = []
        self.documents: List[str] = []

    def _load_and_index_data(self):
        """Loads original FAQ data and builds the FAISS index."""
        from sentence_transformers import SentenceTransformer
        import faiss
        import numpy as np

        logger.info(f"Loading original FAQ data from {SOURCE_DOCUMENTS}...")
        if not os.path.exists(SOURCE_DOCUMENTS):
             error_msg = f"Original FAQ data file not found at {SOURCE_DOCUMENTS}."
             logger.error(error_msg)
             raise RuntimeError(error_msg)

        # Load and group chunks/pairs by type, and build self.documents for retrieval
        with open(SOURCE_DOCUMENTS, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for item in data:
                doc_type = item.get("file_type", "")
                file_title = item.get("file_title", "")
                source_page_url = item.get("source_page_url", "")

                if doc_type == 'FileType.FAQ':
                    question = item.get("question", "")
                    answer = item.get("answer", "")
                    topics = item.get("topics")

                    file_url = item.get("file_url")
                    
                    # Format FAQ with new <doc_metadata> structure
                    topic_str = f'Topic: "{", ".join(topics)}", ' if topics else ''
                    filename_str = f'File: "{item.get("file_name", "")}", '
                    url_str = f'URL: "{file_url}"'
                    
                    formatted_item = f'Q: "{question}", A: "{answer}"<doc_metadata>{topic_str}{filename_str}{url_str}</doc_metadata>\n'
                    self.pairs.append(formatted_item)
                    self.documents.append(formatted_item)  # Add to retrieval

                else:
                    chunk = item.get("chunk", "")
                    topics = item.get("topic")
                    professor = item.get("professor")
                    course = item.get("course")
                    file_name = item.get("file_name", "")
                    file_url = item.get("file_url")
                    
                    # Format chunk with new <doc_metadata> structure
                    topic_str = f'Topic: "{topics}", ' if topics else ''
                    professor_str = f'Professor: "{professor}", ' if professor else ''
                    course_str = f'Course: "{course}", ' if course else ''
                    filename_str = f'File: "{file_name}", '
                    
                    # detect if it's html file
                    is_html_file = file_name.lower().endswith((".html", ".htm"))
                    if not is_html_file and source_page_url:
                        url_str = f'URLs: "{source_page_url} [{file_title}]({file_url})"'
                    else:
                        url_str = f'URL: "[{file_title}]({file_url})"'
                    
                    formatted_item = f'Chunk: "{chunk}"<doc_metadata>\n{topic_str}{professor_str}{course_str}{filename_str}{url_str}\n</doc_metadata>'
                    self.regulars.append(formatted_item)
                    self.documents.append(formatted_item)
            # log one example for each type
            logger.info(f"Example FAQ: {self.pairs[0]}")
            logger.info(f"Example Regular: {self.regulars[0]}")

        logger.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME}...")
        try:
            embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device='cuda') # Use GPU
            logger.info("Embedding model loaded. Generating embeddings for documents...")
            embeddings = embedding_model.encode(self.documents, convert_to_tensor=False, show_progress_bar=True)
            logger.info(f"Generated {embeddings.shape[0]} embeddings of dimension {embeddings.shape[1]}.")

            # Build FAISS index
            logger.info("Building FAISS index...")
            # Normalize embeddings for cosine similarity if using IndexFlatIP
            faiss.normalize_L2(embeddings)
            # index = faiss.IndexFlatL2(embeddings.shape[1]) # L2 distance index
            index = faiss.IndexFlatIP(embeddings.shape[1]) # Inner product index
            index.add(np.array(embeddings).astype('float32')) # Add embeddings to index
            self.retriever = index
            logger.info("FAISS index built successfully.")
            # Store embedding model ref if needed later, but encode is often one-off here
            self.embedding_model = embedding_model # Store model ref
        except Exception as e:
            logger.error(f"Failed to load embedding model or build FAISS index: {e}")
            raise

    @modal.enter()
    def load_model_and_data(self):
        """Load GGUF model AND setup the RAG retriever."""
        from llama_cpp import Llama

        logger.info("Container starting up...")

        # --- Load Original Data and Build Retriever FIRST ---
        try:
            self._load_and_index_data()
        except Exception as e:
            # Handle data loading/indexing failure gracefully
            logger.error(f"Failed to initialize RAG components: {e}", exc_info=True)
            self.retriever = None # Ensure retriever is None if setup fails


        # --- Load GGUF Model ---
        logger.info(f"Attempting to load Llama model from {self.gguf_file_path_in_container}...")
        if not os.path.exists(self.gguf_file_path_in_container):
            error_msg = f"GGUF model file not found at {self.gguf_file_path_in_container}."
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        # --- Load Helper LLM Model for query expansion ---
        logger.info(f"Attempting to load helper Llama model from {self.helper_llm_gguf_file_path_in_container}...")
        if not os.path.exists(self.helper_llm_gguf_file_path_in_container):
            logger.info(f"Helper LLM GGUF model file not found at {self.helper_llm_gguf_file_path_in_container}.")
            self.helper_llm = None
        else:
            self.helper_llm = Llama(
                model_path=self.helper_llm_gguf_file_path_in_container,
                n_gpu_layers=30, n_ctx=512, n_batch=512,
                f16_kv=True, verbose=True, seed=42, offload_kqv=True, use_mlock=True,
            )
            logger.info("Successfully loaded Helper LLM model into container.")

        try:
            self.llm = Llama(
                model_path=self.gguf_file_path_in_container,
                n_gpu_layers=60, n_ctx=CONTEXT_SIZE, n_batch=512,
                f16_kv=True, verbose=True, seed=42, offload_kqv=True, use_mlock=True,
            )
            logger.info("Successfully loaded Llama model into container.")
        except Exception as e:
            logger.error(f"Failed to load Llama model: {str(e)}", exc_info=True)
            self.llm = None # Ensure llm is None if loading fails
            # Optional: Raise error to prevent endpoint start if model load fails
            # raise RuntimeError(f"Failed to load LLM model: {e}") from e

    def _retrieve_context(self, query: str, k: int) -> List[str]:
        """Retrieve top-k relevant document chunks for the query."""
        import numpy as np
        import faiss

        if self.retriever is None or self.embedding_model is None:
            logger.error("Retriever or embedding model not initialized.")
            return []
        try:
            logger.info(f"Encoding query for retrieval...")
            query_embedding = self.embedding_model.encode([query], convert_to_tensor=False)
            query_embedding_np = np.array(query_embedding).astype('float32')
            faiss.normalize_L2(query_embedding_np) # Normalize if using IndexFlatIP

            logger.info(f"Searching FAISS index for top {k} documents...")
            distances, indices = self.retriever.search(query_embedding_np, k)
            logger.info(f"Retrieved indices: {indices[0]}")

            # Get the actual document content for the retrieved indices
            retrieved_docs = [self.documents[i] for i in indices[0]]
            return retrieved_docs
        except Exception as e:
            logger.error(f"Error during retrieval: {e}", exc_info=True)
            return []

    # Helper functions for intent detection and query expansion
    def _classify_intent(self, text: str) -> str:
        """Detects whether the message is an actual university question
        ('domain_query') or just small talk ('non_domain_query'). Falls back
        to 'domain_query' when the helper model isn't available."""
        if not self.helper_llm:
            return "domain_query"

        prompt = (
            "Você é um classificador de intenções para um chatbot da UnB.\n"
            "Separe em dois tipos:\n"
            "  • non_domain_query — saudações, agradecimentos, small talk, insultos, aleatoriedade, subjetividade\n"
            "  • domain_query (padrão) — qualquer outra coisa (pergunta factual sobre qualquer assunto relacionado à universidade)\n\n"
            "Agora, classifique somente esta mensagem e retorne apenas o tipo:\n"
            f"{text}\n"
            "Tipo:"
        )

        resp = self.helper_llm(prompt, max_tokens=16, temperature=0.1)
        return resp["choices"][0]["text"].strip().lower()

    def _expand_query(self, user_query: str) -> str:
        """Asks the helper model for a few alternative phrasings of the user
        question to improve retrieval."""
        if not self.helper_llm:
            return ""

        expansion_resp = self.helper_llm(
            QUERY_EXPANSION_PROMPT.format(user_query=user_query),
            max_tokens=512,
            temperature=1.0,
            top_p=0.95,
            top_k=64,
            min_p=0.01,
        )

        if expansion_resp and expansion_resp.get("choices"):
            return expansion_resp["choices"][0]["text"]
        return ""

    @modal.fastapi_endpoint(method="POST")
    async def generate_web(self, request_data: dict):
        """Handle RAG inference requests."""
        from fastapi import HTTPException

        if self.llm is None:
            logger.error("LLM model was not loaded successfully.")
            raise HTTPException(status_code=503, detail="LLM Model not ready.")
        if self.retriever is None:
             logger.error("Retriever was not initialized successfully.")
             raise HTTPException(status_code=503, detail="Retriever not ready.")

        user_query = request_data.get("prompt")# Rename to query for clarity
        if not user_query:
            raise HTTPException(status_code=400, detail="Missing 'prompt' (user query) in request.")

        # Get generation parameters
        max_tokens = request_data.get("max_tokens", DEFAULT_MAX_TOKENS)
        temperature = request_data.get("temperature", DEFAULT_TEMPERATURE)
        top_p = request_data.get("top_p", DEFAULT_TOP_P)

        logger.info(f"Received query: {user_query}")
        logger.info(f"Generation parameters: max_tokens={max_tokens}, temperature={temperature}, top_p={top_p}")

        # Decide the question type and, if appropriate, expand it
        intent = self._classify_intent(user_query)

        if intent != "domain_query":
            logger.info("Chitchat detected. Skipping query expansion.")
            additional_queries = ""
            max_tokens = 300  # keep answers short for chitchat
        else:
            additional_queries = self._expand_query(user_query)
            logger.info(f"Additional queries: {additional_queries}")

        # === RAFT RAG Logic ===
        assembled_context_str = ""
        # 1. Retrieve Context
        if intent == "domain_query":
            logger.info("Retrieving relevant context...")
            # additional_queries already logged above if generated.
            retrieved_docs = self._retrieve_context(user_query + "\n" + additional_queries, k=TOP_K_RETRIEVAL)
            prompt_to_use = SYSTEM_PROMPT
            if not retrieved_docs:
                logger.warning("No documents retrieved. Proceeding without context.")
                prompt_to_use = ""

            # 2. Format Context for RAFT Model
            for doc_content in retrieved_docs:
                assembled_context_str += f"<DOCUMENT>{doc_content}</DOCUMENT>\n\n"

        else:
            prompt_to_use = "You are a specialized UnB (Universidade de Brasília) chatbot assistant.\n\nPlease just respond in a friendly and engaging way in Portuguese, but be very brief and concise.\n\nOBS: If this requires a factual answer, beware: the question was classified in the non_domain_query category. You were not provided any source documents, so you have no information to correctly answer any UnB factual answer."

        # 3. Construct the Final Prompt for the LLM
        final_llm_prompt = prompt_to_use + "\n" + assembled_context_str + "\n\n" + user_query

       
        formatted_prompt = f"<start_of_turn>user\n{final_llm_prompt}<end_of_turn>\n<start_of_turn>model\n"

        logger.info(f"Constructed final prompt for LLM (length: {len(final_llm_prompt)} chars):")
        logger.info(final_llm_prompt)

        # 4. Generate Response using the RAFT-tuned GGUF model
        logger.info("Generating response using RAFT-tuned model...")
        try:
            output = self.llm(
                formatted_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=64,
                min_p=0.01,
                repeat_penalty=1.0,
                echo=False
            )

            # 5. Extract and Parse Response
            raw_response_text = ""
            if output and output.get('choices') and len(output['choices']) > 0:
                raw_response_text = output['choices'][0]['text']
                if usage := output.get('usage'):
                    logger.info(f"Token usage: {usage}")

            final_answer = raw_response_text

        except Exception as e:
            logger.error(f"Error during generation: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Error during generation: {str(e)}")

        logger.info("Generation complete.")
        return {"response": final_answer}

# Optional local entrypoint for testing