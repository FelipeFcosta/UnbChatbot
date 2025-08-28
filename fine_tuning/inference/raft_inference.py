#!/usr/bin/env python3
"""
Run RAFT inference using a GGUF LLM model stored in a Modal Volume,
exposed as a persistent web endpoint using llama-cpp-python with CUDA support.
Includes Retrieval-Augmented Generation (RAG) logic.
"""

import modal
import os
import logging
from pathlib import Path
from typing import List, Dict

# Package context
from .config import *
from .data_handler import DataHandler
from .query_processor import QueryProcessor
from .prompt_builder import PromptBuilder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

app = modal.App(APP_NAME)

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
        self.gguf_file_path_in_container = str(Path(MODEL_MOUNT_PATH) / MODEL_DIR_IN_VOLUME / CHECKPOINT_FOLDER / GGUF_FILENAME)
        self.helper_llm_gguf_file_path_in_container = str(Path(HELPER_LLM_MODEL_MOUNT_PATH) / HELPER_LLM_MODEL_DIR_IN_VOLUME / GGUF_FILENAME)
        self.llm = None
        self.helper_llm = None
        self.data_handler = DataHandler(logger)
        self.query_processor = None
        self.prompt_builder = None

    @modal.enter()
    def load_model_and_data(self):
        """Load GGUF models, data, and initialize handlers."""
        from llama_cpp import Llama
        logger.info("Container starting up...")

        try:
            self.data_handler.load_and_index_data()
        except Exception as e:
            logger.error(f"Failed to initialize RAG components: {e}", exc_info=True)

        self._load_llm_models()

        self.query_processor = QueryProcessor(self.helper_llm, logger)
        self.prompt_builder = PromptBuilder(self.query_processor.get_non_domain_hashes(), logger)

    def _load_llm_models(self):
        """Loads the main and helper LLM models."""
        from llama_cpp import Llama
        logger.info(f"Attempting to load Llama model from {self.gguf_file_path_in_container}...")
        if not os.path.exists(self.gguf_file_path_in_container):
            error_msg = f"GGUF model file not found at {self.gguf_file_path_in_container}."
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        logger.info(f"Attempting to load helper Llama model from {self.helper_llm_gguf_file_path_in_container}...")
        if os.path.exists(self.helper_llm_gguf_file_path_in_container):
            self.helper_llm = Llama(
                model_path=self.helper_llm_gguf_file_path_in_container,
                n_gpu_layers=30, n_ctx=2048, n_batch=512,
                f16_kv=True, verbose=True, seed=42, offload_kqv=True, use_mlock=True,
            )
            logger.info("Successfully loaded Helper LLM model into container.")
        else:
            logger.info(f"Helper LLM GGUF model file not found at {self.helper_llm_gguf_file_path_in_container}.")
            self.helper_llm = None

        try:
            self.llm = Llama(
                model_path=self.gguf_file_path_in_container,
                n_gpu_layers=60, n_ctx=CONTEXT_SIZE, n_batch=512,
                f16_kv=True, verbose=True, seed=42, offload_kqv=True, use_mlock=True,
            )
            logger.info("Successfully loaded Llama model into container.")
        except Exception as e:
            logger.error(f"Failed to load Llama model: {str(e)}", exc_info=True)
            self.llm = None
    
    def _get_generation_params(self, request_data: Dict) -> tuple[int, float, float]:
        """Extract generation parameters from request data."""
        max_tokens = request_data.get("max_tokens", DEFAULT_MAX_TOKENS)
        temperature = request_data.get("temperature", DEFAULT_TEMPERATURE)
        top_p = request_data.get("top_p", DEFAULT_TOP_P)
        return max_tokens, temperature, top_p

    

    def _prepare_context_and_prompt(self, intent: str, user_query: str, contextualized_query: str) -> tuple[str, str]:
        """Prepare context (documents) and system prompt"""
        if intent == "domain_query":
            additional_queries = self.query_processor.expand_query(contextualized_query)
            logger.info(f"Additional queries: {additional_queries}")
            retrieval_input = user_query + "\n" + contextualized_query + "\n" + additional_queries
            logger.info("Retrieving relevant context...")
            retrieved_docs = self.data_handler.retrieve_context(retrieval_input, k=TOP_K_RETRIEVAL)
            
            system_prompt = SYSTEM_PROMPT
            if not retrieved_docs:
                logger.warning("No documents retrieved. Proceeding without context.")
                system_prompt = ""

            assembled_context_str = "".join(f"<DOCUMENT>{doc}</DOCUMENT>\n\n" for doc in retrieved_docs)
        else:
            logger.info("Chitchat detected. Skipping query expansion and retrieval.")
            assembled_context_str = ""
            system_prompt = CHITCHAT_PROMPT
            retrieved_docs = []
        
        return assembled_context_str, system_prompt, retrieved_docs


    @modal.fastapi_endpoint(method="POST")
    async def generate_web(self, request_data: dict):
        """Handle RAG inference requests."""
        from fastapi import HTTPException

        if self.llm is None:
            logger.error("LLM model was not loaded successfully.")
            raise HTTPException(status_code=503, detail="Model not ready.")
        if self.query_processor is None or self.prompt_builder is None:
            logger.error("Query processor or prompt builder not initialized.")
            raise HTTPException(status_code=503, detail="Dependencies not ready.")
        if self.data_handler.retriever is None:
            logger.error("Retriever was not initialized successfully.")
            raise HTTPException(status_code=503, detail="Retriever not ready.")

        # get the messages array from the request
        messages = request_data.get("messages")
        if not isinstance(messages, list) or not messages or messages[-1].get("role") != "user":
            raise HTTPException(status_code=400, detail="Request must include a non-empty 'messages' array ending with a user message.")

        # identify user query
        user_query = (messages[-1].get("content") or "").strip()
        if not user_query:
            raise HTTPException(status_code=400, detail="Last user message has no content.")

        # get generation parameters for the main LLM
        max_tokens, temperature, top_p = self._get_generation_params(request_data)
        logger.info(f"Received query: {user_query}")
        logger.info(f"Generation parameters: max_tokens={max_tokens}, temperature={temperature}, top_p={top_p}")

        # classify intent based on the last in-domain exchange + current query
        in_domain_history = self.prompt_builder.process_history(messages)
        previous_question = ""
        previous_response = ""
        if len(in_domain_history) >= 2:
            previous_response = (messages[-3].get("content") or "").strip()
            for i in range(len(in_domain_history) - 2, -1, -1):
                if (
                    in_domain_history[i].get("role") == "user"
                    and i + 1 < len(in_domain_history)
                    and in_domain_history[i + 1].get("role") == "assistant"
                ):
                    previous_question = (in_domain_history[i].get("content") or "").strip()
                    previous_response = (in_domain_history[i + 1].get("content") or "").strip()
                    break
        intent = self.query_processor.classify_intent(user_query, previous_question, previous_response)

        contextualized_query = user_query
        if intent != "domain_query":
            max_tokens = 500
        else:
            # chat history for contextualization (reuse processed history)
            chat_history_str = self.prompt_builder.build_chat_history_str(in_domain_history)

            # make query self-contained
            if chat_history_str:
                contextualized_query = self.query_processor.contextualize_query(chat_history_str, user_query)
                logger.info(f"Contextualized query: {contextualized_query}")

        # gather context documents from contextualized query and its expansions
        assembled_context_str, system_prompt, retrieved_docs = self._prepare_context_and_prompt(intent, user_query, contextualized_query)

        formatted_prompt = self.prompt_builder.build_prompt(messages, system_prompt, assembled_context_str, contextualized_query)
        logger.info(f"Constructed final prompt for LLM (length: {len(formatted_prompt)} chars). Prompt content follows:\n{formatted_prompt}")

        try:
            output = self.llm(
                formatted_prompt, max_tokens=max_tokens, temperature=temperature, top_p=top_p,
                top_k=64, min_p=0.01, repeat_penalty=1.0, echo=False
            )
            raw_response_text = output["choices"][0]["text"] if output and output.get("choices") else ""
            if usage := output.get("usage"):
                logger.info(f"Token usage: {usage}")
            
            logger.info("Generation complete.")
            return {"response": raw_response_text, "retrieved_chunks": retrieved_docs}
        except Exception as e:
            logger.error(f"Error during generation: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Error during generation: {str(e)}")
