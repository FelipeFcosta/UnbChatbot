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

SYSTEM_PROMPT = "Você é um assistente especializado que responde diretamente à perguntas do usuário com base no seu conhecimento sobre dados de FAQ da UnB. Seja preciso e factual de acordo com o material de origem. Não invente informações. Utilize a tag `<REASON>:` para raciocínio com referências ao contexto e a tag `<ANSWER>:` para a resposta ao usuário em português.\n\n"
# SYSTEM_PROMPT = ""

# --- Configuration ---
APP_NAME = "unb-chatbot-raft-gguf-web-endpoint" # Updated name
# --- GGUF Model Details ---
MODEL_DIR_IN_VOLUME = "faq_raft_gemma4b_run3" # IMPORTANT: Point to your RAFT-tuned model directory
GGUF_FILENAME = "merged_model.Q8_0.gguf" # Check if path is correct within MODEL_DIR_IN_VOLUME
VOLUME_NAME = "faq-unb-chatbot-gemma-raft" # Volume where RAFT model and potentially data are stored
DATA_VOLUME_NAME = "faq-unb-chatbot-gemma-raft-data" # Volume where RAFT model and potentially data are stored
GPU_CONFIG = "A10G"
MODEL_MOUNT_PATH = "/model_files" # Where the GGUF model is mounted
DATA_MOUNT_PATH = "/data" # Added: Where original FAQ data will be mounted
CONTEXT_SIZE = 4096 # Keep same as training
ANSWER_TAG = "<ANSWER>:" # Tag used in training to mark the final answer

# --- RAG Configuration ---
FAQ_DOCUMENTS = f"{DATA_MOUNT_PATH}/extracted_faq_combined.json"
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-large-instruct"
# infly/inf-retriever-v1
TOP_K_RETRIEVAL = 5 # chunks to retrieve

# Default generation parameters
DEFAULT_MAX_TOKENS = 2048
DEFAULT_TEMPERATURE = 0.7 # Adjust as needed for RAFT model
DEFAULT_TOP_P = 0.9
MINUTES = 60



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
        gpu="A10G" # Build requires GPU access if CUDA is enabled
    )
    .entrypoint([])
)

model_volume = modal.Volume.from_name(VOLUME_NAME)
document_volume = modal.Volume.from_name(DATA_VOLUME_NAME)

@app.cls(
    image=image,
    gpu=GPU_CONFIG,
    volumes={
        MODEL_MOUNT_PATH: model_volume,
        DATA_MOUNT_PATH: document_volume # Example: using the same volume
    },
    timeout=60*6,
    allow_concurrent_inputs=3,
    min_containers=2
)
class ModelEndpoint:
    def __init__(self):
        self.gguf_file_path_in_container = str(
            Path(MODEL_MOUNT_PATH) / MODEL_DIR_IN_VOLUME / GGUF_FILENAME
        )
        self.llm = None

        self.retriever = None
        self.original_faqs: List[Dict] = []
        self.documents: List[str] = []

    def _load_and_index_data(self):
        """Loads original FAQ data and builds the FAISS index."""
        from sentence_transformers import SentenceTransformer
        import faiss
        import numpy as np

        logger.info(f"Loading original FAQ data from {FAQ_DOCUMENTS}...")
        if not os.path.exists(FAQ_DOCUMENTS):
             error_msg = f"Original FAQ data file not found at {FAQ_DOCUMENTS}."
             logger.error(error_msg)
             raise RuntimeError(error_msg)

        try:
            with open(FAQ_DOCUMENTS, 'r', encoding='utf-8') as f:
                # dict JSON -> {'question': '...', 'answer': '...'}
                # Adjust loading if your format is different (e.g., jsonl)
                self.original_faqs = json.load(f)
                # Format documents for retrieval (combine Q and A for better context)
                self.documents = [f'Q:"{faq["question"]}", A:"{faq["answer"]}"' for faq in self.original_faqs]
                self.documents = [
                    f'Q:"{faq["question"]}", A:"{faq["answer"]}"' + 
                    (f', Topics: "{faq["topics"]}"' if faq.get("topics") else '')
                    for faq in self.original_faqs
                ]
                
            logger.info(f"Loaded {len(self.original_faqs)} original FAQ pairs.")
        except Exception as e:
            logger.error(f"Failed to load or parse original FAQ data: {e}")
            raise

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

        try:
            self.llm = Llama(
                model_path=self.gguf_file_path_in_container,
                n_gpu_layers=-1, n_ctx=CONTEXT_SIZE, n_batch=512,
                f16_kv=True, verbose=True, seed=42, offload_kqv=True, use_mlock=True
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

        if self.retriever is None or self.embedding_model is None:
            logger.error("Retriever or embedding model not initialized.")
            return []
        try:
            logger.info(f"Encoding query for retrieval...")
            query_embedding = self.embedding_model.encode([query], convert_to_tensor=False)
            query_embedding_np = np.array(query_embedding).astype('float32')
            # faiss.normalize_L2(query_embedding_np) # Normalize if using IndexFlatIP

            logger.info(f"Searching FAISS index for top {k} documents...")
            distances, indices = self.retriever.search(query_embedding_np, k)
            logger.info(f"Retrieved indices: {indices[0]}")

            # Get the actual document content for the retrieved indices
            retrieved_docs = [self.documents[i] for i in indices[0]]
            return retrieved_docs
        except Exception as e:
            logger.error(f"Error during retrieval: {e}", exc_info=True)
            return []

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

        user_query = request_data.get("prompt") # Rename to query for clarity
        if not user_query:
            raise HTTPException(status_code=400, detail="Missing 'prompt' (user query) in request.")

        # Get generation parameters
        max_tokens = request_data.get("max_tokens", DEFAULT_MAX_TOKENS)
        temperature = request_data.get("temperature", DEFAULT_TEMPERATURE)
        top_p = request_data.get("top_p", DEFAULT_TOP_P)

        logger.info(f"Received query: {user_query}")
        logger.info(f"Generation parameters: max_tokens={max_tokens}, temperature={temperature}, top_p={top_p}")

        # === RAFT RAG Logic ===
        # 1. Retrieve Context
        logger.info("Retrieving relevant context...")
        retrieved_docs = self._retrieve_context(user_query, k=TOP_K_RETRIEVAL)
        prompt_to_use = SYSTEM_PROMPT
        if not retrieved_docs:
            logger.warning("No documents retrieved. Proceeding without context.")
            prompt_to_use = ""
            # Handle case with no context - maybe just use system prompt + query
            # Or return a specific message

        # 2. Format Context for RAFT Model
        assembled_context_str = ""
        for doc_content in retrieved_docs:
            # Using the same <DOCUMENT> tag as in training data generation
            # Assuming your original data used Q:/A: format within the context
            assembled_context_str += f"<DOCUMENT>{doc_content}</DOCUMENT>\n"

        # 3. Construct the Final Prompt for the LLM
        final_llm_prompt = prompt_to_use + assembled_context_str + "\n" + user_query

        # Prepare messages for llama-cpp chat format
        messages = [
            {"role": "user", "content": final_llm_prompt}
        ]
        logger.info(f"Constructed final prompt for LLM (length: {len(final_llm_prompt)} chars):")
        logger.info(final_llm_prompt)
        # logger.debug(f"Final Prompt:\n{final_llm_prompt}") # Optional: log full prompt for debug

        # 4. Generate Response using the RAFT-tuned GGUF model
        logger.info("Generating response using RAFT-tuned model...")
        try:
            output = self.llm.create_chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=["<end_of_turn>"], # Stop before ANSWER_TAG if parsing based on it
                repeat_penalty=1.1
            )

            # 5. Extract and Parse Response
            raw_response_text = ""
            if output and output.get('choices') and len(output['choices']) > 0:
                raw_response_text = output['choices'][0]['message']['content']
                if usage := output.get('usage'):
                    logger.info(f"Token usage: {usage}")

            # Extract only the text AFTER the <ANSWER>: tag
            final_answer = raw_response_text # Default to full response if tag not found
            # if ANSWER_TAG in raw_response_text:
            #     try:
            #         # Split at the *last* occurrence of the tag in case it appears in reasoning
            #         parts = raw_response_text.rsplit(ANSWER_TAG, 1)
            #         if len(parts) > 1:
            #             final_answer = parts[1].strip()
            #             logger.info(f"Successfully extracted answer after '{ANSWER_TAG}' tag.")
            #         else:
            #              logger.warning(f"'{ANSWER_TAG}' tag found, but splitting failed unexpectedly.")
            #     except Exception as parse_err:
            #          logger.warning(f"Error parsing response for '{ANSWER_TAG}': {parse_err}")
            # else:
            #      logger.warning(f"'{ANSWER_TAG}' tag not found in model response. Returning full response.")

        except Exception as e:
            logger.error(f"Error during generation: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Error during generation: {str(e)}")

        logger.info("Generation complete.")
        return {"response": final_answer} # Return only the parsed answer

# Optional local entrypoint for testing
@app.local_entrypoint()
def main(prompt: str = "O Enade é obrigatório pra quem é formando?"):
    """Test the RAG endpoint locally"""
    logger.info(f"Testing RAG endpoint locally with prompt: '{prompt}'")
    # Simulate request data structure
    request_data = {"prompt": prompt}
    # Instantiate the class locally (will load model/data if run with `modal run`)
    endpoint = ModelEndpoint()
    # Manually trigger the loading process if needed for pure local test (won't work fully without Modal env)
    # endpoint.load_model_and_data()
    # Note: Calling .remote() here will execute it in Modal cloud if script run directly
    response = endpoint.generate_web.remote(request_data)
    print(f"\n--- Response ---")
    print(response.get('response', 'No response generated.'))
    print("----------------\n")