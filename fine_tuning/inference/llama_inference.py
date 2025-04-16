#!/usr/bin/env python3
"""
Run inference using a GGUF LLM model stored in a Modal Volume,
exposed as a persistent web endpoint using llama-cpp-python.
"""

import modal
import os
import logging
from pathlib import Path

# --- Configuration ---
APP_NAME = "unb-chatbot-gguf-web-endpoint"
# Directory *inside* the volume where the fine-tuning output (including GGUF) was saved
MODEL_DIR_IN_VOLUME = "faq_gemma4b_run9" # <--- CHANGE THIS
# Exact filename of the GGUF model within the MODEL_DIR_IN_VOLUME
# *** VERIFY THIS FILENAME is correct in your volume ***
GGUF_FILENAME = "merged_model.Q8_0.gguf" # <--- CHANGE THIS (Added dot before gguf based on common 
VOLUME_NAME = "faq-unb-chatbot-gemma"       # Name of the Modal Volume used during training
GPU_CONFIG = "A10G" # Or A10G() if T4 is insufficient VRAM modal.gpu.A10G()
MODEL_MOUNT_PATH = "/model_files"          # Where the volume will be mounted inside the container
CONTEXT_SIZE = 4096                        # Context size used during training/model capacity
SYSTEM_PROMPT = "Você é um assistente especializado que responde perguntas *estritamente* com base nos dados de FAQ da UnB fornecidos no contexto. Seja preciso e factual de acordo com o material de origem. Não invente informações.\n\n"# ---------------------

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# Define Modal app and resources
app = modal.App(APP_NAME)

# Image definition - Installs llama-cpp-python with potential GPU support
# Modal's base CUDA images usually handle the compilation details well.
image = (
    modal.Image.debian_slim(python_version="3.10") # Match python version if needed
    .apt_install(
        "git"  # <--- ADD THIS LINE TO INSTALL GIT
    )
    .pip_install(
        "modal-client",
        "fastapi[standard]",
        # Installs llama-cpp-python; Modal's build process often handles BLAS/CUDA correctly.
        # Add extras if needed, e.g., "[server]" if using its server features elsewhere.
        "llama-cpp-python",
        "huggingface_hub", # Good practice, sometimes used by llama-cpp for metadata
    )
)

# Access the Volume where the GGUF model is stored
model_volume = modal.Volume.from_name(VOLUME_NAME)

# --- Global cache for the Llama object ---
model_cache = {}

def load_llama_model(model_path_in_container: str):
    """Loads the Llama model using llama-cpp-python, caching it."""
    from llama_cpp import Llama

    if model_path_in_container in model_cache:
        logger.info(f"Using cached Llama model for {model_path_in_container}")
        return model_cache[model_path_in_container]

    logger.info(f"Loading Llama model from {model_path_in_container}...")
    try:
        # n_gpu_layers=-1 means offload all possible layers to GPU
        # Check llama-cpp-python docs for optimal settings for your GPU type if needed
        llm = Llama(
            model_path=model_path_in_container,
            n_gpu_layers=-1,
            n_ctx=CONTEXT_SIZE,
            verbose=True, # Log llama.cpp details
            # chat_format="gemma" # Specify chat format if needed and supported by your llama-cpp version
                                 # Alternatively, format manually. Gemma template might be auto-detected.
        )
        model_cache[model_path_in_container] = llm
        logger.info(f"Successfully loaded and cached Llama model.")
        return llm
    except Exception as e:
        logger.error(f"!!! Failed to load Llama model from {model_path_in_container} !!!")
        logger.exception(e)
        raise # Re-raise the exception to signal failure

@app.function(
    image=image,
    gpu=GPU_CONFIG,
    volumes={MODEL_MOUNT_PATH: model_volume},
    timeout=60 * 3,             # 3 minutes timeout
    allow_concurrent_inputs=5,  # Adjust based on expected load and GPU memory
    min_containers=1,           # Keep at least one container warm
    # keep_warm=1, # Alternative to min_containers=1
)
@modal.fastapi_endpoint(method="POST")
async def generate_web(request_data: dict):
    """Generates a response using the GGUF model via a web endpoint."""
    from fastapi import HTTPException
    # llama_cpp is imported within load_llama_model

    # --- Construct the full path to the GGUF file ---
    gguf_file_path_in_container = str(Path(MODEL_MOUNT_PATH) / MODEL_DIR_IN_VOLUME / GGUF_FILENAME)

    # --- Validate GGUF file existence ---
    if not os.path.exists(gguf_file_path_in_container):
         error_msg = f"GGUF model file not found at {gguf_file_path_in_container}. Check VOLUME_NAME, MODEL_DIR_IN_VOLUME, and GGUF_FILENAME."
         logger.error(error_msg)
         raise HTTPException(status_code=500, detail=error_msg)

    # --- Extract prompt from request ---
    user_prompt = request_data.get("prompt")
    if not user_prompt:
        raise HTTPException(status_code=400, detail="Missing 'prompt' in request JSON body.")

    logger.info(f"Received prompt: {user_prompt[:100]}...") # Log truncated prompt

    # --- Load Model (using cache) ---
    try:
        llm = load_llama_model(gguf_file_path_in_container)
    except Exception as e:
        logger.error(f"Failed to load GGUF model: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load GGUF model: {e}")

    # --- Prepare the prompt using llama-cpp's chat handler (preferred) ---
    # This attempts to use the model's built-in chat template if available.
    # Your fine-tuning script used "gemma-3" template. llama-cpp might detect it.
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt}
    ]

    logger.info("Generating response...")
    try:
        output = llm.create_chat_completion(
            messages=messages,
            max_tokens=1024,         # Max tokens to generate *in the response*
            temperature=0.6,       # Adjust creativity/determinism
            top_p=0.9,             # Nucleus sampling
            stop=["<end_of_turn>"], # Explicitly stop at Gemma's EOT token if needed
                                   # (check if llama-cpp handles this automatically)
            # stream=False, # Set to True if you want streaming output (requires client changes)
        )

        # --- Extract the response ---
        if output and output.get('choices') and len(output['choices']) > 0:
            response_text = output['choices'][0]['message']['content']
            # Log token usage if available
            if usage := output.get('usage'):
                logger.info(f"Token usage: {usage}")
        else:
            logger.warning("No response generated or unexpected output format.")
            response_text = "" # Or an error message

    except Exception as e:
        logger.error(f"Error during Llama generation: {e}")
        raise HTTPException(status_code=500, detail=f"Error during generation: {e}")

    logger.info("Generation complete.")

    # --- Return JSON response ---
    return {"response": response_text.strip()}

# --- No local_entrypoint needed for a web endpoint ---