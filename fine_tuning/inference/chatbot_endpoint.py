#!/usr/bin/env python3
"""
Run inference using a fine-tuned Unsloth LLM model stored in a Modal Volume,
exposed as a persistent web endpoint.
The base model is automatically detected from the saved adapter configuration.
"""

import modal
import os
import logging
import json # Import json to read config files

# --- Configuration ---
APP_NAME = "unb-chatbot-gemma-web-endpoint" # Updated app name
DEFAULT_MODEL_DIR_NAME = "faq_gemma12b_run4/checkpoint-260" # Directory *inside* the volume
VOLUME_NAME = "faq-unb-chatbot-gemma"       # Name of the Modal Volume
GPU_CONFIG = "T4"                        # GPU for inference
# ---------------------

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# Define Modal app and resources
app = modal.App(APP_NAME)

# Image definition - Added fastapi
image = (
    modal.Image.debian_slim()
    .env({"LC_ALL": "C.UTF-8", "LANG": "C.UTF-8", "PYTHONIOENCODING": "utf-8"})
    .apt_install("git")
    .pip_install(
        "torch",
        "unsloth",
        "datasets",
        "xformers",
        "transformers",
        "trl",
        "triton",
        "huggingface_hub",
        "bitsandbytes",
        "accelerate",
        "sentencepiece",
        "protobuf",
        "fastapi[standard]" # <--- Added FastAPI dependency
    )
)

# Access the Volume where the model is stored
model_volume = modal.Volume.from_name(VOLUME_NAME)
MODEL_MOUNT_PATH = "/model_outputs"

# --- Global cache for model and tokenizer to avoid reloading on every request ---
# Note: This basic caching works well with min_containers=1 or keep_warm=1
# More advanced caching might be needed for complex scaling scenarios.
model_cache = {}

def load_model_and_tokenizer(saved_model_path):
    """Loads model and tokenizer, caching them."""
    from unsloth import FastModel
    import torch

    if saved_model_path in model_cache:
        logger.info(f"Using cached model and tokenizer for {saved_model_path}")
        return model_cache[saved_model_path]

    logger.info(f"Loading model and tokenizer from {saved_model_path}...")
    try:
        model, tokenizer = FastModel.from_pretrained(
            model_name=saved_model_path,
            max_seq_length=4096,
            dtype=None,
            load_in_4bit=True,
        )
        FastModel.for_inference(model) # Ensure model is ready for inference
        model_cache[saved_model_path] = (model, tokenizer)
        logger.info(f"Successfully loaded and cached model and tokenizer.")
        return model, tokenizer
    except Exception as e:
        logger.error(f"!!! Failed to load model from {saved_model_path} !!!")
        logger.exception(e)
        raise # Re-raise the exception to signal failure

@app.function(
    image=image,
    gpu=GPU_CONFIG,
    volumes={MODEL_MOUNT_PATH: model_volume},
    timeout=60 * 5, # Adjusted timeout slightly
    allow_concurrent_inputs=10, # Allow multiple requests per container
    min_containers=1,           # Keep at least one container warm to reduce cold starts
    # keep_warm=1, # Alternative to min_containers=1 for keeping one instance ready
)
@modal.fastapi_endpoint(method="POST") # <--- Changed to web endpoint, expecting POST
async def generate_web(request_data: dict): # <--- Changed function signature to accept JSON dict
    """Generates a response using the fine-tuned model via a web endpoint."""
    from unsloth.chat_templates import get_chat_template
    import torch
    # Removed TextStreamer import

    # --- Get model directory (can eventually be dynamic via request_data if needed) ---
    model_dir_name = DEFAULT_MODEL_DIR_NAME # Or potentially get from request_data
    saved_model_path = os.path.join(MODEL_MOUNT_PATH, model_dir_name)

    # --- Validate adapter config path (optional but good practice) ---
    adapter_config_path = os.path.join(saved_model_path, "adapter_config.json")
    if not os.path.exists(adapter_config_path):
         error_msg = f"adapter_config.json not found at {adapter_config_path}."
         logger.error(error_msg)
         # Return FastAPI compatible error response
         from fastapi import HTTPException
         raise HTTPException(status_code=500, detail=error_msg)

    # --- Extract prompt from request ---
    user_prompt = request_data.get("prompt")
    if not user_prompt:
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail="Missing 'prompt' in request JSON body.")

    logger.info(f"Received prompt: {user_prompt[:50]}...") # Log truncated prompt

    # --- Load Model (using cache) ---
    # This ensures the model isn't reloaded unnecessarily for subsequent requests
    # handled by the same warm container.
    try:
        model, tokenizer = load_model_and_tokenizer(saved_model_path)
    except Exception as e:
        from fastapi import HTTPException
        raise HTTPException(status_code=500, detail=f"Failed to load model: {e}")

    # --- Prepare the prompt ---
    messages = [{
        "role": "user",
        "content": [{"type": "text", "text": user_prompt}]
    }]

    # Re-get tokenizer with template inside the request function if needed
    # or ensure the cached tokenizer has the template applied (better)
    # For simplicity here, we re-apply. A more robust cache would handle this.
    tokenizer = get_chat_template(
        tokenizer,
        chat_template = "gemma-3",
    )

    text = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt = True,
    )
    inputs = tokenizer([text], return_tensors="pt").to("cuda")

    # --- Generate the response (No Streaming) ---
    logger.info("Generating response...")
    outputs = model.generate(
        **inputs,
        max_new_tokens = 3096,
        temperature = 1.0, top_p = 0.95, top_k = 64,
        eos_token_id = tokenizer.eos_token_id # Important for stopping generation
    )

    # --- Decode the response (Only the generated part) ---
    input_token_len = inputs.input_ids.shape[1]
    # Handle potential edge case where nothing new is generated
    if outputs.shape[1] > input_token_len:
        generated_tokens = outputs[0][input_token_len:]
        decoded_response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    else:
        decoded_response = "" # Or some indicator that nothing was generated

    logger.info("Generation complete.")

    # --- Return JSON response ---
    return {"response": decoded_response}

# --- Remove the local_entrypoint ---
# @app.local_entrypoint()
# def main(...): ...