#!/usr/bin/env python3
"""
Run inference using a fine-tuned Unsloth LLM model stored in a Modal Volume.
The base model is automatically detected from the saved adapter configuration.
"""

import modal
import os
import logging
import argparse
import json # Import json to read config files

# --- Configuration ---
APP_NAME = "unb-chatbot-gemma12b-a10g"
DEFAULT_MODEL_DIR_NAME = "unb_chatbot_gemma12b_a10g_patched" # Directory *inside* the volume
VOLUME_NAME = "unb-chatbot-models"       # Name of the Modal Volume
GPU_CONFIG = "T4"                        # GPU for inference
# BASE_MODEL variable is removed - it will be inferred from saved_model_path
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

# Image definition
image = (
    modal.Image.debian_slim()
    .env({"LC_ALL": "C.UTF-8", "LANG": "C.UTF-8", "PYTHONIOENCODING": "utf-8"}) # Set UTF-8 Env Vars
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
        "protobuf"
    )
)

# Access the Volume where the model is stored
model_volume = modal.Volume.from_name(VOLUME_NAME)
MODEL_MOUNT_PATH = "/model_outputs" # Where the volume is mounted

@app.function(
    image=image,
    gpu=GPU_CONFIG,
    volumes={MODEL_MOUNT_PATH: model_volume},
    timeout=60 * 10,
    allow_concurrent_inputs=10,
    min_containers=1,
)
def generate(user_prompt: str, model_dir_name: str = DEFAULT_MODEL_DIR_NAME):
    """Generates a response using the fine-tuned model."""
    from unsloth import FastLanguageModel
    import torch

    # --- 1. Construct the path to your saved model ---
    saved_model_path = os.path.join(MODEL_MOUNT_PATH, model_dir_name)
    logger.info(f"Target saved model path (in container): {saved_model_path}")

    # --- Check for essential adapter config file ---
    adapter_config_path = os.path.join(saved_model_path, "adapter_config.json")
    if not os.path.exists(adapter_config_path):
         error_msg = f"adapter_config.json not found at {adapter_config_path}. Cannot infer base model. Check volume contents and model_dir_name."
         logger.error(error_msg)
         raise FileNotFoundError(error_msg)
    logger.info(f"Found adapter_config.json at: {adapter_config_path}")

    # --- Log info before loading ---
    logger.info("="*30)
    logger.info("Preparing to load fine-tuned model (base + adapters)...")
    logger.info(f"  Loading from saved path: {saved_model_path}")
    logger.info(f"  Base model will be automatically inferred from configuration files in this path.")

    # Try to read the base model from the adapter config for logging confirmation
    base_model_from_config = "N/A" # Default value
    try:
        with open(adapter_config_path, 'r') as f:
            adapter_config = json.load(f)
            base_model_from_config = adapter_config.get("base_model_name_or_path", "N/A")
            logger.info(f"  Base model specified in adapter_config.json: {base_model_from_config}")
    except Exception as e:
        logger.warning(f"Could not read base model from adapter_config.json for logging: {e}")

    logger.info("="*30)

    # --- 2. Load the fine-tuned model ---
    # Unsloth automatically loads the base model specified in the config files
    # within `saved_model_path` and then applies the LoRA adapters from the same path.
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=saved_model_path, # <--- Load from the saved directory
            max_seq_length=2048,
            dtype=None,
            load_in_4bit=True,
        )
        logger.info(f"Successfully loaded model based on '{base_model_from_config}' and applied adapters.")
        # Log model details *after* loading
        logger.info(f"  Loaded model class: {model.__class__.__name__}")
        logger.info(f"  Model device map: {getattr(model, 'hf_device_map', 'N/A')}")
        try:
           logger.info(f"  Model footprint (approx GB): {model.get_memory_footprint() / (1024**3):.2f} GB")
           total_params = sum(p.numel() for p in model.parameters())
           trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
           logger.info(f"  Total parameters (base + active adapters): {total_params:,}")
           logger.info(f"  Trainable parameters (LoRA): {trainable_params:,}")
        except Exception as e:
           logger.warning(f"Could not get detailed model stats: {e}")

    except Exception as e:
        logger.error(f"!!! Failed to load model from {saved_model_path} !!!")
        logger.exception(e)
        raise

    # Ensure PEFT model is used for inference
    FastLanguageModel.for_inference(model)

    # --- 3. Prepare the prompt ---
    messages = [
        {"role": "user", "content": user_prompt},
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(next(model.parameters()).device)

    # --- 4. Generate the response ---
    logger.info("Generating response...")
    # (Generation code remains the same)
    outputs = model.generate(
        input_ids=inputs,
        max_new_tokens=512,
        use_cache=True,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id
    )

    # --- 5. Decode the response ---
    response_tokens = outputs[0][inputs.shape[-1]:]
    decoded_response = tokenizer.decode(response_tokens, skip_special_tokens=True)

    logger.info("Generation complete.")
    return decoded_response

@app.local_entrypoint()
def main(prompt: str = "ENADE é obrigatório?", model_dir: str = DEFAULT_MODEL_DIR_NAME):
    """Local entrypoint to test the inference function."""
    if not prompt:
        print("Error: Please provide a prompt using --prompt 'Your question here'")
        return

    print(f"\nSending prompt to the fine-tuned model in volume directory: '{model_dir}'...")
    print("(Base model is automatically detected from the saved model files)")

    try:
        response = generate.remote(user_prompt=prompt, model_dir_name=model_dir)
        print("\nModel Response:")
        print("-" * 20)
        print(response)
        print("-" * 20)
    except Exception as e:
        print("\n--- ERROR DURING REMOTE EXECUTION ---")
        print(f"An error occurred: {e}")
        print("Check the Modal logs for function 'generate' for more details.")