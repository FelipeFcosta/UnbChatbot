#!/usr/bin/env python3
"""
Load a previously merged Hugging Face model from a Modal Volume
and export it to GGUF format using Unsloth.
"""

import os
import logging
from pathlib import Path
import modal
from huggingface_hub import login # For potential push to hub

# --- Configuration ---
GPU = "A10G" # Might still need GPU for loading large models, though conversion is CPU-based
INPUT_VOLUME_NAME = "faq-unb-chatbot-gemma-dpo" # Volume WHERE MERGED MODEL IS SAVED
OUTPUT_VOLUME_NAME = "faq-unb-chatbot-gemma-dpo" # Volume WHERE GGUF WILL BE SAVED (can be the same)
# ---

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# Define Modal app and resources
app = modal.App("unb-chatbot-gguf-export") # New app name for clarity

# Reuse the same image definition from your DPO training script
# Ensure it includes all necessary dependencies (unsloth, transformers, torch etc.)
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git")
    .pip_install(
        "ipython",
        "torch",
        "bitsandbytes",
        "accelerate",
        "xformers==0.0.29.post3",
        "triton",
        "unsloth",
        "unsloth_zoo",
        "cut_cross_entropy",
        "git+https://github.com/huggingface/transformers@v4.49.0-Gemma-3",
        "peft",
        "trl==0.15.2",
        "datasets",
        "huggingface_hub",
        "hf_transfer",
        "sentencepiece",
        "protobuf",
    )
)

# Volume containing the merged model (read access)
input_volume = modal.Volume.from_name(INPUT_VOLUME_NAME)
# Volume to save the GGUF file (write access)
output_volume = modal.Volume.from_name(OUTPUT_VOLUME_NAME)

INPUT_MOUNT_PATH = "/input_model"
OUTPUT_MOUNT_PATH = "/output_gguf"

huggingface_secret = modal.Secret.from_name("huggingface")

@app.function(
    image=image,
    gpu=GPU, # Keep GPU for loading flexibility
    timeout = 60 * 60, # 1 hour timeout should be plenty
    volumes={
        INPUT_MOUNT_PATH: input_volume,
        OUTPUT_MOUNT_PATH: output_volume,
        },
    cpu=2, # Can adjust based on needs
    memory=24576, # Keep memory reasonable for loading
    secrets=[huggingface_secret], # Needed if pushing to hub
)
def export_to_gguf(
    merged_model_dir_name: str, # e.g., "unb_chatbot_gemma4b_dpo/merged_model"
    output_dir_name: str,      # e.g., "unb_chatbot_gemma4b_dpo" (GGUF will be saved here)
    quantization_method: str = "q8_0", # Desired quantization (e.g., f16, q4_k_m)
    push_to_hub: bool = False,
    hf_repo_id: str = None,
    hf_token_env_var: str = "HF_TOKEN", # Allow specifying token env var name
):
    """Loads a merged model and exports it to GGUF."""
    from unsloth import FastModel # Assuming FastModel can load standard HF models
    import torch
    from transformers import AutoTokenizer # Use standard tokenizer loading
    from unsloth import is_bfloat16_supported

    logger.info(f"Starting GGUF export process...")
    logger.info(f"  Merged Model Directory (relative to volume): {merged_model_dir_name}")
    logger.info(f"  Output Directory (relative to volume): {output_dir_name}")
    logger.info(f"  Quantization Method: {quantization_method}")
    logger.info(f"  Push to Hub: {push_to_hub}")
    if push_to_hub:
        logger.info(f"  HF Repo ID: {hf_repo_id}")

    merged_model_path = os.path.join(INPUT_MOUNT_PATH, merged_model_dir_name)
    output_path = os.path.join(OUTPUT_MOUNT_PATH, output_dir_name)
    os.makedirs(output_path, exist_ok=True) # Ensure output dir exists

    hf_token = os.environ.get(hf_token_env_var)
    if push_to_hub and not hf_token:
        logger.warning(f"Push to Hub requested but Hugging Face token not found in environment variable '{hf_token_env_var}'.")
        raise ValueError(f"Hugging Face token needed for push_to_hub=True.")
    if push_to_hub:
        logger.info(f"Logging in to Hugging Face Hub.")
        login(token=hf_token)

    try:
        logger.info(f"Loading merged model from: {merged_model_path}")
        # Load the merged model - use appropriate precision and avoid device map issues
        # Loading in full precision (fp16/bf16) is generally needed before quantization
        dtype = torch.bfloat16 if is_bfloat16_supported() else torch.float16
        model, _ = FastModel.from_pretrained(
            model_name=merged_model_path,
            load_in_4bit=False, # Load in full precision
            load_in_8bit=False,
            dtype=dtype,
            device_map = "auto", # Let accelerate handle placement, safer than forcing CPU if OOM
            # device_map = None, # Try None if auto fails and you have enough RAM/VRAM
            # token = hf_token, # Usually not needed for local loading
        )
        # Load the tokenizer associated with the merged model
        tokenizer = AutoTokenizer.from_pretrained(merged_model_path)
        logger.info("Merged model and tokenizer loaded successfully.")

        # --- Save GGUF ---
        logger.info(f"Exporting model to GGUF format ({quantization_method})...")
        logger.info(f"Saving GGUF to: {output_path}")
        model.save_pretrained_gguf(
            save_directory=output_path,
            tokenizer=tokenizer,
            quantization_method=quantization_method.lower() # Ensure lowercase e.g., 'q8_0'
        )
        logger.info("GGUF export completed successfully.")
        gguf_files = [f for f in os.listdir(output_path) if f.endswith('.gguf')]

        # --- Push to Hub (Optional) ---
        if push_to_hub and hf_repo_id:
            if gguf_files:
                logger.info(f"Pushing GGUF model to Hugging Face Hub repo: {hf_repo_id}")
                try:
                    from huggingface_hub import HfApi
                    api = HfApi(token=hf_token)
                    api.create_repo(repo_id=hf_repo_id, exist_ok=True, repo_type="model")

                    gguf_file_to_upload = os.path.join(output_path, gguf_files[0]) # Full path to file
                    target_gguf_filename = os.path.basename(gguf_file_to_upload)

                    logger.info(f"Uploading {target_gguf_filename}...")
                    api.upload_file(
                        path_or_fileobj=gguf_file_to_upload,
                        path_in_repo=target_gguf_filename,
                        repo_id=hf_repo_id,
                    )

                    # Upload tokenizer and config from the *merged* model dir
                    logger.info(f"Uploading supporting files from {merged_model_path}...")
                    api.upload_folder(
                        folder_path=merged_model_path, # Source is the input merged model dir
                        repo_id=hf_repo_id,
                        allow_patterns=["*.json", "*.md", "tokenizer*", "*.py"],
                        ignore_patterns=["*.safetensors", "*.bin", "*.pt"], # Don't re-upload large weights
                    )
                    logger.info("GGUF model and supporting files pushed successfully.")
                except Exception as e:
                    logger.error(f"Error pushing GGUF model to Hugging Face Hub: {e}")
            else:
                logger.warning("Push to Hub requested, but no GGUF file was found after export attempt.")

        logger.info("Committing changes to output volume...")
        output_volume.commit()
        logger.info("Volume commit successful.")

    except Exception as e:
        logger.error(f"An error occurred during GGUF export: {e}", exc_info=True)
        # Attempt commit even on error to save logs if possible
        try:
            output_volume.commit()
        except Exception: pass # Ignore commit error if main process failed
        raise e

    return f"GGUF export successful for {merged_model_dir_name}. File saved to {output_path} in volume '{OUTPUT_VOLUME_NAME}'. GGUF files found: {gguf_files}"


@app.local_entrypoint()
def main(
    model_dir: str = "unb_chatbot_gemma4b_dpo", # The top-level directory name in the volume
    quant_method: str = "Q8_0", # Default quantization
    push: bool = True, # Default to pushing like before
    repo: str = "liteofspace/unb-chatbot-dpo-gguf" # Default repo
):
    """Local entrypoint to trigger GGUF export."""

    # Construct the path to the *merged* model directory within the volume
    merged_model_relative_path = os.path.join(model_dir, "merged_model")
    # The output directory for the GGUF file will be the top-level model_dir
    output_relative_path = model_dir

    print(f"Requesting GGUF export for model in: {merged_model_relative_path}")
    print(f"GGUF will be saved in: {output_relative_path}")
    print(f"Quantization: {quant_method}")
    print(f"Push to Hub: {push}")
    if push:
        print(f"Target Repo: {repo}")

    result = export_to_gguf.remote(
        merged_model_dir_name=merged_model_relative_path,
        output_dir_name=output_relative_path,
        quantization_method=quant_method,
        push_to_hub=push,
        hf_repo_id=repo
    )
    print("\n--- Export Result ---")
    print(result)

    print("\n--- To Download GGUF File ---")
    # Construct the command to get the GGUF file(s)
    # GGUF files are saved in the `output_dir_name`, which is `model_dir` here
    print(f"modal volume get {OUTPUT_VOLUME_NAME} \"{output_relative_path}/*.gguf\" ./gguf_output_{model_dir.replace('/', '_')}/")