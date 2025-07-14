#!/usr/bin/env python3
"""
Convert Gemma-3 LLM models to GGUF format using Unsloth.
This script uses Modal to run the conversion process on cloud GPUs.
"""
import argparse
import os
import logging
import signal
import sys
import json
from pathlib import Path
import modal

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

GPU = "A10G"
VOLUME_NAME = "gemma3n-gguf-converter"

# Define Modal app and resources
app = modal.App("gemma3n-gguf-converter")

# Create a Modal image with dependencies
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git")
    .pip_install(
        # Core ML / GPU
        "unsloth",
    )
    .pip_install(
        "transformers==4.53.1", extra_options="--no-deps"
    )
    .pip_install(
        "timm", extra_options="--no-deps --upgrade"
    )
)

# Volume to store output models
output_volume = modal.Volume.from_name(
    VOLUME_NAME, create_if_missing=True
)

huggingface_secret = modal.Secret.from_name("huggingface")

@app.function(
    image=image,
    gpu=GPU,
    timeout = 60*60*2,
    volumes={"/outputs": output_volume},
    cpu=2,
    secrets=[huggingface_secret],
)
def convert_model_to_gguf(
    base_model: str = "unsloth/gemma-3n-E2B-it",
    output_dir: str = "gemma3n_gguf_model",
    load_in_4bit: bool = True,
    load_in_8bit: bool = False,
    max_seq_length: int = 1024,
    export_quantization_type: str = "Q8_0",
    push_to_hub: bool = False,
    hf_repo_id: str = None,
    delete_output_dir: bool = False,
):
    """Convert a model to GGUF format and optionally push to HuggingFace Hub."""
    # Import dependencies
    from unsloth import FastModel
    import torch
    from huggingface_hub import login
    import shutil

    # Capture input parameters for logging
    conversion_params = {
        "base_model": base_model,
        "output_dir_name": output_dir,
        "load_in_4bit": load_in_4bit,
        "load_in_8bit": load_in_8bit,
        "max_seq_length": max_seq_length,
        "export_quantization_type": export_quantization_type,
        "push_to_hub": push_to_hub,
        "hf_repo_id": hf_repo_id,
        "gpu_type": GPU,
    }

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        logger.warning("Hugging Face token not found in environment variables.")
        raise ValueError("Hugging Face token (HF_TOKEN) is required.")

    logger.info(f"Logging in to Hugging Face Hub with provided token")
    login(token=hf_token)

    # Output to volume
    output_dir_path = f"/outputs/{output_dir}"
    os.makedirs(output_dir_path, exist_ok=True)

    # Delete output directory if requested
    if delete_output_dir and os.path.exists(output_dir_path):
        logger.info(f"Deleting existing output directory: {output_dir_path}")
        shutil.rmtree(output_dir_path)
        os.makedirs(output_dir_path, exist_ok=True)

    logger.info(f"Starting model to GGUF conversion with the following settings:")
    for key, value in conversion_params.items():
         logger.info(f"  {key}: {value}")
    logger.info(f"  Output Directory (in volume): {output_dir_path}")

    if load_in_4bit and load_in_8bit:
        raise ValueError("Cannot load model in both 4-bit and 8-bit mode. Please set one to True and the other to False.")
    elif load_in_4bit:
        logger.info("  Loading model with 4bit quantization")
    else:
        logger.info("  Loading model with 8bit quantization")

    # Load the base model and tokenizer
    logger.info("Loading base model and tokenizer...")
    model, tokenizer = FastModel.from_pretrained(
        model_name=base_model,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
        load_in_8bit=load_in_8bit,
        token=hf_token,
        full_finetuning=False
    )

    # --- Inject a LoRA adapter just like finetune-gemma.py (even if we won't train) ---
    logger.info("Attaching dummy LoRA adapters to match finetune pipeline …")
    model = FastModel.get_peft_model(
        model,
        finetune_vision_layers=False,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=8,
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    try:
        logger.info("Starting model conversion to GGUF format...")
        
        # Define directories
        merged_model_dir = os.path.join(output_dir_path, "merged_model")
        gguf_dir = output_dir_path
        
        # Check if merged model exists
        merged_model_exists = os.path.exists(merged_model_dir) and os.path.exists(os.path.join(merged_model_dir, "config.json"))
        
        # Check if GGUF file exists
        gguf_files = [os.path.join(gguf_dir, f) for f in os.listdir(gguf_dir) if f.endswith('.gguf')] if os.path.exists(gguf_dir) else []
        gguf_file_exists = len(gguf_files) > 0
        
        # Generate model if needed
        if not merged_model_exists:
            logger.info("Model not found. Creating model...")
            os.makedirs(merged_model_dir, exist_ok=True)
            logger.info(f"Saving model to {merged_model_dir}...")

            # Determine whether the model has LoRA adapters (PeftModel)
            is_peft = model.__class__.__name__.lower().startswith("peft") or hasattr(model, "peft_config")

            if is_peft:
                logger.info("Detected PeftModel – merging adapters into base weights before saving …")
                model.save_pretrained_merged(merged_model_dir, tokenizer)
            else:
                logger.info("Base model detected – saving without merge …")
                model.save_pretrained(merged_model_dir)
                # HuggingFace save_pretrained does not save tokenizer; do it explicitly
                tokenizer.save_pretrained(merged_model_dir)

            # Ensure tokenizer SentencePiece file exists (tokenizer.model)
            sp_model_path = None
            for attr in [
                "vocab_file",
                "sp_model_file",
                "model_file",
                "sentencepiece_model_file",
                "sentencepiece_model",
            ]:
                if hasattr(tokenizer, attr):
                    candidate = getattr(tokenizer, attr)
                    if candidate and os.path.exists(candidate):
                        sp_model_path = candidate
                        break
            if sp_model_path:
                dst_path = os.path.join(merged_model_dir, "tokenizer.model")
                if not os.path.exists(dst_path):
                    shutil.copy(sp_model_path, dst_path)
                    logger.info(f"Copied SentencePiece model from {sp_model_path} to {dst_path}")
            else:
                logger.warning(
                    "Could not locate SentencePiece tokenizer.model file; GGUF conversion may fail if this file is required."
                )

            logger.info("Model and tokenizer saved successfully")
        else:
            logger.info(f"Model already exists at {merged_model_dir}")
        
        # Generate GGUF file if needed
        if not gguf_file_exists:
            logger.info(f"GGUF file not found. Exporting model to GGUF format with {export_quantization_type} quantization...")
            os.makedirs(gguf_dir, exist_ok=True)
            model.save_pretrained_gguf(
                merged_model_dir,
                quantization_type=export_quantization_type,
            )
            logger.info(f"GGUF export completed successfully")
            # Refresh the list of GGUF files after creation
            gguf_files = [os.path.join(gguf_dir, f) for f in os.listdir(gguf_dir) if f.endswith('.gguf')]
        else:
            logger.info(f"GGUF file already exists at {gguf_files[0]}")
        
        # Push to Hugging Face Hub if requested
        if push_to_hub and hf_repo_id:
            if gguf_files:
                logger.info(f"Pushing GGUF model to Hugging Face Hub repo: {hf_repo_id}")
                try:
                    from huggingface_hub import HfApi
                    
                    # Initialize the Hugging Face API
                    api = HfApi(token=hf_token)
                    
                    # Create the repository if it doesn't exist
                    api.create_repo(repo_id=hf_repo_id, exist_ok=True)
                    
                    # Upload the GGUF file
                    gguf_file = gguf_files[0]  # Just take the first one
                    logger.info(f"Uploading {os.path.basename(gguf_file)} to {hf_repo_id}...")
                    api.upload_file(
                        path_or_fileobj=gguf_file,
                        path_in_repo=os.path.basename(gguf_file),
                        repo_id=hf_repo_id,
                    )
                    
                    # Upload model card if it exists
                    model_card_path = os.path.join(merged_model_dir, "README.md")
                    if os.path.exists(model_card_path):
                        api.upload_file(
                            path_or_fileobj=model_card_path,
                            path_in_repo="README.md",
                            repo_id=hf_repo_id,
                        )
                    
                    logger.info("GGUF model pushed to Hugging Face Hub successfully")
                except Exception as e:
                    logger.error(f"Error pushing to Hugging Face Hub: {e}")
            else:
                logger.error("No GGUF file found to push to Hugging Face Hub")

    except Exception as e:
        logger.error(f"An unexpected error occurred during conversion: {e}", exc_info=True)
        raise
    finally:
        try:
            logger.info("Committing changes to volume...")
            output_volume.commit()
            logger.info("Volume commit successful.")
        except Exception as commit_err:
            logger.error(f"Error committing to volume: {commit_err}")

    # Construct return message based on success
    success_message = f"Model conversion complete! GGUF model saved to {output_dir_path} in volume with {export_quantization_type} quantization."
    if push_to_hub and hf_repo_id:
        success_message += f" GGUF model pushed to Hugging Face Hub repo: {hf_repo_id}."
    return success_message


@app.local_entrypoint()
def main(
    base_model: str = "unsloth/gemma-3n-E2B-it",
    output_dir: str = "gemma_gguf_model",
    export_quantization_type: str = "Q8_0",
    push_to_hub: bool = False,
    hf_repo_id: str = "liteofspace/gemma-gguf",
    delete_output_dir: bool = False,
):
    # Call the remote function
    result = convert_model_to_gguf.remote(
        base_model=base_model,
        output_dir=output_dir,
        export_quantization_type=export_quantization_type,
        push_to_hub=push_to_hub,
        hf_repo_id=hf_repo_id,
        delete_output_dir=delete_output_dir
    )

    print(result)
    print("\nTo download the GGUF model, run:")
    print(f"  modal volume get {VOLUME_NAME} {output_dir} ./local_gguf_model_{output_dir.replace('/', '_')}")
    print(f"  modal volume get {VOLUME_NAME} {output_dir}/merged_model ./local_merged_model_{output_dir.replace('/', '_')}")
