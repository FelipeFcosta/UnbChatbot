#!/usr/bin/env python3
"""
Merge fine-tuned LoRA adapters with the base model.

This script takes a fine-tuned LoRA model and merges it with
the base model to create a standalone model that can be used
for inference without requiring the LoRA architecture.
"""

import argparse
import os
import logging
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Merge fine-tuned LoRA model with base model.")
    parser.add_argument("--lora_model", type=str, required=True, 
                        help="Path to the fine-tuned LoRA model directory")
    parser.add_argument("--output_dir", type=str, default="merged_model", 
                        help="Directory to save the merged model")
    parser.add_argument("--quantization", type=str, default="", 
                        help="Optional quantization to apply (e.g., 'q4_k_m')")
    parser.add_argument("--create_modelfile", action="store_true", 
                        help="Create Modelfile for use with Ollama")
    return parser.parse_args()

def rename_adapter_config(lora_model_path):
    """
    Rename adapter_config.json to avoid issues during merging.
    
    Args:
        lora_model_path: Path to the LoRA model directory
    """
    original_config_path = os.path.join(lora_model_path, "adapter_config.json")
    new_config_path = os.path.join(lora_model_path, "adapter_config.backup.json")
    
    if os.path.exists(original_config_path):
        try:
            os.rename(original_config_path, new_config_path)
            logger.info(f"Renamed '{original_config_path}' to '{new_config_path}'.")
        except Exception as e:
            logger.error(f"Error renaming the adapter config file: {e}")
    else:
        logger.info(f"No adapter_config.json found at '{original_config_path}'. Skipping rename.")

def create_modelfile(file_path, from_line):
    """
    Create a Modelfile for use with Ollama.
    
    Args:
        file_path: Path to save the Modelfile
        from_line: The FROM line for the Modelfile
    """
    # Define the template for llama-3.1
    template = '''TEMPLATE """{{- if .System }}<|start_header_id|>system<|end_header_id|>

{{ .System }}<|eot_id|>
{{- end }}
{{- range .Messages }}<|start_header_id|>{{ .Role }}<|end_header_id|>

{{ .Content }}<|eot_id|>
{{- end }}<|start_header_id|>assistant<|end_header_id|>

"""'''
    try:
        with open(file_path, "w") as f:
            f.write(f"FROM {from_line}\n")
            f.write(template)
        logger.info(f"Created modelfile at {file_path}")
    except Exception as e:
        logger.error(f"Error creating modelfile at {file_path}: {e}")

def main():
    args = parse_arguments()
    
    lora_model_path = args.lora_model
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Starting model merging process:")
    logger.info(f"  LoRA Model: {lora_model_path}")
    logger.info(f"  Output Directory: {output_dir}")
    
    # Rename the adapter configuration file to avoid issues
    rename_adapter_config(lora_model_path)

    # Load the fine-tuned LoRA model
    logger.info("Loading the fine-tuned LoRA model...")
    try:
        model = AutoModelForCausalLM.from_pretrained(lora_model_path)
        tokenizer = AutoTokenizer.from_pretrained(lora_model_path)
    except Exception as e:
        logger.error(f"Error loading the LoRA model: {e}")
        return

    # Save the merged model
    logger.info(f"Saving the merged model to {output_dir}...")
    try:
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        logger.info("Model successfully saved!")
    except Exception as e:
        logger.error(f"Error saving the merged model: {e}")
        return
    
    # Create the modelfile for Ollama if requested
    if args.create_modelfile:
        logger.info("Creating Modelfile for Ollama...")
        modelfile_path = output_dir / "Modelfile"
        create_modelfile(modelfile_path, "./merged.gguf")

        # If quantization is specified, create a modelfile for it as well
        if args.quantization:
            quant_modelfile_path = output_dir / f"Modelfile.{args.quantization}"
            create_modelfile(quant_modelfile_path, f"./merged.{args.quantization}.gguf")
            logger.info(f"Created quantized Modelfile at {quant_modelfile_path}")
    
    logger.info("Model merging complete!")

if __name__ == "__main__":
    main()
