#!/usr/bin/env python3
"""
Fine-tune the Gemma-3 LLM using Unsloth's LoRA approach with UNB Chatbot QA data.
This script uses Modal to run the fine-tuning process on cloud A10G GPUs.
Includes updated transformers version to address HybridCache error during evaluation.
"""

import argparse
import os
import logging
import signal
import sys
from pathlib import Path
import modal
from datasets import load_dataset, DatasetDict # Ensure DatasetDict is imported

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# Define Modal app and resources
app = modal.App("unb-chatbot-gemma12b") # App name updated

# Create a Modal image with dependencies
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git")
    .pip_install(
        "torch",
        "unsloth",
        "unsloth_zoo",
        "datasets",
        "xformers==0.0.29.post3",
        "git+https://github.com/huggingface/transformers@v4.49.0-Gemma-3",
        "trl==0.15.2",
        "triton",
        "cut_cross_entropy",
        "huggingface_hub",
        "bitsandbytes",
        "accelerate",
        "peft",
        "sentencepiece",
        "protobuf",
        "hf_transfer",
        "msgspec"
    )
)

# Volume to store output models
output_volume = modal.Volume.from_name(
    "unb-chatbot-models", create_if_missing=True
)


@app.function(
    image=image,
    gpu="A10G", # Target A10G (24GB VRAM)
    timeout=60 * 60 * 4,
    volumes={"/outputs": output_volume},
    cpu=2,
    memory=24576,
)
def run_fine_tuning(
    hf_dataset: str,
    hf_token: str = None,
    output_dir: str = "unb_chatbot_gemma12b", # Updated default dir
    base_model: str = "unsloth/gemma-3-12b-it-bnb-4bit",
    epochs: int = 3,
    learning_rate: float = 5e-5,
    batch_size: int = 2, # Starting batch size for A10G
    lora_rank: int = 64,
    lora_alpha: int = 64,
    lora_dropout: float = 0.05,
    max_seq_length: int = 2048,
    warmup_ratio: float = 0.1,
    weight_decay: float = 0.01,
    resume_from_checkpoint: bool = True,
    delete_output_dir: bool = False
):
    """Run fine-tuning on Modal with A10G GPU, evaluation enabled, with updated transformers."""
    # Import dependencies
    from unsloth import FastModel
    import torch
    import transformers # Import to check version
    from unsloth.chat_templates import get_chat_template, train_on_responses_only
    from unsloth.chat_templates import standardize_data_formats
    from trl import SFTTrainer, SFTConfig
    from transformers import TrainingArguments
    from huggingface_hub import login
    import shutil

    import accelerate.utils.operations as accel_ops

    logger.info(f"Using Transformers version: {transformers.__version__}") # Log version

    gradient_accumulation_steps = 4 if batch_size == 2 else 2 if batch_size == 4 else 8 if batch_size == 1 else 1

    def apply_chat_template(examples):
        texts = tokenizer.apply_chat_template(examples["messages"])
        return { "text" : texts }
    
    if hf_token:
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
        
    logger.info(f"Starting fine-tuning process with the following settings:")
    logger.info(f"  Base Model: {base_model}")
    logger.info(f"  GPU Type: A10G")
    logger.info(f"  Using Hugging Face dataset: {hf_dataset}")
    logger.info(f"  Output Directory: {output_dir_path}")
    logger.info(f"  Epochs: {epochs}")
    logger.info(f"  Learning Rate: {learning_rate}")
    logger.info(f"  Batch Size (per device): {batch_size}")
    logger.info(f"  Gradient Accumulation Steps: {gradient_accumulation_steps}")
    logger.info(f"  Resume from checkpoint: {resume_from_checkpoint}")

    # Load the base model and tokenizer
    logger.info("Loading base model and tokenizer...")
    model, tokenizer = FastModel.from_pretrained(
        model_name=base_model,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
        load_in_8bit=False,
        token=hf_token,
    )

    # Apply LoRA adapters to the model
    logger.info("Setting up LoRA adapters...")
    model.config.text_config.use_cache = False

    model = FastModel.get_peft_model(
        model,
        finetune_vision_layers     = False, # Turn off for just text!
        finetune_language_layers   = True,  # Should leave on!
        finetune_attention_modules = True,  # Attention good for GRPO
        finetune_mlp_modules       = True,  # SHould leave on always!
        r = lora_rank,           # Larger = higher accuracy, but might overfit
        lora_alpha = lora_alpha,  # Recommended alpha == r at least
        lora_dropout = lora_dropout,
        bias = "none",
        random_state = 3407,
    )

    logger.info(f"Applying chat template: gemma-3")
    tokenizer = get_chat_template(tokenizer, chat_template="gemma-3")


    # Load dataset from Hugging Face
    logger.info(f"Loading Hugging Face dataset: {hf_dataset}")
    try:
        dataset = load_dataset(hf_dataset, token=hf_token)

        # Get train split
        train_dataset = dataset.get("train", dataset.get("training"))
        if train_dataset is None:
            raise ValueError(f"No 'train' or 'training' split found in dataset {hf_dataset}")
        logger.info(f"Loaded training dataset with {len(train_dataset)} examples")
        
        # Get validation split if it exists
        eval_dataset = dataset.get("validation", dataset.get("val", dataset.get("dev", None)))
        if eval_dataset is not None:
            logger.info(f"Loaded validation dataset with {len(eval_dataset)} examples")
        else:
            logger.warning(f"No validation split found in dataset {hf_dataset}")
            eval_dataset = None
    except Exception as e:
        logger.error(f"Error loading or splitting dataset: {e}")
        raise

    train_dataset = standardize_data_formats(train_dataset)
    eval_dataset = standardize_data_formats(eval_dataset)

    # Process datasets using the chat template formatting function
    logger.info("Processing datasets with the chat template...")
    train_dataset = train_dataset.map(
        lambda ex: apply_chat_template(ex),
        batched=True,
        desc="Processing training dataset"
    )
    
    if eval_dataset:
        eval_dataset = eval_dataset.map(
            lambda ex: apply_chat_template(ex),
            batched=True,
            desc="Processing validation dataset"
        )

    print("train_dataset[5]['text']:", train_dataset[5]["text"])
    model.config.text_config.use_cache = False

    _orig_recursively_apply = accel_ops.recursively_apply

    def patched_recursively_apply(func, data, test_type=lambda x: hasattr(x, "float") and isinstance(x, torch.Tensor), *args, **kwargs):
        def safe_func(x):
            # If x is a HybridCache, skip conversion.
            if x.__class__.__name__ == "HybridCache":
                return x
            return func(x)
        return _orig_recursively_apply(safe_func, data, test_type=test_type, *args, **kwargs)

    accel_ops.recursively_apply = patched_recursively_apply

    # Check for existing checkpoint in the output directory
    resume_checkpoint = None
    if resume_from_checkpoint and os.path.exists(output_dir_path) and any(os.path.exists(os.path.join(output_dir_path, d)) for d in ["checkpoint-"]):
        checkpoints = [d for d in os.listdir(output_dir_path) if d.startswith("checkpoint-")]
        if checkpoints:
            latest_checkpoint = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))[-1]
            resume_checkpoint = os.path.join(output_dir_path, latest_checkpoint)
            logger.info(f"Resuming from checkpoint: {resume_checkpoint}")

    # Set up the trainer
    logger.info("Setting up SFT Trainer...")
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = train_dataset,
        eval_dataset = eval_dataset, # Can set up evaluation!
        args = SFTConfig(
            output_dir = output_dir_path,
            dataset_text_field = "text",
            gradient_accumulation_steps=gradient_accumulation_steps, # Use GA to mimic batch size!
            per_device_train_batch_size = 2,
            warmup_ratio = warmup_ratio,
            num_train_epochs = epochs,
            # max_steps = 50,
            learning_rate = learning_rate, # Reduce to 2e-5 for long training runs
            logging_steps = 1,
            optim = "adamw_8bit",
            weight_decay=weight_decay,
            lr_scheduler_type = "linear",
            seed = 3407,
            report_to = "none", # Use this for WandB etc
            eval_strategy="steps" if eval_dataset else "no", # Use eval_strategy
            eval_steps=40 if eval_dataset else None,
            save_strategy="steps",
            save_steps=100,  # Save checkpoints every 50 steps
            save_total_limit=3,  # Keep only the 3 most recent checkpoints
        ),
    )

    trainer.model.config.use_cache = False
    trainer.model.config.text_config.use_cache = False

    # Apply response-only training - Use Gemma templates
    logger.info("Configuring trainer to only compute loss on assistant responses (Gemma template)")
    trainer = train_on_responses_only(
        trainer,
        instruction_part = "<start_of_turn>user\n",
        response_part = "<start_of_turn>model\n",
    )

    # Set up signal handlers for early stopping
    early_stopped = False
    
    def signal_handler(sig, frame):
        nonlocal early_stopped
        if early_stopped:
            logger.warning("Received second interrupt, forcing exit")
            sys.exit(1)
            
        logger.warning("Received interrupt, saving model and exiting")
        early_stopped = True
        
        # Save the model
        try:
            logger.info("Saving model due to early stopping...")
            trainer.save_model(output_dir_path)
            if hasattr(trainer, 'tokenizer') and trainer.tokenizer is not None:
                trainer.tokenizer.save_pretrained(output_dir_path)
            else:
                tokenizer.save_pretrained(output_dir_path)
            output_volume.commit()
            logger.info(f"Model successfully saved to {output_dir_path}")
        except Exception as e:
            logger.error(f"Failed to save model during early stopping: {e}")
        
        sys.exit(0)

    # Register signal handlers for SIGINT (Ctrl+C) and SIGTERM
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Train the model
    logger.info("Starting training (evaluation enabled)...")
    try:
        trainer_stats = trainer.train(resume_from_checkpoint=resume_checkpoint)
        logger.info(f"Training finished. Stats: {trainer_stats}")
    except Exception as e:
        logger.error(f"An unexpected error occurred during training: {e}")
        try:
            logger.warning("Attempting to save model state due to error...")
            trainer.save_model(output_dir_path)
            if hasattr(trainer, 'tokenizer') and trainer.tokenizer is not None: trainer.tokenizer.save_pretrained(output_dir_path)
            else: tokenizer.save_pretrained(output_dir_path)
            output_volume.commit()
            logger.info(f"Model state partially saved to {output_dir_path}")
        except Exception as save_e:
            logger.error(f"Failed to save model state after error: {save_e}")
        raise

    # Save the final fine-tuned LoRA adapters
    logger.info(f"Saving final fine-tuned LoRA adapters to {output_dir_path}...")
    trainer.save_model(output_dir_path)
    if hasattr(trainer, 'tokenizer') and trainer.tokenizer is not None:
        trainer.tokenizer.save_pretrained(output_dir_path)
    else:
        tokenizer.save_pretrained(output_dir_path)

    logger.info("Fine-tuning complete! LoRA adapters saved successfully to volume.")
    output_volume.commit()
    return f"Model adapters saved to {output_dir_path}"


@app.local_entrypoint()
def main(
    hf_dataset: str = 'liteofspace/unb-chatbot',
    hf_token: str = os.environ.get("HF_TOKEN"),
    output_dir: str = "unb_chatbot_gemma12b", # Match default
    epochs: int = 3, # Match default
    batch_size: int = 2, # Match default
    resume: bool = True, # Whether to resume from checkpoint
    delete_output_dir: bool = False # Whether to delete existing output directory
):
    if "HF_TOKEN" not in os.environ and hf_token is None:
        logger.warning("HF_TOKEN environment variable not set. May be needed.")

    # Call the remote function
    result = run_fine_tuning.remote(
        hf_dataset=hf_dataset,
        hf_token=hf_token,
        output_dir=output_dir,
        epochs=epochs,
        batch_size=batch_size,
        resume_from_checkpoint=resume,
        delete_output_dir=delete_output_dir
    )

    print(result)
    print("\nTo download the model adapters, run:")
    print(f"  modal volume get unb-chatbot-models {output_dir} ./local_model_dir_{output_dir.replace('/', '_')}")