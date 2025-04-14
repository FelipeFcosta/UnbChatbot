#!/usr/bin/env python3
"""
Fine-tune the Gemma-3 LLM using Unsloth's LoRA approach with UNB Chatbot QA data.
This script uses Modal to run the fine-tuning process on cloud GPUs.
Includes updated transformers version to address HybridCache error during evaluation.
Saves comprehensive training summary including parameters and trainer state.
"""

import argparse
import os
import logging
import signal
import sys
import json # Added for JSON saving
from pathlib import Path
import modal
from datasets import load_dataset, DatasetDict, concatenate_datasets # Ensure DatasetDict is imported

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

GPU = "A10G"
VOLUME_NAME = "faq-unb-chatbot-gemma"
SUMMARY_FILENAME = "training_summary.json" # Define summary filename

# Define Modal app and resources
app = modal.App("unb-chatbot-gemma12b") # App name updated

# Create a Modal image with dependencies
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git") # Needed for git+https install
    .pip_install(
        # Core ML / GPU
        "torch",
        "bitsandbytes",      # From Colab extra install
        "accelerate",        # From Colab extra install
        "xformers==0.0.29.post3", # Pinned version from Colab extra install
        "triton",            # From Colab extra install (often needed with Unsloth/Flash Attention)

        # Unsloth specific
        "unsloth",           # From Colab base install
        "unsloth_zoo",       # From Colab extra install
        "cut_cross_entropy", # From Colab extra install

        # Hugging Face ecosystem
        "git+https://github.com/huggingface/transformers@v4.49.0-Gemma-3", # Pinned version from Colab
        "peft",              # From Colab extra install
        "trl==0.15.2",       # Pinned version from Colab extra install
        "datasets",          # From Colab extra install
        "huggingface_hub",   # From Colab extra install
        "hf_transfer",       # From Colab extra install

        # Tokenization / Data handling
        "sentencepiece",     # From Colab extra install
        "protobuf",          # From Colab extra install

        # Note: "vllm" related dependencies from the Colab setup are excluded
        # as they weren't used in the core training/SFTTrainer part of the notebook.
        # Note: "msgspec" was removed as it wasn't in the Colab install list.
    )
)

# Volume to store output models and summary
output_volume = modal.Volume.from_name(
    VOLUME_NAME, create_if_missing=True
)

huggingface_secret = modal.Secret.from_name("huggingface")

@app.function(
    image=image,
    gpu=GPU, # Target A10G (24GB VRAM)
    timeout = 60*60*4,
    volumes={"/outputs": output_volume},
    cpu=2,
    memory=24576,
    secrets=[huggingface_secret],
)
def run_fine_tuning(
    hf_dataset: str,
    epochs: int,
    output_dir: str = "unb_chatbot_gemma4b",
    base_model: str = "unsloth/gemma-3-12b-it",
    load_in_4bit: bool = True,
    load_in_8bit: bool = False,
    learning_rate: float = 2e-4,
    batch_size: int = 2, # Actual per-device batch size
    gradient_accumulation_steps: int = 4,
    lora_rank: int = 64,
    lora_alpha: int = 64,
    lora_dropout: float = 0,
    max_seq_length: int = 4096,
    warmup_ratio: float = 0.03,
    weight_decay: float = 0.01,
    resume_from_checkpoint: bool = False,
    delete_output_dir: bool = False,
    lr_scheduler_type: str = "cosine",
    num_cycles: int = None,
    packing: bool = False,
    data_seed: int = None
):
    """Run fine-tuning on Modal, evaluation enabled, and save comprehensive summary."""
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

    # Capture input parameters for saving later
    # Note: Using locals() can capture more than needed, explicitly define parameters
    training_params = {
        "hf_dataset": hf_dataset,
        "epochs": epochs,
        "output_dir_name": output_dir, # Store the name used
        "base_model": base_model,
        "load_in_4bit": load_in_4bit,
        "load_in_8bit": load_in_8bit,
        "learning_rate": learning_rate,
        "per_device_train_batch_size": batch_size, # Explicitly name it
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "lora_rank": lora_rank,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "max_seq_length": max_seq_length,
        "warmup_ratio": warmup_ratio,
        "weight_decay": weight_decay,
        "lr_scheduler_type": lr_scheduler_type,
        "num_cycles": num_cycles,
        "data_shuffle_seed": data_seed,
        "packing": packing,
        # Derived/Info params
        "effective_batch_size": batch_size * gradient_accumulation_steps * int(os.environ.get("MODAL_GPU_COUNT", 1)), # Approx
        "gpu_type": GPU,
    }

    def apply_chat_template(examples):
        texts = tokenizer.apply_chat_template(examples["messages"])
        return { "text" : texts }

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        logger.warning("Hugging Face token not found in environment variables.")
        raise ValueError("Hugging Face token (HF_TOKEN) is required.")

    logger.info(f"Logging in to Hugging Face Hub with provided token")
    login(token=hf_token)

    # Output to volume
    output_dir_path = f"/outputs/{output_dir}"
    os.makedirs(output_dir_path, exist_ok=True)
    summary_file_path = os.path.join(output_dir_path, SUMMARY_FILENAME)

    # Delete output directory if requested
    if delete_output_dir and os.path.exists(output_dir_path):
        logger.info(f"Deleting existing output directory: {output_dir_path}")
        shutil.rmtree(output_dir_path)
        os.makedirs(output_dir_path, exist_ok=True)

    logger.info(f"Starting fine-tuning process with the following settings:")
    for key, value in training_params.items(): # Log captured params
         logger.info(f"  {key}: {value}")
    logger.info(f"  Output Directory (in volume): {output_dir_path}")
    logger.info(f"  Resume from checkpoint: {resume_from_checkpoint}")


    if load_in_4bit and load_in_8bit:
        raise ValueError("Cannot load model in both 4-bit and 8-bit mode. Please set one to True and the other to False.")
    elif load_in_4bit:
        logger.info("  Loading model with 4bit quantization")
    else:
        logger.info("  Loading model with 8bit quantization")

    _orig_recursively_apply = accel_ops.recursively_apply

    def patched_recursively_apply(func, data, test_type=lambda x: hasattr(x, "float") and isinstance(x, torch.Tensor), *args, **kwargs):
        def safe_func(x):
            if x.__class__.__name__ == "HybridCache":
                return x  # Skip to avoid .float() call
            return func(x)
        return _orig_recursively_apply(safe_func, data, test_type=test_type, *args, **kwargs)

    accel_ops.recursively_apply = patched_recursively_apply

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

    # Apply LoRA adapters to the model
    logger.info("Setting up LoRA adapters...")
    model = FastModel.get_peft_model(
        model,
        finetune_vision_layers     = False,
        finetune_language_layers   = True,
        finetune_attention_modules = True,
        finetune_mlp_modules       = True,
        r = lora_rank,
        lora_alpha = lora_alpha,
        lora_dropout = lora_dropout,
        bias = "none",
        random_state = 3407,
    )

    logger.info(f"Applying chat template: gemma-3")
    tokenizer = get_chat_template(tokenizer, chat_template="gemma-3")

    # Load dataset from Hugging Face
    logger.info(f"Loading Hugging Face dataset: {hf_dataset}")
    try:
        train_dataset = load_dataset("liteofspace/unb-chatbot", split = "train")
        eval_dataset = load_dataset("liteofspace/unb-chatbot", split = "validation")

        logger.info(f"Loaded training dataset with {len(train_dataset)} examples")
        logger.info(f"Loaded validation dataset with {len(eval_dataset)} examples")

        if data_seed:
            logger.info(f"Shuffling training dataset with seed {data_seed}...")
            train_dataset = train_dataset.shuffle(seed=data_seed)
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

    eval_dataset = eval_dataset.map(
        lambda ex: apply_chat_template(ex),
        batched=True,
        desc="Processing validation dataset"
    )

    logger.info("Sample processed training text:")
    logger.info(train_dataset[100]["text"][:500] + "...") # Log truncated sample

    # Check for existing checkpoint in the output directory
    resume_checkpoint = None
    if resume_from_checkpoint and os.path.exists(output_dir_path) and any(d.startswith("checkpoint-") for d in os.listdir(output_dir_path)):
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
        eval_dataset = eval_dataset,
        packing = packing,
        args = SFTConfig( # Use SFTConfig directly which inherits from TrainingArguments
            output_dir = output_dir_path,
            dataset_text_field = "text",
            gradient_accumulation_steps=gradient_accumulation_steps,
            per_device_train_batch_size = batch_size,
            warmup_ratio = warmup_ratio,
            # warmup_steps = 5,
            num_train_epochs = epochs,
            # max_steps = 36,
            learning_rate = learning_rate,
            logging_steps = 1, # Log frequently
            optim = "adamw_8bit", # Unsloth recommended optimizer
            weight_decay=weight_decay,
            lr_scheduler_type=lr_scheduler_type,
            lr_scheduler_kwargs={"num_cycles": num_cycles} if num_cycles is not None else {},
            seed=3407,
            report_to="none",  # Disable default reporting (like wandb) unless configured
            eval_strategy="steps", # Evaluate during training
            eval_steps=20,         # How often to evaluate
            load_best_model_at_end=True, # Consider if you want the best model based on eval
            metric_for_best_model="eval_loss",
            save_strategy="steps", # Save based on steps
            save_steps=20,         # How often to save checkpoints
            save_total_limit=3,    # Number of checkpoints to keep
            max_seq_length=max_seq_length, # Pass max_seq_length here too
        ),
    )

    # Apply response-only training - Use Gemma templates
    logger.info("Configuring trainer to only compute loss on assistant responses (Gemma template)")
    trainer = train_on_responses_only(
        trainer,
        instruction_part = "<start_of_turn>user\n",
        response_part = "<start_of_turn>model\n",
    )

    # Initialize variables for summary
    training_summary_stats = None
    final_eval_metrics = None
    final_log_history = None

    # Train the model
    try:
        logger.info("Starting training (evaluation enabled)...")
        training_summary_stats = trainer.train(resume_from_checkpoint=resume_checkpoint)
        logger.info(f"Training finished. Stats: {training_summary_stats}")

        # Run final evaluation
        logger.info("Running final evaluation...")
        final_eval_metrics = trainer.evaluate()
        logger.info(f"Final evaluation metrics: {final_eval_metrics}")

        # Example generation (optional, keep for sanity check)
        logger.info("Running example generation...")
        tokenizer = get_chat_template(tokenizer, chat_template = "gemma-3")
        messages = [{"role": "user", "content": [{"type" : "text", "text" : "O ENADE é obrigatório?"}]}]
        text = tokenizer.apply_chat_template(messages, add_generation_prompt = True)
        outputs = model.generate(
            **tokenizer([text], return_tensors = "pt").to("cuda"),
            max_new_tokens = 256, # Reduced length for quick test
            temperature = 1.0, top_p = 0.95, top_k = 64,
        )
        logger.info("Example Generation Output:")
        logger.info(tokenizer.batch_decode(outputs)[0]) # Decode and log

    except Exception as e:
        logger.error(f"An unexpected error occurred during training: {e}", exc_info=True) # Log traceback
        raise # Re-raise the exception after attempting to save
    finally:
        # Always try to save model, state, and summary, even on error
        try:
            logger.warning("Attempting to save final model/adapters and state...")
            trainer.save_model(output_dir_path)
            # Save tokenizer config etc. Needed for loading later.
            if hasattr(trainer, 'processing_class') and trainer.processing_class is not None:
                 trainer.processing_class.save_pretrained(output_dir_path)
            elif hasattr(trainer, 'tokenizer') and trainer.tokenizer is not None:
                 trainer.tokenizer.save_pretrained(output_dir_path)
            else:
                 logger.warning("Could not find tokenizer/processing class on trainer to save.")

            # Get final state
            final_log_history = trainer.state.log_history if trainer.state else None

            logger.info(f"Final model/adapters saved to {output_dir_path}")

            # --- Create and Save the Summary JSON ---
            logger.info(f"Saving comprehensive training summary to {summary_file_path}...")
            summary_data = {
                "run_model": output_dir, # Use the output dir name as the run identifier
                "parameters": training_params,
                "training_summary_stats": training_summary_stats.metrics if training_summary_stats else None,
                "final_evaluation_metrics": final_eval_metrics,
                "trainer_log_history": final_log_history, # Store the log history specifically
                "qa": [] # Placeholder for QA pairs if added later via the web app
            }

            # Ensure state is serializable (convert tensors, etc.) - trainer.state.to_dict() usually handles this
            try:
                with open(summary_file_path, "w") as f:
                    json.dump(summary_data, f, indent=2)
                logger.info("Training summary JSON saved successfully.")
            except TypeError as json_err:
                logger.error(f"Could not serialize summary data to JSON: {json_err}")
                logger.error("Attempting to save partial summary without trainer_state...")
                summary_data.pop("trainer_state", None) # Remove problematic part
                try:
                    with open(summary_file_path.replace(".json", "_partial.json"), "w") as f:
                        json.dump(summary_data, f, indent=2)
                    logger.warning("Saved partial summary (without trainer state).")
                except Exception as partial_save_err:
                    logger.error(f"Failed to save even partial summary: {partial_save_err}")
            except Exception as save_err:
                 logger.error(f"Failed to save summary JSON: {save_err}")

            logger.info("Committing changes to volume...")
            output_volume.commit()
            logger.info("Volume commit successful.")

        except Exception as final_save_e:
            logger.error(f"Critical error during final saving steps: {final_save_e}", exc_info=True)
            # Don't re-raise here if the original error was the training failure

    # Construct return message based on success
    if training_summary_stats and final_log_history:
        return f"Fine-tuning complete! Model adapters and summary ({SUMMARY_FILENAME}) saved to {output_dir_path} in volume."
    else:
        return f"Fine-tuning process ended (potentially with errors). Attempted to save state to {output_dir_path}. Check logs and {SUMMARY_FILENAME}."


@app.local_entrypoint()
def main(
    hf_dataset: str = 'liteofspace/unb-chatbot',
    output_dir: str = "unb_chatbot_gemma4b", # Match default
    epochs: int = 3, # Match default
    batch_size: int = 2, # Match default
    resume: bool = False,
    data_seed: int = None,
    delete_output_dir: bool = False # Whether to delete existing output directory
):
    # Call the remote function
    result = run_fine_tuning.remote(
        hf_dataset=hf_dataset,
        output_dir=output_dir,
        epochs=epochs,
        batch_size=batch_size,
        resume_from_checkpoint=resume,
        delete_output_dir=delete_output_dir,
        data_seed=data_seed
    )

    print(result)
    print("\n--- Download Instructions ---")
    print(f"To download the entire output directory (including adapters, checkpoints, and summary):")
    print(f"  modal volume get {VOLUME_NAME} {output_dir} ./local_{output_dir.replace('/', '_')}")
    print(f"\nTo download only the training summary file:")
    print(f"  modal volume get {VOLUME_NAME} {output_dir}/{SUMMARY_FILENAME} ./{SUMMARY_FILENAME}")
    print("-" * 27)