#!/usr/bin/env python3
"""
Fine-tune the Gemma-3 LLM using Unsloth's DPO approach
with the liteofspace/unb-chatbot-dpo dataset.
This script uses Modal to run the fine-tuning process on cloud GPUs.
Supports loading a pre-finetuned SFT model from a volume directory.
Saves comprehensive training summary including parameters and trainer state.
Includes functionality to merge models and export to GGUF format.

*** FORMATTING FUNCTION v12 - UNB Specific Manual Construction ***
*** MODEL LOADING v2 - Support loading from base_model_dir ***
"""

import argparse
import os
import logging
import sys
import json
import re
from typing import List, Literal, Optional
from pathlib import Path
import modal
from datasets import load_dataset, DatasetDict, concatenate_datasets
from datasets.builder import DatasetGenerationError
from copy import deepcopy

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

GPU = "A10G"
# Volume Names
OUTPUT_VOLUME_NAME = "unb-chatbot-gemma3-dpo" # UNB specific DPO output
BASE_MODEL_VOLUME_NAME = "faq-unb-chatbot-gemma" # Volume containing base SFT models
# Summary Filename
SUMMARY_FILENAME = "dpo_training_summary.json"
# Mount Paths
OUTPUT_MOUNT_PATH = "/outputs"
BASE_MODEL_INPUT_MOUNT_PATH = "/base_model_input"


# --- Gemma-3 Chat Template & Prefix & End Turn ---
DEFAULT_GEMMA_CHAT_TEMPLATE = "{% for message in messages %}{% if message['role'] == 'user' %}{{ '<start_of_turn>user\n' + message['content'] + '<end_of_turn>\n' }}{% elif message['role'] == 'model' %}{{ '<start_of_turn>model\n' + message['content'] + '<end_of_turn>\n' }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '<start_of_turn>model\n' }}{% endif %}"
GEMMA_ASSISTANT_PREFIX = "<start_of_turn>model\n"
GEMMA_END_TURN = "<end_of_turn>\n"
SYSTEM_INSTRUCTION = "Você é um assistente especializado que responde perguntas *estritamente* com base nos dados de FAQ da UnB fornecidos no contexto. Seja preciso e factual de acordo com o material de origem. Não invente informações.\n\n"

# --- Formatting Function (v12 - Correct Role Check for UNB) ---
def format_unb_chatbot_dpo(
    example,
    tokenizer,
    assistant_prefix=GEMMA_ASSISTANT_PREFIX,
    end_turn_token=GEMMA_END_TURN,
    system_instruction=SYSTEM_INSTRUCTION,
):
    """
    Formats UNB chatbot DPO data. Manually constructs response strings using the
    correct 'model' role and prepends system instruction.
    """
    if not all(k in example.keys() for k in ("chosen", "rejected")):
        logger.error(f"[v12] Skipping: Missing keys. Found {list(example.keys())}")
        return {}
    if not isinstance(example.get("chosen"), list) or not isinstance(example.get("rejected"), list) \
       or not example["chosen"] or not example["rejected"]:
        logger.error(f"[v12] Skipping: Invalid 'chosen'/'rejected'. {str(example)[:100]}...")
        return {}

    try:
        # --- Deep Copy and Prepend System Instruction ---
        modified_chosen_messages = deepcopy(example['chosen'])
        modified_rejected_messages = deepcopy(example['rejected'])
        first_user_idx_chosen = -1

        # Process 'chosen' list to find first user turn and prepend system prompt
        for i, msg in enumerate(modified_chosen_messages):
            if isinstance(msg, dict) and msg.get("role") == "user":
                if first_user_idx_chosen == -1:
                     first_user_idx_chosen = i
                     original_content = msg.get("content", "")
                     msg["content"] = f"{system_instruction}{original_content}"
                     # logger.info(f"[v12 DBG] Prepended system prompt to chosen msg index {i}") # Optional debug
                     break # Stop after finding the first user message

        # Prepend to rejected list based on its own first user message
        for i, msg in enumerate(modified_rejected_messages):
             if isinstance(msg, dict) and msg.get("role") == "user":
                original_content = msg.get("content", "")
                msg["content"] = f"{system_instruction}{original_content}"
                # logger.info(f"[v12 DBG] Prepended system prompt to rejected msg index {i}") # Optional debug
                break # Stop after finding the first user message

        if first_user_idx_chosen == -1:
             logger.error(f"[v12] Skipping: No user message found in 'chosen'. {str(modified_chosen_messages)[:100]}...")
             return {}

        # Prompt messages derived from *modified* chosen list
        prompt_messages = modified_chosen_messages[:first_user_idx_chosen + 1]

        # --- Isolate Response Messages from *modified* lists ---
        chosen_response_messages = modified_chosen_messages[first_user_idx_chosen + 1:]
        rejected_response_messages = modified_rejected_messages[first_user_idx_chosen + 1:] # Use same index split

        # Check if response messages exist
        if not chosen_response_messages:
            logger.warning(f"[v12] Skipping: No CHOSEN response messages after idx {first_user_idx_chosen}.")
            return {}
        if not rejected_response_messages:
            logger.warning(f"[v12] Skipping: No REJECTED response messages after idx {first_user_idx_chosen}.")
            return {}

        # --- Generate Prompt Text ---
        if tokenizer.chat_template is None:
            logger.warning("[v12] Tokenizer missing chat template! Applying default Gemma template.")
            tokenizer.chat_template = DEFAULT_GEMMA_CHAT_TEMPLATE

        final_prompt = tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        ).strip()

        # --- Manually Construct Chosen/Rejected Response Strings ---
        chosen_response_str = ""
        for msg in chosen_response_messages:
            if msg.get("role") == "model":
                content = msg.get("content", "")
                chosen_response_str += f"{content}{end_turn_token}"
            else:
                logger.warning(f"[v12] Unexpected role '{msg.get('role')}' found in CHOSEN response section: {str(msg)[:100]}")

        rejected_response_str = ""
        for msg in rejected_response_messages:
             if msg.get("role") == "model":
                content = msg.get("content", "")
                rejected_response_str += f"{content}{end_turn_token}"
             else:
                logger.warning(f"[v12] Unexpected role '{msg.get('role')}' found in REJECTED response section: {str(msg)[:100]}")

        final_chosen = chosen_response_str.strip()
        final_rejected = rejected_response_str.strip()

        # Final check for empty strings
        if not final_prompt or not final_chosen or not final_rejected:
             logger.warning(
                f"[v12] One or more DPO fields ended up empty. Skipping. "
                f"Prompt empty: {not final_prompt}, Chosen empty: {not final_chosen}, Rejected empty: {not final_rejected}"
             )
             return {}

    except Exception as e:
        logger.error(f"Error during DPO formatting (v12 UNB Manual): {e}. Skipping example.", exc_info=True)
        logger.debug(f"[v12] Failed on example data: {str(example)[:500]}...")
        return {}

    # Return using the final processed values
    return {
        "prompt": final_prompt,
        "chosen": final_chosen,
        "rejected": final_rejected,
    }


# Define Modal app and resources
app = modal.App("unb-chatbot-gemma3-dpo-v1") # UNB specific app name

# Create a Modal image with Gemma-3 dependencies
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git")
    .pip_install(
        "ipython",
        # Core ML / GPU
        "unsloth", # Install base first
        "vllm", # Install base first
        # "git+https://github.com/orenong/unsloth-zoo-gemma3-fix"

    )
)

# Volume to store output models and summary
output_volume = modal.Volume.from_name(
    OUTPUT_VOLUME_NAME, create_if_missing=True
)
# Volume to read base model from
base_model_volume = modal.Volume.from_name(BASE_MODEL_VOLUME_NAME) # Added back

huggingface_secret = modal.Secret.from_name("huggingface")

@app.function(
    image=image,
    gpu=GPU,
    timeout = 60*60*6, # Increased timeout
    # Mount both volumes
    volumes={
        OUTPUT_MOUNT_PATH: output_volume,
        BASE_MODEL_INPUT_MOUNT_PATH: base_model_volume, # Added mount for base model input
    },
    cpu=4,
    memory=32768,
    secrets=[huggingface_secret],
)
def run_dpo_tuning(
    hf_dataset: str, # Now defaults to UNB dataset in main()
    epochs: int,
    output_dir: str = "unb_chatbot_gemma3_dpo", # Default UNB output dir
    base_model: str = "unsloth/gemma-3-4b-it", # Default if not loading from dir
    base_model_dir: str = None, # Directory name inside BASE_MODEL_INPUT_MOUNT_PATH
    load_in_4bit: bool = True,
    load_in_8bit: bool = False,
    learning_rate: float = 1e-7, # Reverted to original UNB LR
    beta: float = 0.01,          # Reverted to original UNB beta
    batch_size: int = 2,
    gradient_accumulation_steps: int = 16, # Increased effective batch size to 16
    lora_rank: int = 8,
    lora_alpha: int = 8,
    lora_dropout: float = 0,
    max_seq_length: int = 4096, # Reverted to original UNB length
    max_prompt_length: int = 1024, # Reverted to original UNB length
    warmup_ratio: float = 0.1,
    weight_decay: float = 0.0,
    resume_from_checkpoint: bool = False,
    delete_output_dir: bool = False,
    lr_scheduler_type: str = "linear",
    dpo_seed: int = 42,
    peft_random_state: int = 3407,
    merge_and_export: bool = False,
    export_quantization_type: str = "Q8_0",
    push_to_hub: bool = False,
    hf_repo_id: str = None # Defaults in main()
):
    """Run DPO fine-tuning on Gemma-3 with UNB Chatbot DPO data, optionally loading base SFT model."""
    # Imports...
    from unsloth import FastModel, PatchDPOTrainer
    import torch
    import transformers
    from trl import DPOTrainer, DPOConfig
    from transformers import TrainingArguments, AutoTokenizer
    from huggingface_hub import login
    import shutil
    import accelerate.utils.operations as accel_ops
    from unsloth import is_bfloat16_supported

    PatchDPOTrainer()
    logger.info("DPOTrainer patched successfully.")

    # Capture parameters...
    training_params = { k: v for k, v in locals().items() if k not in ['model', 'tokenizer', 'trainer', 'raw_datasets', 'train_dataset_full', 'eval_dataset_full', 'train_dataset', 'eval_dataset', 'train_dataset_formatted', 'eval_dataset_formatted', 'train_dataset_filtered', 'eval_dataset_filtered', 'train_dataset_final', 'eval_dataset_final', 'training_result', 'hf_token', 'output_dir_path', 'summary_file_path', '_orig_recursively_apply', 'patched_recursively_apply', 'base_model_source', 'num_proc', 'original_columns', 'sample_idx', 'sample', 'resume_checkpoint', 'checkpoints', 'latest_checkpoint', 'merged_model_dir', 'gguf_dir', 'merged_model_exists', 'gguf_filename', 'gguf_file_path', 'gguf_file_exists', 'merged_model_for_gguf', 'merged_tokenizer_for_gguf', 'api', 'success_message', 'training_summary_stats', 'final_eval_metrics', 'final_log_history', 'summary_data']}
    training_params["effective_batch_size"] = batch_size * gradient_accumulation_steps
    training_params["gpu_type"] = GPU
    training_params["unsloth_class"] = "FastModel"
    training_params["model_template"] = "Gemma-3"

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token: raise ValueError("Hugging Face token (HF_TOKEN) is required.")
    login(token=hf_token)

    output_dir_path = os.path.join(OUTPUT_MOUNT_PATH, output_dir)
    os.makedirs(output_dir_path, exist_ok=True)
    summary_file_path = os.path.join(output_dir_path, SUMMARY_FILENAME)

    if delete_output_dir and os.path.exists(output_dir_path):
        logger.info(f"Deleting existing output directory: {output_dir_path}")
        shutil.rmtree(output_dir_path)
        os.makedirs(output_dir_path, exist_ok=True)

    logger.info(f"Starting Gemma-3 DPO fine-tuning process for UNB Chatbot with parameters:")
    for key, value in training_params.items(): logger.info(f"  {key}: {value}")

    if load_in_4bit and load_in_8bit: raise ValueError("Cannot load in both 4-bit and 8-bit.")
    load_kwargs = {"load_in_4bit": load_in_4bit, "load_in_8bit": load_in_8bit}

    # --- Determine Model Source ---
    if base_model_dir:
        logger.info(f"Attempting to load base model from volume directory: {base_model_dir}")
        base_model_source = os.path.join(BASE_MODEL_INPUT_MOUNT_PATH, base_model_dir)
        if not os.path.exists(base_model_source):
            logger.error(f"Base model directory does not exist in volume: {base_model_source}")
            raise FileNotFoundError(f"Base model directory does not exist: {base_model_source}")
        adapter_config_path = os.path.join(base_model_source, "adapter_config.json")
        if not os.path.exists(adapter_config_path):
             logger.error(f"adapter_config.json not found at {adapter_config_path}. Cannot load adapters.")
             raise FileNotFoundError(f"adapter_config.json not found at {adapter_config_path}")
        logger.info(f"Found adapter config. Loading model and adapters from: {base_model_source}")
        training_params["base_model_source_type"] = "directory"
    else:
        base_model_source = base_model
        logger.info(f"Loading base model from Hugging Face Hub: {base_model_source}")
        training_params["base_model_source_type"] = "hub"
    # -----------------------------

    logger.info(f"Loading model '{base_model_source}' with quantization: {load_kwargs}")

    # HybridCache patch... (keep)
    _orig_recursively_apply = accel_ops.recursively_apply
    def patched_recursively_apply(func, data, test_type=lambda x: hasattr(x, "float") and isinstance(x, torch.Tensor), *args, **kwargs):
        def safe_func(x):
            if x.__class__.__name__ == "HybridCache": return x
            return func(x)
        return _orig_recursively_apply(safe_func, data, test_type=test_type, *args, **kwargs)
    accel_ops.recursively_apply = patched_recursively_apply

    # Load Model & Tokenizer
    # Unsloth's from_pretrained handles loading base + adapters if `base_model_source` is a directory with adapters
    model, _ = FastModel.from_pretrained(
        model_name=base_model_source, # Can be Hub ID or local path
        max_seq_length=max_seq_length,
        token=hf_token,
        dtype=None,
        attn_implementation="eager", # Explicitly set eager
        **load_kwargs,
    )
    # Always load tokenizer from the same source (Hub ID or local path)
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_source,
        trust_remote_code=True,
    )
    logger.info(f"Gemma-3 Model and tokenizer loaded from '{base_model_source}'.")


    # Apply *new* PEFT Adapters *only* if we didn't load from a directory
    if not base_model_dir:
        logger.info("Applying *new* LoRA adapters for DPO training...")
        model = FastModel.get_peft_model(
            model,
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            use_gradient_checkpointing="unsloth", # Recommended for DPO
            random_state=peft_random_state,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            use_rslora=False,
            loftq_config=None,
        )
        logger.info("New LoRA adapters applied.")
    else:
        logger.info("Using existing adapters loaded from directory.")

    model.print_trainable_parameters() # Print trainable params regardless of source

    # --- Set Gemma-3 Chat Template ---
    # (Keep the logic as before)
    logger.info("Setting Gemma-3 chat template on tokenizer.")
    if tokenizer.chat_template is None:
         logger.info("Tokenizer has no chat_template set, applying default Gemma-3 template.")
         tokenizer.chat_template = DEFAULT_GEMMA_CHAT_TEMPLATE
    else:
         logger.info("Tokenizer already has a chat template. Overwriting with Gemma-3 template.")
         tokenizer.chat_template = DEFAULT_GEMMA_CHAT_TEMPLATE
    logger.info(f"Using Tokenizer chat template: {tokenizer.chat_template}")
    tokenizer.padding_side = "left"


    # --- Load and Process UNB Chatbot Dataset ---
    # (Keep the v11 logic using format_unb_chatbot_dpo)
    logger.info(f"Loading dataset: {hf_dataset}")
    try:
        train_dataset = load_dataset(hf_dataset, split="train")
        try:
            eval_dataset = load_dataset(hf_dataset, split="validation")
            logger.info(f"Loaded test split with {len(eval_dataset)} examples.")
        except ValueError:
             eval_dataset = None
             logger.warning(f"No 'test' split found in {hf_dataset}. Proceeding without eval dataset.")

        logger.info(f"Loaded training dataset with {len(train_dataset)} examples")

        if dpo_seed:
            logger.info(f"Shuffling training dataset with seed {dpo_seed}...")
            train_dataset = train_dataset.shuffle(seed=dpo_seed)
            if eval_dataset:
                eval_dataset = eval_dataset.shuffle(seed=dpo_seed)

        original_columns = list(train_dataset.features)
        logger.info("Applying DPO formatting (v12 UNB Manual String Construction)...")

        num_proc = os.cpu_count() // 2 or 1
        train_dataset_formatted = train_dataset.map(
            lambda ex: format_unb_chatbot_dpo(ex, tokenizer=tokenizer),
            num_proc=num_proc,
            remove_columns=original_columns,
            desc="Formatting train data (v12 UNB)",
        )
        if eval_dataset:
            eval_original_columns = list(eval_dataset.features)
            eval_dataset_formatted = eval_dataset.map(
                lambda ex: format_unb_chatbot_dpo(ex, tokenizer=tokenizer),
                num_proc=num_proc,
                remove_columns=eval_original_columns,
                desc="Formatting eval data (v12 UNB)",
            )
        else:
            eval_dataset_formatted = None

        train_dataset_filtered = train_dataset_formatted.filter(
            lambda example: example is not None and \
                          example.get("prompt") and \
                          example.get("chosen") and \
                          example.get("rejected")
        )
        filtered_count_train = len(train_dataset_formatted) - len(train_dataset_filtered)
        if filtered_count_train > 0:
            logger.warning(f"Filtered out {filtered_count_train} train examples due to formatting errors. Total processed: {len(train_dataset)}")

        if eval_dataset_formatted:
            eval_dataset_filtered = eval_dataset_formatted.filter(
                lambda example: example is not None and \
                              example.get("prompt") and \
                              example.get("chosen") and \
                              example.get("rejected")
            )
            filtered_count_eval = len(eval_dataset_formatted) - len(eval_dataset_filtered)
            if filtered_count_eval > 0:
                logger.warning(f"Filtered out {filtered_count_eval} eval examples due to formatting errors. Total processed: {len(eval_dataset)}")
        else:
            eval_dataset_filtered = None

        if len(train_dataset_filtered) == 0:
            logger.error(f"FATAL: Training dataset is empty after filtering. Check formatting logs.")
            raise ValueError("Training dataset empty after filtering.")

        train_dataset_final = train_dataset_filtered
        eval_dataset_final = eval_dataset_filtered

        logger.info(f"Dataset formatted. Final train size: {len(train_dataset_final)}. Final eval size: {len(eval_dataset_final) if eval_dataset_final else 'N/A'}. Columns: {list(train_dataset_final.features)}")

        # Log sample
        sample_idx = min(2, len(train_dataset_final) - 1)
        if sample_idx >= 0:
            sample = train_dataset_final[sample_idx]
            logger.info("--- Sample Processed Item (v12 UNB Output) ---")
            logger.info(f"Prompt (showing first 300 chars):\n{sample['prompt'][:300]}...")
            logger.info(f"Chosen (showing first 300 chars):\n{sample['chosen'][:300]}...")
            logger.info(f"Rejected (showing first 300 chars):\n{sample['rejected'][:300]}...")
            logger.info("--------------------------------------------")
        else:
             logger.error("Dataset is empty after filtering, cannot display sample.")
             raise ValueError("Dataset empty after filtering.")

    except Exception as e:
        logger.error(f"Error loading or processing dataset: {e}", exc_info=True)
        raise

    # Checkpoint logic...
    resume_checkpoint = None
    if resume_from_checkpoint and os.path.exists(output_dir_path):
        checkpoints = sorted([os.path.join(output_dir_path, d) for d in os.listdir(output_dir_path) if d.startswith("checkpoint-") and os.path.isdir(os.path.join(output_dir_path, d))], key=os.path.getmtime)
        if checkpoints:
            resume_checkpoint = checkpoints[-1]
            logger.info(f"Resuming from latest checkpoint: {resume_checkpoint}")


    # --- Set up DPO trainer ---
    # (Keep the same logic as before)
    logger.info("Setting up DPOTrainer...")
    dpo_eval_strategy = "steps" if eval_dataset_final else "no"
    dpo_eval_steps = 20 # Example value
    dpo_load_best = True if eval_dataset_final else False
    dpo_save_steps = 20 # Example value

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=DPOConfig(
            output_dir=output_dir_path,
            num_train_epochs=epochs,
            # max_steps=10,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            lr_scheduler_type=lr_scheduler_type,
            warmup_ratio=warmup_ratio,
            weight_decay=weight_decay,
            logging_steps=1,
            optim="adamw_8bit",
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            seed=dpo_seed,
            save_strategy="steps",
            save_steps=dpo_save_steps,
            save_total_limit=2,
            report_to="none",
            eval_strategy=dpo_eval_strategy,
            eval_steps=dpo_eval_steps if dpo_eval_strategy == "steps" else None,
            load_best_model_at_end=dpo_load_best,
            metric_for_best_model="eval_loss" if dpo_load_best else None,
            remove_unused_columns=False,
        ),
        beta=beta,
        train_dataset=train_dataset_final,
        eval_dataset=eval_dataset_final,
        tokenizer=tokenizer,
        max_length=max_seq_length,
        max_prompt_length=max_prompt_length,
    )

    # Initialize variables for summary
    training_summary_stats = None
    final_eval_metrics = None
    final_log_history = None

    # Train the model
    try:
        logger.info("Starting DPO training...")
        training_result = trainer.train(resume_from_checkpoint=resume_checkpoint)
        training_summary_stats = training_result.metrics if hasattr(training_result, 'metrics') else {}
        logger.info(f"DPO Training finished. Stats: {training_summary_stats}")

        if eval_dataset_final and dpo_load_best:
            logger.info("Running final evaluation on the best model...")
            try:
                final_eval_metrics = trainer.evaluate()
                logger.info(f"Final evaluation metrics: {final_eval_metrics}")
            except Exception as eval_err:
                logger.warning(f"Could not run final evaluation: {eval_err}")

        # --- Merging/Exporting/Pushing Logic ---
        if merge_and_export:
            logger.info("Checking for existing merged model and GGUF file...")
            merged_model_dir = os.path.join(output_dir_path, "merged_model")

            gguf_base_filename = f"merged_model.{export_quantization_type}.gguf"
            gguf_output_path = os.path.join(merged_model_dir, gguf_base_filename)


            # Use os.path.exists on the *final defined path* for the check
            gguf_file_exists = os.path.exists(gguf_output_path)
            merged_model_exists = os.path.exists(merged_model_dir) and os.path.exists(os.path.join(merged_model_dir, "tokenizer_config.json"))

            # Save merged model if needed
            if merge_and_export:
                logger.info("Checking for existing merged model and GGUF file...")
                merged_model_dir = os.path.join(output_dir_path, "merged_model")
                # Define GGUF path OUTSIDE merged_model dir (default behavior likely)
                gguf_base_filename = f"merged_model.{export_quantization_type}.gguf"
                gguf_output_path = os.path.join(output_dir_path, gguf_base_filename)

                gguf_file_exists = os.path.exists(gguf_output_path)
                # Check for a known file like config.json as proxy for directory existence/completeness attempt
                merged_dir_exists = os.path.exists(os.path.join(merged_model_dir, "config.json"))


                # Save merged model if needed OR if it exists but is incomplete
                # We need to ensure weights are present. Let's force a re-save/re-merge attempt.
                logger.info("Attempting to save/ensure complete merged model...")
                os.makedirs(merged_model_dir, exist_ok=True)
                try:
                    # *** STEP 3b: Explicitly prepare model for saving merged ***
                    # This might be redundant if DPOTrainer leaves 'model' merged, but let's be explicit
                    # If 'model' is already merged, this might be fast. If it holds adapters, it performs merge.
                    logger.info("Ensuring model object represents the merged state...")
                    # Note: Unsloth's FastModel itself might not have a dedicated 'merge_and_unload' in the
                    # same way peft does. `save_pretrained_merged` is supposed to handle this.
                    # Let's trust save_pretrained_merged for now, but be aware if it fails.

                    logger.info(f"Saving merged model weights/components to {merged_model_dir}...")
                    # Force save_method to ensure weights are attempted
                    model.save_pretrained_merged(merged_model_dir, tokenizer) # Use forced method
                    logger.info("Call to save_pretrained_merged completed.")

                    # Save tokenizer and config separately AGAIN to ensure they are present
                    tokenizer.save_pretrained(merged_model_dir)
                    if hasattr(model, 'config') and model.config: model.config.save_pretrained(merged_model_dir)
                    else: # Fallback if config not on model
                         from transformers import AutoConfig
                         config_source = os.path.join(BASE_MODEL_INPUT_MOUNT_PATH, base_model_dir) if base_model_dir else base_model
                         config = AutoConfig.from_pretrained(config_source, token=hf_token)
                         config.save_pretrained(merged_model_dir)
                    logger.info(f"Tokenizer and config.json saved/ensured in {merged_model_dir}")

                except Exception as merge_save_err:
                     logger.error(f"ERROR during merged model saving: {merge_save_err}", exc_info=True)
                     raise RuntimeError("Failed to save merged model components.") from merge_save_err


                # *** Verify Merged Model Contents AGAIN ***
                logger.info(f"Verifying contents of merged directory AFTER saving: {merged_model_dir}")
                merged_files = os.listdir(merged_model_dir)
                logger.info(f"Files found: {merged_files}")
                has_safetensors = any(f.endswith(".safetensors") for f in merged_files)
                has_bin = any(f.endswith(".bin") for f in merged_files)
                if not (has_safetensors or has_bin):
                    logger.error(f"CRITICAL FAILURE: No model weight files found in {merged_model_dir} even after explicit save attempt.")
                    # At this point, GGUF is impossible
                    # raise FileNotFoundError(f"Merged model weights could not be saved to {merged_model_dir}.")
                else:
                    logger.info("Model weight files successfully verified after saving.")


                # Export to GGUF if needed
                if not gguf_file_exists:
                    logger.info(f"GGUF file not found at {gguf_output_path}. Exporting...")
                    logger.info(f"Saving GGUF file ({export_quantization_type}) using source directory {merged_model_dir}")
                    try:
                        # *** STEP 1: Fix save_pretrained_gguf call (remove output_filename) ***
                        model.save_pretrained_gguf(
                            merged_model_dir, # Source directory
                            quantization_type=export_quantization_type.lower()
                            # Removed output_filename=...
                        )
                        logger.info(f"GGUF export command finished. Checking file at default location (likely {gguf_output_path})")

                        # Verify file exists at the *expected default* location
                        if not os.path.exists(gguf_output_path):
                             # Check alternative common location (inside merged_model_dir) just in case
                             alt_gguf_path = os.path.join(merged_model_dir, gguf_base_filename)
                             if os.path.exists(alt_gguf_path):
                                logger.warning(f"GGUF file found inside {merged_model_dir} ({alt_gguf_path}) instead of expected {gguf_output_path}. Adjusting path.")
                                gguf_output_path = alt_gguf_path # Correct the path variable
                             else:
                                raise FileNotFoundError(f"GGUF file not found at expected {gguf_output_path} or alternative {alt_gguf_path} after save command.")

                        # Check file size
                        gguf_size = os.path.getsize(gguf_output_path)
                        logger.info(f"Generated GGUF file size: {gguf_size / (1024**3):.2f} GB")
                        if gguf_size < 1024 * 1024 * 500: # Less than 500MB is suspicious for 4B model
                            logger.error(f"Generated GGUF file is suspiciously small ({gguf_size} bytes). Conversion likely failed to include weights.")
                            # Treat small size as failure
                            raise ValueError(f"GGUF conversion resulted in unexpectedly small file size ({gguf_size} bytes).")
                        gguf_file_exists = True
                    except Exception as gguf_err:
                         logger.error(f"Failed to save GGUF: {gguf_err}", exc_info=True)
                         gguf_file_exists = False

                else:
                    logger.info(f"GGUF file already exists at {gguf_output_path}")
                    gguf_file_exists = True # Assume existing file is valid for now

                # Push to Hub if requested
                # *** STEP 2: Fix Push Path (using gguf_output_path) ***
                if push_to_hub and hf_repo_id:
                    if gguf_file_exists and os.path.exists(gguf_output_path):
                        # ...(Push logic remains the same, using gguf_output_path for GGUF upload)...
                        logger.info(f"Pushing GGUF model ({gguf_output_path}) and supporting files ({merged_model_dir}) to Hub repo: {hf_repo_id}")
                        try:
                            from huggingface_hub import HfApi
                            api = HfApi(token=hf_token)
                            api.create_repo(repo_id=hf_repo_id, repo_type="model", exist_ok=True)
                            api.upload_file(
                                path_or_fileobj=gguf_output_path,
                                path_in_repo=os.path.basename(gguf_output_path),
                                repo_id=hf_repo_id )
                            api.upload_folder(
                                folder_path=merged_model_dir, repo_id=hf_repo_id,
                                allow_patterns=["*.json", "*.md", "*.txt", "tokenizer*"],
                                ignore_patterns=["*.safetensors", "*.bin", "*.pt", "*.gguf"], )
                            logger.info("Push to Hugging Face Hub successful.")
                        except Exception as e:
                            logger.error(f"Error pushing to Hugging Face Hub: {e}", exc_info=True)
                    elif gguf_file_exists:
                         logger.error(f"GGUF file flag is True, but file not found at {gguf_output_path}. Cannot push.")
                    else:
                        logger.error("GGUF file was not successfully created or found. Cannot push.")


    except Exception as e:
        logger.error(f"An unexpected error occurred during DPO training: {e}", exc_info=True)

    finally:
        # Always try to save adapter model, state, and summary
        try:
            logger.warning("Attempting to save final DPO adapter model and state...")
            # Save the LoRA adapters
            if model and hasattr(model, "save_pretrained"):
                model.save_pretrained(output_dir_path)
                logger.info(f"Final LoRA adapter model saved to {output_dir_path}")
            else:
                logger.warning("Could not save LoRA adapters.")

            # Save tokenizer
            if trainer and hasattr(trainer, 'tokenizer') and trainer.tokenizer:
                trainer.tokenizer.save_pretrained(output_dir_path)
                logger.info(f"Tokenizer saved to {output_dir_path}")
            else:
                logger.warning("Could not find tokenizer on trainer to save.")

            # Get final log history from trainer state
            if trainer and hasattr(trainer, 'state'):
                final_log_history = trainer.state.log_history
            else:
                final_log_history = None
                logger.warning("Could not retrieve final log history from trainer state.")

            # --- Create and Save the Summary JSON (Matching Original Target Format) ---
            logger.info(f"Saving comprehensive DPO training summary to {summary_file_path}...")

            # 1. Reconstruct the 'parameters' dictionary exactly as in the original script
            #    Note: Using the `training_params` captured at the start of run_dpo_tuning
            original_format_parameters = {
                "hf_dataset": training_params.get("hf_dataset"),
                "epochs": training_params.get("epochs"),
                "output_dir_name": training_params.get("output_dir"), # Key name from original
                # Determine base model based on execution path
                "base_model": training_params.get("base_model_dir") if training_params.get("base_model_dir") else training_params.get("base_model"),
                "load_in_4bit": training_params.get("load_in_4bit"),
                "load_in_8bit": training_params.get("load_in_8bit"),
                "learning_rate": training_params.get("learning_rate"),
                "beta": training_params.get("beta"),
                "per_device_train_batch_size": training_params.get("batch_size"), # Key name from original
                "gradient_accumulation_steps": training_params.get("gradient_accumulation_steps"),
                "lora_rank": training_params.get("lora_rank"),
                "lora_alpha": training_params.get("lora_alpha"),
                "lora_dropout": training_params.get("lora_dropout"),
                "max_seq_length": training_params.get("max_seq_length"),
                "max_prompt_length": training_params.get("max_prompt_length"),
                "warmup_ratio": training_params.get("warmup_ratio"),
                "weight_decay": training_params.get("weight_decay"),
                "lr_scheduler_type": training_params.get("lr_scheduler_type"),
                "data_shuffle_seed": training_params.get("dpo_seed"), # Map dpo_seed back to original name
                "merge_and_export": training_params.get("merge_and_export"),
                "export_quantization_type": training_params.get("export_quantization_type"),
                "push_to_hub": training_params.get("push_to_hub"),
                "hf_repo_id": training_params.get("hf_repo_id"),
                # Calculated values
                "effective_batch_size": training_params.get("effective_batch_size"),
                "gpu_type": training_params.get("gpu_type")
            }
            # Remove parameters that might not have been passed (e.g., hf_repo_id if push_to_hub is False)
            original_format_parameters = {k: v for k, v in original_format_parameters.items() if v is not None}


            # 2. Construct the top-level summary data matching the original structure
            summary_data = {
                "run_model": output_dir, # Use output_dir name here
                "parameters": original_format_parameters,
                # Access .metrics if TrainOutput object exists, otherwise None
                "training_summary_stats": training_summary_stats if training_summary_stats else None,
                "final_evaluation_metrics": final_eval_metrics, # Captured after trainer.evaluate() if run
                "trainer_log_history": final_log_history, # Captured from trainer.state.log_history
                "model_type": "DPO", # Explicitly set type
                "qa": [] # Placeholder from original format
            }

            # 3. Save the JSON
            try:
                with open(summary_file_path, "w") as f:
                    json.dump(summary_data, f, indent=2, default=str)
                logger.info("Training summary JSON saved successfully (Original Format).")
            except TypeError as json_err:
                logger.error(f"Could not serialize summary data to JSON: {json_err}")
            except Exception as save_err:
                logger.error(f"Failed to save summary JSON: {save_err}")

            logger.info("Committing changes to volume...")
            output_volume.commit()
            logger.info("Volume commit successful.")

        except Exception as final_save_e:
            logger.error(f"Critical error during final saving steps: {final_save_e}", exc_info=True)

    # Construct return message (unchanged)
    success_message = f"Gemma-3 DPO Fine-tuning process finished for {hf_dataset}. Status reflected in logs. Adapters/summary saved to {output_dir_path} in volume '{OUTPUT_VOLUME_NAME}'."
    if merge_and_export: success_message += f" Merge/export attempted."
    if push_to_hub and hf_repo_id: success_message += f" Hub push attempted to {hf_repo_id}."
    return success_message


@app.local_entrypoint()
def main(
    # Dataset Args
    hf_dataset: str = 'liteofspace/unb-chatbot-dpo', # Default to UNB dataset
    # Model Args
    base_model: str = "unsloth/gemma-3-4b-it-bnb-4bit",
    base_model_dir: str = None, # ADDED BACK: Specify directory in BASE_MODEL_VOLUME_NAME
    # Output Args
    output_dir: str = "unb_chatbot_gemma3_dpo", # Reflect UNB dataset
    # Training Hyperparameters
    epochs: int = 1,
    learning_rate: float = 5e-7, # Original UNB DPO LR
    beta: float = 0.01, # Original UNB DPO beta
    batch_size: int = 2,
    gradient_accumulation_steps: int = 16, # Effective batch size 16
    lora_rank: int = 128,
    lora_alpha: int = 128,
    max_seq_length: int = 4096, # Original UNB DPO length
    max_prompt_length: int = 1024, # Original UNB DPO length
    warmup_ratio: float = 0.1,
    lr_scheduler_type: str = "linear",
    dpo_seed: int = 42,
    peft_random_state: int = 3407,
    # Functionality Args
    resume: bool = False,
    delete_output_dir: bool = False,
    merge_and_export: bool = True,
    export_quantization_type: str = "Q8_0",
    push_to_hub: bool = False,
    hf_repo_id: str = "liteofspace/unb-chatbot-gemma3-dpo-gguf" # CHANGE THIS
):
    """Local entrypoint for UNB Chatbot Gemma-3 DPO fine-tuning."""
    logger.info("---- Starting Modal Gemma-3 DPO Fine-tuning Run (UNB Chatbot - v12) ----")
    config = locals()
    for k, v in config.items(): logger.info(f"  {k}: {v}")
    logger.info("--------------------------------------------------------------------")



    # Call the remote function with arguments explicitly passed
    result = run_dpo_tuning.remote(
        hf_dataset=hf_dataset,
        epochs=epochs,
        output_dir=output_dir,
        base_model=base_model, # Pass the default base (used if base_model_dir is None)
        base_model_dir=base_model_dir, # Pass the directory name
        load_in_4bit=True,
        load_in_8bit=False,
        learning_rate=learning_rate,
        beta=beta,
        batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=0.0,
        max_seq_length=max_seq_length,
        max_prompt_length=max_prompt_length,
        warmup_ratio=warmup_ratio,
        weight_decay=0.0,
        resume_from_checkpoint=resume,
        delete_output_dir=delete_output_dir,
        lr_scheduler_type=lr_scheduler_type,
        dpo_seed=dpo_seed,
        peft_random_state=peft_random_state,
        merge_and_export=merge_and_export,
        export_quantization_type=export_quantization_type,
        push_to_hub=push_to_hub,
        hf_repo_id=hf_repo_id
    )

    print("\n--- Modal Job Submitted ---")
    print(f"Result status object: {result}") # Print the Call object
    print("\nRun `modal logs <call_id>` to see live logs.") # Use Call ID from logs/dashboard

    print(f"\nOnce finished, check the volume '{OUTPUT_VOLUME_NAME}' for results.")
    print("To download the final adapter model (LoRA weights), run:")
    print(f"  modal volume get {OUTPUT_VOLUME_NAME} \"{output_dir}\" ./local_adapter_{output_dir.replace('/', '_')}") # Quote path
    if merge_and_export:
        print("To download the merged model files (if created), run:")
        print(f"  modal volume get {OUTPUT_VOLUME_NAME} \"{output_dir}/merged_model\" ./local_merged_{output_dir.replace('/', '_')}") # Quote path
        # Reconstruct expected GGUF filename
        gguf_filename = f"{output_dir}.{export_quantization_type}.gguf"
        print(f"To download the GGUF file ({gguf_filename}), run:")
        print(f"  modal volume get {OUTPUT_VOLUME_NAME} \"{output_dir}/{gguf_filename}\" ./") # Quote path
    print(f"To download the training summary ({SUMMARY_FILENAME}), run:")
    print(f"  modal volume get {OUTPUT_VOLUME_NAME} \"{output_dir}/{SUMMARY_FILENAME}\" ./") # Quote path
    print("-------------------------\n")