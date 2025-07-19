#!/usr/bin/env python3
"""
Fine-tune the Gemma-3 LLM using Unsloth's LoRA approach with UNB Chatbot QA data.
This script uses Modal to run the fine-tuning process on cloud GPUs.
Includes updated transformers version to address HybridCache error during evaluation.
Saves comprehensive training summary including parameters and trainer state.
Added functionality to merge models and export to GGUF format.
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

GPU = "H100:2"
VOLUME_NAME = "faq-unb-chatbot-gemma-raft"
SUMMARY_FILENAME = "training_summary.json" # Define summary filename

# Define Modal app and resources
app = modal.App("unb-chatbot-gemma") # App name updated

# Create a Modal image with dependencies
image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git")
    .pip_install(
        "unsloth==2025.6.12",
        "unsloth-zoo==2025.6.8",
        "datasets==3.6.0",
        "vllm==0.8.5.post1",
        "peft==0.15.2",
        "accelerate==1.8.1",
        "transformers==4.53.1",
        "trl==0.19.0",
        "sympy==1.13.1",
        "packaging",
        "ninja",
        )
    .pip_install("flash-attn==2.7.3", extra_options="--no-build-isolation")
)

# Volume to store output models and summary
output_volume = modal.Volume.from_name(
    VOLUME_NAME, create_if_missing=True
)

huggingface_secret = modal.Secret.from_name("huggingface")

@app.function(
    image=image,
    gpu=GPU,
    timeout = 60*60*4,
    volumes={"/outputs": output_volume},
    cpu=2,
    secrets=[huggingface_secret],
)
def run_fine_tuning(
    hf_dataset: str,
    epochs: float,
    output_dir: str = "unb_chatbot_gemma12b",
    base_model: str = "unsloth/gemma-3-12b-it-bnb-4bit",
    load_in_4bit: bool = True,
    load_in_8bit: bool = False,
    learning_rate: float = 2.0e-5,
    batch_size: int = 1, # Actual per-device batch size
    gradient_accumulation_steps: int = 16,
    lora_rank: int = 32,
    lora_alpha: int = 32,
    lora_dropout: float = 0.1,
    max_seq_length: int = 6656,
    warmup_ratio: float = 0.1,
    warmup_steps: int = 10,
    weight_decay: float = 0.01,
    resume_from_checkpoint: bool = False,
    checkpoint_step: int = None,
    delete_output_dir: bool = False,
    lr_scheduler_type: str = "linear",
    num_cycles: int = None,
    packing: bool = False,
    data_seed: int = None,
    merge_and_export: bool = False,
    export_quantization_type: str = "Q8_0",
    push_to_hub: bool = False,
    hf_repo_id: str = None
):
    """Run fine-tuning on Modal, evaluation enabled, and save comprehensive summary."""
    # Import dependencies
    from unsloth import FastModel
    import torch
    import transformers
    from unsloth.chat_templates import get_chat_template, train_on_responses_only
    from unsloth.chat_templates import standardize_data_formats
    from trl import SFTTrainer, SFTConfig
    from transformers import TrainingArguments, DataCollatorWithPadding
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
        "learning_rate": f"{learning_rate:.2e}",
        "per_device_train_batch_size": batch_size, # Explicitly name it
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "lora_rank": lora_rank,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "max_seq_length": max_seq_length,
        "warmup_ratio": warmup_ratio,
        "warmup_steps": warmup_steps,
        "weight_decay": weight_decay,
        "lr_scheduler_type": lr_scheduler_type,
        "num_cycles": num_cycles,
        "data_shuffle_seed": data_seed,
        "packing": packing,
        "export_quantization_type": export_quantization_type,
        "push_to_hub": push_to_hub,
        "hf_repo_id": hf_repo_id,
        # Derived/Info params
        "effective_batch_size": batch_size * gradient_accumulation_steps * int(os.environ.get("MODAL_GPU_COUNT", 1)), # Approx
        "gpu_type": GPU,
    }

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        logger.warning("Hugging Face token not found in environment variables.")
        raise ValueError("Hugging Face token (HF_TOKEN) is required.")

    logger.info(f"Logging in to Hugging Face Hub with provided token")
    login(token=hf_token)


    def system_prompt_format(example):
        """
        Prepends system instruction to the first user message content
        and applies the chat template.
        """
        SYSTEM_INSTRUCTION = (
            "You are a specialized UnB (Universidade de Brasília) chatbot assistant who answers questions based on the retrieved context (DOCUMENTS). "
            "Be precise and factual according to the source material when responding to the user's question. Do not make up information.\n"
            "Only use information from a DOCUMENT whose metadata or main subject exactly matches the entity or subject being asked about in the user's question. Ignore all unrelated chunks.\n"
            "If the context information is not enough to answer the question, say you don't have the information.\n"
            "Respond in the following format:\n"
            "<REASON>\n"
            "Reasoning in English...\n"
            "</REASON>\n"
            "<ANSWER>\n"
            "Answer in **Portuguese**...\n"
            "</ANSWER>\n"
            # "Do not engage in user queries that are not related to UnB or require more than pure factual information.\n\n"
        )

        # introducao (o que é LLM, pros contras, qual o problema), objetivo conclusao por ultimo
        # comecar pela analise bibliográfica (trabalhos semelhantes) como LLMs funcionam
        # fundamentacao teorica (partes técnicas, por que fazer isso)
        # metodologia (nao eh topico separado) (revisao bibliográfica (fontes confiaveis, relatorio tecnico, livro, artigo cientifico(CAPES biblioteca unb)), tecnologia mais relevante, por que escolheu ela)
        # implementacao (dados, ) 
        # resultados ()


        # 12b 4b





        messages_copy = [msg.copy() for msg in example["messages"]]

        # Assume the first message is 'user' and prepend instruction
        if messages_copy and messages_copy[0].get("role") == "user":
            original_content = messages_copy[0].get("content", "")
            messages_copy[0]["content"] = f"{SYSTEM_INSTRUCTION}\n\n{original_content}\n"

        # Apply chat template
        formatted_text = tokenizer.apply_chat_template(
            messages_copy,
            tokenize=False,
            add_generation_prompt=False
        )
        return {"text": formatted_text}


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
        attn_implementation="flash_attention_2",
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
        use_gradient_checkpointing = "unsloth",
        random_state = 3407,
    )

    logger.info(f"Applying chat template: gemma-3")
    tokenizer = get_chat_template(tokenizer, chat_template="gemma-3")

    # Load dataset from Hugging Face
    logger.info(f"Loading Hugging Face dataset: {hf_dataset}")
    try:
        train_dataset = load_dataset(hf_dataset, split = "train")
        eval_dataset = load_dataset(hf_dataset, split = "validation")
        

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
    logger.info("Adding system instructions to first user turn and applying chat template (simplified)...")
    train_dataset = train_dataset.map(
        system_prompt_format,
        batched=False,       # Keep processing individually
        desc="Formatting train data"
    )
    eval_dataset = eval_dataset.map(
        system_prompt_format,
        batched=False,       # Keep processing individually
        desc="Formatting eval data"
    )

    logger.info("Sample processed training text:")
    logger.info(train_dataset[100]["text"])

    # Check for existing checkpoint in the output directory
    resume_checkpoint = None
    if resume_from_checkpoint and os.path.exists(output_dir_path):
        if "checkpoint_step" in locals() and checkpoint_step is not None:
            # Load from specific checkpoint if checkpoint_step is provided
            specific_checkpoint = f"checkpoint-{checkpoint_step}"
            specific_checkpoint_path = os.path.join(output_dir_path, specific_checkpoint)
            if os.path.exists(specific_checkpoint_path):
                resume_checkpoint = specific_checkpoint_path
                logger.info(f"Resuming from specified checkpoint step: {resume_checkpoint}")
            else:
                logger.warning(f"Specified checkpoint {specific_checkpoint} not found")
        # Fall back to finding latest checkpoint
        elif any(d.startswith("checkpoint-") for d in os.listdir(output_dir_path)):
            checkpoints = [d for d in os.listdir(output_dir_path) if d.startswith("checkpoint-")]
            if checkpoints:
                latest_checkpoint = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))[-1]
                resume_checkpoint = os.path.join(output_dir_path, latest_checkpoint)
                logger.info(f"Resuming from latest checkpoint: {resume_checkpoint}")

    data_collator = DataCollatorWithPadding(tokenizer, padding="longest")

    # Set up the trainer
    logger.info("Setting up SFT Trainer...")
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = train_dataset,
        eval_dataset = eval_dataset,
        args = SFTConfig( # Use SFTConfig directly which inherits from TrainingArguments
            output_dir = output_dir_path,
            dataset_text_field = "text",
            gradient_accumulation_steps=gradient_accumulation_steps,
            packing = packing,
            per_device_train_batch_size = batch_size,
            # warmup_ratio = warmup_ratio,
            warmup_steps = warmup_steps,
            num_train_epochs = epochs,
            # max_steps = 0,
            learning_rate = learning_rate,
            logging_steps = 1, # Log frequently
            optim = "adamw_8bit", # Unsloth recommended optimizer
            weight_decay=weight_decay,
            lr_scheduler_type=lr_scheduler_type,
            lr_scheduler_kwargs={"num_cycles": num_cycles} if num_cycles is not None else {},
            seed=3407,
            report_to="none",  # Disable default reporting (like wandb) unless configured
            eval_strategy="steps", # Evaluate during training
            eval_steps=80,         # How often to evaluate
            load_best_model_at_end=False, # Consider if you want the best model based on eval
            metric_for_best_model="eval_loss",
            save_strategy="steps", # Save based on steps
            save_steps=40,         # How often to save checkpoints
            save_total_limit=3,    # Number of checkpoints to keep
        ),
        data_collator=data_collator,
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
    initial_eval_metrics = None
    final_log_history = None

    # Train the model
    try:
        # logger.info("Running initial evaluation...")
        # initial_eval_metrics = trainer.evaluate()
        # logger.info(f"Initial evaluation metrics: {initial_eval_metrics}")

        logger.info("Starting training (evaluation enabled)...")
        training_summary_stats = trainer.train(resume_from_checkpoint=resume_checkpoint)
        logger.info(f"Training finished. Stats: {training_summary_stats}")

        # Run final evaluation
        # logger.info("Running final evaluation...")
        # final_eval_metrics = trainer.evaluate()
        # logger.info(f"Final evaluation metrics: {final_eval_metrics}")

        # Example generation
        # logger.info("Running example generation...")
        # tokenizer = get_chat_template(tokenizer, chat_template = "gemma-3")
        # messages = [{"role": "user", "content": [{"type" : "text", "text" : "O ENADE é obrigatório?"}]}]
        # text = tokenizer.apply_chat_template(messages, add_generation_prompt = True)
        # outputs = model.generate(
        #     **tokenizer([text], return_tensors = "pt").to("cuda"),
        #     max_new_tokens = 256, # Reduced length for quick test
        #     temperature = 1.0, top_p = 0.95, top_k = 64,
        # )
        # logger.info("Example Generation Output:")
        # logger.info(tokenizer.batch_decode(outputs)[0]) # Decode and log

        # Add model merging and GGUF export if enabled
        if merge_and_export:
            logger.info("Checking for existing merged model and GGUF file...")
            
            # Define directories
            merged_model_dir = os.path.join(output_dir_path, "merged_model")
            if checkpoint_step:
                merged_model_dir = os.path.join(output_dir_path, f"checkpoint-{checkpoint_step}", "merged_model")


            gguf_dir = output_dir_path
            if checkpoint_step:
                gguf_dir = os.path.join(output_dir_path, f"checkpoint-{checkpoint_step}")
            
            # Check if merged model exists
            merged_model_exists = os.path.exists(merged_model_dir) and os.path.exists(os.path.join(merged_model_dir, "config.json"))
            
            # Check if GGUF file exists
            gguf_files = [os.path.join(gguf_dir, f) for f in os.listdir(gguf_dir) if f.endswith('.gguf')] if os.path.exists(gguf_dir) else []
            gguf_file_exists = len(gguf_files) > 0
            
            # Generate merged model if needed
            if not merged_model_exists:
                logger.info("Merged model not found. Creating merged model...")
                os.makedirs(merged_model_dir, exist_ok=True)
                logger.info(f"Saving merged model to {merged_model_dir}...")
                model.save_pretrained_merged(merged_model_dir, tokenizer)
                logger.info("Merged model saved successfully")
            else:
                logger.info(f"Merged model already exists at {merged_model_dir}")
            
            # Generate GGUF file if needed
            if not gguf_file_exists:
                logger.info(f"GGUF file not found. Exporting merged model to GGUF format with {export_quantization_type} quantization...")
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
        logger.error(f"An unexpected error occurred during training: {e}", exc_info=True) # Log traceback
        raise
    finally:
        # to save model, state, and summary, even on error
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
                "initial_evaluation_metrics": initial_eval_metrics,
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

    # Construct return message based on success
    if training_summary_stats and final_log_history:
        success_message = f"Fine-tuning complete! Model adapters and summary ({SUMMARY_FILENAME}) saved to {output_dir_path} in volume."
        if merge_and_export:
            success_message += f" Model merged and exported to GGUF format with {export_quantization_type} quantization."
            if push_to_hub and hf_repo_id:
                success_message += f" GGUF model pushed to Hugging Face Hub repo: {hf_repo_id}."
        return success_message
    else:
        return f"Fine-tuning process ended (potentially with errors). Attempted to save state to {output_dir_path}. Check logs and {SUMMARY_FILENAME}."


@app.local_entrypoint()
def main(
    hf_dataset: str = 'liteofspace/unb-chatbot',
    output_dir: str = "unb_chatbot_gemma12b",
    base_model: str = "unsloth/gemma-3-12b-it-bnb-4bit",
    epochs: float = 3.0,
    resume: bool = False,
    checkpoint_step: int = None,
    data_seed: int = None,
    delete_output_dir: bool = False, # Whether to delete existing output directory
    merge_and_export: bool = False, # Enable model merging and GGUF export
    export_quantization_type: str = "Q8_0", # Quantization type for GGUF export
    push_to_hub: bool = False, # Push to Hugging Face Hub
    hf_repo_id: str = "liteofspace/unb-chatbot-gguf" # Repository ID for HF Hub
):
    # Call the remote function
    result = run_fine_tuning.remote(
        hf_dataset=hf_dataset,
        output_dir=output_dir,
        base_model=base_model,
        epochs=epochs,
        resume_from_checkpoint=resume,
        checkpoint_step=checkpoint_step,
        delete_output_dir=delete_output_dir,
        data_seed=data_seed,
        merge_and_export=merge_and_export,
        export_quantization_type=export_quantization_type,
        push_to_hub=push_to_hub,
        hf_repo_id=hf_repo_id
    )

    print(result)
    print("\nTo download the model adapters, run:")
    print(f"  modal volume get {VOLUME_NAME} {output_dir} ./local_model_dir_{output_dir.replace('/', '_')}")
    
    if merge_and_export:
        print("\nTo download the merged model and GGUF files, run:")
        print(f"  modal volume get {VOLUME_NAME} {output_dir}/merged_model ./local_merged_model_{output_dir.replace('/', '_')}")
        print(f"  modal volume get {VOLUME_NAME} {output_dir}/gguf_model ./local_gguf_model_{output_dir.replace('/', '_')}")