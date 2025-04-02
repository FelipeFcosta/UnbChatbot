#!/usr/bin/env python3
"""
Fine-tune an LLM using Unsloth's LoRA approach with synthetic QA data.
This script uses Modal to run the fine-tuning process on cloud T4 GPUs.
"""

import argparse
import os
import logging
from pathlib import Path
import modal

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# Define Modal app and resources
app = modal.App("unb-chatbot")

# Create a Modal image with dependencies
image = (
    modal.Image.debian_slim()
    .apt_install("git")
    .pip_install(
        "torch",
        "unsloth",
        "datasets",
        "xformers==0.0.29.post3",
        "transformers",
        "trl",
        "triton",
        "huggingface_hub",
        "bitsandbytes",
        "accelerate",
        "cut_cross_entropy",
        "peft",
        "unsloth_zoo",
        "sentencepiece",
        "protobuf",
        "hf_transfer"
    )
)

# Volume to store output models
output_volume = modal.Volume.from_name(
    "unb-chatbot-models", create_if_missing=True
)


@app.function(
    image=image,
    gpu="T4",
    timeout=60 * 60 * 3,  # 3 hour timeout
    volumes={"/outputs": output_volume},
)
def run_fine_tuning(
    hf_dataset: str,
    hf_token: str = None,
    output_dir: str = "unb_chatbot",
    base_model: str = "Qwen2.5-7B-Instruct",
    epochs: int = 3,
    learning_rate: float = 5e-5,
    batch_size: int = 4,
    lora_rank: int = 16,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    max_seq_length: int = 2048,
    warmup_ratio: float = 0.1,
    weight_decay: float = 0.01
):
    """Run fine-tuning on Modal with T4 GPU"""
    # Import dependencies that are only needed in the cloud environment
    import torch
    from unsloth import FastLanguageModel, is_bfloat16_supported
    from unsloth.chat_templates import get_chat_template, train_on_responses_only
    from datasets import load_dataset
    from trl import SFTTrainer
    from transformers import TrainingArguments
    from huggingface_hub import login


    def formatting_prompts_func(examples, tokenizer):
        """
        Formats the prompts from the dataset by applying the chat template
        and tokenizing the resulting texts.
        """
        convos = examples["messages"]
        # Apply the chat template to each conversation without tokenizing yet.
        texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False)
                for convo in convos]
        # Tokenize the texts.
        tokenized_texts = tokenizer(texts, padding=False, truncation=True, add_special_tokens=False)
        return {"text": texts, "input_ids": tokenized_texts["input_ids"]}

    # Login to HF if token provided
    if hf_token:
        logger.info(f"Logging in to Hugging Face Hub with provided token")
        login(token=hf_token)
    
    # Output to volume
    output_dir_path = f"/outputs/{output_dir}"
    os.makedirs(output_dir_path, exist_ok=True)
    
    logger.info(f"Starting fine-tuning process with the following settings:")
    logger.info(f"  Base Model: {base_model}")
    logger.info(f"  Using Hugging Face dataset: {hf_dataset}")
    logger.info(f"  Output Directory: {output_dir_path}")
    logger.info(f"  Epochs: {epochs}")
    logger.info(f"  Learning Rate: {learning_rate}")
    logger.info(f"  Batch Size: {batch_size}")
    
    # Load the base model and tokenizer
    logger.info("Loading base model and tokenizer...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model,
        max_seq_length=max_seq_length,
        dtype=None,  # Auto detection of dtype
        load_in_4bit=True,  # Use 4-bit quantization for efficient training
    )

    # Enable memory-efficient training
    model.gradient_checkpointing_enable()
    if torch.cuda.is_available():
        # Optimize CUDA memory usage
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True
    
    # Apply LoRA adapters to the model
    logger.info("Setting up LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_rank,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        use_gradient_checkpointing=True,  # Enable for memory efficiency
        random_state=42,
        use_rslora=False,
        loftq_config=None,
    )

    # Apply chat template for conversation formatting
    logger.info(f"Applying chat template: llama-3.1")
    tokenizer = get_chat_template(tokenizer, chat_template="llama-3.1")

    # Load dataset from Hugging Face
    logger.info(f"Loading Hugging Face dataset: {hf_dataset}")
    try:
        # Load from Hugging Face Hub
        dataset = load_dataset(hf_dataset)

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
        logger.error(f"Error loading dataset from Hugging Face: {e}")
        raise
        
    # Process datasets to prepare for training
    logger.info("Processing datasets with the chat template...")
    train_dataset = train_dataset.map(
        lambda ex: formatting_prompts_func(ex, tokenizer=tokenizer),
        batched=True,
        desc="Processing training dataset"
    )
    
    if eval_dataset:
        eval_dataset = eval_dataset.map(
            lambda ex: formatting_prompts_func(ex, tokenizer=tokenizer),
            batched=True,
            desc="Processing validation dataset"
        )
    
    # Print sample data
    logger.info("Sample training example:")
    sample_idx = min(0, len(train_dataset) - 1)
    if sample_idx >= 0 and len(train_dataset) > 0:
        logger.info(f"Input: {train_dataset[sample_idx]['messages'][0]['content'][:100]}...")
        logger.info(f"Target: {train_dataset[sample_idx]['messages'][1]['content'][:100]}...")

    # Configure training arguments
    logger.info("Setting up training configuration...")
    training_args = TrainingArguments(
        output_dir=output_dir_path,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=1,
        warmup_ratio=warmup_ratio,  # Use ratio instead of steps
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        optim="adamw_8bit",  # More memory efficient optimizer
        save_strategy="steps",
        save_steps=200,
        save_total_limit=3,
        weight_decay=weight_decay,
        lr_scheduler_type="linear",  # Better than linear for domain adaptation
        seed=42,
        report_to="none",
        eval_strategy="steps" if eval_dataset else "no",
        eval_steps=50 if eval_dataset else None,
    )

    # Set up the trainer
    logger.info("Setting up SFT Trainer...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        dataset_num_proc=2,
        packing=False,  # Better for question-answer pairs to avoid context bleeding
        args=training_args,
    )

    # Apply response-only training - only compute loss on assistant responses
    logger.info("Configuring trainer to only compute loss on assistant responses")
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|start_header_id|>user<|end_header_id|>\n\n",
        response_part="<|start_header_id|>assistant<|end_header_id|>\n\n",
    )

    # Train the model
    logger.info("Starting training...")
    trainer_stats = trainer.train()

    # Save the fine-tuned model
    logger.info(f"Saving fine-tuned model to {output_dir_path}...")
    model.save_pretrained(output_dir_path)
    tokenizer.save_pretrained(output_dir_path)

    logger.info("Fine-tuning complete! Model saved successfully to volume.")
    return f"Model saved to {output_dir_path}"


@app.local_entrypoint()
def main(
    hf_dataset: str = 'liteofspace/unb-chatbot',
    hf_token: str = os.environ.get("HF_TOKEN"),
    output_dir: str = "unb_chatbot",
    base_model: str = "Qwen2.5-7B-Instruct",
    epochs: int = 3,
    batch_size: int = 4
):
    # Call the remote function with the provided arguments
    result = run_fine_tuning.remote(
        hf_dataset=hf_dataset,
        hf_token=hf_token,
        output_dir=output_dir,
        base_model=base_model,
        epochs=epochs,
        batch_size=batch_size
    )
    
    print(result)
    print("\nTo download the model, run:")
    print(f"  modal volume get unb-chatbot-models {output_dir} ./local_model_dir")


# To download the model, run:
  # modal volume get unb-chatbot-models unb_chatbot local_model_dir

# cd ./local_model_dir
# ollama create unb-chatbot -f Modelfile