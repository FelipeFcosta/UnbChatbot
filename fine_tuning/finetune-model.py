#!/usr/bin/env python3
"""
Fine-tune an LLM using Unsloth's LoRA approach with synthetic QA data.

This script adapts the Unsloth library for fine-tuning on 
synthetic QA data generated specifically for institutional chatbots.
"""

import argparse
import os
import logging
from pathlib import Path
import torch

from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import get_chat_template
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
from huggingface_hub import login

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Fine-tune a language model using PEFT LoRA on institutional QA data.")

    # Dataset loading options - either local files or Hugging Face dataset
    data_group = parser.add_argument_group('Dataset Options')
    data_group.add_argument("--train_data", type=str, default=None, help="Path to training data file (jsonl format).")
    data_group.add_argument("--val_data", type=str, default=None, help="Path to validation data file (jsonl format).")
    data_group.add_argument("--hf_dataset", type=str, default=None, 
                          help="Hugging Face dataset name (e.g., 'username/unb-qa-dataset')")
    data_group.add_argument("--hf_token", type=str, default=None, 
                          help="Hugging Face API token for accessing private datasets")
    
    parser.add_argument("--output_dir", type=str, default="unsloth_finetuned", help="Output directory for fine-tuned model.")
    
    # Optimized defaults for institutional QA fine-tuning
    parser.add_argument("--base_model", type=str, default="unsloth/Llama-3.1-1B-Instruct-bnb-4bit", 
                        help="Base model path or HF identifier.")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for training.")
    parser.add_argument("--chat_template", type=str, default="llama-3.1", help="Chat template for the model.")
    parser.add_argument("--lora_rank", type=int, default=16, help="LoRA rank parameter.")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha parameter.")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout probability.")
    parser.add_argument("--max_seq_length", type=int, default=2048, help="Maximum sequence length.")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Portion of steps used for warmup.")
    parser.add_argument("--save_steps", type=int, default=200, help="Save checkpoint every N steps.")
    parser.add_argument("--save_total_limit", type=int, default=3, help="Maximum number of checkpoints to save.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for AdamW optimizer.")
    parser.add_argument("--use_checkpoint", action="store_true", help="Resume from checkpoint if available.")
    parser.add_argument("--quantization", type=str, default="q4_k_m", 
                        help="Quantization method for GGUF export (e.g. q4_k_m)")
    parser.add_argument("--eval_steps", type=int, default=200, help="Evaluate every N steps.")
    parser.add_argument("--log_steps", type=int, default=10, help="Log every N steps.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, 
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--export_gguf", action="store_true", 
                        help="Export model to GGUF format after training")
    
    return parser.parse_args()



def formatting_prompts_func(examples, tokenizer):
    """
    Formats the prompts from the dataset by applying the chat template
    and tokenizing the resulting texts.

    Args:
        examples (dict): A dictionary with a key "messages" containing conversation data.
        tokenizer: The tokenizer that includes the chat template method.

    Returns:
        dict: A dictionary with tokenized texts under the key "text".
    """
    convos = examples["messages"]
    # Apply the chat template to each conversation without tokenizing yet.
    texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False)
             for convo in convos]
    # Tokenize the texts.
    tokenized_texts = tokenizer(texts, padding=False, truncation=True, add_special_tokens=False)
    return {"text": texts, "input_ids": tokenized_texts["input_ids"]}


def main():
    args = parse_arguments()
    
    # Validate dataset arguments
    if args.hf_dataset is None and args.train_data is None:
        raise ValueError("You must provide either --hf_dataset or --train_data")
    
    # Login to Hugging Face if using their dataset and a token is provided
    if args.hf_dataset and args.hf_token:
        logger.info(f"Logging in to Hugging Face Hub with provided token")
        login(token=args.hf_token)
    
    logger.info(f"Starting fine-tuning process with the following settings:")
    logger.info(f"  Base Model: {args.base_model}")
    if args.hf_dataset:
        logger.info(f"  Using Hugging Face dataset: {args.hf_dataset}")
    else:
        logger.info(f"  Training Data: {args.train_data}")
        logger.info(f"  Validation Data: {args.val_data if args.val_data else 'None'}")
    logger.info(f"  Output Directory: {args.output_dir}")
    logger.info(f"  Epochs: {args.epochs}")
    logger.info(f"  Learning Rate: {args.learning_rate}")
    logger.info(f"  Batch Size: {args.batch_size}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load the base model and tokenizer
    logger.info("Loading base model and tokenizer...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.base_model,
        max_seq_length=args.max_seq_length,
        dtype=None,  # Auto detection of dtype
        load_in_4bit=True,  # Use 4-bit quantization for efficient training
    )

    # Enable memory-efficient training
    model.gradient_checkpointing_enable()
    if torch.cuda.is_available():
        # Optimize CUDA memory usage
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True
    
    # Add Portuguese special tokens
    logger.info("Optimizing tokenizer for Portuguese language...")
    pt_special_tokens = ['ç', 'á', 'à', 'â', 'ã', 'é', 'ê', 'í', 'ó', 'ô', 'õ', 'ú']
    special_tokens_dict = {'additional_special_tokens': pt_special_tokens}
    num_added_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    if num_added_tokens > 0:
        logger.info(f"Added {num_added_tokens} special tokens for Portuguese language")
        model.resize_token_embeddings(len(tokenizer))

    # Apply LoRA adapters to the model
    logger.info("Setting up LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_rank,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        use_gradient_checkpointing=True,  # Enable for memory efficiency
        random_state=args.seed,
        use_rslora=False,
        loftq_config=None,
    )

    # Apply chat template for conversation formatting
    logger.info(f"Applying chat template: {args.chat_template}")
    tokenizer = get_chat_template(tokenizer, chat_template=args.chat_template)

    # Load dataset: either from Hugging Face or local files
    if args.hf_dataset:
        logger.info(f"Loading Hugging Face dataset: {args.hf_dataset}")
        try:
            # Load from Hugging Face Hub
            dataset = load_dataset(args.hf_dataset)
            
            # Get train split
            train_dataset = dataset.get("train", dataset.get("training"))
            if train_dataset is None:
                raise ValueError(f"No 'train' or 'training' split found in dataset {args.hf_dataset}")
            logger.info(f"Loaded training dataset with {len(train_dataset)} examples")
            
            # Get validation split if it exists
            eval_dataset = dataset.get("validation", dataset.get("val", dataset.get("dev", None)))
            if eval_dataset is not None:
                logger.info(f"Loaded validation dataset with {len(eval_dataset)} examples")
            else:
                logger.warning(f"No validation split found in dataset {args.hf_dataset}")
                
        except Exception as e:
            logger.error(f"Error loading dataset from Hugging Face: {e}")
            raise
    else:
        # Load from local files
        logger.info(f"Loading training dataset from {args.train_data}")
        train_dataset = load_dataset("json", data_files=args.train_data, split="train")
        logger.info(f"Loaded training dataset with {len(train_dataset)} examples")
        
        # Load validation dataset if provided
        if args.val_data:
            logger.info(f"Loading validation dataset from {args.val_data}")
            eval_dataset = load_dataset("json", data_files=args.val_data, split="train")
            logger.info(f"Loaded validation dataset with {len(eval_dataset)} examples")
        else:
            eval_dataset = None
        
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

    # Configure training arguments with optimized settings
    logger.info("Setting up training configuration...")
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_ratio=args.warmup_ratio,  # Use ratio instead of steps
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=args.log_steps,
        optim="adamw_8bit",  # More memory efficient optimizer
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        weight_decay=args.weight_decay,
        lr_scheduler_type="cosine_with_restarts",  # Better than linear for domain adaptation
        seed=args.seed,
        report_to="none",
        evaluation_strategy="steps" if eval_dataset else "no",
        eval_steps=args.eval_steps if eval_dataset else None,
    )

    # Set up the trainer
    logger.info("Setting up SFT Trainer...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="input_ids",
        max_seq_length=args.max_seq_length,
        dataset_num_proc=max(1, os.cpu_count() // 2),
        packing=False,  # Better for question-answer pairs to avoid context bleeding
        args=training_args,
    )

    # Train the model
    logger.info("Starting training...")
    trainer_stats = trainer.train(resume_from_checkpoint=args.use_checkpoint)

    # Save the fine-tuned model
    logger.info("Saving fine-tuned model...")
    output_dir_str = str(output_dir)
    model.save_pretrained(output_dir_str)
    tokenizer.save_pretrained(output_dir_str)

    # Export to GGUF format with optimization for Portuguese
    if args.export_gguf:
        logger.info(f"Exporting model to GGUF format with {args.quantization} quantization...")
        try:
            model.save_pretrained_gguf(
                output_dir_str, 
                tokenizer, 
                quantization_method=args.quantization.lower()
            )
            
            # Generate a modelfile for Ollama with Portuguese optimization
            modelfile_path = os.path.join(output_dir_str, f"Modelfile")
            with open(modelfile_path, "w") as f:
                f.write(f"FROM ./unsloth.{args.quantization.lower()}.gguf\n")
                # Include comments about Portuguese optimization
                f.write(f"# Model fine-tuned for Portuguese institutional content\n")
                # Standard template for LLaMA 3.1
                f.write('TEMPLATE """{{- if .System }}<|start_header_id|>system<|end_header_id|>\n\n{{ .System }}<|eot_id|>\n{{- end }}\n{{- range .Messages }}<|start_header_id|>{{ .Role }}<|end_header_id|>\n\n{{ .Content }}<|eot_id|>\n{{- end }}<|start_header_id|>assistant<|end_header_id|>\n\n"""')
            
            logger.info(f"Created Modelfile at {modelfile_path}")
            
        except Exception as e:
            logger.error(f"Failed to export to GGUF: {e}")
            logger.info("You can still use the saved model with transformers or export it to GGUF manually.")

    logger.info("Fine-tuning complete! Model saved successfully.")


if __name__ == "__main__":
    main()