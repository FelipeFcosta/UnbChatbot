# improved_upload_to_hf.py
import datasets
from datasets import load_dataset
from huggingface_hub import login
import argparse
import os

parser = argparse.ArgumentParser(description="Upload dataset to Hugging Face")
parser.add_argument("--train", default="output/unb_training_data.jsonl", help="Training data file")
parser.add_argument("--val", default="output/unb_training_data_val.jsonl", help="Validation data file")
parser.add_argument("--repo", default="liteofspace/unb-chatbot", help="Hugging Face repository ID")
parser.add_argument("--message", default="Update dataset", help="Commit message")
args = parser.parse_args()

# Log in to Hugging Face
login(token=os.environ.get("HF_TOKEN"))

# Load dataset
dataset = load_dataset("json", data_files={
    "train": args.train,
    "validation": args.val
})

# Push to Hub with commit message
dataset.push_to_hub(args.repo, commit_message=args.message)

print(f"Dataset updated at {args.repo} with message: {args.message}")