import modal
import os

# Define Modal app
app = modal.App("unb-chatbot-export")

# Minimal image with essential packages
image = (
    modal.Image.debian_slim()
    .apt_install('git')
    .pip_install(
        "torch",
        "transformers>=4.34.0",
        "peft>=0.5.0",
        "huggingface_hub",
        "accelerate",
        "bitsandbytes"
    )
)

# Use the same volume where your model is stored
output_volume = modal.Volume.from_name("unb-chatbot-models")

@app.function(
    image=image,
    gpu="T4",
    timeout=60 * 30,  # 30 minute timeout
    volumes={"/outputs": output_volume},
)
def push_model_to_hf(
    model_dir: str = "unb_chatbot",
    hf_token: str = None
):
    """Push model to HuggingFace in standard format (no GGUF)"""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import logging
    from huggingface_hub import login
    import os
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Login to HuggingFace
    if hf_token:
        login(token=hf_token)
        logger.info("Logged in to HuggingFace")
    else:
        logger.warning("No HF token provided - repository will be public")
    
    model_path = f"/outputs/{model_dir}"
    logger.info(f"Loading model from {model_path}")
    
    # Check if model exists
    if not os.path.exists(model_path):
        available = os.listdir("/outputs")
        return f"Model not found at {model_path}. Available: {available}"
    
    # Load model and tokenizer (using standard transformers, not unsloth)
    logger.info("Loading with standard transformers library")
    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Push to HuggingFace without GGUF conversion
    username = "liteofspace"  # Replace with your actual username
    hf_repo = f"{username}/unb-chatbot"
    
    logger.info(f"Pushing model to HuggingFace as {hf_repo}...")
    model.push_to_hub(hf_repo, token=hf_token)
    tokenizer.push_to_hub(hf_repo, token=hf_token)
    
    return f"""
Success! Model pushed to HuggingFace at: https://huggingface.co/{hf_repo}

To use with Ollama:
1. Install llama.cpp locally: https://github.com/ggml-org/llama.cpp
2. Clone your model: git clone https://huggingface.co/{hf_repo}
3. Convert to GGUF: ./llama.cpp/convert.py /path/to/cloned/model --outfile model.gguf
4. Create Modelfile and use with Ollama

Alternatively, you can download GGUF from HuggingFace's web interface after they process it.
"""

@app.local_entrypoint()
def main(
    model_dir: str = "unb_chatbot",
    hf_token: str = os.environ.get("HF_TOKEN")
):
    """Run the export function"""
    if not hf_token:
        print("HF_TOKEN environment variable not set")
        print("Get a token at: https://huggingface.co/settings/tokens")
        hf_token = input("Enter HuggingFace token: ")
    
    result = push_model_to_hf.remote(
        model_dir=model_dir,
        hf_token=hf_token
    )
    print(result)