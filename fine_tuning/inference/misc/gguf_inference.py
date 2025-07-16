#!/usr/bin/env python3
"""
Run inference using a GGUF LLM model stored in a Modal Volume,
exposed as a persistent web endpoint using llama-cpp-python with CUDA support.
"""

import modal
import os
import logging
from pathlib import Path

# --- Configuration ---
APP_NAME = "unb-chatbot-gguf-web-endpoint"
MODEL_DIR_IN_VOLUME = "faq_gemma4b_dpo_run4"
GGUF_FILENAME = "merged_model.Q8_0.gguf"
VOLUME_NAME = "unb-chatbot-gemma3-dpo"
GPU_CONFIG = "A10G"
MODEL_MOUNT_PATH = "/model_files"
CONTEXT_SIZE = 4096
SYSTEM_PROMPT = "Você é um assistente especializado que responde perguntas *estritamente* com base nos dados de FAQ da UnB fornecidos no contexto. Seja preciso e factual de acordo com o material de origem. Não invente informações.\n\n"
# SYSTEM_PROMPT = ""

# Default generation parameters
DEFAULT_MAX_TOKENS = 1024
DEFAULT_TEMPERATURE = 0.6
DEFAULT_TOP_P = 0.9
MINUTES = 60

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

app = modal.App(APP_NAME)

# IMPORTANT: Build custom image with CUDA support properly configured
cuda_version = "12.1.0"  # Match with Modal's CUDA version
flavor = "devel"
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.10")
    .apt_install(
        "git",
        "build-essential", 
        "cmake",
        "curl",
        "libcurl4-openssl-dev",
        "libopenblas-dev",
        "libomp-dev",
        "clang",
        "gcc-11",
        "g++-11"
    )
    # Install Python dependencies
    .pip_install(
        "modal-client",
        "fastapi[standard]",
        "huggingface_hub",
        "ninja",
        "wheel",
        "packaging"
    )
    # Build llama-cpp-python with CUDA support
    .run_commands(
        'CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=75" pip install --no-cache-dir llama-cpp-python',
        gpu="A10G"
    )
    .entrypoint([])  # remove NVIDIA base container entrypoint
)

# Access the Volume where model is stored
model_volume = modal.Volume.from_name(VOLUME_NAME)

@app.cls(
    image=image,
    gpu=GPU_CONFIG,
    volumes={MODEL_MOUNT_PATH: model_volume},
    timeout=60*6,
    allow_concurrent_inputs=3,  # Reduced for A10G VRAM
    min_containers=2
)
class ModelEndpoint:
    def __init__(self):
        self.gguf_file_path_in_container = str(
            Path(MODEL_MOUNT_PATH) / MODEL_DIR_IN_VOLUME / GGUF_FILENAME
        )
        self.llm = None

    @modal.enter()
    def load_model(self):
        """Load the GGUF model when container starts."""
        from llama_cpp import Llama

        logger.info("Container starting up...")
        logger.info(f"Attempting to load Llama model from {self.gguf_file_path_in_container}...")

        if not os.path.exists(self.gguf_file_path_in_container):
            error_msg = f"GGUF model file not found at {self.gguf_file_path_in_container}."
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        try:
            # Configure for optimal GPU usage
            self.llm = Llama(
                model_path=self.gguf_file_path_in_container,
                n_gpu_layers=-1,  # Use all possible GPU layers
                n_ctx=CONTEXT_SIZE,
                n_batch=512,       # Larger batch size for better throughput
                f16_kv=True,       # Use half precision for key/value cache
                verbose=True,
                seed=42,           # Deterministic output for testing
                offload_kqv=True,  # Offload key/query/value tensors to GPU
                use_mlock=True     # Lock memory to prevent swapping
            )
            logger.info("Successfully loaded Llama model into container.")
        except Exception as e:
            logger.error(f"Failed to load Llama model during container startup: {str(e)}")
            logger.exception(e)
            raise

    @modal.fastapi_endpoint(method="POST")
    async def generate_web(self, request_data: dict):
        """Handle inference requests."""
        from fastapi import HTTPException

        if self.llm is None:
            logger.error("Model was not loaded successfully during startup.")
            raise HTTPException(status_code=503, detail="Model not ready.")

        user_prompt = request_data.get("prompt")
        if not user_prompt:
            raise HTTPException(status_code=400, detail="Missing 'prompt' in request.")

        # Get generation parameters
        max_tokens = request_data.get("max_tokens", DEFAULT_MAX_TOKENS)
        temperature = request_data.get("temperature", DEFAULT_TEMPERATURE)
        top_p = request_data.get("top_p", DEFAULT_TOP_P)

        logger.info(f"Received prompt: { SYSTEM_PROMPT + user_prompt}...")
        logger.info(f"Generation parameters: max_tokens={max_tokens}, temperature={temperature}, top_p={top_p}")

        text = SYSTEM_PROMPT + user_prompt

        messages = [
            {"role": "user", "content": text}
        ]

        logger.info("Generating response...")
        try:
            output = self.llm.create_chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=["<end_of_turn>"],
                repeat_penalty=1.1
            )

            # Extract response
            if output and output.get('choices') and len(output['choices']) > 0:
                response_text = output['choices'][0]['message']['content']
                if usage := output.get('usage'):
                    logger.info(f"Token usage: {usage}")
            else:
                logger.warning("No response generated or unexpected output format.")
                response_text = ""

        except Exception as e:
            logger.error(f"Error during generation: {e}")
            logger.exception(e)
            raise HTTPException(status_code=500, detail=f"Error during generation: {str(e)}")

        logger.info("Generation complete.")
        return {"response": response_text.strip()}

# Optional local entrypoint for testing
@app.local_entrypoint()
def main(prompt: str = "Como funciona o processo de matrícula na UnB?"):
    """Test the endpoint locally"""
    response = ModelEndpoint().generate_web.remote({"prompt": prompt})
    print(f"Response: {response['response']}")