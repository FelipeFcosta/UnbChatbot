#!/usr/bin/env python3
"""
Run retrieval endpoint using Modal, exposed as a persistent web endpoint.
Returns list of retrieved documents for distractor creation.
Assumes source_json_combined.json is present in the document volume.
"""

import modal
import os
import logging
from pathlib import Path
from typing import List, Dict
from fastapi import Query, HTTPException

# Package context
from .config import *
from .data_handler import DataHandler

GPU_CONFIG = "A10G"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

app = modal.App("retrieval-endpoint")

cuda_version = "12.1.0"
flavor = "devel"
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.10")
    .apt_install(
        "git", "build-essential", "cmake", "curl",
        "libcurl4-openssl-dev", "libopenblas-dev", "libomp-dev", "clang",
        "gcc-11", "g++-11"
    )
    .pip_install(
        "modal-client",
        "fastapi[standard]",
        "huggingface_hub",
        "ninja",
        "wheel",
        "packaging",
        # --- For RAG ---
        "sentence-transformers",
        "faiss-cpu", # Use faiss-cpu for simplicity
        "numpy",     # Dependency for FAISS/embeddings
        # ---------------------
    )
    .run_commands(
        'CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=75" pip install --no-cache-dir llama-cpp-python',
        gpu=GPU_CONFIG
    )
    .entrypoint([])
)

document_volume = modal.Volume.from_name(DATA_VOLUME_NAME)

@app.cls(
    image=image,
    gpu=GPU_CONFIG,
    volumes={
        DATA_MOUNT_PATH: document_volume, # where documents source is mounted
    },
    timeout=60*6,
    min_containers=1
)
@modal.concurrent(max_inputs=3)
class RetrievalEndpoint:
    def __init__(self):
        self.data_handler = DataHandler(logger)

    @modal.enter()
    def load_data(self):
        """Load and index data."""
        logger.info("Container starting up...")

        try:
            self.data_handler.load_and_index_data()
        except Exception as e:
            logger.error(f"Failed to initialize RAG components: {e}", exc_info=True)

    @modal.fastapi_endpoint(method="POST")
    async def retrieve(self, request_data: dict):
        """Retrieve documents based on query and return as list."""
        if self.data_handler.retriever is None:
            logger.error("Retriever was not initialized successfully.")
            raise HTTPException(status_code=503, detail="Retriever not ready.")

        query = request_data.get("query")
        if not query:
            raise HTTPException(status_code=400, detail="'query' is required in the request body.")

        k = request_data.get("k", 5)

        try:
            retrieved_docs = self.data_handler.retrieve_raw(query, k)
            return {"docs": retrieved_docs}
        except Exception as e:
            logger.error(f"Error during retrieval: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Error during retrieval: {str(e)}") 