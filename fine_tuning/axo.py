import modal

# Create volumes for persistent storage
cache_vol = modal.Volume.from_name("hf-cache", create_if_missing=True)
output_vol = modal.Volume.from_name("axolotl-output", create_if_missing=True)

# Define the Modal app
app = modal.App("axolotl-llama3")

# Set up CUDA version parameters for the base image
cuda_version = "12.8.0"
flavor = "devel"  # includes full CUDA toolkit
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

# Create the image with CUDA support and all necessary dependencies
image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.11")
    .apt_install("git", "build-essential", "cmake", "ninja-build")
    # Install build dependencies first - needs to happen before flash-attn
    .run_commands("pip install pip==25.1", force_build=True)
    .pip_install("packaging", "setuptools", "wheel", "ninja", "pybind11", "einops")
    # Install PyTorch with CUDA support - specific version required by axolotl
    .pip_install("torch==2.6.0")
    # Set environment variables for CUDA
    .env({"CUDA_HOME": "/usr/local/cuda"})
    # Install flash-attn separately before axolotl
    .pip_install("flash-attn==2.7.4.post1")
    .pip_install("xformers==0.0.29.post2")
    # Install Axolotl without flash-attn (as we installed it separately)
    .run_commands(
        "git clone https://github.com/axolotl-ai-cloud/axolotl.git",
        "cd axolotl && pip install --use-pep517 -e '.[deepspeed]'"
    )
)

# Define the training function
@app.function(
    gpu="A10G",  # Use A10G GPU
    volumes={
        "/cache": cache_vol,
        "/outputs": output_vol
    },
    image=image,
    timeout=6 * 60 * 60,  # 6 hour timeout
)
def train_model():
    import os
    import subprocess
    
    # Create example config directory
    os.makedirs("examples", exist_ok=True)
    
    # Fetch example configs from Axolotl
    subprocess.run(["axolotl", "fetch", "examples"], check=True)
    
    # Run the training with the Llama-3 LoRA example
    cmd = ["axolotl", "train", "examples/llama-3/lora-1b.yml"]
    
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    
    # Stream logs in real-time
    for line in process.stdout:
        print(line, end="")
    
    process.wait()
    
    if process.returncode != 0:
        return "Training failed"
    
    return "Training completed successfully"

# Entry point for the application
@app.local_entrypoint()
def main():
    print("Starting Axolotl training on Modal...")
    result = train_model.remote()
    print(result)

if __name__ == "__main__":
    with app.run():
        main()