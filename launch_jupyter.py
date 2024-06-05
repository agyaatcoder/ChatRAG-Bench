import os
import subprocess
import modal 
import time 

MODEL_DIR = "/model"
MODEL_NAME = "nvidia/Llama3-ChatQA-1.5-8B"  # Add your model name 

DATASET_ID = "nvidia/ChatRAG-Bench"
HF_SECRET_NAME = "huggingface-llama" #Required for gated models, secret name from modal 


def download_model_to_image(model_dir, model_name):
    from huggingface_hub import snapshot_download
    from transformers.utils import move_cache

    os.makedirs(model_dir, exist_ok=True)

    snapshot_download(
        model_name,
        local_dir=model_dir,
        token=os.environ["HF_TOKEN"],
        ignore_patterns=["*.pt", "*.gguf"],  # Using safetensors
    )
    move_cache()

def download_benchmark_dataset_to_image():
    """This will download all the datafiles to modal image. This files will be downloaded to hf cache
    """
    from datasets import load_dataset
    dataset_names = [
                     'coqa', 'inscit', 'topiocqa', 'hybridial', 
                     'doc2dial', 'quac', 'qrecc', 'doqa_cooking',
                     'doqa_movies', 'doqa_travel', 'sqa'
                     ]
    for dataset in dataset_names:
        load_dataset("nvidia/ChatRAG-Bench", dataset)
        

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.1.1-devel-ubuntu22.04", add_python="3.10"
    )
    .apt_install("git", "git-lfs")
    .pip_install(
        "vllm==0.2.5",
        "huggingface_hub==0.19.4",
        "hf-transfer==0.1.4",
        "torch==2.1.2",
        "datasets"
    )
    # Use the barebones hf-transfer package for maximum download speeds. No progress bar, but expect 700MB/s.
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .run_function(
        download_model_to_image,
        secrets=[modal.Secret.from_name(HF_SECRET_NAME)],
        timeout=60 * 20,
        kwargs={"model_dir": MODEL_DIR, "model_name": MODEL_NAME},
    )
    .run_function(download_benchmark_dataset_to_image)
    .pip_install("jupyterlab")
)

stub = modal.Stub("rag_bench", image=image)


GPU_CONFIG = modal.gpu.H100(count=1)  #too many prompts - will do batch processing - faster 


JUPYTER_TOKEN = "1234"  # Change me to something non-guessable!


#secrets=[Secret.from_name("mistral-secret")
@stub.function(concurrency_limit=1, timeout= 30000, gpu = GPU_CONFIG)
def run_jupyter(timeout: int):
    jupyter_port = 8888
    with modal.forward(jupyter_port) as tunnel:
        jupyter_process = subprocess.Popen(
            [
                "jupyter",
                "lab",
                "--no-browser",
                "--allow-root",
                "--ip=0.0.0.0",
                f"--port={jupyter_port}",
                "--NotebookApp.allow_origin='*'",
                "--NotebookApp.allow_remote_access=1",
            ],
            env={**os.environ, "JUPYTER_TOKEN": JUPYTER_TOKEN},
        )

        print(f"Jupyter available at => {tunnel.url}")

        try:
            end_time = time.time() + timeout
            while time.time() < end_time:
                time.sleep(5)
            print(f"Reached end of {timeout} second timeout period. Exiting...")
        except KeyboardInterrupt:
            print("Exiting...")
        finally:
            jupyter_process.kill()


@stub.local_entrypoint()
def main(timeout: int = 30000):
    # Run the Jupyter Notebook server
    run_jupyter.remote(timeout=timeout)

#modal run launch_jupter.py
