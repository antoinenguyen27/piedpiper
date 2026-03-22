import modal

# 1. Define the environment and install the missing library
image = modal.Image.debian_slim().pip_install("huggingface_hub")

# Your personal volume name
volume = modal.Volume.from_name("model-weights-vol")

app = modal.App("hf-uploader")


@app.function(
    image=image,
    volumes={"/data": volume},
    # Ensure you've added your HF token to Modal Secrets
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=600  # Increased timeout for the 500MiB upload
)
def upload():
    from huggingface_hub import HfApi
    api = HfApi()

    repo_id = "HuggingFacer112358/piedpiper"

    # Ensure the repo exists before uploading
    print(f"Checking if repo {repo_id} exists...")
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)

    print("Uploading to Hugging Face...")
    api.upload_folder(
        folder_path="/data",
        repo_id=repo_id,
        repo_type="model",
    )
    print(f"Done! Access your model at: https://huggingface.co/{repo_id}")


@app.local_entrypoint()
def main():
    upload.remote()