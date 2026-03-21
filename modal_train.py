"""
Modal entrypoint for training the VideoCompressor on an H100 GPU.

Usage:
    modal run modal_train.py

Data setup (one-time):
    modal volume create piedpiper-data
    modal volume put piedpiper-data ./datasets /datasets
    modal volume put piedpiper-data ./models  /models
"""

import modal

# ---------------------------------------------------------------------------
# Modal configuration
# ---------------------------------------------------------------------------

app = modal.App("piedpiper-train")

# Persistent volume for datasets, model weights, and checkpoints
volume = modal.Volume.from_name("piedpiper-data", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libgl1-mesa-glx", "libglib2.0-0")
    .pip_install(
        # Core ML
        "torch==2.4.0",
        "torchvision==0.19.0",
        "torchaudio==2.4.0",
        # Video I/O
        "decord>=0.6.0",
        "opencv-python-headless",
        "av",
        # Model ecosystem
        "transformers>=4.44.0",
        "accelerate>=0.33.0",
        # Utilities
        "numpy<2.0.0",
        "pandas",
        "tqdm",
    )
    .add_local_dir(".", remote_path="/app")  # copy project source into the image
)

VOLUME_MOUNT = "/data"


# ---------------------------------------------------------------------------
# Training function
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    gpu="H100",
    volumes={VOLUME_MOUNT: volume},
    timeout=60 * 60 * 12,  # 12-hour max
)
def train(
    num_epochs: int = 10,
    batch_size: int = 4,
    learning_rate: float = 1e-3,
    save_freq: int = 2,
    model_weights: str = "/data/models/flax_base_f16r288_repeated.npz",
    data_dir: str = "/data/datasets/eastgate/",
    checkpoint_dir: str = "/data/checkpoints",
):
    """Run the full training loop on a Modal H100."""

    import sys
    sys.path.insert(0, "/app")

    from pathlib import Path

    import numpy as np
    import torch
    from torch import optim
    from torch.utils.data import DataLoader

    from cnn_compress import VideoCompressor
    from stream_processing import StreamingVideoDataset
    from loss_func import SemanticCompressionLoss
    from train_helpers import train_model, TrainConfig

    # --- Inline teacher (same as train.py) so we don't import the module-level code ---
    class VideoPrismTeacher(torch.nn.Module):
        def __init__(self, weights_path: str):
            super().__init__()
            self.weights_path = str(weights_path)
            self._weights = np.load(self.weights_path, allow_pickle=False)
            self._device = torch.device("cpu")

        def to(self, device):
            self._device = torch.device(device)
            return self

        def eval(self):
            return self

        def forward(self, video: torch.Tensor):
            pooled = video.mean(dim=(2, 3, 4))
            if "projection" in self._weights.files:
                proj = torch.from_numpy(self._weights["projection"]).to(
                    video.device, dtype=video.dtype
                )
                if proj.ndim == 2 and proj.shape[0] == pooled.shape[-1]:
                    orig = pooled @ proj
                    return orig, orig
            return pooled, pooled

    # --- Validate paths ---
    weights_path = Path(model_weights)
    if not weights_path.is_file():
        raise FileNotFoundError(
            f"VideoPrism weights not found at {model_weights}. "
            "Upload them to the volume: modal volume put piedpiper-data ./models /models"
        )

    data_path = Path(data_dir)
    if not data_path.is_dir():
        raise FileNotFoundError(
            f"Dataset directory not found at {data_dir}. "
            "Upload data to the volume: modal volume put piedpiper-data ./datasets /datasets"
        )

    print(f"=== PiedPiper Training on Modal (H100) ===")
    print(f"  Weights : {model_weights}")
    print(f"  Data    : {data_dir}")
    print(f"  Epochs  : {num_epochs}")
    print(f"  Batch   : {batch_size}")
    print(f"  LR      : {learning_rate}")
    print(f"  Ckpt dir: {checkpoint_dir}")
    print(f"  Device  : cuda ({torch.cuda.get_device_name(0)})")
    print()

    # --- Build components ---
    compressor = VideoCompressor()
    teacher_model = VideoPrismTeacher(str(weights_path))

    train_dataset = StreamingVideoDataset(
        video_dir=data_dir,
        clip_length=16,
        sample_every_n=4,
        resolution=240,
    )
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, num_workers=4, pin_memory=True
    )

    config = TrainConfig(
        loss_fn=SemanticCompressionLoss(),
        optimizer_cls=optim.Adam,
        train_loader=train_dataloader,
        test_loader=None,
        learning_rate=learning_rate,
        batch_size=batch_size,
        num_epochs=num_epochs,
        save_freq=save_freq,
        checkpoint_folder=checkpoint_dir,
    )

    epoch_history = train_model(compressor, teacher_model, config)

    # Persist checkpoints back to the volume
    volume.commit()

    print("\n=== Training complete. Checkpoints saved to volume. ===")
    return epoch_history


# ---------------------------------------------------------------------------
# Data Setup
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    volumes={VOLUME_MOUNT: volume},
    timeout=60 * 60,  # 1 hour to download
)
def setup_data():
    """Downloads dataset from Google Drive and model weights from HuggingFace."""
    import os
    import gdown
    from huggingface_hub import hf_hub_download

    print("=== Setting up data in Modal Volume ===")
    
    # 1. Download Model Weights
    model_dir = "/data/models"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "flax_base_f16r288_repeated.npz")
    
    if not os.path.exists(model_path):
        print("Downloading VideoPrism weights from HuggingFace...")
        hf_hub_download(
            repo_id="google/videoprism-base-f16r288",
            filename="flax_base_f16r288_repeated.npz",
            local_dir=model_dir
        )
    else:
        print("Model weights already exist.")

    # 2. Download Dataset
    dataset_dir = "/data/datasets/eastgate"
    os.makedirs(dataset_dir, exist_ok=True)
    
    # Simple check if dataset is empty
    if not os.listdir(dataset_dir):
        print("Downloading Eastgate dataset from Google Drive...")
        gdown.download_folder(
            id="1cR1VwoAvEjFLRaUzeYph-bxx4LoM6pOH",
            output=dataset_dir,
            quiet=False,
            use_cookies=False
        )
    else:
        print("Dataset already appears to be downloaded.")
        
    volume.commit()
    print("=== Data setup complete ===")

# ---------------------------------------------------------------------------
# Local entrypoint  (modal run modal_train.py)
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(setup: bool = False):
    if setup:
        setup_data.remote()
    
    result = train.remote()
    print("Epoch history:", result)
