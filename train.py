from pathlib import Path

import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader

from cnn_compress import VideoCompressor
from stream_processing import StreamingVideoDataset
from loss_func import SemanticCompressionLoss
from train_helpers import train_model, TrainConfig


class VideoPrismTeacher(torch.nn.Module):
    """Lightweight NPZ-backed teacher wrapper for local VideoPrism embeddings."""

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
        # Placeholder deterministic embedding projection until full VideoPrism runtime is wired.
        pooled = video.mean(dim=(2, 3, 4))
        if "projection" in self._weights.files:
            proj = torch.from_numpy(self._weights["projection"]).to(video.device, dtype=video.dtype)
            if proj.ndim == 2 and proj.shape[0] == pooled.shape[-1]:
                orig = pooled @ proj
                masked = orig
                return orig, masked
        orig = pooled
        masked = pooled
        return orig, masked


compressor = VideoCompressor()
model_path = Path("/workspace/models/videoprism_public_v1_base.npz")
if not model_path.is_file():
    raise FileNotFoundError(
        f"VideoPrism weights not found at: {model_path}. "
        "Expected file: /workspace/models/videoprism_public_v1_base.npz"
    )
teacher_model = VideoPrismTeacher(str(model_path))
train_dataset = StreamingVideoDataset(
    video_dir="/workspace/datasets/eastgate/",
    clip_length=16,
    sample_every_n=4,
    resolution=240,
)

optimizer_cls = optim.Adam
loss_fn = SemanticCompressionLoss()
train_dataloader = DataLoader(train_dataset, batch_size=4, num_workers=4, pin_memory=True)
test_loader = None
learning_rate = 0.001
batch_size = 4
num_epochs = 10
save_freq = 2
checkpoint_folder = "./model_checkpoints"

train_config = TrainConfig(
    loss_fn=loss_fn,
    optimizer_cls=optimizer_cls,
    train_loader=train_dataloader,
    test_loader=test_loader,
    learning_rate=learning_rate,
    batch_size=batch_size,
    num_epochs=num_epochs,
    save_freq=save_freq,
    checkpoint_folder=checkpoint_folder,
)

epoch_history = train_model(compressor, teacher_model, train_config)
