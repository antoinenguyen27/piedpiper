"""
Modal entrypoint for running inference with a trained VideoCompressor checkpoint.

Usage:
    modal run modal_inference.py --checkpoint /data/checkpoints/checkpoint_model_VideoCompressor_epoch_10.pth
    modal run modal_inference.py --checkpoint /data/checkpoints/checkpoint_model_VideoCompressor_epoch_10.pth --video-path /data/datasets/eastgate/sample.mp4
"""

import modal

# ---------------------------------------------------------------------------
# Modal configuration (shared with training)
# ---------------------------------------------------------------------------

app = modal.App("piedpiper-inference")

volume = modal.Volume.from_name("piedpiper-data", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libgl1-mesa-glx", "libglib2.0-0")
    .pip_install(
        "torch==2.4.0",
        "torchvision==0.19.0",
        "torchaudio==2.4.0",
        "decord>=0.6.0",
        "opencv-python-headless",
        "av",
        "numpy<2.0.0",
        "tqdm",
    )
    .copy_local_dir(".", "/app")
)

VOLUME_MOUNT = "/data"


# ---------------------------------------------------------------------------
# Inference function
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    gpu=modal.gpu.H100(count=1),
    volumes={VOLUME_MOUNT: volume},
    timeout=60 * 60,  # 1-hour max
)
def infer(
    checkpoint: str,
    video_path: str = "/data/datasets/eastgate/",
    clip_length: int = 16,
    sample_every_n: int = 4,
    resolution: int = 240,
):
    """Load a trained checkpoint and run the compressor on video data."""

    import sys
    sys.path.insert(0, "/app")

    from pathlib import Path

    import torch
    from torch.utils.data import DataLoader

    from cnn_compress import VideoCompressor
    from stream_processing import StreamingVideoDataset

    ckpt_path = Path(checkpoint)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    device = torch.device("cuda")
    print(f"=== PiedPiper Inference on Modal (H100) ===")
    print(f"  Checkpoint: {checkpoint}")
    print(f"  Video path: {video_path}")
    print(f"  Device    : {torch.cuda.get_device_name(0)}")
    print()

    # --- Load model ---
    model = VideoCompressor()
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    # --- Build dataset ---
    dataset = StreamingVideoDataset(
        video_dir=video_path,
        clip_length=clip_length,
        sample_every_n=sample_every_n,
        resolution=resolution,
    )
    loader = DataLoader(dataset, batch_size=1, num_workers=2)

    # --- Run inference ---
    results = []
    with torch.no_grad():
        for batch_idx, video in enumerate(loader):
            video = video.to(device, non_blocking=True)
            mask = model(video)
            keep_ratio = mask.mean().item()
            results.append({
                "batch": batch_idx,
                "mask_shape": list(mask.shape),
                "keep_ratio": round(keep_ratio, 4),
            })
            print(
                f"  Batch {batch_idx}: mask {tuple(mask.shape)}, "
                f"keep_ratio={keep_ratio:.4f}"
            )

    print(f"\n=== Processed {len(results)} clip(s). ===")
    return results


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(
    checkpoint: str = "/data/checkpoints/checkpoint_model_VideoCompressor_epoch_10.pth",
    video_path: str = "/data/datasets/eastgate/",
):
    result = infer.remote(checkpoint=checkpoint, video_path=video_path)
    for r in result:
        print(r)
