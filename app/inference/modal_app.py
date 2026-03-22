from __future__ import annotations

import modal


app = modal.App("pied-piper-inference")
model_cache = modal.Volume.from_name("pied-piper-model-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("ffmpeg", "libgl1-mesa-glx", "libglib2.0-0")
    .pip_install(
        "fastapi>=0.115,<1.0",
        "pydantic>=2.9,<3.0",
        "python-multipart>=0.0.9,<1.0",
        "llmlingua>=0.2.2,<1.0",
        "transnetv2-pytorch>=1.0.5,<2.0",
        "torch>=2.8,<3.0",
        "transformers>=4.57,<5.0",
        "ffmpeg-python>=0.2.0,<1.0",
        "opencv-python-headless>=4.10,<5.0",
        "numpy>=1.26,<3.0",
        "pillow>=10.0,<12.0",
        "python-docx>=1.1,<2.0",
        "python-pptx>=1.0,<2.0",
        "pypdf>=5.0,<6.0",
    )
    .env(
        {
            "HF_HOME": "/models/hf",
            "PIED_PIPER_PRELOAD_MODELS": "1",
        }
    )
)


@app.function(
    image=image,
    gpu="T4",
    cpu=4.0,
    memory=16384,
    secrets=[modal.Secret.from_name("pied-piper-backend")],
    volumes={"/models": model_cache},
    min_containers=0,
)
@modal.asgi_app()
def fastapi_app():
    from .api import create_app

    return create_app()
