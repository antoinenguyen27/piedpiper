from __future__ import annotations

import modal

from .api import create_app


app = modal.App("pied-piper-inference")

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "fastapi>=0.115,<1.0",
        "pydantic>=2.9,<3.0",
        "python-multipart>=0.0.9,<1.0",
        "llmlingua>=0.2.2,<1.0",
        "python-docx>=1.1,<2.0",
        "python-pptx>=1.0,<2.0",
        "pypdf>=5.0,<6.0",
    )
)


@app.function(
    image=image,
    gpu="T4",
    cpu=4.0,
    memory=8192,
    secrets=[modal.Secret.from_name("pied-piper-backend")],
    min_containers=0,
)
@modal.asgi_app()
def fastapi_app():
    return create_app()
