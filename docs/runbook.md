# Pied Piper Runbook

This repo has two product surfaces:

- `app/inference`: the Modal-hosted FastAPI service
- `app/packaging`: the pip-installable Python SDK

Current package names:

- PyPI distribution: `piedpiper-sdk`
- Python import: `pied_piper`

Current backend auth contract:

- Modal Secret name: `pied-piper-backend`
- Required secret key: `PIED_PIPER_API_KEY`
- SDK/frontend base URL env var: `PIED_PIPER_BASE_URL`
- Compression endpoint: `POST /v1/compress`

## 1. Prerequisites

Install the basic tooling:

```bash
python3 -m pip install --upgrade modal build twine
```

Authenticate Modal once on your machine:

```bash
modal setup
```

For PyPI publishing, create a PyPI account and an API token. If you want a dry run first, also create a TestPyPI token.

## 2. Local SDK install

From the repo root:

```bash
python3 -m pip install -e app/packaging
```

Quick smoke test:

```bash
python3 - <<'PY'
import pied_piper

print(pied_piper.__all__)
PY
```

## 3. Modal local development

Create the backend secret once per Modal environment:

```bash
modal secret create pied-piper-backend PIED_PIPER_API_KEY=replace-me
```

Start the local dev server on Modal:

```bash
modal serve app/inference/modal_app.py
```

What to do next:

1. Copy the ephemeral HTTPS URL printed by Modal.
2. Export it as the SDK base URL.
3. Use the same bearer token value locally when testing clients.

Example shell setup:

```bash
export PIED_PIPER_BASE_URL="https://<your-dev-url>.modal.run"
export PIED_PIPER_API_KEY="replace-me"
```

Health check:

```bash
curl "$PIED_PIPER_BASE_URL/health"
```

You should see `auth_configured: true` and `text_backend: "llmlingua"` when the image is healthy.

## 4. Modal production deployment

Deploy the ASGI app:

```bash
modal deploy app/inference/modal_app.py
```

After deploy:

1. Copy the deployed web endpoint URL from the CLI output or Modal dashboard.
2. Set that URL wherever the SDK or Python frontend is configured.
3. Keep `PIED_PIPER_API_KEY` synchronized between the Modal Secret and any calling app.

Recommended verification:

```bash
curl "$PIED_PIPER_BASE_URL/health"
```

If you need separate dev and prod stacks, use separate Modal environments and create `pied-piper-backend` in each environment with the correct `PIED_PIPER_API_KEY` value.

## 5. Python frontend / caller configuration

Any Python caller needs these env vars:

```bash
export PIED_PIPER_BASE_URL="https://<your-modal-url>.modal.run"
export PIED_PIPER_API_KEY="replace-me"
```

Minimal caller example:

```python
import pied_piper

result = pied_piper.compress("Long input text")
print(result.status)
print(result.text)
```

## 6. Release the SDK to PyPI

### 6.1 Bump the version

Update the version in [app/packaging/pyproject.toml](/Users/an/Documents/piedpiper/app/packaging/pyproject.toml).

### 6.2 Build the package

Run the build from the packaging directory:

```bash
cd app/packaging
python3 -m build
```

This creates:

- `dist/*.tar.gz`
- `dist/*.whl`

Optional validation:

```bash
python3 -m twine check dist/*
```

### 6.3 Upload to TestPyPI first

Recommended dry run:

```bash
python3 -m twine upload --repository testpypi dist/*
```

Test install from TestPyPI:

```bash
python3 -m pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple piedpiper-sdk
```

### 6.4 Upload to PyPI

When the TestPyPI install looks right:

```bash
python3 -m twine upload dist/*
```

### 6.5 Verify the published package

Install it from PyPI in a clean environment:

```bash
python3 -m pip install piedpiper-sdk
```

Then confirm the import path:

```bash
python3 - <<'PY'
import pied_piper

print(pied_piper.__all__)
PY
```

## 7. Release checklist

- Bump `version` in `app/packaging/pyproject.toml`.
- Confirm `app/packaging/README.md` still matches current behavior.
- Build with `python3 -m build`.
- Run `python3 -m twine check dist/*`.
- Upload to TestPyPI.
- Install from TestPyPI in a clean env and verify `import pied_piper`.
- Upload to PyPI.
- Update any downstream app or frontend to the new version if you are pinning dependencies.

## 8. Notes

- Plain strings are treated as inline text, not file paths.
- Use `pathlib.Path(...)` for file uploads through the SDK.
- The Modal service currently exposes `GET /`, `GET /health`, and `POST /v1/compress`.
- The deployed service is configured in [app/inference/modal_app.py](/Users/an/Documents/piedpiper/app/inference/modal_app.py).

Useful references:

- Modal apps and deploys: [Apps, Functions, and entrypoints](https://modal.com/docs/guide/apps)
- Modal ASGI endpoints: [modal.asgi_app](https://modal.com/docs/reference/modal.asgi_app)
- Modal secrets: [Secrets](https://modal.com/docs/guide/secrets)
