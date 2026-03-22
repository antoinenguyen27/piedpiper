# Pied Piper Runbook

This repo has two independent product surfaces:

- `app/inference`: the Modal-hosted FastAPI service
- `app/packaging`: the pip-installable Python SDK

Treat them as separate operational tracks:

- Modal work changes or deploys the backend service.
- PyPI work builds and publishes the SDK package.
- One does not automatically release the other.

Current package names:

- PyPI distribution: `piedpiper-sdk`
- Python import: `pied_piper`

Current backend auth contract:

- Modal Secret name: `pied-piper-backend`
- Required secret key: `PIED_PIPER_API_KEY`
- SDK/frontend base URL override env var: `PIED_PIPER_BASE_URL`
- Compression endpoint: `POST /v1/compress`

## 1. Modal Service Operations

Use this section when you are changing, testing, or deploying the backend in `app/inference`.

### 1.1 Modal prerequisites

Install only the backend tooling you need:

```bash
python3 -m pip install --upgrade modal
```

Modal imports the backend module on your local machine before it builds the remote image. Run Modal commands from the repo root and use a Python environment that can import `app.inference.modal_app` and its top-level dependencies.

Authenticate Modal once on your machine:

```bash
modal setup
```

If your workspace uses multiple Modal environments, choose one explicitly with `-e/--env` on CLI commands or by setting `MODAL_ENVIRONMENT`.

### 1.2 Create or update the backend secret

Create the backend secret once per Modal environment:

```bash
modal secret create pied-piper-backend PIED_PIPER_API_KEY=replace-me
```

If you need to target a non-default environment:

```bash
modal secret create -e dev pied-piper-backend PIED_PIPER_API_KEY=replace-me
```

If the secret already exists and you need to rotate the value, re-run the command with `--force`.

### 1.3 Local development with `modal serve`

Start the live-reloading dev app:

```bash
modal serve -m app.inference.modal_app
```

What to do next:

1. Copy the HTTPS URL printed by Modal for the served app.
2. Export it as `PIED_PIPER_BASE_URL` if you want your local caller to hit the dev app instead of production.
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

### 1.4 Production deployment with `modal deploy`

Deploy the ASGI app:

```bash
modal deploy -m app.inference.modal_app
```

After deploy:

1. Copy the deployed web endpoint URL from the CLI output or Modal dashboard.
2. If you want callers to hit this deployment instead of the baked-in production service, set it as `PIED_PIPER_BASE_URL`.
3. Keep `PIED_PIPER_API_KEY` synchronized between the Modal Secret and every calling app.

Recommended verification:

```bash
export PIED_PIPER_BASE_URL="https://<your-modal-url>.modal.run"
curl "$PIED_PIPER_BASE_URL/health"
```

If you need separate dev and prod stacks, use separate Modal environments and create `pied-piper-backend` in each environment with the correct `PIED_PIPER_API_KEY` value.

### 1.5 Python caller configuration

Any Python caller needs the API key. The base URL env var is only needed to override the baked-in production endpoint:

```bash
export PIED_PIPER_API_KEY="replace-me"
```

Optional override for local development or a non-production deployment:

```bash
export PIED_PIPER_BASE_URL="https://<your-modal-url>.modal.run"
```

Minimal caller example:

```python
import pied_piper

result = pied_piper.compress("Long input text")
print(result.status)
print(result.text)
```

## 2. Python SDK Packaging And PyPI Release

Use this section when you are changing, building, validating, or publishing `app/packaging`.

Publishing the SDK does not deploy the Modal backend. Deploying the Modal backend does not publish a new SDK.

### 2.1 SDK prerequisites

For local SDK development only:

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

For package builds and manual uploads:

```bash
python3 -m pip install --upgrade build twine
```

### 2.2 Preferred publishing approach: Trusted Publishing

Current PyPI guidance prefers Trusted Publishing for automated releases because it avoids long-lived API tokens in CI.

If you automate releases for this repo, configure Trusted Publishing on PyPI and TestPyPI for the release workflow, then publish from CI. This repo does not currently include a publishing workflow, so the manual Twine flow below remains the operational fallback.

### 2.3 Manual release flow with Twine

Use this only when you are publishing manually from a trusted local machine.

Create a PyPI account and a project-scoped API token. If you want a dry run first, also create a TestPyPI token.

For Twine authentication, use `__token__` as the username and the API token as the password. Prefer keyring or environment variables over storing tokens in plain-text config files.

### 2.4 Bump the version

Update the version in [app/packaging/pyproject.toml](/Users/an/Documents/piedpiper/app/packaging/pyproject.toml).

### 2.5 Build the package

Run the build from the packaging directory:

```bash
cd app/packaging
python3 -m build
```

This creates:

- `dist/*.tar.gz`
- `dist/*.whl`

Validate the built metadata before uploading:

```bash
python3 -m twine check --strict dist/*
```

### 2.6 Upload to TestPyPI first

Recommended dry run:

```bash
python3 -m twine upload --repository testpypi dist/*
```

Test install from TestPyPI:

```bash
python3 -m pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple piedpiper-sdk
```

### 2.7 Upload to PyPI

When the TestPyPI install looks right:

```bash
python3 -m twine upload dist/*
```

### 2.8 Verify the published package

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

### 2.9 SDK release checklist

- Bump `version` in `app/packaging/pyproject.toml`.
- Confirm `app/packaging/README.md` still matches current behavior.
- Build with `python3 -m build`.
- Run `python3 -m twine check --strict dist/*`.
- Upload to TestPyPI.
- Install from TestPyPI in a clean env and verify `import pied_piper`.
- Upload to PyPI.
- Update any downstream app that pins `piedpiper-sdk`.

## 3. Notes

- Plain strings are treated as inline text, not file paths.
- Use `pathlib.Path(...)` for file uploads through the SDK.
- The Modal service currently exposes `GET /`, `GET /health`, and `POST /v1/compress`.
- The deployed service is configured in [app/inference/modal_app.py](/Users/an/Documents/piedpiper/app/inference/modal_app.py).

## 4. References

Modal:

- Apps, Functions, and entrypoints: [https://modal.com/docs/guide/apps](https://modal.com/docs/guide/apps)
- Developing and debugging with `modal serve`: [https://modal.com/docs/guide/developing-debugging](https://modal.com/docs/guide/developing-debugging)
- `modal.asgi_app`: [https://modal.com/docs/reference/modal.asgi_app](https://modal.com/docs/reference/modal.asgi_app)
- `modal secret` CLI: [https://modal.com/docs/reference/cli/secret](https://modal.com/docs/reference/cli/secret)

PyPI and packaging:

- Packaging and distributing projects: [https://packaging.python.org/en/latest/guides/distributing-packages-using-setuptools/](https://packaging.python.org/en/latest/guides/distributing-packages-using-setuptools/)
- Using TestPyPI: [https://packaging.python.org/en/latest/guides/using-testpypi/](https://packaging.python.org/en/latest/guides/using-testpypi/)
- Trusted Publishers: [https://docs.pypi.org/trusted-publishers/](https://docs.pypi.org/trusted-publishers/)
- Adding a Trusted Publisher to an existing PyPI project: [https://docs.pypi.org/trusted-publishers/adding-a-publisher/](https://docs.pypi.org/trusted-publishers/adding-a-publisher/)
