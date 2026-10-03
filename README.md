# Multilingual E5 Embedding Server

A local REST API that turns text in any language into a 768-dimension embedding vector, using the [`multilingual-e5-base`](https://ollama.com/qllama/multilingual-e5-base) model running in [Ollama](https://ollama.com). Use the vectors for semantic search, RAG, similarity matching or clustering. Everything runs on your own machine with no cloud API.

## Quick start (Windows, no Python needed)

1. Download `MultilingualE5Server.exe` from the [Releases](../../releases) page.
2. Double-click it and keep the window open.

On first run it will:

1. Install Ollama from ollama.com if it isn't already installed (per-user, no admin rights needed).
2. Download the embedding model (about 560 MB).
3. Set up PowerShell so typing a URL at the prompt fetches it.
4. Start the server on port **8007**.

Later runs start in a few seconds.

> Windows SmartScreen may warn about an unsigned app. Click **More info → Run anyway**.

## Usage

Open this in a browser:

```
http://127.0.0.1:8007/embed/hello
```

Or in PowerShell:

```powershell
# After the exe has set up PowerShell, you can type the URL directly:
http://127.0.0.1:8007/embed/hello

# This works on any Windows PC with no setup:
Invoke-RestMethod http://127.0.0.1:8007/embed/hello
```

Example response (the embedding is shortened here):

```json
{
  "input_text": "hello",
  "model": "qllama/multilingual-e5-base",
  "dimensions": 768,
  "time_taken_seconds": 0.63,
  "embedding": [0.0963, 0.7772, -0.0725, "..."]
}
```

The response also includes `embeddings` (in Ollama's format) and `data` (in OpenAI's format), so existing clients for either API can read it.

### Endpoints

| Method | Path | Body |
|---|---|---|
| GET | `/` | Health check |
| GET | `/embed/{text}` | – |
| POST | `/embed` | `{"text": "..."}` |
| GET | `/api/embed/{text}` | – |
| POST | `/api/embed` | `{"text": "..."}` or `{"input": "..."}` |
| GET | `/docs` | Interactive Swagger UI |

### Access from other devices

The server listens on all network interfaces, so other devices on the same network can use `http://<this-pc-ip>:8007/...`. Windows Firewall blocks this by default. To allow it, run this in an Administrator PowerShell:

```powershell
New-NetFirewallRule -DisplayName "Multilingual E5 API (port 8007)" -Direction Inbound -Protocol TCP -LocalPort 8007 -Action Allow -Profile Any
```

## Running from source

Requires Python 3.10+ and [Ollama](https://ollama.com/download).

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python main.py                      # serves on 127.0.0.1:8000
# or
python client_launcher.py           # full launcher: Ollama check/install, 0.0.0.0:8007
```

Set the `EMBED_API_PORT` environment variable to change the launcher's port. Pass `--no-ps-setup` to skip the PowerShell changes.

## Building the exe

```bash
pip install pyinstaller
pyinstaller --noconfirm MultilingualE5Server.spec
# -> dist/MultilingualE5Server.exe  (~19 MB)
```

## Project files

| File | Purpose |
|---|---|
| `main.py` | FastAPI app: endpoints, Ollama auto-start, model auto-download |
| `client_launcher.py` | Exe entry point: PowerShell setup, Ollama find/install, starts `main.py` on port 8007 |
| `MultilingualE5Server.spec` | PyInstaller build for the small (~19 MB) launcher exe |
| `bundle_launcher.py`, `MultilingualEmbeddingServer.spec` | Older fully offline build that bundles Ollama and the model inside the exe (~340 MB; needs a local `bundle_staging/` folder that isn't in git) |
| `main.spec` | PyInstaller build of `main.py` alone (needs Ollama already installed) |
