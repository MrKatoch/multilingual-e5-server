from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ValidationError
import httpx
import time
import os
import socket
import subprocess
import threading
import json
import logging
import traceback

# ---------------------------------------------------------------------------
# Logging setup - writes to a file so failures are visible on client machines
# even when the console window is hidden/suppressed.
# ---------------------------------------------------------------------------
import sys

if getattr(sys, "frozen", False):
    # Running as a PyInstaller-built exe: __file__ points into the temp
    # extraction folder (_MEIxxxxx), which is deleted on exit. Use the
    # actual exe's real location instead, so the log survives.
    _APP_DIR = os.path.dirname(os.path.abspath(sys.executable))
else:
    _APP_DIR = os.path.dirname(os.path.abspath(__file__))

LOG_FILE = os.path.join(_APP_DIR, "embedding_api.log")

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
# Also echo to console if one is attached (harmless if not).
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
logging.getLogger().addHandler(console_handler)

logging.info("=== Embedding API starting up ===")

app = FastAPI(title="Ollama Embedding API")

OLLAMA_URL = "http://localhost:11434/api/embeddings"
MODEL_NAME = "qllama/multilingual-e5-base"
OLLAMA_EXE = os.path.expandvars(r"%LOCALAPPDATA%\Programs\Ollama\ollama.exe")
_restart_lock = threading.Lock()


def _port_open(host, port, timeout=0.5):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(timeout)
        return s.connect_ex((host, port)) == 0


def _ensure_ollama_running():
    with _restart_lock:
        if _port_open("localhost", 11434):
            return

        if not os.path.isfile(OLLAMA_EXE):
            raise RuntimeError(f"Ollama executable not found at: {OLLAMA_EXE}")

        logging.warning("Ollama not running. Attempting to start: %s", OLLAMA_EXE)
        try:
            subprocess.Popen(
                [OLLAMA_EXE, "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
        except OSError as e:
            raise RuntimeError(f"Failed to launch Ollama process: {e}")

        for _ in range(20):
            if _port_open("localhost", 11434):
                logging.info("Ollama came back up successfully.")
                return
            time.sleep(0.5)

        raise RuntimeError("Ollama failed to start within the timeout window (10s).")


def _model_exists(model_name: str) -> bool:
    """Check whether the model is already pulled into this Ollama instance."""
    try:
        resp = httpx.get("http://localhost:11434/api/tags", timeout=10.0)
        resp.raise_for_status()
        tags = resp.json().get("models", [])
        # Ollama tags usually look like "name:latest" - match on the base name too.
        names = {t.get("name", "") for t in tags}
        base_names = {n.split(":")[0] for n in names}
        return model_name in names or model_name in base_names
    except Exception as e:
        logging.warning("Could not check installed models: %s", e)
        return False


def _pull_model(model_name: str, timeout_seconds: float = 1800.0) -> None:
    """
    Pull a model into Ollama, showing live progress in both the console
    (for the person watching the exe window on first run) and the log file.
    Blocks until the pull finishes, fails, or times out (default 30 min,
    since large embedding models can take a while on first run).
    """
    banner = f"Downloading embedding model '{model_name}' (first run only)..."
    print(banner, flush=True)
    logging.info(banner)

    try:
        with httpx.stream(
            "POST",
            "http://localhost:11434/api/pull",
            json={"model": model_name, "stream": True},
            timeout=timeout_seconds,
        ) as resp:
            resp.raise_for_status()
            last_status = None
            last_pct_printed = -1
            for line in resp.iter_lines():
                if not line:
                    continue
                try:
                    evt = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if evt.get("error"):
                    raise RuntimeError(f"Ollama pull error: {evt['error']}")

                status = evt.get("status", "")
                total = evt.get("total")
                completed = evt.get("completed")

                if total and completed is not None and total > 0:
                    pct = int((completed / total) * 100)
                    # Print at most every 5% to avoid spamming the console.
                    if pct != last_pct_printed and pct % 5 == 0:
                        mb_done = completed / (1024 * 1024)
                        mb_total = total / (1024 * 1024)
                        line_out = f"  {status}: {pct}%  ({mb_done:.0f} MB / {mb_total:.0f} MB)"
                        print(line_out, flush=True)
                        logging.info(line_out.strip())
                        last_pct_printed = pct
                elif status and status != last_status:
                    print(f"  {status}", flush=True)
                    logging.info("Pull status for '%s': %s", model_name, status)
                    last_status = status

    except httpx.HTTPStatusError as e:
        raise RuntimeError(f"Failed to pull model '{model_name}': HTTP {e.response.status_code}")
    except httpx.RequestError as e:
        raise RuntimeError(f"Failed to pull model '{model_name}': {e}")

    if not _model_exists(model_name):
        raise RuntimeError(f"Pull for '{model_name}' finished but model still not listed by Ollama.")

    done_msg = f"Model '{model_name}' downloaded successfully. Ready to serve requests."
    print(done_msg, flush=True)
    logging.info(done_msg)


def _ensure_model_available(model_name: str) -> None:
    """Make sure Ollama is running and the required model is pulled, pulling it if missing."""
    _ensure_ollama_running()
    if not _model_exists(model_name):
        _pull_model(model_name)


class EmbedRequest(BaseModel):
    text: str


@app.on_event("startup")
def _startup_ensure_model():
    """
    On server startup: make sure Ollama is running and the embedding model
    is pulled. This means a client never has to run any manual setup step -
    the first launch on a fresh machine pulls the model automatically
    (with visible progress), and every launch after that is instant because
    the model is already cached by Ollama.
    """
    print("=" * 60, flush=True)
    print("  Multilingual Embedding Server", flush=True)
    print("=" * 60, flush=True)
    print("Checking embedding model status...", flush=True)

    try:
        _ensure_ollama_running()
        if _model_exists(MODEL_NAME):
            print(f"Model '{MODEL_NAME}' already installed. Starting instantly.", flush=True)
            logging.info("Model '%s' already installed - no download needed.", MODEL_NAME)
        else:
            _pull_model(MODEL_NAME)
    except Exception as e:
        # Don't crash the server if this fails at startup (e.g. no internet yet) -
        # call_ollama() below will retry the pull on the first real request instead.
        print(f"Warning: could not prepare model at startup ({e}). Will retry on first request.", flush=True)
        logging.error("Startup model check/pull failed (will retry on first request): %s", e)

    print("=" * 60, flush=True)
    print(f"Server ready at http://127.0.0.1:{os.environ.get('EMBED_API_PORT', 8000)}", flush=True)
    print("Keep this window open while using the server.", flush=True)
    print("=" * 60, flush=True)


@app.get("/")
def home():
    return {"message": "Ollama Embedding API is running", "model": MODEL_NAME}


@app.get("/embed/{text}")
def embed_via_get(text: str):
    return call_ollama(text)


@app.post("/embed")
def embed_via_post(req: EmbedRequest):
    return call_ollama(req.text)


@app.get("/api/embed/{text}")
def embed_via_get_api(text: str):
    return call_ollama(text)


@app.post("/api/embed")
async def embed_via_post_api(request: Request):
    raw_body = await request.body()
    logging.info("RAW BODY RECEIVED FROM CLIENT: %r", raw_body)
    logging.info("Headers: %s", dict(request.headers))

    try:
        parsed = json.loads(raw_body)
    except json.JSONDecodeError as e:
        logging.error("Body is not valid JSON: %s. Raw body: %r", e, raw_body)
        raise HTTPException(
            status_code=422,
            detail=f"Body is not valid JSON: {str(e)}. Raw body was: {raw_body!r}",
        )

    if "text" not in parsed and "input" in parsed:
        parsed["text"] = parsed["input"]

    try:
        req = EmbedRequest(**parsed)
    except ValidationError as e:
        logging.error("Validation failed. Parsed: %s. Errors: %s", parsed, e.errors())
        raise HTTPException(
            status_code=422,
            detail=(
                f"JSON parsed but didn't match expected shape {{'text': str}}. "
                f"You sent: {parsed}. Validation errors: {e.errors()}"
            ),
        )

    return call_ollama(req.text)


def call_ollama(text: str, _retry: bool = True):
    start_time = time.time()
    try:
        response = httpx.post(
            OLLAMA_URL,
            json={
                "model": MODEL_NAME,
                "prompt": f"query: {text}",
            },
            timeout=120.0,
        )
        response.raise_for_status()

    except httpx.ConnectError:
        logging.warning("Ollama ConnectError (retry=%s).", _retry)
        if _retry:
            try:
                _ensure_ollama_running()
            except RuntimeError as e:
                logging.error("Ollama restart attempt failed: %s", e)
                raise HTTPException(status_code=503, detail=f"Ollama unavailable: {e}")
            return call_ollama(text, _retry=False)
        logging.error("Ollama still unreachable after restart attempt.")
        raise HTTPException(
            status_code=503,
            detail="Cannot connect to Ollama. Make sure Ollama is running.",
        )

    except httpx.TimeoutException:
        logging.error("Ollama timed out embedding %d characters.", len(text))
        raise HTTPException(
            status_code=504,
            detail=f"Ollama took too long to respond (>120s) embedding {len(text)} characters of text.",
        )

    except httpx.HTTPStatusError as e:
        try:
            ollama_error = e.response.json().get("error", "")
        except Exception:
            ollama_error = e.response.text
        logging.error("Ollama returned HTTP error: %s", ollama_error)

        if "context length" in ollama_error.lower():
            raise HTTPException(
                status_code=413,
                detail=(
                    f"Text too long for the embedding model's context window "
                    f"({len(text)} characters). Please shorten the input and try again."
                ),
            )

        if "not found" in ollama_error.lower() and _retry:
            logging.warning("Model missing mid-request. Attempting automatic pull.")
            try:
                _pull_model(MODEL_NAME)
            except RuntimeError as pull_err:
                logging.error("Auto-pull failed: %s", pull_err)
                raise HTTPException(
                    status_code=503,
                    detail=f"Embedding model not available and automatic download failed: {pull_err}",
                )
            return call_ollama(text, _retry=False)

        raise HTTPException(status_code=500, detail=ollama_error or str(e))

    except Exception as e:
        logging.error("Unhandled exception in call_ollama: %s\n%s", e, traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Unexpected server error: {e}")

    try:
        data = response.json()
        embedding = data["embedding"]
    except (json.JSONDecodeError, KeyError) as e:
        logging.error("Malformed response from Ollama: %s. Body: %s", e, response.text)
        raise HTTPException(
            status_code=502,
            detail=f"Ollama returned an unexpected response shape: {e}",
        )

    end_time = time.time()
    logging.info(
        "Embedded %d chars in %.3fs, dims=%d",
        len(text), end_time - start_time, len(embedding),
    )

    return {
        "input_text": text,
        "model": MODEL_NAME,
        "dimensions": len(embedding),
        "time_taken_seconds": round(end_time - start_time, 3),
        "embedding": embedding,
        # Extra keys below are non-breaking additions so we don't have to
        # guess/change the client's exact expected JSON shape - they mirror
        # Ollama's native /api/embed response ("embeddings": [...]) and
        # OpenAI's embeddings API response ("data": [{"embedding": ...}]).
        "embeddings": [embedding],
        "data": [{"embedding": embedding, "index": 0}],
    }


# ---------------------------------------------------------------------------
# Global safety net: guarantees every unhandled exception is logged and
# returns a structured JSON body instead of a bare, unexplained 500.
# ---------------------------------------------------------------------------
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logging.error(
        "UNHANDLED EXCEPTION on %s %s: %s\n%s",
        request.method, request.url, exc, traceback.format_exc(),
    )
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {exc}"},
    )


if __name__ == "__main__":
    import uvicorn
    # FastAPI's startup event (model check/download) runs automatically
    # before uvicorn starts accepting connections - so by the time
    # "Server ready" prints, the model is guaranteed to be usable.
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")