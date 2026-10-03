import os
import sys
import time
import subprocess
import socket
import atexit
import threading
import json
import logging
import traceback

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ValidationError


def resource_path(relative):
    base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, relative)


def app_dir():
    # sys.executable is the exe's own location when frozen; _MEIPASS is a
    # temp extraction dir that's wiped on exit, so logs must not go there.
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# Logging setup - writes to a file so failures are visible on client machines
# even when the console window is hidden/suppressed.
# ---------------------------------------------------------------------------
LOG_FILE = os.path.join(app_dir(), "embedding_api.log")

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
logging.getLogger().addHandler(console_handler)

logging.info("=== Embedding API starting up ===")


def port_open(host, port, timeout=0.5):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(timeout)
        return s.connect_ex((host, port)) == 0


OLLAMA_PORT = int(os.environ.get("BUNDLE_OLLAMA_PORT", 11434))
API_PORT = int(os.environ.get("BUNDLE_API_PORT", 8000))
OLLAMA_URL = f"http://127.0.0.1:{OLLAMA_PORT}/api/embeddings"
MODEL_NAME = "multilingual-e5"

ollama_proc = None


def start_bundled_ollama():
    global ollama_proc
    if port_open("127.0.0.1", OLLAMA_PORT):
        print(f"Ollama already running on port {OLLAMA_PORT}, using it.")
        return

    ollama_exe = resource_path(os.path.join("ollama_runtime", "ollama.exe"))
    models_dir = resource_path("ollama_models")

    env = os.environ.copy()
    env["OLLAMA_MODELS"] = models_dir
    env["OLLAMA_HOST"] = f"127.0.0.1:{OLLAMA_PORT}"

    print("Starting bundled embedding engine...")
    ollama_proc = subprocess.Popen(
        [ollama_exe, "serve"],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )

    for _ in range(60):
        if port_open("127.0.0.1", OLLAMA_PORT):
            print("Embedding engine ready.")
            return
        time.sleep(0.5)

    raise RuntimeError("Bundled embedding engine failed to start in time.")


def stop_bundled_ollama():
    if ollama_proc is not None and ollama_proc.poll() is None:
        ollama_proc.terminate()
        try:
            ollama_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            ollama_proc.kill()


atexit.register(stop_bundled_ollama)

_restart_lock = threading.Lock()


def _ensure_ollama_running():
    # start_bundled_ollama() already no-ops if the port is up, so concurrent
    # requests hitting a dead Ollama just queue on the lock and share one restart.
    with _restart_lock:
        start_bundled_ollama()

app = FastAPI(title="Ollama Embedding API")


class EmbedRequest(BaseModel):
    text: str


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
            detail=f"Body is not valid JSON: {str(e)}. Raw body was: {raw_body!r}"
        )

    if "text" not in parsed and "input" in parsed:
        parsed["text"] = parsed["input"]

    try:
        req = EmbedRequest(**parsed)
    except ValidationError as e:
        logging.error("Validation failed. Parsed: %s. Errors: %s", parsed, e.errors())
        raise HTTPException(
            status_code=422,
            detail=f"JSON parsed but didn't match expected shape {{'text': str}}. "
                   f"You sent: {parsed}. Validation errors: {e.errors()}"
        )

    return call_ollama(req.text)


def call_ollama(text: str, _retry: bool = True):
    start_time = time.time()
    try:
        response = httpx.post(
            OLLAMA_URL,
            json={"model": MODEL_NAME, "prompt": f"query: {text}"},
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
            detail="Cannot connect to the embedding engine.",
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
                detail=f"Text too long for the embedding model's context window "
                       f"({len(text)} characters). Please shorten the input and try again.",
            )
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
    print("=" * 60)
    print("  Multilingual Embedding Server")
    print("=" * 60)
    start_bundled_ollama()
    print(f"\nServer starting at http://127.0.0.1:{API_PORT}")
    print("Keep this window open while using the server.")
    print("Close this window to stop the server.\n")
    uvicorn.run(app, host="0.0.0.0", port=API_PORT, log_level="warning")
