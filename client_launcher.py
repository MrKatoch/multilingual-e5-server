"""
One-file launcher for client PCs.

On every run it:
  1. Sets up PowerShell so typing a URL at the prompt fetches it
     (installs a profile hook + allows profile scripts for this user).
  2. Finds Ollama; if it isn't installed, downloads the official installer
     from ollama.com and installs it silently (per-user, no admin needed).
  3. Starts the FastAPI server from main.py on 0.0.0.0:8000. main.py's
     startup hook pulls the embedding model through Ollama if it's missing.

Flags:
  --no-ps-setup   skip step 1 (don't touch the PowerShell profile/policy)
"""
import os
import sys
import shutil
import socket
import subprocess
import tempfile
import time

import httpx

# Default 8007 so it doesn't clash with the start_server.bat service on 8000.
# Exported so main.py's "Server ready" banner shows the same port.
os.environ.setdefault("EMBED_API_PORT", "8007")
API_PORT = int(os.environ["EMBED_API_PORT"])
OLLAMA_PORT = 11434
OLLAMA_INSTALLER_URL = "https://ollama.com/download/OllamaSetup.exe"

PROFILE_MARKER = "# >>> Multilingual E5 URL hook >>>"
PROFILE_HOOK = PROFILE_MARKER + r"""
# Typing a URL (http://... or https://...) at the prompt fetches it instead of
# failing with "not recognized as the name of a cmdlet".
# Embedding API responses are shown as a short summary instead of 768 numbers.
$ExecutionContext.InvokeCommand.CommandNotFoundAction = {
    param($CommandName, $EventArgs)

    $url = $CommandName -replace '^get-', ''   # PowerShell retries unknown names as get-<name>
    if ($url -notmatch '^https?://') { return }

    $EventArgs.StopSearch = $true
    $EventArgs.CommandScriptBlock = {
        try {
            $r = Invoke-RestMethod -Uri $url -TimeoutSec 120
        } catch {
            Write-Host "Request failed: $($_.Exception.Message)" -ForegroundColor Red
            return
        }

        if ($r.PSObject.Properties.Name -contains 'embedding') {
            $global:LastEmbedding = $r.embedding
            [pscustomobject]@{
                input_text         = $r.input_text
                model              = $r.model
                dimensions         = $r.dimensions
                time_taken_seconds = $r.time_taken_seconds
                embedding_preview  = (($r.embedding | Select-Object -First 5) -join ', ') + ', ...'
            } | Format-List
            Write-Host 'Full vector saved in $LastEmbedding' -ForegroundColor DarkGray
        } else {
            $r
        }
    }.GetNewClosure()
}
# <<< Multilingual E5 URL hook <<<
"""


def port_open(port, host="127.0.0.1", timeout=0.5):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(timeout)
        return s.connect_ex((host, port)) == 0


def lan_ip():
    # No packets are sent; this just asks the OS which interface it would route through.
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except OSError:
        return None


def fail(msg):
    print(f"\nERROR: {msg}", flush=True)
    input("\nPress Enter to close...")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Step 1: PowerShell URL hook
# ---------------------------------------------------------------------------
def run_powershell(command):
    return subprocess.run(
        ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", command],
        capture_output=True, text=True, timeout=60,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )


def setup_powershell():
    print("[1/3] Setting up PowerShell URL shortcut...", flush=True)
    try:
        profile_path = run_powershell("$PROFILE").stdout.strip()
        if not profile_path:
            raise RuntimeError("could not find the PowerShell profile path")

        existing = ""
        if os.path.isfile(profile_path):
            with open(profile_path, encoding="utf-8-sig") as f:
                existing = f.read()

        if PROFILE_MARKER in existing:
            print("      Profile hook already installed.", flush=True)
        else:
            os.makedirs(os.path.dirname(profile_path), exist_ok=True)
            with open(profile_path, "a", encoding="utf-8") as f:
                f.write(("\n" if existing and not existing.endswith("\n") else "") + PROFILE_HOOK)
            print(f"      Installed hook in {profile_path}", flush=True)

        policy = run_powershell("Get-ExecutionPolicy").stdout.strip()
        if policy in ("RemoteSigned", "Unrestricted", "Bypass"):
            print(f"      Script policy already allows profiles ({policy}).", flush=True)
        else:
            r = run_powershell("Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned -Force")
            if r.returncode == 0:
                print("      Enabled PowerShell profile scripts for this user.", flush=True)
            else:
                print("      WARNING: could not enable PowerShell scripts (probably blocked by "
                      "company policy). Typing URLs in PowerShell won't work, but the server "
                      "and browser still will.", flush=True)
    except Exception as e:
        print(f"      WARNING: PowerShell setup skipped: {e}", flush=True)


# ---------------------------------------------------------------------------
# Step 2: Ollama
# ---------------------------------------------------------------------------
def find_ollama():
    candidates = [
        shutil.which("ollama"),
        os.path.expandvars(r"%LOCALAPPDATA%\Programs\Ollama\ollama.exe"),
        os.path.expandvars(r"%ProgramFiles%\Ollama\ollama.exe"),
        os.path.expandvars(r"%ProgramFiles(x86)%\Ollama\ollama.exe"),
    ]
    for path in candidates:
        if path and os.path.isfile(path):
            return path
    return None


def download(url, dest):
    with httpx.stream("GET", url, follow_redirects=True, timeout=60.0) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        done, last_pct = 0, -1
        with open(dest, "wb") as f:
            for chunk in r.iter_bytes(1024 * 1024):
                f.write(chunk)
                done += len(chunk)
                if total:
                    pct = done * 100 // total
                    if pct != last_pct and pct % 5 == 0:
                        print(f"      {pct}%  ({done >> 20} MB / {total >> 20} MB)", flush=True)
                        last_pct = pct


def install_ollama():
    print("      Ollama not found. Downloading installer from ollama.com...", flush=True)
    installer = os.path.join(tempfile.gettempdir(), "OllamaSetup.exe")
    try:
        download(OLLAMA_INSTALLER_URL, installer)
    except httpx.HTTPError as e:
        fail(f"Could not download Ollama ({e}). Check the internet connection, "
             f"or install it manually from https://ollama.com/download")

    print("      Installing Ollama silently (this can take a minute)...", flush=True)
    r = subprocess.run([installer, "/VERYSILENT", "/NORESTART", "/SUPPRESSMSGBOXES"])
    try:
        os.remove(installer)
    except OSError:
        pass
    if r.returncode != 0:
        fail(f"Ollama installer exited with code {r.returncode}.")

    path = find_ollama()
    if not path:
        fail("Ollama installed but ollama.exe could not be found.")
    print(f"      Ollama installed: {path}", flush=True)
    return path


def ensure_ollama():
    print("[2/3] Checking Ollama...", flush=True)
    path = find_ollama()
    if port_open(OLLAMA_PORT):
        print("      Ollama is already running.", flush=True)
        return path
    if path:
        print(f"      Found Ollama: {path}", flush=True)
    else:
        path = install_ollama()

    if not port_open(OLLAMA_PORT):
        print("      Starting Ollama...", flush=True)
        subprocess.Popen(
            [path, "serve"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        for _ in range(60):
            if port_open(OLLAMA_PORT):
                break
            time.sleep(0.5)
        else:
            fail("Ollama did not start within 30 seconds.")
    print("      Ollama is running.", flush=True)
    return path


# ---------------------------------------------------------------------------
# Step 3: API server
# ---------------------------------------------------------------------------
def run_server(ollama_path):
    print("[3/3] Starting embedding server...", flush=True)
    if port_open(API_PORT):
        fail(f"Port {API_PORT} is already in use - the server may already be running. "
             f"Try http://127.0.0.1:{API_PORT} in a browser, or close the other program.")

    import uvicorn
    import main  # imported late so its log file/startup banner come after setup output

    if ollama_path:
        main.OLLAMA_EXE = ollama_path

    ip = lan_ip()
    print(f"\n      This PC:        http://127.0.0.1:{API_PORT}/embed/hello", flush=True)
    if ip:
        print(f"      Other devices:  http://{ip}:{API_PORT}/embed/hello", flush=True)
    print("", flush=True)

    uvicorn.run(main.app, host="0.0.0.0", port=API_PORT, log_level="warning")


if __name__ == "__main__":
    if "--no-ps-setup" not in sys.argv:
        setup_powershell()
    else:
        print("[1/3] PowerShell setup skipped (--no-ps-setup).", flush=True)
    run_server(ensure_ollama())
