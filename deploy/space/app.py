"""Entry point of the QuetHoaDon API on a free Hugging Face Space with the Gradio SDK (Docker Spaces need a paid plan).

A Gradio Space runs ``python app.py`` and expects a web server on port 7860; it does not have to be Gradio. This file
starts the FastAPI app of ``deploy/api/app.py`` (GET /health, POST /extract) right away, and in the background:

1. clones the project (branch ``BRANCH``, default ``thinh-dev``);
2. installs what pip cannot install from requirements.txt: ``craft-text-detector`` and ``vietocr`` without their
   pinned dependencies, and detectron2, which must be compiled against the installed torch. The compiled wheel is
   cached in the weights repo (``wheels/``) when ``HF_TOKEN`` may write there, so later starts skip the 10-20 minute
   compilation;
3. downloads the weights and loads the model. ``/health`` reports the stage meanwhile.
"""
import glob
import importlib.util
import os
import subprocess
import sys
import threading
import time
import traceback

import uvicorn

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(HERE, "QuetHoaDon")
BRANCH = os.environ.get("BRANCH", "thinh-dev")
GIT_URL = "https://github.com/CaoLePhucThinh2203/QuetHoaDon.git"
DETECTRON2 = "git+https://github.com/facebookresearch/detectron2.git"


def run(*cmd: str) -> None:
    print("[space] $", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def pip(*args: str) -> None:
    run(sys.executable, "-m", "pip", "install", "--no-cache-dir", *args)


if not os.path.exists(REPO):                       # the code is needed now: the API module lives in it
    run("git", "clone", "--depth", "1", "-b", BRANCH, GIT_URL, REPO)
os.environ["REPO_DIR"] = REPO
os.environ["LOAD_ON_STARTUP"] = "0"
spec = importlib.util.spec_from_file_location("receipt_api", os.path.join(REPO, "deploy", "api", "app.py"))
api = importlib.util.module_from_spec(spec)
spec.loader.exec_module(api)


def detectron2_wheel_name() -> str:
    import torch
    return f"wheels/detectron2-torch{torch.__version__.split('+')[0]}-py{sys.version_info.major}{sys.version_info.minor}"


def install_detectron2() -> None:
    try:
        import detectron2  # noqa: F401
        return
    except ImportError:
        pass
    from huggingface_hub import HfApi, hf_hub_download
    repo, token = os.environ.get("WEIGHTS_REPO"), os.environ.get("HF_TOKEN") or None
    prefix = detectron2_wheel_name()
    hub = HfApi(token=token)
    try:                                           # a wheel compiled by an earlier start of this Space
        cached = [f for f in hub.list_repo_files(repo) if f.startswith(prefix + "/") and f.endswith(".whl")]
        if cached:
            api.STATE["stage"] = "installing detectron2 (cached wheel)"
            pip(hf_hub_download(repo, cached[0], token=token))
            return
    except Exception:
        traceback.print_exc()
    api.STATE["stage"] = "compiling detectron2 (first start, 10-20 minutes)"
    out = os.path.join(HERE, "wheels")
    run(sys.executable, "-m", "pip", "wheel", "--no-build-isolation", "--no-deps", "-w", out, DETECTRON2)
    wheel = glob.glob(os.path.join(out, "detectron2-*.whl"))[0]
    pip(wheel)
    try:                                           # needs a token that may write to WEIGHTS_REPO; optional
        hub.upload_file(path_or_fileobj=wheel, path_in_repo=f"{prefix}/{os.path.basename(wheel)}", repo_id=repo)
        print("[space] cached the detectron2 wheel in", repo, flush=True)
    except Exception as exc:
        print("[space] could not cache the detectron2 wheel (read-only token?):", exc, flush=True)


def prepare_and_load() -> None:
    t0 = time.time()
    if os.environ.get("SKIP_INSTALL") == "1":     # local test: packages already installed
        api._load()
        return
    try:
        api.STATE["stage"] = "installing OCR packages"
        pip("--no-deps", "craft-text-detector", "vietocr")
        install_detectron2()
        print(f"[space] packages ready in {time.time() - t0:.0f}s", flush=True)
    except Exception as exc:
        traceback.print_exc()
        api.STATE.update(status="error", error=f"install failed: {exc}")
        return
    api._load()


threading.Thread(target=prepare_and_load, daemon=True).start()

if __name__ == "__main__":
    uvicorn.run(api.app, host="0.0.0.0", port=int(os.environ.get("PORT", 7860)))
