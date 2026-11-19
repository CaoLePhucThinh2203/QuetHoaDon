"""QuetHoaDon on a free Hugging Face Space (Gradio SDK, ZeroGPU hardware).

Free accounts get Gradio Spaces on ZeroGPU only, and ZeroGPU accepts nothing but a real Gradio app with at least one
``@spaces.GPU`` function. So this file is a Gradio app whose functions double as the API the Vercel page calls:

* ``/gradio_api/call/extract_b64``: a base64 JPEG in, the compact JSON out (total, items, check, or the unreadable
  warning); ``/gradio_api/call/health``: whether the model is loaded and what the start-up is doing;
* a small page (upload a photo, see the JSON) for trying it in the browser.

The model runs on CPU (as on any machine without a GPU), so no CUDA build is involved; the ``@spaces.GPU`` function is
only there because ZeroGPU requires one. At start-up, in the background: clone the project (branch ``BRANCH``), install
``craft-text-detector`` and ``vietocr`` without their pinned dependencies, install detectron2 (compiled once against the
installed torch; the wheel is cached in the weights repo when ``HF_TOKEN`` may write there), download the weights from
``WEIGHTS_REPO`` and load the model. Photos are processed in a temporary folder and not kept. ``API_KEY``, when set,
must be sent as the second input of ``extract_b64``.
"""
import base64
import glob
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import traceback

import gradio as gr

try:
    import spaces

    @spaces.GPU
    def _zero_gpu_probe() -> bool:                 # ZeroGPU stops a Space that declares no GPU function
        return True
except ImportError:
    spaces = None

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(HERE, "QuetHoaDon")
BRANCH = os.environ.get("BRANCH", "thinh-dev")
GIT_URL = "https://github.com/CaoLePhucThinh2203/QuetHoaDon.git"
DETECTRON2 = "git+https://github.com/facebookresearch/detectron2.git"
MAX_BYTES = 10 * 1024 * 1024


def run(*cmd: str) -> None:
    print("[space] $", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def pip(*args: str) -> None:
    run(sys.executable, "-m", "pip", "install", "--no-cache-dir", *args)


if not os.path.exists(REPO):                       # the loader below lives in the project
    run("git", "clone", "--depth", "1", "-b", BRANCH, GIT_URL, REPO)
os.environ["REPO_DIR"] = REPO
os.environ["LOAD_ON_STARTUP"] = "0"
os.environ.setdefault("DEVICE", "cpu")
_spec = importlib.util.spec_from_file_location("receipt_api", os.path.join(REPO, "deploy", "api", "app.py"))
api = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(api)                      # gives STATE, LOCK and _load (weights + model)
STATE, LOCK = api.STATE, api.LOCK


def install_detectron2() -> None:
    try:
        import detectron2  # noqa: F401
        return
    except ImportError:
        pass
    import torch
    from huggingface_hub import HfApi, hf_hub_download
    repo, token = os.environ.get("WEIGHTS_REPO"), os.environ.get("HF_TOKEN") or None
    prefix = f"wheels/detectron2-torch{torch.__version__.split('+')[0]}-py{sys.version_info.major}{sys.version_info.minor}"
    hub = HfApi(token=token)
    try:                                           # a wheel compiled by an earlier start
        cached = [f for f in hub.list_repo_files(repo) if f.startswith(prefix + "/") and f.endswith(".whl")]
        if cached:
            STATE["stage"] = "installing detectron2 (cached wheel)"
            pip(hf_hub_download(repo, cached[0], token=token))
            return
    except Exception:
        traceback.print_exc()
    STATE["stage"] = "compiling detectron2 (first start, 10-20 minutes)"
    out = os.path.join(HERE, "wheels")
    env = dict(os.environ, FORCE_CUDA="0", CUDA_VISIBLE_DEVICES="")
    print("[space] $ pip wheel detectron2", flush=True)
    subprocess.run([sys.executable, "-m", "pip", "wheel", "--no-build-isolation", "--no-deps", "-w", out, DETECTRON2],
                   check=True, env=env)
    wheel = glob.glob(os.path.join(out, "detectron2-*.whl"))[0]
    pip(wheel)
    try:                                           # needs a token that may write to WEIGHTS_REPO; optional
        hub.upload_file(path_or_fileobj=wheel, path_in_repo=f"{prefix}/{os.path.basename(wheel)}", repo_id=repo)
        print("[space] cached the detectron2 wheel in", repo, flush=True)
    except Exception as exc:
        print("[space] could not cache the detectron2 wheel (read-only token?):", exc, flush=True)


def prepare_and_load() -> None:
    t0 = time.time()
    if os.environ.get("SKIP_INSTALL") != "1":      # local test: packages already installed
        try:
            STATE["stage"] = "installing OCR packages"
            pip("--no-deps", "craft-text-detector", "vietocr")
            install_detectron2()
            print(f"[space] packages ready in {time.time() - t0:.0f}s", flush=True)
        except Exception as exc:
            traceback.print_exc()
            STATE.update(status="error", error=f"install failed: {exc}")
            return
    api._load()
    print("[space] model:", STATE["status"], STATE.get("error") or "", flush=True)


threading.Thread(target=prepare_and_load, daemon=True).start()


# ---------------------------------------------------------------------------------------------------------- functions
def health() -> dict:
    return {k: v for k, v in STATE.items() if k != "extractor"}


def _extract_bytes(data: bytes, name: str = "receipt") -> dict:
    if STATE["status"] != "ready":
        return {"error": f"model {STATE['status']}", "stage": STATE.get("stage"), "detail": STATE.get("error")}
    if not data:
        return {"error": "empty image"}
    if len(data) > MAX_BYTES:
        return {"error": "image larger than 10 MB"}
    from src.pipeline.demo_view import prepare_upload
    from src.pipeline.extractor import compact_result
    t0 = time.time()
    with tempfile.TemporaryDirectory() as tmp:
        raw = os.path.join(tmp, "upload")
        with open(raw, "wb") as f:
            f.write(data)
        try:
            path = prepare_upload(raw, os.path.join(tmp, "receipt.jpg"))
        except Exception:
            return {"error": "not an image"}
        with LOCK:
            result = STATE["extractor"].extract(path)
    out = compact_result(result)
    out["file_id"] = name
    out["seconds"] = round(time.time() - t0, 2)
    out["timing_sec"] = result.get("timing_sec")
    return out


def extract_b64(image_b64: str, api_key: str = "") -> dict:
    """API for the Vercel page: base64 JPEG (a data URL prefix is accepted) and the API key."""
    if os.environ.get("API_KEY") and api_key != os.environ["API_KEY"]:
        return {"error": "missing or wrong api key"}
    try:
        data = base64.b64decode((image_b64 or "").split(",")[-1], validate=False)
    except Exception:
        return {"error": "bad base64"}
    return _extract_bytes(data)


def extract_file(path: str) -> dict:
    """The test page: a photo uploaded in the browser."""
    if not path:
        return {"error": "no image"}
    with open(path, "rb") as f:
        return _extract_bytes(f.read(), os.path.splitext(os.path.basename(path))[0])


with gr.Blocks(title="QuetHoaDon API") as demo:
    gr.Markdown("# QuetHoaDon API\nTải ảnh hóa đơn để thử; trang web dùng API `extract_b64`. Trạng thái máy chủ:")
    status = gr.JSON(label="health")
    gr.Button("Kiểm tra trạng thái").click(health, None, status, api_name="health")
    with gr.Row():
        image = gr.Image(type="filepath", label="Ảnh hóa đơn")
        result = gr.JSON(label="Kết quả")
    gr.Button("Quét").click(extract_file, image, result, api_name="extract_file")
    b64_in, key_in = gr.Textbox(visible=False), gr.Textbox(visible=False)
    b64_btn = gr.Button(visible=False)
    b64_btn.click(extract_b64, [b64_in, key_in], gr.JSON(visible=False), api_name="extract_b64")
    demo.load(health, None, status)

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
