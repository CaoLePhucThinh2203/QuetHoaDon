"""QuetHoaDon receipt API for a Hugging Face Docker Space (FastAPI, CPU).

    GET  /health   -> {"status": "loading" | "ready" | "error", ...}
    POST /extract  multipart field "file" (a receipt photo) -> compact JSON:
         {"file_id", "fields": {"TOTAL_COST": {...}, ...}, "items": [...], "items_check": {...}, "seconds": ...}
         or {"file_id", "fields": {}, "warning": "unreadable"} when the OCR repeats invented phrases.

The model is the final configuration (fine-tuned VietOCR, nested-box filter 0.5, M1b v3, date / time and SELLER /
ADDRESS rules, rule-based items) plus the two later options: the total from the verified item sum and the
unreadable-photo check. It is loaded once, in the background, when the server starts; the weights come from a
Hugging Face model repo (WEIGHTS_REPO). Uploaded photos are processed in a temporary folder and never stored.

Environment:
    WEIGHTS_REPO   model repo with m1b_v3_checkpoint/best/ and vietocr_finetuned.pth (required)
    HF_TOKEN       token able to read WEIGHTS_REPO when it is private (a Space secret)
    API_KEY        when set, every /extract call must send the header "x-api-key: <API_KEY>"
    DEVICE         cpu (default) or cuda
"""
import os
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, File, Header, HTTPException, UploadFile

REPO_DIR = os.environ.get("REPO_DIR", str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, REPO_DIR)

MAX_BYTES = 10 * 1024 * 1024
STATE: Dict[str, Any] = {"status": "loading", "stage": "starting", "error": None, "extractor": None,
                         "loaded_in_sec": None}
LOCK = threading.Lock()                      # one receipt at a time: the CPU is the bottleneck anyway

app = FastAPI(title="QuetHoaDon receipt API", version="1.0")


def _load() -> None:
    t0 = time.time()
    try:
        from huggingface_hub import snapshot_download
        from src.pipeline.extractor import ReceiptExtractor
        STATE["stage"] = "downloading weights"
        local = snapshot_download(os.environ["WEIGHTS_REPO"], token=os.environ.get("HF_TOKEN") or None,
                                  allow_patterns=["m1b_v3_checkpoint/*", "m1b_v3_checkpoint/**", "vietocr_finetuned.pth"])
        STATE["stage"] = "loading model"
        STATE["extractor"] = ReceiptExtractor.from_checkpoint(
            os.path.join(local, "m1b_v3_checkpoint", "best"), os.path.join(REPO_DIR, "configs"),
            os.environ.get("DEVICE", "cpu"), with_ocr=True, postprocess=True,
            ocr_weights=os.path.join(local, "vietocr_finetuned.pth"), items=True, nested_threshold=0.5,
            total_from_items=True, quality_check=True)
        STATE.update(status="ready", stage="ready", loaded_in_sec=round(time.time() - t0, 1))
    except Exception as exc:                 # keep the server up so /health can say what went wrong
        STATE.update(status="error", error=f"{type(exc).__name__}: {exc}")


@app.on_event("startup")
def start_loading() -> None:
    if os.environ.get("LOAD_ON_STARTUP", "1") == "1":     # deploy/space/app.py loads after installing detectron2
        threading.Thread(target=_load, daemon=True).start()


@app.get("/")
def root() -> Dict[str, Any]:
    return {"service": "QuetHoaDon receipt API", "health": "/health", "extract": "POST /extract (field: file)"}


@app.get("/health")
def health() -> Dict[str, Any]:
    return {k: v for k, v in STATE.items() if k != "extractor"}


@app.post("/extract")
async def extract(file: UploadFile = File(...), x_api_key: Optional[str] = Header(default=None)) -> Dict[str, Any]:
    if os.environ.get("API_KEY") and x_api_key != os.environ["API_KEY"]:
        raise HTTPException(401, "missing or wrong x-api-key")
    if STATE["status"] != "ready":
        raise HTTPException(503, f"model {STATE['status']}" + (f": {STATE['error']}" if STATE["error"] else ""))
    data = await file.read()
    if not data:
        raise HTTPException(400, "empty file")
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "image larger than 10 MB")

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
            raise HTTPException(400, "not an image")
        with LOCK:
            result = STATE["extractor"].extract(path)
    out = compact_result(result)
    out["file_id"] = os.path.splitext(file.filename or "receipt")[0]
    out["seconds"] = round(time.time() - t0, 2)
    out["timing_sec"] = result.get("timing_sec")
    return out
