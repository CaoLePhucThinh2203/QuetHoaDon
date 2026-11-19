---
title: QuetHoaDon API
emoji: 🧾
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 5.9.1
python_version: "3.12"
app_file: app.py
pinned: false
---

# QuetHoaDon receipt API

Free CPU Space (Gradio SDK) that serves a FastAPI app on port 7860:

- `GET /health`: `status` (`loading`, `ready`, `error`) and the current `stage` while starting
- `POST /extract` with a receipt photo (multipart field `file`): total, item lines (name, quantity, unit price, amount)
  and whether they add up to the printed total; `"warning": "unreadable"` for photos whose OCR is not trustworthy

Code: https://github.com/CaoLePhucThinh2203/QuetHoaDon (`deploy/space/app.py`, `deploy/api/app.py`).
Settings: variable `WEIGHTS_REPO`, secrets `HF_TOKEN` (read; write to also cache the compiled detectron2) and `API_KEY`.
