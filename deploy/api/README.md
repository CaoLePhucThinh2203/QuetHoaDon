---
title: QuetHoaDon API
emoji: 🧾
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# QuetHoaDon receipt API

`POST /extract` with a receipt photo (multipart field `file`) returns the total, the item lines (name, quantity, unit
price, amount) and whether the items add up to the printed total. `GET /health` says whether the model is loaded.
Code: https://github.com/CaoLePhucThinh2203/QuetHoaDon (`deploy/api/app.py`).

Copy this file and `Dockerfile` into the Space repository. Set the variable `WEIGHTS_REPO` and, for a private weights
repo, the secret `HF_TOKEN`; optionally the secret `API_KEY`.
