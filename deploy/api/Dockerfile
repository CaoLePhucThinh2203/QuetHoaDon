# QuetHoaDon receipt API: Hugging Face Docker Space, CPU. Build takes about 15-25 minutes (detectron2 is compiled).
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends git build-essential libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Spaces run the container as user 1000
RUN useradd -m -u 1000 user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2
WORKDIR /home/user

# CPU PyTorch first (detectron2 compiles against it). transformers is the version M1b v3 was saved with.
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir "transformers==5.20.2" sentencepiece "numpy<3" opencv-python-headless pillow pyyaml \
        scikit-learn tqdm pandas fastapi "uvicorn[standard]" python-multipart huggingface_hub
RUN pip install --no-cache-dir craft-text-detector vietocr --no-deps \
    && pip install --no-cache-dir gdown einops prefetch_generator albumentations lmdb
RUN pip install --no-cache-dir --no-build-isolation "git+https://github.com/facebookresearch/detectron2.git"

# Code of the project (branch with the item rules, the total check and the unreadable-photo check)
ARG BRANCH=thinh-dev
RUN git clone --depth 1 -b ${BRANCH} https://github.com/CaoLePhucThinh2203/QuetHoaDon.git /home/user/QuetHoaDon \
    && chown -R user /home/user
ENV REPO_DIR=/home/user/QuetHoaDon

USER user
EXPOSE 7860
CMD ["uvicorn", "app:app", "--app-dir", "/home/user/QuetHoaDon/deploy/api", "--host", "0.0.0.0", "--port", "7860"]
