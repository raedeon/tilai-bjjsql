# ✅ Base image with CUDA 12.8 and cuDNN support
FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04

# ✅ Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

# ✅ Set working directory
WORKDIR /workspace

# ✅ Install Python 3.10 and required tools
RUN apt-get update && apt-get install -y \
    python3.10 python3.10-venv python3.10-distutils ffmpeg libsndfile1 curl git \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python

# ✅ Install PyTorch 2.7.0 with CUDA 12.8 support manually
RUN pip install --upgrade pip
RUN pip install torch==2.7.0+cu128 torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu128

# ✅ Install Python dependencies (excluding torch to avoid conflicts)
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir -r requirements.txt

# ✅ Copy application source code and model directory
COPY src .
COPY Faster/ ./Faster/

# ✅ Expose FastAPI port
EXPOSE 5001

# ✅ Launch the FastAPI ASR server
CMD ["uvicorn", "asr_server:app", "--host", "0.0.0.0", "--port", "5001"]
