# Base image with PyTorch (from NVIDIA NGC)
FROM nvcr.io/nvidia/pytorch:25.03-py3

# Environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies (for OCR libraries)
# WARNING: The options to allow unauthenticated packages are a security risk.
RUN apt-get update -o Acquire::AllowInsecureRepositories=true -o Acquire::AllowDowngradeToInsecureRepositories=true && \
    apt-get install -y --allow-unauthenticated --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libpango-1.0-0 \
    libpangoft2-1.0-0 \
    libpangocairo-1.0-0 \
    libcairo2 \
    libgdk-pixbuf2.0-0 \
    libffi-dev \
    shared-mime-info \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Copy application code
WORKDIR /workspace
COPY src/ /workspace/

RUN echo "Pre-downloading models..." && \
    python -c "from ocr_manager import OCRManager; manager = OCRManager()"

EXPOSE 5003

CMD ["python", "-m", "uvicorn", "ocr_server:app", "--port", "5003", "--host", "0.0.0.0"]