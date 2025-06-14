# Dockerfile for building the CV image.

# Note: .dockerignore file is hidden. contains src/train/ in cv folder. ls -la to see it
# command: docker build -t bjjsql-cv:v1 . 
# If no space, run: docker system prune -a
# To run image: docker run -p 5002:5002 --gpus all bjjsql-cv:v4.7
# To run offline: docker run --network none --gpus all bjjsql-cv:latest

# The base image, an example deep learning VM.
# For a full list, see https://us-docker.pkg.dev/deeplearning-platform-release/gcr.io/
# For info, see https://cloud.google.com/deep-learning-vm/docs/images#supported-frameworks
FROM us-docker.pkg.dev/deeplearning-platform-release/gcr.io/pytorch-gpu.2-2.py310

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_ROOT_USER_ACTION=ignore
WORKDIR /workspace

# Upgrade pip
RUN pip install -U pip

# Copy files needed for installing dependencies
COPY requirements.txt .
COPY models/ ./models/

# Install PyTorch with CUDA 12.1 support
RUN pip install torch==2.7.1+cu128 torchvision==0.22.1+cu128 torchaudio==2.7.1+cu128 \
    --extra-index-url https://download.pytorch.org/whl/cu128

# Install JAX with CUDA 12.1 support
RUN pip install jax[cuda12]==0.6.1 \
    -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Install remaining project dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy your source code
COPY src .

# Start the model server
CMD ["uvicorn", "cv_server:app", "--port", "5002", "--host", "0.0.0.0"]