# Dockerfile for building the RL image.


# The base image. Using a regular Python base because RL agents don't typically
# require much GPU. This will speed up your evaluation.
# Use a slim Python base for performance
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_ROOT_USER_ACTION=ignore
WORKDIR /workspace

# Upgrade pip
RUN pip install -U pip

# Copy dependency files
COPY requirements.txt .
COPY ./til-25-environment ./til-25-environment

# Install PyTorch with CUDA 12.8 (cu128) support
RUN pip install torch==2.7.1+cu128 torchvision==0.22.1+cu128 torchaudio==2.7.1+cu128 \
    --extra-index-url https://download.pytorch.org/whl/cu128

# Install JAX with CUDA 12 support (optional, only if needed by your code)
#RUN pip install jax[cuda12]==0.6.1 \
    #-f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Install remaining dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy your source code
COPY src .

# Start your model server
CMD ["uvicorn", "rl_server:app", "--port", "5004", "--host", "0.0.0.0"]
