# Base image with CUDA for GPU acceleration
FROM nvidia/cuda:12.3.2-base-ubuntu22.04

# Prevents interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    python3-pip \
    python3-dev \
    python3-opencv \
    libglib2.0-0 \
    python3-venv && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Create a virtual environment
RUN python3 -m venv /app/venv

# Activate the virtual environment and upgrade pip
RUN /app/venv/bin/pip install --upgrade pip

# Copy only requirements.txt first to leverage Docker caching
COPY requirements.txt /tmp/requirements.txt

# Install dependencies inside the virtual environment
RUN /app/venv/bin/pip install --default-timeout=1000 --no-cache-dir -r /tmp/requirements.txt

# Copy the rest of the application
COPY . /app

# Set the virtual environment as the default Python environment
ENV PATH="/app/venv/bin:$PATH"

# Start the Gunicorn server
ENTRYPOINT ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "-b", "0.0.0.0:8080", "app:app"]
