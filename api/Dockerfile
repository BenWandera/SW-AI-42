# Dockerfile for EcoWaste AI API - Crane Cloud Deployment
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies for PyTorch and image processing
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY real_api.py .
COPY model_loader.py .
COPY gnn_loader.py .
COPY models.py .

# Create directories for model cache and uploads
RUN mkdir -p /app/uploads /app/hf_cache /app/data

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/app/hf_cache
ENV TRANSFORMERS_CACHE=/app/hf_cache

# Expose port (Crane Cloud will set this via PORT env var)
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/').read()"

# Run the API server
# Use PORT environment variable from Crane Cloud
CMD uvicorn real_api:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1
