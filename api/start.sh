#!/bin/bash
# Startup script for Render deployment
# Downloads model file before starting the API

echo "🚀 Starting EcoWaste AI API..."

# Download model if not present or if it's a Git LFS pointer
python download_model.py

# Start the API server
exec uvicorn real_api:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1
