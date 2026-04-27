# Delegating Dockerfile for Hugging Face
# This file lives in the root but builds the backend folder

FROM python:3.11-slim

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    ffmpeg \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the backend code from its subdirectory
COPY colorization-app/backend/requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy all backend files
COPY colorization-app/backend/ .

# Ensure output directories exist
RUN mkdir -p static/uploads static/results models

# Download necessary models during build
RUN python download_models.py --all

# Pre-create the user (required for some HF environments)
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=7860 \
    FLASK_ENV=production

EXPOSE 7860

CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--workers", "2", "--threads", "1", "--timeout", "300", "--worker-class", "sync", "app:app"]
