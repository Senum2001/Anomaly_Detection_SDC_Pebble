# Optimized Dockerfile for Hugging Face Spaces - Reduced size
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies for OpenCV (minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    wget \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy requirements first for better caching
COPY requirements.txt ./

# Install Python dependencies with optimizations
RUN pip install --no-cache-dir -r requirements.txt && \
    pip cache purge && \
    rm -rf /root/.cache/pip

# Download model checkpoint from Google Drive
RUN pip install --no-cache-dir gdown && \
    gdown --id 1ftzxTJUnlxpQFqPlaUozG_JUbl1Qi5tQ -O /app/model_checkpoint.ckpt && \
    pip uninstall -y gdown && \
    rm -rf /root/.cache/pip

# Copy application files
COPY app.py inference_core.py ./
COPY scripts/ ./scripts/
COPY configs/ ./configs/

# Create output directories
RUN mkdir -p api_inference_pred_masks_pipeline \
    api_inference_filtered_pipeline \
    api_inference_labeled_boxes_pipeline

# Expose port for Hugging Face Spaces
EXPOSE 7860

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PORT=7860 \
    PYTHONDONTWRITEBYTECODE=1

# Start Flask app
CMD ["python", "app.py"]
