# Optimized Dockerfile for Hugging Face Spaces - Reduced size
FROM python:3.10-slim

# Create non-root user
RUN useradd -m -u 1000 user

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
COPY --chown=user:user requirements.txt ./

# Switch to user before installing Python packages
USER user
ENV PATH="/home/user/.local/bin:$PATH"

# Install Python dependencies with optimizations
RUN pip install --no-cache-dir -r requirements.txt && \
    pip cache purge

# Download model checkpoint from Google Drive
RUN pip install --no-cache-dir gdown && \
    gdown --id 1ftzxTJUnlxpQFqPlaUozG_JUbl1Qi5tQ -O /app/model_checkpoint.ckpt && \
    pip uninstall -y gdown && \
    pip cache purge

# Copy application files
COPY --chown=user:user app.py inference_core.py ./
COPY --chown=user:user scripts/ ./scripts/
COPY --chown=user:user configs/ ./configs/

# Create output directories
RUN mkdir -p api_inference_pred_masks_pipeline \
    api_inference_filtered_pipeline \
    api_inference_labeled_boxes_pipeline

# Expose port for Hugging Face Spaces
EXPOSE 7860

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PORT=7860 \
    PYTHONDONTWRITEBYTECODE=1 \
    MPLCONFIGDIR=/tmp/matplotlib

# Start Flask app
CMD ["python", "app.py"]
