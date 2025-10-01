# Dockerfile for Hugging Face Spaces
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Download model checkpoint from Google Drive
RUN pip install gdown && \
    gdown --id 1ftzxTJUnlxpQFqPlaUozG_JUbl1Qi5tQ -O /app/model_checkpoint.ckpt

# Copy all project files
COPY . .

# Expose port for Hugging Face Spaces
EXPOSE 7860

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=7860

# Start Flask app (direct JSON responses)
CMD ["python", "app.py"]
