# Use official Python image with CUDA support if needed, else use python:3.10
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0

# Copy all project files
COPY . .

# Expose port (Flask default)
EXPOSE 5000

# Set environment variables (optional)
ENV PYTHONUNBUFFERED=1

# Start the Flask app using Gunicorn for production
CMD ["gunicorn", "pipeline:app", "--bind", "0.0.0.0:5000", "--timeout", "600"]
