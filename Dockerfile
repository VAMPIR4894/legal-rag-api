# Use a Python base image suitable for heavy computation
# We choose a version that supports the required dependencies well
FROM python:3.10-slim

# Set environment variables for non-interactive commands
ENV PYTHONUNBUFFERED 1
ENV APP_HOME /app

# Create and set the working directory
RUN mkdir $APP_HOME
WORKDIR $APP_HOME

# Install system dependencies needed for various Python packages (like 'scipy', 'numpy')
# We need build-essential for compiling some libraries, then clean up.
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file first to leverage Docker layer caching
COPY requirements.txt .

# Install Python dependencies, including production server (gunicorn)
# Note: For Apple Silicon (M-series), installing certain packages (like PyTorch) 
# in the Docker environment can be tricky; this assumes a standard Linux environment.
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . $APP_HOME

# Create necessary runtime directories for ChromaDB persistence
RUN mkdir -p data/vectordb
RUN mkdir -p data/bm25_indexes

# Expose the port for Cloud Run (uses PORT env variable)
EXPOSE 8080

# Command to run the application using Uvicorn (FastAPI server)
# Cloud Run sets PORT environment variable, default to 8080
# --host 0.0.0.0 is required for Cloud Run
# --workers 2 for concurrent request handling
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080} --workers 2