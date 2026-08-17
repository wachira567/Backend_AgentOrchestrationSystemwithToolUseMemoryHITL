# Multi-Agent Backend Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for build tools and PostgreSQL clients
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libpq-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install python packages
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application codebase
COPY . .

# Expose FastAPI default port
EXPOSE 8080

# Default command runs FastAPI server via Uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
