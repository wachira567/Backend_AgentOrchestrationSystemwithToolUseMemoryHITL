#!/usr/bin/env bash
# ==============================================================================
# Multi-Agent Orchestration Infrastructure Quick Setup Script
# ==============================================================================
set -e

echo "=========================================================="
echo "⚡ Initializing Multi-Agent Orchestration Infrastructure ⚡"
echo "=========================================================="

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    echo "❌ Error: Docker is not installed. Please install Docker and Docker Compose."
    exit 1
fi

echo "📦 1. Copying configuration files..."
if [ ! -f .env ]; then
    cp config/.env.template .env
    echo "  -> Created .env from config/.env.template"
fi

echo "🐳 2. Spinning up Infrastructure Containers (PostgreSQL, Redis, ChromaDB)..."
docker compose up -d postgres redis chromadb

echo "⏳ 3. Waiting for database readiness..."
sleep 5

echo "🏗️ 4. Launching Backend and Celery Worker..."
docker compose up -d backend celery_worker

echo "🔍 5. Verifying container health..."
docker compose ps

echo "=========================================================="
echo "✅ All Infrastructure Services are UP and Running!"
echo "📡 FastAPI Gateway:  http://localhost:8080/api/v1/docs"
echo "🧠 ChromaDB Server:  http://localhost:8000/api/v1/heartbeat"
echo "⚡ Redis Cache:      localhost:6379"
echo "🐘 PostgreSQL DB:    localhost:5432"
echo "=========================================================="
