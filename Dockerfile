FROM python:3.11-slim

WORKDIR /app

# System dependencies needed by chromadb / sentence-transformers
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

# ingest.py now calls the Gemini embedding API (no local model), so it
# needs GEMINI_API_KEY -- which is only available at runtime on
# Render/Railway, not during `docker build`. Run it at container start.
# Shell form (not exec form) is required for $PORT expansion and && chaining.
CMD python ingest.py && uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}