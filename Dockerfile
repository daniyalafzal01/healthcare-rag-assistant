FROM python:3.11-slim

WORKDIR /app

# System dependencies needed by chromadb / sentence-transformers
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Build the vector store at image build time so the container is
# ready to serve immediately on startup.


EXPOSE 8000

CMD python ingest.py && uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}