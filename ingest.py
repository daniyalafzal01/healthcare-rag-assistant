"""
Ingestion pipeline for the Healthcare RAG Assistant.

Loads every .md, .txt, and .pdf file from DATA_DIR, splits them into
overlapping chunks, embeds each chunk with a local sentence-transformers
model (no API cost), and persists everything to a local ChromaDB store.

Run this once whenever you add or change documents in sample_data/:
    python ingest.py
"""

import os
import sys
import logging

from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    DirectoryLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

from config import get_settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_documents(data_dir: str):
    """Load all supported documents from the data directory."""
    if not os.path.isdir(data_dir):
        logger.error(f"Data directory '{data_dir}' does not exist.")
        sys.exit(1)

    documents = []

    md_txt_loader = DirectoryLoader(
        data_dir,
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=True,
    )
    documents.extend(md_txt_loader.load())

    txt_loader = DirectoryLoader(
        data_dir,
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=True,
    )
    documents.extend(txt_loader.load())

    pdf_loader = DirectoryLoader(
        data_dir,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True,
    )
    documents.extend(pdf_loader.load())

    if not documents:
        logger.error(
            f"No .md, .txt, or .pdf files found in '{data_dir}'. "
            f"Add at least one document before running ingestion."
        )
        sys.exit(1)

    logger.info(f"Loaded {len(documents)} source document(s).")
    return documents


def chunk_documents(documents, chunk_size: int, chunk_overlap: int):
    """Split documents into overlapping chunks for retrieval."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)

    # Tag each chunk with a stable chunk index within its source file.
    # This is what lets the assistant cite "Section X, chunk N" later.
    counters = {}
    for chunk in chunks:
        source = chunk.metadata.get("source", "unknown")
        counters[source] = counters.get(source, -1) + 1
        chunk.metadata["chunk_index"] = counters[source]

    logger.info(f"Split into {len(chunks)} chunk(s).")
    return chunks


def build_vector_store(chunks, embedding_model: str, persist_dir: str):
    """Embed chunks and persist them to a local ChromaDB collection."""
    logger.info(f"Loading embedding model '{embedding_model}' (first run downloads it)...")
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)

    logger.info(f"Embedding {len(chunks)} chunk(s) and writing to '{persist_dir}'...")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir,
        collection_name="healthcare_billing_docs",
    )
    logger.info("Ingestion complete. Vector store is ready to query.")
    return vector_store


def main():
    settings = get_settings()
    documents = load_documents(settings.data_dir)
    chunks = chunk_documents(documents, settings.chunk_size, settings.chunk_overlap)
    build_vector_store(chunks, settings.embedding_model, settings.chroma_persist_dir)


if __name__ == "__main__":
    main()
