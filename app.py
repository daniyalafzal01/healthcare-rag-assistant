"""
FastAPI wrapper for the Healthcare RAG Assistant.

This turns the RAG pipeline into a deployable web API instead of a
notebook, which is what production-grade portfolio projects need to
demonstrate.

Run locally with:
    uvicorn app:app --reload

Then open http://127.0.0.1:8000/docs for interactive API documentation.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from rag_chain import HealthcareRagAssistant

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

assistant_instance: HealthcareRagAssistant | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the assistant once at startup, not on every request.
    global assistant_instance
    logger.info("Loading Healthcare RAG Assistant (embeddings + vector store)...")
    assistant_instance = HealthcareRagAssistant()
    logger.info("Assistant ready.")
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="Healthcare Billing RAG Assistant",
    description=(
        "Answers billing and claims questions for independent healthcare "
        "practices, grounded in reference documents with source citations."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


class QueryRequest(BaseModel):
    question: str = Field(
        ..., min_length=3, max_length=500, description="The billing/claims question to ask."
    )


class SourceChunk(BaseModel):
    source: str
    chunk_index: int
    excerpt: str


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceChunk]


@app.get("/health")
def health_check():
    return {"status": "ok", "assistant_loaded": assistant_instance is not None}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    if assistant_instance is None:
        raise HTTPException(status_code=503, detail="Assistant is still starting up.")

    try:
        result = assistant_instance.answer(request.question)
    except Exception as exc:
        logger.exception("Error while answering question")
        raise HTTPException(status_code=500, detail=f"Internal error: {exc}") from exc

    sources = [
        SourceChunk(
            source=chunk.metadata.get("source", "unknown").split("/")[-1],
            chunk_index=chunk.metadata.get("chunk_index", -1),
            excerpt=chunk.page_content[:200] + "...",
        )
        for chunk in result.retrieved_chunks
    ]

    return QueryResponse(question=result.question, answer=result.answer, sources=sources)
