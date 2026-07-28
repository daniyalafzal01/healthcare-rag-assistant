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
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
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

# Allows a standalone test frontend (or any browser-based client) to call
# this API directly. Fine for a portfolio demo; for a real production
# deployment you'd restrict allow_origins to your actual frontend's domain.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
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


SIMPLE_UI_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Healthcare Billing Assistant</title>
<style>
  * { box-sizing: border-box; }
  body {
    margin: 0;
    min-height: 100vh;
    display: flex;
    align-items: center;
    justify-content: center;
    background: #F6F4EE;
    font-family: -apple-system, "Segoe UI", Inter, sans-serif;
    color: #1E2A26;
    padding: 20px;
  }
  .card {
    width: 100%;
    max-width: 560px;
    background: #fff;
    border: 1px solid #DCD8CC;
    border-radius: 16px;
    padding: 28px;
  }
  h1 {
    font-size: 20px;
    margin: 0 0 6px;
  }
  p.sub {
    color: #6B7570;
    font-size: 14px;
    margin: 0 0 20px;
    line-height: 1.5;
  }
  textarea {
    width: 100%;
    min-height: 70px;
    border: 1px solid #DCD8CC;
    border-radius: 10px;
    padding: 12px 14px;
    font-size: 15px;
    font-family: inherit;
    resize: vertical;
    outline: none;
  }
  textarea:focus { border-color: #2F5D50; }
  button {
    margin-top: 12px;
    background: #2F5D50;
    color: #fff;
    border: none;
    border-radius: 10px;
    padding: 11px 20px;
    font-size: 15px;
    font-weight: 500;
    cursor: pointer;
    width: 100%;
  }
  button:disabled { background: #C7D6CC; cursor: default; }
  #answerBox {
    margin-top: 20px;
    padding-top: 18px;
    border-top: 1px solid #DCD8CC;
    display: none;
  }
  #answerText {
    font-size: 15.5px;
    line-height: 1.65;
    white-space: pre-wrap;
  }
  details {
    margin-top: 14px;
    font-size: 13px;
    color: #6B7570;
  }
  summary { cursor: pointer; color: #2F5D50; font-weight: 500; }
  .source {
    margin-top: 8px;
    padding-top: 8px;
    border-top: 1px solid #EEEBE0;
  }
  .source b { color: #B4762B; font-weight: 500; }
  .error { color: #8A3B2A; }
</style>
</head>
<body>
  <div class="card">
    <h1>Ask a billing or claims question</h1>
    <p class="sub">Type your question in plain English below. The answer comes only from the reference guidelines document, with sources cited.</p>
    <textarea id="q" placeholder="e.g. What happens if a claim is missing prior authorization?"></textarea>
    <button id="askBtn" onclick="ask()">Ask</button>
    <div id="answerBox">
      <div id="answerText"></div>
      <details id="sourcesDetails" style="display:none">
        <summary>View sources</summary>
        <div id="sourcesList"></div>
      </details>
    </div>
  </div>

<script>
async function ask() {
  const q = document.getElementById('q').value.trim();
  if (!q) return;
  const btn = document.getElementById('askBtn');
  const box = document.getElementById('answerBox');
  const answerText = document.getElementById('answerText');
  const sourcesDetails = document.getElementById('sourcesDetails');
  const sourcesList = document.getElementById('sourcesList');

  btn.disabled = true;
  btn.textContent = 'Thinking...';
  box.style.display = 'block';
  answerText.textContent = '';
  sourcesDetails.style.display = 'none';

  try {
    const res = await fetch('/query', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question: q })
    });
    if (!res.ok) {
      const errText = await res.text();
      answerText.innerHTML = '<span class="error">Something went wrong (' + res.status + '). If this is the first request in a while, the server may still be waking up -- wait 30-60s and try again.</span>';
      return;
    }
    const data = await res.json();
    answerText.textContent = data.answer;
    if (data.sources && data.sources.length) {
      sourcesList.innerHTML = data.sources.map(s =>
        '<div class="source"><b>' + s.source + ', chunk ' + s.chunk_index + '</b><br>"' + s.excerpt + '"</div>'
      ).join('');
      sourcesDetails.style.display = 'block';
    }
  } catch (err) {
    answerText.innerHTML = '<span class="error">Could not reach the server: ' + err.message + '</span>';
  } finally {
    btn.disabled = false;
    btn.textContent = 'Ask';
  }
}

document.getElementById('q').addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); ask(); }
});
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def root():
    return SIMPLE_UI_HTML


@app.get("/info")
def info():
    return {
        "service": "Healthcare Billing RAG Assistant",
        "docs": "/docs",
        "health": "/health",
    }


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