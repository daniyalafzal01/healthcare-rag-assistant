# Healthcare Billing RAG Assistant

A retrieval-augmented generation (RAG) system that answers insurance billing and
claims questions for independent healthcare practices, with every answer grounded
in cited source material and measured for hallucination rate using RAGAS.

## Problem

Independent and small healthcare practices lose significant revenue to preventable
claim denials — missing prior authorizations, timely-filing misses, and diagnosis/
procedure code mismatches are among the most common and most avoidable causes.
Billing staff need fast, accurate answers grounded in actual payer policy, not a
general-purpose chatbot that might hallucinate a rule that doesn't exist.

## What this project demonstrates

- **RAG architecture**: document ingestion, chunking, embedding, and retrieval
- **Grounded generation**: the model is instructed to answer only from retrieved
  context and to refuse when the answer isn't present, with mandatory source
  citations on every claim
- **Measured evaluation, not vibes**: a RAGAS evaluation harness scores
  faithfulness (hallucination rate), answer relevancy, and context precision
  against a fixed test set
- **Production shape**: a deployable FastAPI service with a `/health` and
  `/query` endpoint, containerized with Docker, not a notebook

## Architecture

```
sample_data/*.md  --->  ingest.py  --->  ChromaDB (local vector store)
                                              |
User question  --->  rag_chain.py  --->  retrieve top-k chunks
                            |
                    Gemini API (generation, grounded + cited)
                            |
                       app.py (FastAPI) --->  JSON response with sources
                            |
                     evaluate.py (RAGAS) --->  eval_report.json
```

## Tech stack

| Component | Tool | Why |
|---|---|---|
| Orchestration | LangChain | Standard document loading/splitting utilities |
| Embeddings | sentence-transformers (local) | Free, no API cost, runs offline |
| Vector store | ChromaDB | Simple, local-first, production-capable |
| Generation | Google Gemini API (free tier) | No credit card required; strong instruction-following |
| Evaluation | RAGAS | Industry-standard RAG evaluation metrics |
| API | FastAPI | Async, typed, auto-generated docs |
| Deployment | Docker | Portable, reproducible builds |

## Setup

### 1. Clone and install

```bash
git clone <your-repo-url>
cd healthcare-rag-assistant
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure environment variables

```bash
cp .env.example .env
```

Open `.env` and add your Gemini API key (get one for free, no credit card,
at https://aistudio.google.com -- click "Get API key").

**Cost note:** This project runs entirely on free tiers. Google AI Studio's
Gemini 2.5 Flash free tier requires no credit card and no payment, and
embeddings run locally on your own machine for zero cost. Building, testing,
and running the full evaluation suite costs $0.

### 3. Ingest the sample documents

```bash
python ingest.py
```

This reads everything in `sample_data/`, chunks it, embeds it locally, and
builds a ChromaDB store in `./chroma_db`. Replace the sample document with
your own reference material (public payer policy documents, not real patient
data) and re-run this step whenever documents change.

### 4. Ask a question directly

```bash
python rag_chain.py
```

### 5. Run the evaluation suite

```bash
python evaluate.py
```

This produces `eval_report.json` with real, defensible numbers — faithfulness
score, hallucination rate, answer relevancy, and context precision — across a
fixed 5-question test set. Use these numbers in interviews and on LinkedIn
instead of unverified claims.

### 6. Run the API locally

```bash
uvicorn app:app --reload
```

Visit `http://127.0.0.1:8000/docs` for interactive API documentation.

### 7. Deploy with Docker

```bash
docker build -t healthcare-rag-assistant .
docker run -p 8000:8000 --env-file .env healthcare-rag-assistant
```

## Example evaluation output

After running `evaluate.py`, you should see a report similar to this
(your exact numbers will depend on your documents and test questions):

```
==================================================
EVALUATION REPORT
==================================================
faithfulness_avg: 0.94
answer_relevancy_avg: 0.91
context_precision_avg: 0.88
hallucination_rate_pct: 6.0
num_questions_tested: 5
==================================================
```

## What I'd improve next

- Add a re-ranking step (e.g., Cohere Rerank) before generation to improve
  context precision on ambiguous questions
- Expand the test set beyond 5 questions for a more statistically stable
  evaluation
- Add a feedback loop where flagged wrong answers are added back into the
  test set (regression tracking)

## Disclaimer

This project uses synthetic, publicly-styled reference documents for
demonstration purposes only. It does not process or store real patient
data and is not a substitute for professional billing or legal advice.
