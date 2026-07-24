"""
Central configuration for the Healthcare RAG Assistant.
All environment variables are loaded here so the rest of the codebase
never calls os.getenv() directly. This keeps configuration in one place
and makes the project easy to reconfigure or deploy.
"""

import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


def _require_env(key: str) -> str:
    value = os.getenv(key)
    if not value or value.startswith("your_"):
        raise EnvironmentError(
            f"Missing required environment variable: {key}. "
            f"Copy .env.example to .env and fill in your real values."
        )
    return value


@dataclass(frozen=True)
class Settings:
    gemini_api_key: str
    gemini_model: str
    embedding_model: str
    chroma_persist_dir: str
    data_dir: str
    chunk_size: int
    chunk_overlap: int
    top_k_results: int


def get_settings() -> Settings:
    return Settings(
        # NOTE: not validated eagerly here. Both ingest.py and
        # rag_chain.py call get_settings(), and both now need
        # gemini_api_key (ingest.py uses it for Gemini's embedding API,
        # rag_chain.py for generation). Validation happens at the
        # actual point of use in HealthcareRagAssistant.__init__ and
        # in ingest.py's main(), with a clear error message either way.
        gemini_api_key=os.getenv("GEMINI_API_KEY", ""),
        gemini_model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
        embedding_model=os.getenv(
            "EMBEDDING_MODEL", "models/gemini-embedding-001"
        ),
        chroma_persist_dir=os.getenv("CHROMA_PERSIST_DIR", "./chroma_db"),
        data_dir=os.getenv("DATA_DIR", "./sample_data"),
        chunk_size=int(os.getenv("CHUNK_SIZE", "1000")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "150")),
        top_k_results=int(os.getenv("TOP_K_RESULTS", "4")),
    )