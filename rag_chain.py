"""
Core retrieval-augmented generation logic.

Given a user question, this module:
1. Retrieves the top-k most relevant chunks from ChromaDB
2. Sends them to Gemini with an instruction to answer ONLY from the
   provided context and to cite the exact source file + chunk number
3. Returns both the answer and the raw retrieved chunks (needed later
   for RAGAS evaluation)

This separation (retrieve -> generate, with retrieved chunks returned
alongside the answer) is what makes the pipeline evaluable, not just
"a chatbot that sounds right."
"""

from dataclasses import dataclass
from typing import List

from google import genai
from google.genai import types as genai_types
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

from config import get_settings

SYSTEM_PROMPT = """You are a billing and claims reference assistant for independent \
healthcare practices. You answer questions ONLY using the reference material provided \
in the context below. 

Rules you must follow strictly:
1. If the answer is not contained in the provided context, say exactly: \
"I don't have information about that in the reference material provided." \
Do not guess or use outside knowledge.
2. Every claim in your answer must be followed by a citation in the format \
[Source: <filename>, chunk <chunk_index>].
3. Keep answers concise and practical, written for a billing staff member, \
not a technical audience.
4. Never state a specific dollar amount, timeline, or rule that is not explicitly \
present in the context.
"""


@dataclass
class RagResult:
    answer: str
    retrieved_chunks: List[Document]
    question: str


class HealthcareRagAssistant:
    def __init__(self):
        self.settings = get_settings()
        if not self.settings.gemini_api_key or self.settings.gemini_api_key.startswith("your_"):
            raise EnvironmentError(
                "Missing required environment variable: GEMINI_API_KEY. "
                "Set it in your .env file (local) or your host's dashboard (Render/Railway)."
            )
        self.embeddings = HuggingFaceEmbeddings(model_name=self.settings.embedding_model)
        self.vector_store = Chroma(
            collection_name="healthcare_billing_docs",
            embedding_function=self.embeddings,
            persist_directory=self.settings.chroma_persist_dir,
        )
        self.client = genai.Client(api_key=self.settings.gemini_api_key)
    def retrieve(self, question: str) -> List[Document]:
        return self.vector_store.similarity_search(
            question, k=self.settings.top_k_results
        )

    def _format_context(self, chunks: List[Document]) -> str:
        blocks = []
        for chunk in chunks:
            source = chunk.metadata.get("source", "unknown").split("/")[-1]
            index = chunk.metadata.get("chunk_index", "?")
            blocks.append(
                f"[Source: {source}, chunk {index}]\n{chunk.page_content}"
            )
        return "\n\n---\n\n".join(blocks)

    def answer(self, question: str) -> RagResult:
        chunks = self.retrieve(question)

        if not chunks:
            return RagResult(
                answer="I don't have information about that in the reference material provided.",
                retrieved_chunks=[],
                question=question,
            )

        context = self._format_context(chunks)

        response = self.client.models.generate_content(
            model=self.settings.gemini_model,
            contents=f"Context:\n\n{context}\n\nQuestion: {question}",
            config=genai_types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                max_output_tokens=600,
            ),
        )

        answer_text = response.text or ""

        return RagResult(answer=answer_text, retrieved_chunks=chunks, question=question)


if __name__ == "__main__":
    assistant = HealthcareRagAssistant()
    test_question = "What happens if a claim is missing prior authorization?"
    result = assistant.answer(test_question)
    print(f"Q: {result.question}\n")
    print(f"A: {result.answer}\n")
    print(f"Retrieved {len(result.retrieved_chunks)} chunk(s).")
