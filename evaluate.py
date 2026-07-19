"""
Evaluation script using RAGAS.

This is the single most important file in the project for portfolio purposes.
It runs a fixed set of test questions through the RAG pipeline, then scores:

- Faithfulness: does the answer only contain claims supported by the
  retrieved context? (low faithfulness = hallucination)
- Answer Relevancy: does the answer actually address the question asked?
- Context Precision: are the retrieved chunks actually relevant?

Run with:
    python evaluate.py

The script prints a report and saves results to eval_report.json so you
can commit real numbers to your GitHub README instead of vague claims.
"""

import json
import logging

from datasets import Dataset
from ragas import evaluate
from ragas.run_config import RunConfig
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings

from config import get_settings
from rag_chain import HealthcareRagAssistant

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Fixed test set. Add more questions here as you add more source documents.
# ground_truth is what a correct answer should contain, used by RAGAS to
# score relevancy and faithfulness against.
TEST_QUESTIONS = [
    {
        "question": "What happens if a claim is missing prior authorization?",
        "ground_truth": (
            "Claims for procedures requiring prior authorization will be denied "
            "if the authorization number is missing, expired, or does not match "
            "the billed procedure code."
        ),
    },
    {
        "question": "How long do practices have to submit a claim after the date of service?",
        "ground_truth": (
            "Most commercial payers require claims to be submitted within 90 to "
            "180 days of the date of service, and Medicaid programs often have "
            "shorter windows, sometimes as low as 90 days."
        ),
    },
    {
        "question": "What are the four things a practice should verify before submitting a claim?",
        "ground_truth": (
            "Active eligibility on the date of service, a valid prior authorization "
            "number if applicable, a diagnosis code supporting medical necessity, "
            "and confirmation the claim has not already been submitted."
        ),
    },
    {
        "question": "Does physical therapy always require prior authorization?",
        "ground_truth": (
            "Physical therapy generally does not require authorization for the "
            "initial visit allotment, but many plans require authorization renewal "
            "after 12 visits."
        ),
    },
    {
        "question": "What is the average cost of an MRI for a knee injury?",
        "ground_truth": (
            "This information is not present in the reference material, so the "
            "assistant should state it does not have this information rather than "
            "guessing."
        ),
    },
]


def run_pipeline_on_test_set(assistant: HealthcareRagAssistant):
    """Run every test question through the RAG pipeline and collect results."""
    questions, answers, contexts, ground_truths = [], [], [], []

    for item in TEST_QUESTIONS:
        logger.info(f"Running: {item['question']}")
        result = assistant.answer(item["question"])

        questions.append(item["question"])
        answers.append(result.answer)
        contexts.append([chunk.page_content for chunk in result.retrieved_chunks])
        ground_truths.append(item["ground_truth"])

    return Dataset.from_dict(
        {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        }
    )


def main():
    settings = get_settings()
    assistant = HealthcareRagAssistant()

    dataset = run_pipeline_on_test_set(assistant)

    # RAGAS defaults to OpenAI for its judge LLM and embeddings unless told
    # otherwise. We explicitly point it at the free Gemini model and the
    # same local embeddings used elsewhere, so evaluation stays $0-cost.
    judge_llm = LangchainLLMWrapper(
        ChatGoogleGenerativeAI(
            model=settings.gemini_model,
            google_api_key=settings.gemini_api_key,
        )
    )
    judge_embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name=settings.embedding_model)
    )

    logger.info("Running RAGAS evaluation (this calls the free Gemini API for each metric)...")
    # Gemini's free tier has a low requests-per-minute limit. RAGAS runs many
    # concurrent judge calls by default, which blows through that limit and
    # causes timeouts. max_workers=1 forces calls to run one at a time, and
    # a longer timeout + more retries gives Gemini room to recover from
    # rate-limit backoff instead of failing outright.
    run_config = RunConfig(timeout=120, max_retries=5, max_workers=1)

    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision],
        llm=judge_llm,
        embeddings=judge_embeddings,
        run_config=run_config,
    )

    scores = result.to_pandas()
    summary = {
        "faithfulness_avg": float(scores["faithfulness"].mean()),
        "answer_relevancy_avg": float(scores["answer_relevancy"].mean()),
        "context_precision_avg": float(scores["context_precision"].mean()),
        "hallucination_rate_pct": round((1 - scores["faithfulness"].mean()) * 100, 2),
        "num_questions_tested": len(TEST_QUESTIONS),
    }

    print("\n" + "=" * 50)
    print("EVALUATION REPORT")
    print("=" * 50)
    for key, value in summary.items():
        print(f"{key}: {value}")
    print("=" * 50)

    with open("eval_report.json", "w") as f:
        json.dump(summary, f, indent=2)

    scores.to_csv("eval_detailed_results.csv", index=False)

    logger.info(
        "Saved eval_report.json and eval_detailed_results.csv. "
        "Use the hallucination_rate_pct number in your README / LinkedIn Featured post."
    )


if __name__ == "__main__":
    main()