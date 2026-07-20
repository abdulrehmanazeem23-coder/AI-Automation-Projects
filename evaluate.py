"""
evaluate.py
============

Evaluates the deployed RAG pipeline (HybridRetriever + Groq generation, as used
in main.py) with the ragas framework.

Pipeline:
    1. Generate 20 synthetic Q&A pairs relevant to a law firm's internal
       documentation (contracts, precedent memos, disclosures, compliance
       policies, etc.) using Groq as a question/reference-answer generator.
    2. Run each generated question through the actual production pipeline
       (HybridRetriever.search -> Groq answer generation, reusing the exact
       logic in main.py) to get the system's answer and retrieved contexts.
    3. Score the resulting (question, answer, contexts, reference) tuples with
       ragas: Faithfulness, Answer Relevancy, Context Precision, Context Recall.
    4. Export a clean per-question + summary CSV to evaluation_report.csv.

Usage:
    python evaluate.py
    python evaluate.py --num-questions 20 --output evaluation_report.csv

Environment variables (same as main.py / retriever.py):
    GEMINI_API_KEY, PINECONE_API_KEY, PINECONE_INDEX_NAME, PINECONE_NAMESPACE,
    GROQ_API_KEY, CORPUS_PATH
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from typing import Any

import pandas as pd
from dotenv import load_dotenv

from groq import Groq

# Reuses the exact production retrieval + generation logic rather than
# reimplementing it, so the evaluation reflects what's actually deployed.
import main as rag_app
from retriever import HybridRetriever, RetrieverError
from main import QueryRequest, load_bm25_corpus, contains_legal_terms, LEGAL_METADATA_FILTER

from langchain_groq import ChatGroq
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from ragas import evaluate, EvaluationDataset
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("evaluate")

load_dotenv()

GENERATOR_MODEL = "llama-3.3-70b-versatile"
EVALUATOR_MODEL = "llama-3.3-70b-versatile"
EMBEDDING_MODEL = "models/text-embedding-004"

DEFAULT_NUM_QUESTIONS = 20
MAX_RETRIES = 5
RETRY_BASE_DELAY_SECONDS = 2.0

QA_GENERATION_PROMPT = """You are helping build an evaluation set for a law firm's internal
knowledge-base assistant. Generate {n} realistic questions that an attorney, paralegal, or
compliance officer might ask an internal RAG system, drawn from across these document
categories: engagement letters, NDA/confidentiality templates, conflict-of-interest checks,
precedent research memos, regulatory/compliance disclosures, litigation holds, billing and
timekeeping policy, statute-of-limitations tracking, and firm governance policies.

For each question, also provide a concise "reference" answer (2-4 sentences) representing
the ideal, factually correct answer such a system should give if the relevant internal
document exists and is retrieved correctly. Vary question phrasing and difficulty (some
direct lookups, some requiring synthesis across a document).

Respond with ONLY a JSON array of exactly {n} objects, each with keys "question" and
"reference". No preamble, no markdown fences, no trailing commentary.
"""


class EvaluationError(Exception):
    """Raised for unrecoverable evaluation errors."""


def with_retries(fn, *args, retries: int = MAX_RETRIES, description: str = "operation", **kwargs):
    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            delay = RETRY_BASE_DELAY_SECONDS * (2 ** (attempt - 1))
            logger.warning(
                "%s failed (attempt %d/%d): %s. Retrying in %.1fs...",
                description, attempt, retries, exc, delay,
            )
            time.sleep(delay)
    raise EvaluationError(f"{description} failed after {retries} attempts") from last_exc


# ------------------------------------------------------------------------------
# Step 1: synthetic test-set generation
# ------------------------------------------------------------------------------

def extract_json_array(raw_text: str) -> list[dict[str, str]]:
    """Best-effort extraction of a JSON array from an LLM response that may
    include stray preamble/markdown fences despite instructions not to."""
    text = raw_text.strip()
    text = re.sub(r"^```(?:json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        raise EvaluationError("No JSON array found in question-generation response.")
    return json.loads(match.group(0))


def generate_test_qa_pairs(groq_client: Groq, n: int = DEFAULT_NUM_QUESTIONS) -> list[dict[str, str]]:
    prompt = QA_GENERATION_PROMPT.format(n=n)

    def _call() -> list[dict[str, str]]:
        completion = groq_client.chat.completions.create(
            model=GENERATOR_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )
        content = completion.choices[0].message.content or ""
        pairs = extract_json_array(content)
        cleaned = [
            {"question": p["question"].strip(), "reference": p["reference"].strip()}
            for p in pairs
            if isinstance(p, dict) and p.get("question") and p.get("reference")
        ]
        if len(cleaned) < n:
            raise EvaluationError(f"Only got {len(cleaned)}/{n} valid Q&A pairs from generator.")
        return cleaned[:n]

    return with_retries(_call, description="synthetic Q&A pair generation")


# ------------------------------------------------------------------------------
# Step 2: run the production pipeline for each question
# ------------------------------------------------------------------------------

def run_pipeline_for_question(question: str) -> dict[str, Any]:
    """Mirrors main.query()'s logic: applies the same legal-term filter, runs
    the same HybridRetriever, and calls the same Groq answer generation —
    calling main.query() directly so the eval reflects the deployed behavior."""
    filters = LEGAL_METADATA_FILTER if contains_legal_terms(question) else None

    try:
        chunks = rag_app.retriever.search(question, filters=filters)
    except RetrieverError as exc:
        raise EvaluationError(f"Retrieval failed for question '{question}': {exc}") from exc

    response = rag_app.query(QueryRequest(question=question))

    contexts = [c.get("text", "") for c in chunks if c.get("text")]
    return {"answer": response.answer, "contexts": contexts}


# ------------------------------------------------------------------------------
# Step 3: ragas evaluation
# ------------------------------------------------------------------------------

def build_ragas_dataset(rows: list[dict[str, Any]]) -> EvaluationDataset:
    samples = [
        {
            "user_input": row["question"],
            "response": row["answer"],
            "retrieved_contexts": row["contexts"] if row["contexts"] else [""],
            "reference": row["reference"],
        }
        for row in rows
    ]
    return EvaluationDataset.from_list(samples)


def run_ragas_evaluation(dataset: EvaluationDataset) -> pd.DataFrame:
    groq_api_key = os.environ.get("GROQ_API_KEY")
    if not groq_api_key:
        raise EvaluationError("GROQ_API_KEY environment variable is not set.")

    evaluator_llm = LangchainLLMWrapper(
        ChatGroq(model=EVALUATOR_MODEL, api_key=groq_api_key, temperature=0.0)
    )
    evaluator_embeddings = LangchainEmbeddingsWrapper(
        GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
    )

    metrics = [Faithfulness(), AnswerRelevancy(), ContextPrecision(), ContextRecall()]

    try:
        result = evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=evaluator_llm,
            embeddings=evaluator_embeddings,
        )
    except Exception as exc:
        raise EvaluationError(f"ragas evaluation failed: {exc}") from exc

    return result.to_pandas()


# ------------------------------------------------------------------------------
# Step 4: CSV export
# ------------------------------------------------------------------------------

METRIC_COLUMNS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]


def export_report(df: pd.DataFrame, output_path: str) -> None:
    present_metrics = [c for c in METRIC_COLUMNS if c in df.columns]
    if not present_metrics:
        raise EvaluationError("No expected ragas metric columns found in results.")

    display_cols = ["user_input", "response", "reference"] + present_metrics
    display_cols = [c for c in display_cols if c in df.columns]
    report_df = df[display_cols].copy()
    report_df[present_metrics] = report_df[present_metrics].round(4)
    report_df.rename(
        columns={"user_input": "question", "response": "answer", "reference": "ground_truth"},
        inplace=True,
    )

    summary = {col: "AVERAGE" for col in ["question", "answer", "ground_truth"] if col in report_df.columns}
    for metric in present_metrics:
        summary[metric] = round(report_df[metric].mean(), 4)
    report_df = pd.concat([report_df, pd.DataFrame([summary])], ignore_index=True)

    report_df.to_csv(output_path, index=False)
    logger.info("Wrote evaluation report to %s (%d questions + summary row).", output_path, len(df))


# ------------------------------------------------------------------------------
# Orchestration
# ------------------------------------------------------------------------------

def initialize_pipeline() -> None:
    """Populates main.py's module-level retriever/groq_client globals, the
    same objects main.py's @app.on_event('startup') would set up — done here
    manually since we're not running under uvicorn."""
    index_name = os.environ.get("PINECONE_INDEX_NAME")
    if not index_name:
        raise EvaluationError("PINECONE_INDEX_NAME environment variable is not set.")
    groq_api_key = os.environ.get("GROQ_API_KEY")
    if not groq_api_key:
        raise EvaluationError("GROQ_API_KEY environment variable is not set.")

    corpus_path = os.environ.get("CORPUS_PATH", "corpus.jsonl")
    namespace = os.environ.get("PINECONE_NAMESPACE") or None

    try:
        bm25_corpus = load_bm25_corpus(corpus_path)
        rag_app.retriever = HybridRetriever(
            pinecone_index_name=index_name,
            bm25_corpus=bm25_corpus,
            namespace=namespace,
            final_top_k=8,
        )
        rag_app.groq_client = Groq(api_key=groq_api_key)
    except RetrieverError as exc:
        raise EvaluationError(f"Failed to initialize retrieval pipeline: {exc}") from exc


def run(num_questions: int, output_path: str) -> None:
    initialize_pipeline()

    generation_client = Groq(api_key=os.environ["GROQ_API_KEY"])
    logger.info("Generating %d synthetic Q&A pairs...", num_questions)
    qa_pairs = generate_test_qa_pairs(generation_client, n=num_questions)

    rows: list[dict[str, Any]] = []
    for i, pair in enumerate(qa_pairs, start=1):
        question = pair["question"]
        logger.info("[%d/%d] Running pipeline for: %s", i, len(qa_pairs), question)
        try:
            result = with_retries(
                run_pipeline_for_question, question,
                description=f"pipeline run for question {i}",
            )
        except EvaluationError as exc:
            logger.error("Skipping question %d after repeated failures: %s", i, exc)
            continue
        rows.append(
            {
                "question": question,
                "reference": pair["reference"],
                "answer": result["answer"],
                "contexts": result["contexts"],
            }
        )

    if not rows:
        raise EvaluationError("No questions were successfully run through the pipeline. Aborting.")

    logger.info("Scoring %d Q&A pairs with ragas...", len(rows))
    dataset = build_ragas_dataset(rows)
    results_df = run_ragas_evaluation(dataset)

    export_report(results_df, output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the RAG pipeline with ragas.")
    parser.add_argument("--num-questions", type=int, default=DEFAULT_NUM_QUESTIONS, help="Number of synthetic Q&A pairs to generate.")
    parser.add_argument("--output", default="evaluation_report.csv", help="Path to write the CSV report.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        run(args.num_questions, args.output)
    except EvaluationError as exc:
        logger.error("Evaluation failed: %s", exc)
        return 1
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unexpected error during evaluation: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
