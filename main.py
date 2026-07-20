"""
main.py
=======

FastAPI backend for the RAG knowledge base. Exposes a single `/query` endpoint
that:
    1. Runs hybrid retrieval (HybridRetriever: Pinecone dense + BM25 sparse, RRF-fused).
    2. Applies a legal-document metadata filter when the question uses legal terminology.
    3. Sends the top 8 fused chunks to Groq (llama-3.3-70b-versatile) as grounding context,
       instructing the model to answer only from that context with inline citations.
    4. Returns a fixed refusal message if the model determines the question is unrelated
       to the retrieved context.
    5. Appends a low-confidence disclaimer if the top chunk's relevance score is weak.

Run locally:
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload

Deploy on Railway:
    Railway auto-detects `requirements.txt`. Set the start command to:
        uvicorn main:app --host 0.0.0.0 --port $PORT
    (or add a Procfile: `web: uvicorn main:app --host 0.0.0.0 --port $PORT`)

Environment variables (see .env.example):
    GEMINI_API_KEY          Gemini API key (used inside HybridRetriever for query embeddings)
    PINECONE_API_KEY        Pinecone API key
    PINECONE_INDEX_NAME     Name of the existing Pinecone index to query
    PINECONE_NAMESPACE      Optional Pinecone namespace
    GROQ_API_KEY            Groq API key
    CORPUS_PATH             Path to a JSONL file used to build the local BM25 corpus
                             (one JSON object per line: {"text": ..., "metadata": {...}})
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from langchain_core.documents import Document

from groq import Groq

from retriever import HybridRetriever, RetrieverError

# ------------------------------------------------------------------------------
# Configuration & logging
# ------------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("main")

load_dotenv()

GROQ_MODEL = "llama-3.3-70b-versatile"
RELEVANCE_THRESHOLD = 0.55
NOT_RELEVANT_SENTINEL = "NOT_RELEVANT_TO_CONTEXT"

REFUSAL_MESSAGE = (
    "I can only answer questions about the firm's internal documentation. "
    "Please rephrase or ask your manager."
)
LOW_CONFIDENCE_NOTE = (
    "Note: This answer is based on partial matches. Please verify with the source documents."
)

LEGAL_TERMS = {
    "statute", "statutes", "precedent", "precedents", "disclosure", "disclosures",
    "liability", "indemnify", "indemnification", "clause", "jurisdiction",
    "litigation", "plaintiff", "defendant", "tort", "breach of contract",
    "compliance", "regulatory filing", "affidavit", "arbitration", "counsel",
}
LEGAL_METADATA_FILTER = {"document_type": {"$in": ["legal_template", "precedent"]}}

SYSTEM_PROMPT_TEMPLATE = """You are an internal knowledge-base assistant. Answer the user's question
using ONLY the context provided below. Do not use outside knowledge, and do not guess.

Rules:
- Every factual claim in your answer must include an inline citation in this exact format:
  [Source: <filename>, page <page_number>]
  If a chunk has no page number, cite it as [Source: <filename>].
- If the provided context does not contain information relevant to answering the question,
  respond with exactly this token and nothing else: {sentinel}
- Do not fabricate sources, filenames, or page numbers that are not present in the context.
- Be concise and direct.

Context:
{context}
"""


# ------------------------------------------------------------------------------
# Request / response models
# ------------------------------------------------------------------------------

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="The user's natural-language question.")


class QueryResponse(BaseModel):
    answer: str
    used_legal_filter: bool
    top_relevance_score: float | None
    chunks_used: int


# ------------------------------------------------------------------------------
# Corpus loading (for the local BM25 index)
# ------------------------------------------------------------------------------

def load_bm25_corpus(path: str) -> list[Document]:
    """
    Loads the local BM25 corpus from a JSONL file, where each line is:
        {"text": "...", "metadata": {"source_file": ..., "page_number": ..., "chunk_index": ..., ...}}

    This file should be produced/kept in sync with the same chunks ingested into
    Pinecone by ingest.py, so BM25 and dense results reference the same chunk universe.
    """
    if not path or not os.path.exists(path):
        logger.warning(
            "BM25 corpus file '%s' not found. Starting with an empty sparse corpus "
            "(hybrid search will run dense-only).", path,
        )
        return []

    documents: list[Document] = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    documents.append(
                        Document(
                            page_content=record["text"],
                            metadata=record.get("metadata", {}),
                        )
                    )
                except (json.JSONDecodeError, KeyError) as exc:
                    logger.warning("Skipping malformed corpus line %d in '%s': %s", line_number, path, exc)
    except OSError as exc:
        logger.error("Failed to read BM25 corpus file '%s': %s", path, exc)
        return []

    logger.info("Loaded %d documents into local BM25 corpus.", len(documents))
    return documents


# ------------------------------------------------------------------------------
# App setup
# ------------------------------------------------------------------------------

app = FastAPI(title="RAG Knowledge Base API", version="1.0.0")

retriever: HybridRetriever | None = None
groq_client: Groq | None = None


@app.on_event("startup")
def startup() -> None:
    global retriever, groq_client

    index_name = os.environ.get("PINECONE_INDEX_NAME")
    if not index_name:
        raise RuntimeError("PINECONE_INDEX_NAME environment variable is not set.")

    groq_api_key = os.environ.get("GROQ_API_KEY")
    if not groq_api_key:
        raise RuntimeError("GROQ_API_KEY environment variable is not set.")

    corpus_path = os.environ.get("CORPUS_PATH", "corpus.jsonl")
    namespace = os.environ.get("PINECONE_NAMESPACE") or None

    try:
        bm25_corpus = load_bm25_corpus(corpus_path)
        retriever = HybridRetriever(
            pinecone_index_name=index_name,
            bm25_corpus=bm25_corpus,
            namespace=namespace,
            final_top_k=8,
        )
        groq_client = Groq(api_key=groq_api_key)
        logger.info("Startup complete: retriever and Groq client initialized.")
    except RetrieverError as exc:
        logger.exception("Failed to initialize HybridRetriever during startup.")
        raise RuntimeError(f"Startup failed: {exc}") from exc


# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------

def contains_legal_terms(question: str) -> bool:
    lowered = question.lower()
    return any(term in lowered for term in LEGAL_TERMS)


def format_context(chunks: list[dict[str, Any]]) -> str:
    blocks = []
    for i, chunk in enumerate(chunks, start=1):
        metadata = chunk.get("metadata", {})
        source_file = metadata.get("source_file", "unknown")
        page_number = metadata.get("page_number")
        page_label = f"page {page_number}" if page_number not in (None, -1) else "no page number"
        text = chunk.get("text", "")
        blocks.append(f"[Chunk {i} | Source: {source_file}, {page_label}]\n{text}")
    return "\n\n".join(blocks)


def highest_relevance_score(chunks: list[dict[str, Any]]) -> float | None:
    scores = [c.get("dense_score") for c in chunks if c.get("dense_score") is not None]
    if not scores:
        return None
    return max(scores)


def call_groq(question: str, context: str) -> str:
    if groq_client is None:
        raise HTTPException(status_code=503, detail="Groq client is not initialized.")

    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(sentinel=NOT_RELEVANT_SENTINEL, context=context)
    try:
        completion = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ],
            temperature=0.0,
        )
        content = completion.choices[0].message.content
        if not content or not content.strip():
            raise HTTPException(status_code=502, detail="Empty response from Groq.")
        return content.strip()
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Groq API call failed.")
        raise HTTPException(status_code=502, detail=f"Groq API call failed: {exc}") from exc


# ------------------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------------------

@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    if retriever is None:
        raise HTTPException(status_code=503, detail="Retriever is not initialized.")

    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="question must not be empty.")

    use_legal_filter = contains_legal_terms(question)
    filters = LEGAL_METADATA_FILTER if use_legal_filter else None

    try:
        chunks = retriever.search(question, filters=filters)
    except RetrieverError as exc:
        logger.error("Retrieval failed for question '%s': %s", question, exc)
        raise HTTPException(status_code=502, detail=f"Retrieval failed: {exc}") from exc

    if not chunks:
        return QueryResponse(
            answer=REFUSAL_MESSAGE,
            used_legal_filter=use_legal_filter,
            top_relevance_score=None,
            chunks_used=0,
        )

    context = format_context(chunks)
    answer = call_groq(question, context)

    if NOT_RELEVANT_SENTINEL in answer:
        return QueryResponse(
            answer=REFUSAL_MESSAGE,
            used_legal_filter=use_legal_filter,
            top_relevance_score=highest_relevance_score(chunks),
            chunks_used=len(chunks),
        )

    top_score = highest_relevance_score(chunks)
    if top_score is not None and top_score < RELEVANCE_THRESHOLD:
        answer = f"{answer}\n\n{LOW_CONFIDENCE_NOTE}"

    return QueryResponse(
        answer=answer,
        used_legal_filter=use_legal_filter,
        top_relevance_score=top_score,
        chunks_used=len(chunks),
    )
