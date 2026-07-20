"""
retriever.py
=============

Hybrid retrieval for the RAG knowledge base: combines dense vector search
(Pinecone, Gemini gemini-embedding-001) with sparse lexical search
(LangChain's BM25Retriever), merged via Reciprocal Rank Fusion (RRF).

Usage:
    from retriever import HybridRetriever

    retriever = HybridRetriever(
        pinecone_index_name="my-rag-index",
        bm25_corpus=documents,  # list[langchain_core.documents.Document]
    )
    results = retriever.search("What was Q3 revenue growth?", filters={"document_type": "pdf"})

Environment variables:
    GEMINI_API_KEY        Google AI Studio / Gemini API key
    PINECONE_API_KEY      Pinecone API key
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any

from dotenv import load_dotenv

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from google import genai
from google.genai import types
from pinecone import Pinecone

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("retriever")

load_dotenv()

EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_DIM = 768

DEFAULT_DENSE_TOP_K = 20
DEFAULT_SPARSE_TOP_K = 20
DEFAULT_FINAL_TOP_K = 8
DEFAULT_RRF_K = 60  # standard RRF smoothing constant


class RetrieverError(Exception):
    """Raised for unrecoverable retrieval errors."""


@dataclass
class RankedChunk:
    """A chunk with its origin ranks and final fused score."""

    chunk_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    dense_rank: int | None = None
    sparse_rank: int | None = None
    dense_score: float | None = None
    sparse_score: float | None = None
    rrf_score: float = 0.0

    def to_result(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "metadata": self.metadata,
            "rrf_score": round(self.rrf_score, 6),
            "dense_rank": self.dense_rank,
            "sparse_rank": self.sparse_rank,
            "dense_score": self.dense_score,
            "sparse_score": self.sparse_score,
        }


class HybridRetriever:
    """
    Combines Pinecone dense vector search with a local BM25 sparse retriever,
    fusing the two ranked lists with Reciprocal Rank Fusion (RRF).
    """

    def __init__(
        self,
        pinecone_index_name: str,
        bm25_corpus: list[Document],
        namespace: str | None = None,
        dense_top_k: int = DEFAULT_DENSE_TOP_K,
        sparse_top_k: int = DEFAULT_SPARSE_TOP_K,
        final_top_k: int = DEFAULT_FINAL_TOP_K,
        rrf_k: int = DEFAULT_RRF_K,
        text_metadata_key: str = "original_chunk_text",
        id_metadata_key: str | None = None,
    ) -> None:
        """
        Args:
            pinecone_index_name: Name of an existing 768-dim Pinecone serverless index.
            bm25_corpus: Documents to build the local BM25 index over. Each Document's
                `page_content` should match the text stored in Pinecone metadata
                (`text_metadata_key`) so results from both retrievers can be identified
                and deduplicated consistently.
            namespace: Optional Pinecone namespace to query within.
            dense_top_k / sparse_top_k: How many candidates to pull from each retriever
                before fusion.
            final_top_k: How many fused results to return.
            rrf_k: RRF smoothing constant (higher = flatter weighting across ranks).
            text_metadata_key: Metadata field in Pinecone holding the chunk's raw text.
            id_metadata_key: Optional metadata field to use as a stable chunk id for
                fusion matching across the two retrievers. If None, falls back to
                the Pinecone vector id / a hash of the BM25 document content.
        """
        if dense_top_k <= 0 or sparse_top_k <= 0 or final_top_k <= 0:
            raise ValueError("dense_top_k, sparse_top_k, and final_top_k must all be positive.")

        self.namespace = namespace
        self.dense_top_k = dense_top_k
        self.sparse_top_k = sparse_top_k
        self.final_top_k = final_top_k
        self.rrf_k = rrf_k
        self.text_metadata_key = text_metadata_key
        self.id_metadata_key = id_metadata_key

        self._gemini_client = self._configure_gemini()
        self._index = self._connect_pinecone(pinecone_index_name)
        self._bm25 = self._build_bm25(bm25_corpus)

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    @staticmethod
    def _configure_gemini() -> genai.Client:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RetrieverError("GEMINI_API_KEY environment variable is not set.")
        return genai.Client(api_key=api_key)

    @staticmethod
    def _connect_pinecone(index_name: str):
        api_key = os.environ.get("PINECONE_API_KEY")
        if not api_key:
            raise RetrieverError("PINECONE_API_KEY environment variable is not set.")
        try:
            pc = Pinecone(api_key=api_key)
            existing = {idx["name"] for idx in pc.list_indexes()}
            if index_name not in existing:
                raise RetrieverError(f"Pinecone index '{index_name}' does not exist.")
            index = pc.Index(index_name)
            stats = index.describe_index_stats()
            dim = stats.get("dimension")
            if dim is not None and dim != EMBEDDING_DIM:
                raise RetrieverError(
                    f"Pinecone index '{index_name}' has dimension {dim}, expected {EMBEDDING_DIM}."
                )
            return index
        except RetrieverError:
            raise
        except Exception as exc:
            raise RetrieverError(f"Failed to connect to Pinecone index '{index_name}': {exc}") from exc

    def _build_bm25(self, corpus: list[Document]) -> BM25Retriever | None:
        if not corpus:
            logger.warning("BM25 corpus is empty; sparse retrieval will return no results.")
            return None
        try:
            retriever = BM25Retriever.from_documents(corpus)
            retriever.k = self.sparse_top_k
            return retriever
        except Exception as exc:
            raise RetrieverError(f"Failed to build BM25 retriever: {exc}") from exc

    # ------------------------------------------------------------------
    # Query embedding
    # ------------------------------------------------------------------

    def _embed_query(self, query: str) -> list[float]:
        try:
            result = self._gemini_client.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=query,
                config=types.EmbedContentConfig(
                    outputDimensionality=EMBEDDING_DIM,
                    taskType="RETRIEVAL_QUERY",
                ),
            )
            if not result.embeddings or not result.embeddings[0].values:
                raise RetrieverError("Empty embedding returned from Gemini for query.")
            return result.embeddings[0].values
        except RetrieverError:
            raise
        except Exception as exc:
            raise RetrieverError(f"Failed to embed query: {exc}") from exc

    # ------------------------------------------------------------------
    # Dense (Pinecone) retrieval
    # ------------------------------------------------------------------

    def _dense_search(self, query: str, filters: dict[str, Any] | None) -> list[RankedChunk]:
        try:
            query_vector = self._embed_query(query)
            response = self._index.query(
                vector=query_vector,
                top_k=self.dense_top_k,
                include_metadata=True,
                filter=filters or None,
                namespace=self.namespace,
            )
        except Exception as exc:
            raise RetrieverError(f"Pinecone query failed: {exc}") from exc

        ranked: list[RankedChunk] = []
        for rank, match in enumerate(response.get("matches", []), start=1):
            metadata = match.get("metadata", {}) or {}
            chunk_id = self._resolve_id(match_id=match.get("id"), metadata=metadata)
            ranked.append(
                RankedChunk(
                    chunk_id=chunk_id,
                    text=metadata.get(self.text_metadata_key, ""),
                    metadata=metadata,
                    dense_rank=rank,
                    dense_score=match.get("score"),
                )
            )
        return ranked

    # ------------------------------------------------------------------
    # Sparse (BM25) retrieval
    # ------------------------------------------------------------------

    def _sparse_search(self, query: str) -> list[RankedChunk]:
        if self._bm25 is None:
            return []
        try:
            docs = self._bm25.invoke(query)
        except Exception as exc:
            raise RetrieverError(f"BM25 retrieval failed: {exc}") from exc

        ranked: list[RankedChunk] = []
        for rank, doc in enumerate(docs[: self.sparse_top_k], start=1):
            metadata = doc.metadata or {}
            chunk_id = self._resolve_id(match_id=None, metadata=metadata, fallback_text=doc.page_content)
            ranked.append(
                RankedChunk(
                    chunk_id=chunk_id,
                    text=doc.page_content,
                    metadata=metadata,
                    sparse_rank=rank,
                )
            )
        return ranked

    def _resolve_id(
        self,
        match_id: str | None,
        metadata: dict[str, Any],
        fallback_text: str | None = None,
    ) -> str:
        if self.id_metadata_key and self.id_metadata_key in metadata:
            return str(metadata[self.id_metadata_key])
        if match_id:
            return match_id
        # Stable fallback: source_file + chunk_index if present, else a hash of the text.
        source = metadata.get("source_file")
        chunk_index = metadata.get("chunk_index")
        if source is not None and chunk_index is not None:
            return f"{source}::{chunk_index}"
        basis = fallback_text or ""
        return f"bm25::{hash(basis)}"

    # ------------------------------------------------------------------
    # Reciprocal Rank Fusion
    # ------------------------------------------------------------------

    def _fuse(self, dense: list[RankedChunk], sparse: list[RankedChunk]) -> list[RankedChunk]:
        fused: dict[str, RankedChunk] = {}

        for chunk in dense:
            fused[chunk.chunk_id] = chunk
            chunk.rrf_score += 1.0 / (self.rrf_k + chunk.dense_rank)

        for chunk in sparse:
            existing = fused.get(chunk.chunk_id)
            if existing is not None:
                existing.sparse_rank = chunk.sparse_rank
                existing.sparse_score = chunk.sparse_score
                if not existing.text:
                    existing.text = chunk.text
                existing.rrf_score += 1.0 / (self.rrf_k + chunk.sparse_rank)
            else:
                chunk.rrf_score += 1.0 / (self.rrf_k + chunk.sparse_rank)
                fused[chunk.chunk_id] = chunk

        return sorted(fused.values(), key=lambda c: c.rrf_score, reverse=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(self, query: str, filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """
        Run hybrid search for `query`, optionally restricted by Pinecone metadata `filters`
        (e.g. {"document_type": {"$eq": "pdf"}} or {"document_type": "pdf"}).

        Returns the top `final_top_k` fused results, each a dict with:
            chunk_id, text, metadata, rrf_score, dense_rank, sparse_rank,
            dense_score, sparse_score.
        """
        if not query or not query.strip():
            raise ValueError("query must be a non-empty string.")

        try:
            dense_results = self._dense_search(query, filters)
        except RetrieverError as exc:
            logger.error("Dense retrieval failed, continuing with sparse only: %s", exc)
            dense_results = []

        try:
            sparse_results = self._sparse_search(query)
        except RetrieverError as exc:
            logger.error("Sparse retrieval failed, continuing with dense only: %s", exc)
            sparse_results = []

        if not dense_results and not sparse_results:
            raise RetrieverError("Both dense and sparse retrieval failed or returned no results.")

        fused = self._fuse(dense_results, sparse_results)
        top_results = fused[: self.final_top_k]

        logger.info(
            "search('%s'): dense=%d sparse=%d fused=%d returned=%d",
            query, len(dense_results), len(sparse_results), len(fused), len(top_results),
        )

        return [chunk.to_result() for chunk in top_results]


if __name__ == "__main__":
    # Minimal smoke-test / usage example.
    import argparse

    parser = argparse.ArgumentParser(description="Query the hybrid RAG retriever.")
    parser.add_argument("query", help="Search query text.")
    parser.add_argument("--index-name", required=True, help="Pinecone index name.")
    parser.add_argument("--namespace", default=None, help="Optional Pinecone namespace.")
    args = parser.parse_args()

    # In production the BM25 corpus should be loaded from your document store
    # (e.g. re-hydrated from the same source used in ingest.py) rather than left empty.
    retriever = HybridRetriever(pinecone_index_name=args.index_name, bm25_corpus=[], namespace=args.namespace)
    for result in retriever.search(args.query):
        print(result)
