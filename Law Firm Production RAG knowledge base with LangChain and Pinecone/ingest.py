"""
ingest.py
=========

Production ingestion pipeline for a RAG knowledge base implementing
Anthropic's "Contextual Retrieval" technique (https://www.anthropic.com/news/contextual-retrieval).

Pipeline:
    1. Load PDF/DOCX files from a Google Drive folder.
    2. Split each document into chunks (RecursiveCharacterTextSplitter, ~512 tokens, 50 overlap).
    3. For each chunk, ask Gemini (gemini-2.5-flash) to generate a short contextual
       prefix describing the chunk's place within the whole document.
    4. Prepend that prefix to the chunk text.
    5. Embed the resulting text with Gemini's gemini-embedding-001 model.
    6. Upsert vectors + metadata into a Pinecone serverless index.

Usage:
    python ingest.py --drive-folder-id <FOLDER_ID> --index-name my-rag-index

Environment variables (see .env.example):
    GOOGLE_APPLICATION_CREDENTIALS   Path to a Google service-account JSON key
                                      (Drive API scope: https://www.googleapis.com/auth/drive.readonly)
    GEMINI_API_KEY                   Google AI Studio / Gemini API key
    PINECONE_API_KEY                 Pinecone API key
    PINECONE_CLOUD                   e.g. "aws"   (for serverless spec)
    PINECONE_REGION                  e.g. "us-east-1"
"""

from __future__ import annotations

import argparse
import io
import logging
import os
import sys
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Iterable

from dotenv import load_dotenv

# --- LangChain -------------------------------------------------------------
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# --- Google Drive ------------------------------------------------------------
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google.genai import types

# --- Document parsers --------------------------------------------------------
from pypdf import PdfReader
import docx as python_docx

# --- Gemini (Google GenAI) ---------------------------------------------------
from google import genai
from google.genai import errors as genai_errors

# --- Pinecone ------------------------------------------------------------------
from pinecone import Pinecone, ServerlessSpec

# ------------------------------------------------------------------------------
# Configuration & logging
# ------------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("ingest")

load_dotenv()

DRIVE_MIME_TYPES = {
    "application/pdf": "pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
}

GENERATION_MODEL = "gemini-2.5-flash"
EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_DIM = 768  # Forced output dimensionality to match Pinecone

CHUNK_SIZE_TOKENS = 512
CHUNK_OVERLAP_TOKENS = 50

MAX_RETRIES = 5
RETRY_BASE_DELAY_SECONDS = 2.0
DEFAULT_MAX_CONTEXT_CHARS = int(os.environ.get("INGEST_MAX_CONTEXT_CHARS", "12000"))
DEFAULT_GEMINI_CALL_DELAY_SECONDS = float(os.environ.get("INGEST_GEMINI_CALL_DELAY_SECONDS", "4"))
DEFAULT_EMBEDDING_BATCH_SIZE = int(os.environ.get("INGEST_EMBEDDING_BATCH_SIZE", "32"))
DEFAULT_MAX_CHUNKS = int(os.environ.get("INGEST_MAX_CHUNKS", "0"))
DEFAULT_SKIP_CONTEXTUALIZATION = os.environ.get("INGEST_SKIP_CONTEXTUALIZATION", "1").lower() in {
    "1",
    "true",
    "yes",
}

CONTEXT_PROMPT_TEMPLATE = """<document>
{document_text}
</document>
Here is the chunk we want to situate within the whole document:
<chunk>
{chunk_text}
</chunk>
Generate a 2-sentence contextual prefix that explains what this chunk is about
within the context of the overall document. Answer only with the 2 sentences,
no preamble, no labels."""


class IngestionError(Exception):
    """Raised for unrecoverable pipeline errors."""


@dataclass
class SourceFile:
    file_id: str
    name: str
    mime_type: str
    document_type: str
    raw_bytes: bytes


@dataclass
class PageText:
    page_number: int | None
    text: str


@dataclass
class ChunkRecord:
    source_file: str
    document_type: str
    page_number: int | None
    chunk_index: int
    original_chunk_text: str
    contextual_prefix: str = ""
    embedding: list[float] = field(default_factory=list)

    @property
    def contextualized_text(self) -> str:
        return f"{self.contextual_prefix}\n\n{self.original_chunk_text}".strip()


# ------------------------------------------------------------------------------
# Retry helper
# ------------------------------------------------------------------------------

# Client errors that will NEVER succeed on retry (bad model name, bad/missing
# API key, permission denied, malformed request). Retrying these just burns
# time on exponential backoff for a call that is guaranteed to fail again.
# 429 (rate limit) is deliberately excluded here since that IS worth retrying.
NON_RETRYABLE_STATUS_CODES = {400, 401, 403, 404}


def _is_permanent_failure(exc: Exception) -> bool:
    if isinstance(exc, genai_errors.APIError):
        return exc.code in NON_RETRYABLE_STATUS_CODES
    return False


def with_retries(fn, *args, retries: int = MAX_RETRIES, description: str = "operation", **kwargs):
    """Call fn(*args, **kwargs) with exponential backoff on failure.

    Permanent client errors (e.g. 404 model-not-found, 401 bad API key) are
    raised immediately without retrying, since backing off won't fix a
    misconfigured model name or credential.
    """
    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 - intentional broad catch for retry wrapper
            last_exc = exc
            if _is_permanent_failure(exc):
                logger.error(
                    "%s failed with a permanent error (not retrying): %s",
                    description, exc,
                )
                raise IngestionError(f"{description} failed permanently: {exc}") from exc
            delay = RETRY_BASE_DELAY_SECONDS * (2 ** (attempt - 1))
            logger.warning(
                "%s failed (attempt %d/%d): %s. Retrying in %.1fs...",
                description, attempt, retries, exc, delay,
            )
            time.sleep(delay)
    raise IngestionError(f"{description} failed after {retries} attempts") from last_exc


# ------------------------------------------------------------------------------
# Google Drive loading
# ------------------------------------------------------------------------------

def build_drive_service():
    creds_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if not creds_path or not os.path.exists(creds_path):
        raise IngestionError(
            "GOOGLE_APPLICATION_CREDENTIALS must point to a valid service-account JSON file."
        )
    credentials = service_account.Credentials.from_service_account_file(
        creds_path, scopes=["https://www.googleapis.com/auth/drive.readonly"]
    )
    return build("drive", "v3", credentials=credentials, cache_discovery=False)


def list_drive_files(drive_service, folder_id: str) -> list[dict[str, Any]]:
    query = (
        f"'{folder_id}' in parents and trashed = false and "
        "(mimeType='application/pdf' or "
        "mimeType='application/vnd.openxmlformats-officedocument.wordprocessingml.document')"
    )
    files: list[dict[str, Any]] = []
    page_token = None
    try:
        while True:
            response = drive_service.files().list(
                q=query,
                spaces="drive",
                fields="nextPageToken, files(id, name, mimeType)",
                pageToken=page_token,
            ).execute()
            files.extend(response.get("files", []))
            page_token = response.get("nextPageToken")
            if not page_token:
                break
    except Exception as exc:
        raise IngestionError(f"Failed to list files in Drive folder {folder_id}: {exc}") from exc

    logger.info("Found %d candidate files in Drive folder %s", len(files), folder_id)
    return files


def download_drive_file(drive_service, file_id: str) -> bytes:
    request = drive_service.files().get_media(fileId=file_id)
    buffer = io.BytesIO()
    downloader = MediaIoBaseDownload(buffer, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    return buffer.getvalue()


def load_source_files(drive_service, folder_id: str) -> list[SourceFile]:
    raw_files = list_drive_files(drive_service, folder_id)
    sources: list[SourceFile] = []
    for f in raw_files:
        mime_type = f.get("mimeType", "")
        doc_type = DRIVE_MIME_TYPES.get(mime_type)
        if not doc_type:
            continue
        try:
            raw_bytes = with_retries(
                download_drive_file, drive_service, f["id"],
                description=f"download of '{f['name']}'",
            )
        except IngestionError as exc:
            logger.error("Skipping file '%s' — %s", f["name"], exc)
            continue
        sources.append(
            SourceFile(
                file_id=f["id"],
                name=f["name"],
                mime_type=mime_type,
                document_type=doc_type,
                raw_bytes=raw_bytes,
            )
        )
    logger.info("Successfully downloaded %d files", len(sources))
    return sources


# ------------------------------------------------------------------------------
# Text extraction
# ------------------------------------------------------------------------------

def extract_pdf_pages(raw_bytes: bytes) -> list[PageText]:
    pages: list[PageText] = []
    try:
        reader = PdfReader(io.BytesIO(raw_bytes))
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if text.strip():
                pages.append(PageText(page_number=i + 1, text=text))
    except Exception as exc:
        raise IngestionError(f"Failed to parse PDF: {exc}") from exc
    return pages


def extract_docx_pages(raw_bytes: bytes) -> list[PageText]:
    try:
        document = python_docx.Document(io.BytesIO(raw_bytes))
        full_text = "\n".join(p.text for p in document.paragraphs if p.text.strip())
    except Exception as exc:
        raise IngestionError(f"Failed to parse DOCX: {exc}") from exc
    return [PageText(page_number=None, text=full_text)]


def extract_pages(source: SourceFile) -> list[PageText]:
    if source.document_type == "pdf":
        return extract_pdf_pages(source.raw_bytes)
    if source.document_type == "docx":
        return extract_docx_pages(source.raw_bytes)
    raise IngestionError(f"Unsupported document type: {source.document_type}")


# ------------------------------------------------------------------------------
# Chunking
# ------------------------------------------------------------------------------

def build_text_splitter() -> RecursiveCharacterTextSplitter:
    try:
        import tiktoken
        encoding = tiktoken.get_encoding("cl100k_base")

        def token_len(text: str) -> int:
            return len(encoding.encode(text))

        return RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE_TOKENS,
            chunk_overlap=CHUNK_OVERLAP_TOKENS,
            length_function=token_len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
    except ImportError:
        logger.warning("tiktoken not installed; falling back to character-length chunking.")
        return RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE_TOKENS * 4,
            chunk_overlap=CHUNK_OVERLAP_TOKENS * 4,
            separators=["\n\n", "\n", ". ", " ", ""],
        )


def chunk_source(source: SourceFile, splitter: RecursiveCharacterTextSplitter) -> list[ChunkRecord]:
    pages = extract_pages(source)
    if not pages:
        logger.warning("No extractable text found in '%s'; skipping.", source.name)
        return []

    full_document_text = "\n\n".join(p.text for p in pages)
    records: list[ChunkRecord] = []
    global_chunk_index = 0

    for page in pages:
        if not page.text.strip():
            continue
        chunks = splitter.split_text(page.text)
        for chunk_text in chunks:
            if not chunk_text.strip():
                continue
            records.append(
                ChunkRecord(
                    source_file=source.name,
                    document_type=source.document_type,
                    page_number=page.page_number,
                    chunk_index=global_chunk_index,
                    original_chunk_text=chunk_text,
                )
            )
            global_chunk_index += 1

    for record in records:
        record._full_document_text = full_document_text  # type: ignore[attr-defined]

    return records


# ------------------------------------------------------------------------------
# Gemini: contextual prefix generation + embeddings
# ------------------------------------------------------------------------------

def configure_gemini() -> genai.Client:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise IngestionError("GEMINI_API_KEY environment variable is not set.")
    return genai.Client(api_key=api_key)


def wait_for_quota(delay_seconds: float) -> None:
    if delay_seconds > 0:
        time.sleep(delay_seconds)


def generate_contextual_prefix(
    chunk: ChunkRecord,
    client: genai.Client,
    max_context_chars: int,
) -> str:
    document_text = getattr(chunk, "_full_document_text", "")
    if max_context_chars > 0 and len(document_text) > max_context_chars:
        head_chars = max_context_chars // 2
        tail_chars = max_context_chars - head_chars
        document_text = (
            document_text[:head_chars]
            + "\n\n[...document truncated for quota control...]\n\n"
            + document_text[-tail_chars:]
        )

    prompt = CONTEXT_PROMPT_TEMPLATE.format(
        document_text=document_text,
        chunk_text=chunk.original_chunk_text,
    )

    def _call() -> str:
        response = client.models.generate_content(
            model=GENERATION_MODEL,
            contents=prompt,
        )
        text = (response.text or "").strip()
        if not text:
            raise IngestionError("Empty response from Gemini during contextualization.")
        return text

    return with_retries(
        _call,
        description=f"contextual prefix generation for chunk {chunk.chunk_index} of '{chunk.source_file}'",
    )


def embed_text(text: str, client: genai.Client) -> list[float]:
    return embed_texts([text], client)[0]


def embed_texts(texts: list[str], client: genai.Client) -> list[list[float]]:
    def _call() -> list[list[float]]:
        response = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=texts,
            config=types.EmbedContentConfig(outputDimensionality=EMBEDDING_DIM),
        )
        if not response.embeddings:
            raise IngestionError("Empty embedding returned from Gemini.")
        embeddings = [embedding.values for embedding in response.embeddings]
        if len(embeddings) != len(texts) or any(not values for values in embeddings):
            raise IngestionError(
                f"Gemini returned {len(embeddings)} embeddings for {len(texts)} input texts."
            )
        return embeddings

    return with_retries(_call, description="embedding generation")


def contextualize_and_embed_chunks(
    chunks: list[ChunkRecord],
    client: genai.Client,
    *,
    skip_contextualization: bool = False,
    max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
    gemini_call_delay_seconds: float = DEFAULT_GEMINI_CALL_DELAY_SECONDS,
    embedding_batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE,
) -> list[ChunkRecord]:
    processed: list[ChunkRecord] = []

    for chunk in chunks:
        try:
            if skip_contextualization:
                chunk.contextual_prefix = ""
            else:
                chunk.contextual_prefix = generate_contextual_prefix(
                    chunk,
                    client,
                    max_context_chars=max_context_chars,
                )
                wait_for_quota(gemini_call_delay_seconds)
            processed.append(chunk)
        except IngestionError as exc:
            logger.error(
                "Skipping contextualization for chunk %d of '%s' after repeated failures: %s",
                chunk.chunk_index, chunk.source_file, exc,
            )
            chunk.contextual_prefix = ""
            processed.append(chunk)

    embedded: list[ChunkRecord] = []
    effective_batch_size = max(1, embedding_batch_size)
    for batch in batched(processed, effective_batch_size):
        try:
            embeddings = embed_texts([chunk.contextualized_text for chunk in batch], client)
            for chunk, embedding in zip(batch, embeddings):
                chunk.embedding = embedding
                embedded.append(chunk)
            wait_for_quota(gemini_call_delay_seconds)
            logger.info("Embedded %d/%d chunks", len(embedded), len(chunks))
        except IngestionError as exc:
            logger.error("Skipping embedding batch of %d chunks: %s", len(batch), exc)
            continue

    logger.info("Contextualized and embedded %d/%d chunks", len(embedded), len(chunks))
    return embedded


# ------------------------------------------------------------------------------
# Pinecone upsert
# ------------------------------------------------------------------------------

def get_or_create_index(pc: Pinecone, index_name: str):
    existing = {idx["name"] for idx in pc.list_indexes()}
    if index_name not in existing:
        cloud = os.environ.get("PINECONE_CLOUD", "aws")
        region = os.environ.get("PINECONE_REGION", "us-east-1")
        logger.info("Index '%s' not found; creating serverless index (%s/%s).", index_name, cloud, region)
        try:
            pc.create_index(
                name=index_name,
                dimension=EMBEDDING_DIM,
                metric="cosine",
                spec=ServerlessSpec(cloud=cloud, region=region),
            )
            while not pc.describe_index(index_name).status.get("ready", False):
                time.sleep(1)
        except Exception as exc:
            raise IngestionError(f"Failed to create Pinecone index '{index_name}': {exc}") from exc
    return pc.Index(index_name)


def batched(iterable: list[Any], batch_size: int) -> Iterable[list[Any]]:
    for i in range(0, len(iterable), batch_size):
        yield iterable[i : i + batch_size]


def upsert_chunks(index, chunks: list[ChunkRecord], namespace: str | None = None, batch_size: int = 100) -> None:
    vectors = []
    for chunk in chunks:
        vector_id = f"{chunk.source_file}::{chunk.chunk_index}::{uuid.uuid4().hex[:8]}"
        metadata = {
            "source_file": chunk.source_file,
            "page_number": chunk.page_number if chunk.page_number is not None else -1,
            "chunk_index": chunk.chunk_index,
            "document_type": chunk.document_type,
            "original_chunk_text": chunk.original_chunk_text,
            "contextual_prefix": chunk.contextual_prefix,
        }
        vectors.append({"id": vector_id, "values": chunk.embedding, "metadata": metadata})

    total_upserted = 0
    for batch in batched(vectors, batch_size):
        def _upsert(batch=batch):
            index.upsert(vectors=batch, namespace=namespace)

        with_retries(_upsert, description=f"Pinecone upsert batch of {len(batch)}")
        total_upserted += len(batch)
        logger.info("Upserted %d/%d vectors", total_upserted, len(vectors))


# ------------------------------------------------------------------------------
# Orchestration
# ------------------------------------------------------------------------------

def run_pipeline(
    drive_folder_id: str,
    index_name: str,
    namespace: str | None = None,
    *,
    skip_contextualization: bool = DEFAULT_SKIP_CONTEXTUALIZATION,
    max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
    gemini_call_delay_seconds: float = DEFAULT_GEMINI_CALL_DELAY_SECONDS,
    embedding_batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE,
    max_chunks: int = DEFAULT_MAX_CHUNKS,
) -> None:
    clean_folder_id = drive_folder_id.split('?')[0].strip()

    pinecone_api_key = os.environ.get("PINECONE_API_KEY")
    if not pinecone_api_key:
        raise IngestionError("PINECONE_API_KEY environment variable is not set.")

    client = configure_gemini()
    pc = Pinecone(api_key=pinecone_api_key)
    index = get_or_create_index(pc, index_name)

    drive_service = build_drive_service()
    sources = load_source_files(drive_service, clean_folder_id)
    if not sources:
        logger.warning("No PDF/DOCX files found in folder %s. Nothing to ingest.", clean_folder_id)
        return

    splitter = build_text_splitter()

    all_chunks: list[ChunkRecord] = []
    for source in sources:
        try:
            chunks = chunk_source(source, splitter)
            logger.info("'%s' -> %d chunks", source.name, len(chunks))
            all_chunks.extend(chunks)
        except IngestionError as exc:
            logger.error("Failed to chunk '%s': %s", source.name, exc)
            continue

    if not all_chunks:
        logger.warning("No chunks produced from any source file. Aborting.")
        return

    if max_chunks > 0 and len(all_chunks) > max_chunks:
        logger.warning(
            "Limiting ingestion to the first %d/%d chunks because max_chunks is set.",
            max_chunks, len(all_chunks),
        )
        all_chunks = all_chunks[:max_chunks]

    logger.info(
        "Gemini quota controls: contextualization=%s, max_context_chars=%d, "
        "call_delay=%.1fs, embedding_batch_size=%d",
        "off" if skip_contextualization else "on",
        max_context_chars,
        gemini_call_delay_seconds,
        embedding_batch_size,
    )

    processed_chunks = contextualize_and_embed_chunks(
        all_chunks,
        client,
        skip_contextualization=skip_contextualization,
        max_context_chars=max_context_chars,
        gemini_call_delay_seconds=gemini_call_delay_seconds,
        embedding_batch_size=embedding_batch_size,
    )
    if not processed_chunks:
        logger.error("No chunks were successfully embedded. Aborting upsert.")
        return

    upsert_chunks(index, processed_chunks, namespace=namespace)
    logger.info("Ingestion complete: %d chunks upserted into index '%s'.", len(processed_chunks), index_name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest PDF/DOCX files from Google Drive into a Pinecone RAG index using contextual retrieval.")
    parser.add_argument("--drive-folder-id", required=True, help="Google Drive folder ID to ingest.")
    parser.add_argument("--index-name", required=True, help="Pinecone index name (created if it doesn't exist).")
    parser.add_argument("--namespace", default=None, help="Optional Pinecone namespace.")
    parser.add_argument(
        "--skip-contextualization",
        action="store_true",
        default=DEFAULT_SKIP_CONTEXTUALIZATION,
        help="Skip Gemini contextual prefix generation and only embed raw chunks. This is the default unless INGEST_SKIP_CONTEXTUALIZATION=0 is set.",
    )
    parser.add_argument(
        "--contextualize",
        action="store_false",
        dest="skip_contextualization",
        help="Enable Gemini contextual prefix generation for each chunk. Uses more quota.",
    )
    parser.add_argument(
        "--max-context-chars",
        type=int,
        default=DEFAULT_MAX_CONTEXT_CHARS,
        help="Maximum document characters sent to Gemini for each contextual prefix.",
    )
    parser.add_argument(
        "--gemini-call-delay-seconds",
        type=float,
        default=DEFAULT_GEMINI_CALL_DELAY_SECONDS,
        help="Delay between Gemini requests to reduce rate-limit errors.",
    )
    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=DEFAULT_EMBEDDING_BATCH_SIZE,
        help="Number of chunks to embed per Gemini embedding request.",
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=DEFAULT_MAX_CHUNKS,
        help="Optional cap for testing or quota-limited runs; 0 means no cap.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        run_pipeline(
            args.drive_folder_id,
            args.index_name,
            args.namespace,
            skip_contextualization=args.skip_contextualization,
            max_context_chars=args.max_context_chars,
            gemini_call_delay_seconds=args.gemini_call_delay_seconds,
            embedding_batch_size=args.embedding_batch_size,
            max_chunks=args.max_chunks,
        )
    except IngestionError as exc:
        logger.error("Ingestion failed: %s", exc)
        return 1
    except Exception as exc:  # noqa: BLE001 - top-level safety net
        logger.exception("Unexpected error during ingestion: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
