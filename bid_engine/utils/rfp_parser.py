# utils/rfp_parser.py
import os
import json
import re
from groq import Groq
from pypdf import PdfReader
from docx import Document
from dotenv import load_dotenv
from utils.ner_extractor import extract_entities

load_dotenv()
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

LLAMA_MODEL = "llama-3.1-8b-instant"
CHUNK_SIZE = 12000
CHUNK_OVERLAP = 800

# ── Text Extraction ──────────────────────────────────────────────────────────

def extract_text_from_pdf(file_path: str) -> str:
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text.strip()

def extract_text_from_docx(file_path: str) -> str:
    doc = Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs]).strip()

def extract_text(file_path: str) -> str:
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        return extract_text_from_pdf(file_path)
    elif ext in (".docx", ".doc"):
        return extract_text_from_docx(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

# ── LLM Extraction ───────────────────────────────────────────────────────────

EXTRACTION_PROMPT = """You are an expert bid analyst. Analyze the following RFP/tender document and extract structured information.

Return ONLY a valid JSON object with exactly these keys:
{{
  "title": "Project title or name of the tender",
  "client": "Issuing organization name",
  "sector": "One of: IT Services, Construction, Healthcare, Energy, Logistics, Education, Finance, Telecom, Other",
  "deadline": "Submission deadline date if mentioned, else null",
  "budget": "Budget figure if mentioned (e.g. PKR 50M, USD 2 million), else null",
  "mandatory_requirements": [
    "Requirement 1 — start each with MUST, SHALL, or REQUIRED",
    "Requirement 2"
  ],
  "evaluation_criteria": [
    {{"criterion": "Technical Approach", "weight": "40%"}},
    {{"criterion": "Price", "weight": "30%"}}
  ],
  "key_questions": [
    "Question or section the vendor must respond to"
  ],
  "qa_sections": [
    {{
      "question": "Explicit question found in the RFP Q&A or Clarifications section",
      "context": "Brief context about what this question is asking"
    }}
  ],
  "submission_requirements": [
    "Submission format or procedural requirement, e.g. Must submit 3 hard copies"
  ],
  "summary": "2-3 sentence plain English summary of what this bid is about"
}}

Rules:
- Extract up to 10 mandatory_requirements (compliance clauses, minimum qualifications, certifications required).
- Extract up to 6 evaluation_criteria with percentage weights if present.
- Extract up to 8 key_questions (major sections the proposal must address).
- Extract up to 5 qa_sections (explicit Q&A blocks, clarification questions, or questionnaires found in the document).
- Extract up to 5 submission_requirements (formatting, copies, delivery instructions).
- If a field cannot be found use null for strings or [] for arrays.
- Return ONLY the JSON. No explanation, no markdown, no code fences.

RFP TEXT:
{rfp_text}"""


def _clean_llm_json(raw_json: str) -> str:
    raw_json = raw_json.strip()
    raw_json = re.sub(r"^```json\s*", "", raw_json)
    raw_json = re.sub(r"^```\s*", "", raw_json)
    raw_json = re.sub(r"\s*```$", "", raw_json)
    return raw_json


def _split_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end]
        if start > 0:
            boundary = chunk.find("\n")
            if 0 <= boundary < 500:
                chunk = chunk[boundary + 1:]
        chunks.append(chunk)
        if end == len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def _first_present(values):
    for value in values:
        if value not in (None, "", [], {}):
            return value
    return None


def _dedupe_strings(items: list[str], limit: int | None = None) -> list[str]:
    seen = set()
    merged = []
    for item in items:
        if not item:
            continue
        normalized = re.sub(r"\s+", " ", str(item)).strip()
        key = normalized.lower()
        if key in seen:
            continue
        seen.add(key)
        merged.append(normalized)
        if limit and len(merged) >= limit:
            break
    return merged


def _dedupe_dicts(items: list[dict], key_fields: tuple[str, ...], limit: int | None = None) -> list[dict]:
    seen = set()
    merged = []
    for item in items:
        if not isinstance(item, dict):
            continue
        key = tuple(str(item.get(field, "")).strip().lower() for field in key_fields)
        if not any(key) or key in seen:
            continue
        seen.add(key)
        merged.append(item)
        if limit and len(merged) >= limit:
            break
    return merged


def _merge_chunk_results(results: list[dict]) -> dict:
    return {
        "title": _first_present([r.get("title") for r in results]),
        "client": _first_present([r.get("client") for r in results]),
        "sector": _first_present([r.get("sector") for r in results]) or "Other",
        "deadline": _first_present([r.get("deadline") for r in results]),
        "budget": _first_present([r.get("budget") for r in results]),
        "mandatory_requirements": _dedupe_strings(
            [item for r in results for item in r.get("mandatory_requirements", [])], 15
        ),
        "evaluation_criteria": _dedupe_dicts(
            [item for r in results for item in r.get("evaluation_criteria", [])],
            ("criterion", "weight"),
            8,
        ),
        "key_questions": _dedupe_strings(
            [item for r in results for item in r.get("key_questions", [])], 12
        ),
        "qa_sections": _dedupe_dicts(
            [item for r in results for item in r.get("qa_sections", [])],
            ("question", "context"),
            8,
        ),
        "submission_requirements": _dedupe_strings(
            [item for r in results for item in r.get("submission_requirements", [])], 8
        ),
        "summary": _first_present([r.get("summary") for r in results]),
    }


def _extract_chunk(chunk_text: str, chunk_number: int, total_chunks: int) -> dict:
    response = client.chat.completions.create(
        model=LLAMA_MODEL,
        messages=[{
            "role": "user",
            "content": EXTRACTION_PROMPT.format(
                rfp_text=f"[Chunk {chunk_number} of {total_chunks}]\n{chunk_text}"
            )
        }],
        temperature=0.1,
        max_tokens=2500,
    )

    raw_json = _clean_llm_json(response.choices[0].message.content)
    try:
        return json.loads(raw_json)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"LLM returned invalid JSON in chunk {chunk_number}: {e}\n\nRaw output:\n{raw_json[:500]}"
        )


def parse_rfp(file_path: str) -> dict:
    """
    Main entry point: takes a PDF or DOCX path, returns fully structured RFP data.
    """
    raw_text = extract_text(file_path)

    if not raw_text or len(raw_text) < 100:
        raise ValueError(
            "Could not extract meaningful text. The file may be a scanned image-only PDF."
        )

    chunks = _split_text(raw_text)
    chunk_results = [
        _extract_chunk(chunk, index, len(chunks))
        for index, chunk in enumerate(chunks, 1)
    ]
    parsed = _merge_chunk_results(chunk_results)

    # Ensure all expected keys exist with safe defaults
    parsed.setdefault("qa_sections", [])
    parsed.setdefault("submission_requirements", [])
    parsed.setdefault("mandatory_requirements", [])
    parsed.setdefault("evaluation_criteria", [])
    parsed.setdefault("key_questions", [])

    ner_entities = extract_entities(raw_text)
    parsed["ner_entities"] = ner_entities
    if not parsed.get("deadline") and ner_entities.get("deadlines"):
        parsed["deadline"] = ner_entities["deadlines"][0]
    if not parsed.get("budget") and ner_entities.get("budgets"):
        parsed["budget"] = ner_entities["budgets"][0]
    parsed["deadline_ner_match"] = (
        bool(parsed.get("deadline")) and parsed.get("deadline") in ner_entities.get("deadlines", [])
    )
    parsed["budget_ner_match"] = (
        bool(parsed.get("budget")) and parsed.get("budget") in ner_entities.get("budgets", [])
    )

    parsed["raw_text"]    = raw_text
    parsed["source_file"] = os.path.basename(file_path)
    parsed["char_count"]  = len(raw_text)
    parsed["extraction_chunks"] = len(chunks)

    return parsed
