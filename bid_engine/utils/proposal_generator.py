# utils/proposal_generator.py
import os
from groq import Groq
from utils.rag_engine import search_capabilities
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

LLAMA_MODEL = "llama-3.1-8b-instant"

DRAFT_PROMPT = """You are a senior bid writer at a professional IT services company.
Your job is to write ONE section of a proposal response for a government/enterprise RFP.

RFP CONTEXT:
- Project: {title}
- Client: {client}
- Sector: {sector}

QUESTION / SECTION TO ANSWER:
{question}

RELEVANT EVIDENCE FROM OUR CAPABILITY LIBRARY (use this — do not invent facts):
{evidence}

INSTRUCTIONS:
- Write a professional, confident proposal response to the question above.
- Naturally reference the provided evidence (project names, certifications, metrics).
- Length: 3-5 paragraphs. Formal tone. No bullet points — flowing prose only.
- Do NOT start with "Certainly" or "Sure". Start directly with the response.
- End with one sentence that ties back to the client's specific need.
"""

def generate_section(question: str, rfp_data: dict, top_k: int = 3) -> dict:
    """
    Generate a single proposal section for one RFP question.
    Returns dict with the question, evidence used, and drafted text.
    """
    # Step 1: RAG — find relevant capabilities for this question
    matches = search_capabilities(question, top_k=top_k)

    # Step 2: Format evidence block for the prompt
    evidence_lines = []
    for m in matches:
        cert = m['certification'] if str(m['certification']) != 'nan' else 'N/A'
        evidence_lines.append(
            f"- [{m['cap_id']}] {m['domain']}: {m['summary']} "
            f"(Cert: {cert}, Client: {m['client_type']}, Value: {m['contract_value']})"
        )
    evidence_text = "\n".join(evidence_lines) if evidence_lines else "No direct matches found — use general best practices."

    # Step 3: Call Groq Llama LLM
    response = client.chat.completions.create(
        model=LLAMA_MODEL,
        messages=[{
            "role": "user",
            "content": DRAFT_PROMPT.format(
                title=rfp_data.get("title", "N/A"),
                client=rfp_data.get("client", "N/A"),
                sector=rfp_data.get("sector", "N/A"),
                question=question,
                evidence=evidence_text
            )
        }],
        temperature=0.6,   # slightly creative for better prose
        max_tokens=600,
    )

    drafted_text = response.choices[0].message.content.strip()

    return {
        "question": question,
        "drafted_text": drafted_text,
        "evidence_used": matches
    }


def generate_full_proposal(rfp_data: dict) -> list:
    """
    Generate draft sections for ALL key questions in the RFP.
    Returns a list of section dicts.
    """
    questions = rfp_data.get("key_questions", [])
    if not questions:
        return []

    sections = []
    for i, q in enumerate(questions):
        print(f"  ✍️  Drafting section {i+1}/{len(questions)}: {q[:60]}...")
        section = generate_section(q, rfp_data)
        sections.append(section)

    return sections
