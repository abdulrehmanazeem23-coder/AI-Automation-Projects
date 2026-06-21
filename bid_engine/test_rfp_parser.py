import json
import os
import re

from dotenv import load_dotenv
from groq import Groq


LLAMA_MODEL = "llama-3.1-8b-instant"

with open("sample_rfp.txt", "r", encoding="utf-8") as f:
    sample_text = f.read()

print(f"Loaded sample RFP ({len(sample_text)} chars)")

load_dotenv()
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

EXTRACTION_PROMPT = """You are an expert bid analyst. Analyze the following RFP/tender document excerpt and extract structured information.

Return ONLY a valid JSON object with exactly these keys:
{{
  "title": "Project title or name of the tender",
  "client": "Issuing organization name",
  "sector": "One of: IT Services, Construction, Healthcare, Energy, Logistics, Education, Finance, Telecom, Other",
  "deadline": "Submission deadline date if mentioned, else null",
  "budget": "Budget figure if mentioned, else null",
  "mandatory_requirements": [
    "Requirement 1 (start each with MUST, SHALL, or REQUIRED)",
    "Requirement 2"
  ],
  "evaluation_criteria": [
    {{"criterion": "Technical Approach", "weight": "40%"}},
    {{"criterion": "Price", "weight": "30%"}}
  ],
  "key_questions": [
    "Question or section the vendor must respond to"
  ],
  "summary": "2-3 sentence plain English summary of what this bid is about"
}}

Extract up to 10 mandatory requirements, up to 6 evaluation criteria, and up to 8 key questions.
If a field cannot be found, use null for strings or [] for arrays.
Return ONLY the JSON. No explanation, no markdown, no code fences.

RFP TEXT:
{rfp_text}"""

print(f"Calling Groq model: {LLAMA_MODEL}")
response = client.chat.completions.create(
    model=LLAMA_MODEL,
    messages=[{"role": "user", "content": EXTRACTION_PROMPT.format(rfp_text=sample_text)}],
    temperature=0.1,
    max_tokens=2000,
)

raw = response.choices[0].message.content.strip()
raw = re.sub(r"^```json\s*", "", raw)
raw = re.sub(r"^```\s*", "", raw)
raw = re.sub(r"\s*```$", "", raw)

result = json.loads(raw)

print("\nParsed RFP result:")
print(json.dumps(result, indent=2))

print(f"\nSummary: {result['summary']}")
print(f"Budget: {result['budget']}")
print(f"Deadline: {result['deadline']}")
print(f"Mandatory Requirements: {len(result['mandatory_requirements'])}")
print(f"Evaluation Criteria: {len(result['evaluation_criteria'])}")
print(f"Key Questions: {len(result['key_questions'])}")

print("\nRFP parser model smoke test complete.")
