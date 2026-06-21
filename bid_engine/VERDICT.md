# BidIQ vs. CUST Hackathon Problem #1 — Gap Analysis

**Verdict: ~90% satisfied. Two real gaps, one soft gap.**

## Core Challenge — all 5 met ✅

| Requirement | Where |
|---|---|
| Ingest PDF/DOCX, extract requirements, eval criteria, deadlines, Q&A | `utils/rfp_parser.py` — extracts all of these plus submission requirements and summary |
| Match against Capability Library | `utils/rag_engine.py` — FAISS + sentence-transformers RAG |
| Auto-draft structured proposal mapped to each question | `utils/proposal_generator.py` + `utils/docx_exporter.py` |
| Flag compliance gaps | `utils/scorer.py` — Pass / Weak Match / Critical Gap with remediation suggestions |
| Win-probability heuristics (budget alignment, competitor presence, past win rate in similar domains) | `utils/scorer.py` — all three named factors are explicitly in the 6-factor model |

## Expected Outcomes / Deliverables — 6 of 7 met

| Deliverable | Status | Notes |
|---|---|---|
| Working POC: sample RFP → structured draft response | ✅ | Full pipeline in `app.py` |
| Separate workspace per RFP/RFQ/Tender | ✅ | `st.session_state.workspaces`, switchable in sidebar |
| Auto-generated compliance checklist with pass/fail mapping | ✅ | Compliance tab — Pass / Weak Match / Critical Gap per requirement |
| Win-probability dashboard across key criteria | ✅ | Plotly factor-breakdown dashboard |
| GO/NO-GO decision | ✅ | GO / CONDITIONAL GO / NO-GO banners (extra middle tier) |
| Review, edit, approve AI content before final export | ✅ | Editable text areas per draft section, DOCX download |
| **Demonstrated ≥50% reduction in manual effort vs. baseline** | ❌ | No timing metrics, no baseline comparison, nothing in UI or repo |

## AI Components Required — 3 of 4 wired in

| Component | Status | Notes |
|---|---|---|
| LLM for parsing, extraction, narrative generation | ✅ | Groq `llama-3.1-8b-instant` in parser + proposal generator |
| RAG to query Capability Library | ✅ | FAISS flat-L2 + `all-MiniLM-L6-v2` embeddings |
| **NER for deadlines, budgets, weights, clauses** | ⚠️ | `utils/ner_extractor.py` exists (regex-based) but is **imported nowhere — dead code** |
| Scoring/ranking model for win probability | ✅ | Heuristic 6-factor weighted model (PDF says "heuristics", so acceptable) |

## Gaps in detail

### Gap 1 — Effort-reduction deliverable (hard gap)
"Demonstrated reduction in manual bid preparation effort by at least 50% (measured against a baseline manual exercise)" has zero artifacts in the codebase or demo. This is a demo/presentation deliverable, but right now there is nothing to show.

### Gap 2 — NER not integrated (hard gap)
The PDF lists NER as a *required* AI component. The regex extractor exists but is not part of the pipeline. A judge who asks "show me the NER component" will see dead code.

### Gap 3 — Document scale (soft gap)
Sample RFPs are specified as 15–80 pages, but `parse_rfp()` truncates input at 12,000 characters (~4–5 pages). Requirements appearing mid/late in a long document are silently lost.

## Recommended fixes, in priority order

1. **Wire in the NER extractor** (~30 min). Call `extract_entities()` on raw text inside `parse_rfp()`, store as `rfp_data["ner_entities"]`, surface in the RFP Overview tab. Bonus: cross-check the regex-found deadline/budget against the LLM output.
2. **Add a "time saved" artifact** (~30 min). Either an in-app metric card (manual baseline ~6 hrs vs. BidIQ ~3 min, with per-stage breakdown) or a timed manual-vs-app exercise on `sample_rfp.txt` presented as a slide — the latter is literally what the deliverable asks for.
3. **Chunked extraction** for long documents: run extraction per chunk and merge, or section-aware filtering for "Requirements"/"Eligibility" headings. If out of time, demo with a short RFP and have the chunking design ready to whiteboard.
