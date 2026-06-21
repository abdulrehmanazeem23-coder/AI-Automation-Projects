# utils/scorer.py
import re
import pandas as pd
import numpy as np
from utils.rag_engine import search_capabilities
from utils.data_loader import load_bid_history, get_domain_win_rates

# ── Thresholds ────────────────────────────────────────────────────────────────
COMPLIANCE_THRESHOLD = 0.25   # similarity >= this  → Pass
GAP_THRESHOLD        = 0.15   # similarity >= this  → Weak Match; below → Critical Gap

# ── Remediation map ───────────────────────────────────────────────────────────
REMEDIATION_MAP = {
    "iso":          "Obtain ISO certification or partner with a certified subcontractor.",
    "cmmi":         "Engage a CMMI-certified delivery partner and include their credentials.",
    "pmp":          "Assign a PMP-certified project manager to the bid team.",
    "financial":    "Prepare audited financial statements for the last 3 years.",
    "security":     "Document your cybersecurity policies and obtain ISO 27001 if absent.",
    "construction": "Identify a licensed civil/structural engineering subcontractor.",
    "health":       "Prepare a formal HSE plan with incident statistics.",
    "local":        "Identify local SME partners to meet local content requirements.",
    "experience":   "Pull the most relevant 3 past projects from the Capability Library as references.",
    "team":         "Assemble CVs for all key personnel proposed for this bid.",
    "delivery":     "Prepare a detailed Gantt chart and milestone schedule.",
    "support":      "Define SLA terms, warranty periods, and maintenance pricing.",
    "cloud":        "Highlight cloud-certified staff (AWS/Azure/GCP) in the proposal.",
    "data":         "Include a Data Privacy and GDPR/PDPA compliance statement.",
    "budget":       "Prepare a detailed cost breakdown and financial justification.",
    "certif":       "List all relevant certifications held by your organisation.",
}

def _suggest_remediation(requirement: str) -> str:
    req_lower = requirement.lower()
    for keyword, suggestion in REMEDIATION_MAP.items():
        if keyword in req_lower:
            return suggestion
    return (
        "Review the requirement carefully and prepare supporting documentation "
        "or engage a qualified subcontractor."
    )


# ── Compliance Checker ────────────────────────────────────────────────────────

def check_compliance(mandatory_requirements: list) -> dict:
    """
    For each mandatory requirement search the Capability Library via RAG.
    Returns pass / weak-match / critical-gap classification with remediation actions.
    """
    results       = []
    passed        = 0
    critical_gaps = 0

    for req in mandatory_requirements:
        matches = search_capabilities(req, top_k=1)
        if not matches:
            status      = "❌ Critical Gap"
            best_match  = "No capability match found in library."
            score       = 0.0
            remediation = _suggest_remediation(req)
            critical_gaps += 1
        else:
            top   = matches[0]
            score = top["score"]
            if score >= COMPLIANCE_THRESHOLD:
                status      = "✅ Pass"
                best_match  = f"[{top['cap_id']}] {top['domain']} — {top['summary'][:80]}..."
                remediation = None
                passed      += 1
            elif score >= GAP_THRESHOLD:
                status      = "⚠️ Weak Match"
                best_match  = (
                    f"Closest: [{top['cap_id']}] {top['domain']} "
                    f"(low confidence — {score:.2f})"
                )
                remediation = _suggest_remediation(req)
            else:
                status      = "❌ Critical Gap"
                best_match  = (
                    f"Closest: [{top['cap_id']}] {top['domain']} "
                    f"(very low — {score:.2f})"
                )
                remediation = _suggest_remediation(req)
                critical_gaps += 1

        results.append({
            "requirement": req,
            "status":      status,
            "best_match":  best_match,
            "confidence":  round(score, 3),
            "remediation": remediation,
        })

    total           = len(mandatory_requirements)
    compliance_rate = round((passed / total) * 100, 1) if total else 0

    return {
        "items":           results,
        "passed":          passed,
        "total":           total,
        "critical_gaps":   critical_gaps,
        "compliance_rate": compliance_rate,
    }


# ── Competitor presence heuristics ───────────────────────────────────────────
SECTOR_COMPETITION = {
    "IT Services":  {"level": "High",   "score": 35, "note": "Market is crowded with 10+ established IT vendors."},
    "Construction": {"level": "Medium", "score": 55, "note": "3–6 major contractors typically compete."},
    "Healthcare":   {"level": "Low",    "score": 70, "note": "Few specialists; less competitive."},
    "Energy":       {"level": "Medium", "score": 55, "note": "Moderate competition from energy-sector firms."},
    "Logistics":    {"level": "Medium", "score": 60, "note": "Moderate competition; regional players dominate."},
    "Finance":      {"level": "High",   "score": 40, "note": "Heavily contested; large banks have in-house teams."},
    "Telecom":      {"level": "High",   "score": 38, "note": "Dominated by a few large telco vendors."},
    "Education":    {"level": "Low",    "score": 68, "note": "Fewer specialised bidders in this space."},
}


# ── Win Probability Scorer ────────────────────────────────────────────────────

def calculate_win_probability(
    rfp_data:            dict,
    compliance_report:   dict,
    competitor_override: str | None = None,
) -> dict:
    """
    6-factor weighted win probability.

    Factor weights
    ──────────────
    1. Compliance Rate          25%
    2. Sector Win Rate          20%
    3. Domain Win Rate (RAG)    20%   ← requirement #12: past win rate in similar domains
    4. Avg Historical Score     15%
    5. Budget Competitiveness   10%
    6. Competitor Presence      10%
    """
    df     = load_bid_history()
    sector = rfp_data.get("sector", "")

    # ── Factor 1: Compliance Rate (25%) ──────────────────────────────────────
    compliance_score = compliance_report["compliance_rate"]

    # ── Factor 2: Sector Win Rate (20%) ──────────────────────────────────────
    sector_bids = df[df["Sector"].str.lower() == sector.lower()]
    if len(sector_bids) > 0:
        sector_win_rate = round(
            (sector_bids["Outcome"] == "Win").sum() / len(sector_bids) * 100, 1
        )
        sector_sample = len(sector_bids)
    else:
        sector_win_rate = round((df["Outcome"] == "Win").sum() / len(df) * 100, 1)
        sector_sample   = len(df)

    # ── Factor 3: Domain Win Rate from Capability Library (20%) ──────────────
    # This implements requirement #12: "past win rate in similar domains"
    # We look at which capability domains matched for this RFP's requirements,
    # then pull the historical win rate for those domains.
    domain_win_rates = get_domain_win_rates()
    reqs             = rfp_data.get("mandatory_requirements", [])

    matched_domain_scores = []
    for req in reqs[:5]:                          # cap at 5 to keep it fast
        hits = search_capabilities(req, top_k=1)
        if hits:
            domain     = hits[0]["domain"]
            domain_wr  = domain_win_rates.get(domain, sector_win_rate)
            matched_domain_scores.append(domain_wr)

    domain_win_score = (
        round(float(np.mean(matched_domain_scores)), 1)
        if matched_domain_scores
        else sector_win_rate           # fallback to sector rate if no matches
    )
    domain_sample = len(matched_domain_scores)

    # ── Factor 4: Historical Average Score in Sector (15%) ───────────────────
    if len(sector_bids) > 0:
        avg_sector_score = round(sector_bids["Score (%)"].mean(), 1)
    else:
        avg_sector_score = round(df["Score (%)"].mean(), 1)

    # ── Factor 5: Budget Competitiveness (10%) ────────────────────────────────
    budget_score   = 50
    rfp_budget_str = rfp_data.get("budget") or ""
    try:
        nums = re.findall(r"[\d.]+", rfp_budget_str.replace(",", ""))
        if nums:
            rfp_budget_val = float(nums[0])
            won_bids = (
                sector_bids[sector_bids["Outcome"] == "Win"]
                if len(sector_bids) > 0
                else df[df["Outcome"] == "Win"]
            )
            if len(won_bids) > 0:
                avg_won = won_bids["Budget_Numeric"].mean()
                ratio   = rfp_budget_val / avg_won if avg_won > 0 else 1
                if 0.5 <= ratio <= 2.0:
                    budget_score = 78
                elif 0.3 <= ratio <= 3.0:
                    budget_score = 58
                else:
                    budget_score = 35
    except Exception:
        budget_score = 50

    # ── Factor 6: Competitor Presence (10%) ───────────────────────────────────
    comp_info = SECTOR_COMPETITION.get(
        sector, {"level": "Medium", "score": 50, "note": "Competition level unknown."}
    )
    if competitor_override == "Low":
        comp_score, comp_level, comp_note = 75, "Low",    "User indicated low competitor presence."
    elif competitor_override == "Medium":
        comp_score, comp_level, comp_note = 55, "Medium", "User indicated moderate competitor presence."
    elif competitor_override == "High":
        comp_score, comp_level, comp_note = 30, "High",   "User indicated high competitor presence."
    else:
        comp_score = comp_info["score"]
        comp_level = comp_info["level"]
        comp_note  = comp_info["note"]

    # ── Weighted Final Score ──────────────────────────────────────────────────
    final_score = round(
        compliance_score * 0.25
        + sector_win_rate  * 0.20
        + domain_win_score * 0.20
        + avg_sector_score * 0.15
        + budget_score     * 0.10
        + comp_score       * 0.10,
        1,
    )

    # ── Decision ─────────────────────────────────────────────────────────────
    if final_score >= 65:
        decision = "🟢 GO"
    elif final_score >= 45:
        decision = "🟡 CONDITIONAL GO"
    else:
        decision = "🔴 NO-GO"

    # ── Detailed reasons ─────────────────────────────────────────────────────
    reasons = []

    if compliance_score >= 80:
        reasons.append(f"✅ Strong compliance rate ({compliance_score}%).")
    elif compliance_score >= 55:
        reasons.append(
            f"⚠️ Moderate compliance ({compliance_score}%) — "
            f"address {compliance_report.get('critical_gaps', 0)} gap(s) before submission."
        )
    else:
        reasons.append(
            f"❌ Low compliance ({compliance_score}%) — "
            f"{compliance_report.get('critical_gaps', 0)} critical gap(s) must be resolved."
        )

    if sector_win_rate >= 65:
        reasons.append(
            f"✅ Strong historical win rate in {sector} ({sector_win_rate}%, n={sector_sample})."
        )
    elif sector_win_rate >= 45:
        reasons.append(
            f"⚠️ Average win rate in {sector} ({sector_win_rate}%, n={sector_sample})."
        )
    else:
        reasons.append(
            f"❌ Weak win rate in {sector} ({sector_win_rate}%, n={sector_sample}) — "
            "consider capability investment."
        )

    if domain_win_score >= 65:
        reasons.append(
            f"✅ Matched capability domains show strong win history "
            f"({domain_win_score}% avg, {domain_sample} domain(s) matched)."
        )
    elif domain_win_score >= 45:
        reasons.append(
            f"⚠️ Matched domains show moderate win rate "
            f"({domain_win_score}%, {domain_sample} domain(s) matched) — strengthen evidence."
        )
    else:
        reasons.append(
            f"❌ Matched domains show low historical win rate "
            f"({domain_win_score}%, {domain_sample} domain(s) matched) — capability gap risk."
        )

    if comp_level == "Low":
        reasons.append(
            f"✅ Low competitor presence in {sector} — favourable market conditions."
        )
    elif comp_level == "Medium":
        reasons.append(f"⚠️ Moderate competition in {sector}. {comp_note}")
    else:
        reasons.append(f"❌ High competition in {sector}. {comp_note}")

    if budget_score >= 70:
        reasons.append("✅ Budget is well-aligned with our historical winning bids.")
    elif budget_score >= 50:
        reasons.append("⚠️ Budget is somewhat outside our typical win range — price carefully.")
    else:
        reasons.append(
            "❌ Budget is significantly outside our historical winning range — high price risk."
        )

    return {
        "final_score": final_score,
        "decision":    decision,
        "reasons":     reasons,
        "breakdown": {
            "Compliance Rate":      {"score": compliance_score,  "weight": "25%"},
            "Sector Win Rate":      {"score": sector_win_rate,   "weight": "20%", "sample": sector_sample},
            "Domain Win Rate":      {"score": domain_win_score,  "weight": "20%", "sample": domain_sample},
            "Avg Historical Score": {"score": avg_sector_score,  "weight": "15%"},
            "Budget Alignment":     {"score": budget_score,      "weight": "10%"},
            "Competitor Presence":  {"score": comp_score,        "weight": "10%", "level": comp_level},
        },
    }