# utils/data_loader.py
import pandas as pd

DATA_PATH = "data/sample_data.xlsx"

# ── Evaluation Criteria Taxonomy (hardcoded from PS1 problem statement) ──────
# The Excel only has 2 sheets; the taxonomy is defined here per the PDF spec.
EVAL_CRITERIA_TAXONOMY = [
    {"criterion": "Technical Approach & Methodology",  "sector": "All",          "typical_weight": "25%", "description": "Quality and clarity of the proposed technical solution and implementation plan."},
    {"criterion": "Relevant Experience & Past Performance", "sector": "All",     "typical_weight": "20%", "description": "Demonstrated history of similar projects, client references, and outcomes."},
    {"criterion": "Price / Financial Proposal",        "sector": "All",          "typical_weight": "20%", "description": "Competitiveness and completeness of the financial bid."},
    {"criterion": "Team Qualifications & CVs",         "sector": "All",          "typical_weight": "10%", "description": "Expertise, certifications, and seniority of the proposed team members."},
    {"criterion": "Compliance with Specifications",    "sector": "All",          "typical_weight": "10%", "description": "Degree to which the proposal meets all mandatory technical requirements."},
    {"criterion": "Management & Governance Plan",      "sector": "All",          "typical_weight": "5%",  "description": "Project management framework, reporting structure, and risk governance."},
    {"criterion": "Health, Safety & Environment (HSE)","sector": "Construction", "typical_weight": "10%", "description": "HSE plans, incident records, and compliance with safety regulations."},
    {"criterion": "Local Content & SME Participation", "sector": "All",          "typical_weight": "5%",  "description": "Use of local suppliers, subcontractors, and small businesses."},
    {"criterion": "Delivery Timeline & Milestones",    "sector": "All",          "typical_weight": "5%",  "description": "Realism and detail of the proposed project schedule."},
    {"criterion": "Innovation & Value-Add",            "sector": "IT Services",  "typical_weight": "5%",  "description": "Novel approaches, automation, or efficiency gains beyond base requirements."},
    {"criterion": "Financial Stability",               "sector": "All",          "typical_weight": "5%",  "description": "Audited accounts, turnover thresholds, and financial health of the bidder."},
    {"criterion": "Certifications & Accreditations",   "sector": "All",          "typical_weight": "5%",  "description": "ISO, CMMI, PMP, CE Mark, and other relevant third-party certifications."},
    {"criterion": "Data Security & Privacy Controls",  "sector": "IT Services",  "typical_weight": "10%", "description": "Cybersecurity posture, data handling policies, and compliance with privacy laws."},
    {"criterion": "Maintenance & After-Sales Support", "sector": "All",          "typical_weight": "5%",  "description": "Warranty terms, SLA commitments, and long-term support arrangements."},
    {"criterion": "Social Value & Community Impact",   "sector": "All",          "typical_weight": "5%",  "description": "Employment creation, skills development, and community benefit programmes."},
    {"criterion": "Subcontracting Plan",               "sector": "Construction", "typical_weight": "5%",  "description": "Identified subcontractors, their qualifications, and oversight arrangements."},
]


def load_bid_history() -> pd.DataFrame:
    """Load the historical bid outcomes sheet."""
    df = pd.read_excel(DATA_PATH, sheet_name="PS1 \u2013 Bid History", header=2)
    df = df.dropna(how="all")
    # Parse budget to a numeric float (strips 'PKR ' prefix and 'M' suffix)
    df["Budget_Numeric"] = (
        df["Budget"]
        .str.replace("PKR ", "", regex=False)
        .str.replace("M", "", regex=False)
        .str.strip()
        .astype(float)
    )
    return df


def load_capability_library() -> pd.DataFrame:
    """Load the company capability library for RAG."""
    df = pd.read_excel(DATA_PATH, sheet_name="PS1 \u2013 Capability Library", header=2)
    df = df.dropna(how="all")
    # Rich composite text used for embedding — domain + summary + certs + client type
    df["embed_text"] = (
        df["Domain"].fillna("") + ". "
        + df["Project Summary"].fillna("") + ". "
        + "Certification: " + df["Certification"].fillna("None") + ". "
        + "Client Type: " + df["Client Type"].fillna("Unknown") + "."
    )
    return df


def load_eval_criteria() -> list[dict]:
    """Return the structured evaluation criteria taxonomy (15 entries)."""
    return EVAL_CRITERIA_TAXONOMY


def get_domain_win_rates() -> dict:
    """
    Compute win rate per Domain by cross-referencing bid sectors with the
    capability library domains.  Returns {domain: win_rate_pct}.
    """
    df = load_bid_history()
    sector_win = (
        df.groupby("Sector")
        .apply(lambda x: round((x["Outcome"] == "Win").sum() / len(x) * 100, 1))
        .to_dict()
    )
    # Map capability domains to their closest sector for scoring purposes
    domain_to_sector = {
        "Cybersecurity":       "IT Services",
        "ERP Implementation":  "IT Services",
        "Cloud Infrastructure":"IT Services",
        "Network Design":      "IT Services",
        "LMS Development":     "Education",
        "Mobile Banking":      "Finance",
        "Hospital IT":         "Healthcare",
        "Medical Equipment":   "Healthcare",
        "Road Construction":   "Construction",
        "Bridge Engineering":  "Construction",
        "Fleet Management":    "Logistics",
        "Solar Energy":        "Energy",
    }
    return {
        domain: sector_win.get(sector, 50.0)
        for domain, sector in domain_to_sector.items()
    }


def get_top_bid_managers() -> list[dict]:
    """Return top bid managers ranked by win rate."""
    df = load_bid_history()
    stats = (
        df.groupby("Bid Manager")
        .apply(lambda x: {
            "manager": x.name,
            "total": len(x),
            "wins": int((x["Outcome"] == "Win").sum()),
            "win_rate": round((x["Outcome"] == "Win").sum() / len(x) * 100, 1),
            "avg_score": round(x["Score (%)"].mean(), 1),
        })
        .tolist()
    )
    return sorted(stats, key=lambda x: x["win_rate"], reverse=True)