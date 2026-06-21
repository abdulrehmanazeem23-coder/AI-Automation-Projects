def estimate_effort_savings(rfp_data: dict, compliance: dict | None = None) -> dict:
    """Estimate manual-vs-BidIQ prep effort for the demo deliverable."""
    compliance = compliance or {}
    requirement_count = len(rfp_data.get("mandatory_requirements", []))
    question_count = len(rfp_data.get("key_questions", []))
    criteria_count = len(rfp_data.get("evaluation_criteria", []))
    submission_count = len(rfp_data.get("submission_requirements", []))
    qa_count = len(rfp_data.get("qa_sections", []))
    char_count = rfp_data.get("char_count", 0)
    estimated_pages = max(1, round(char_count / 2500))

    manual_minutes = (
        90
        + estimated_pages * 12
        + requirement_count * 12
        + question_count * 28
        + criteria_count * 10
        + submission_count * 8
        + qa_count * 12
    )
    manual_minutes = max(manual_minutes, 360)

    app_minutes = (
        12
        + estimated_pages * 1.5
        + requirement_count * 2
        + question_count * 4
        + criteria_count
        + submission_count
    )
    app_minutes = max(round(app_minutes), 18)

    saved_minutes = max(manual_minutes - app_minutes, 0)
    reduction_percent = round(saved_minutes / manual_minutes * 100, 1) if manual_minutes else 0

    return {
        "manual_baseline_minutes": round(manual_minutes),
        "bidiq_minutes": app_minutes,
        "saved_minutes": round(saved_minutes),
        "reduction_percent": reduction_percent,
        "baseline_basis": {
            "estimated_pages": estimated_pages,
            "requirements": requirement_count,
            "questions": question_count,
            "evaluation_criteria": criteria_count,
            "submission_requirements": submission_count,
            "qa_items": qa_count,
            "compliance_items": compliance.get("total", requirement_count),
        },
        "stage_breakdown_minutes": {
            "manual_review_and_extraction": round(90 + estimated_pages * 12),
            "manual_compliance_mapping": requirement_count * 12,
            "manual_response_drafting": question_count * 28,
            "manual_submission_packaging": submission_count * 8,
            "bidiq_parse_score_match": round(12 + estimated_pages * 1.5 + requirement_count * 2),
            "bidiq_draft_review_export": question_count * 4 + criteria_count + submission_count,
        },
    }
