# test_scorer.py
import json
from utils.scorer import check_compliance, calculate_win_probability

# Simulate what the RFP parser would return for our sample RFP
mock_rfp = {
    "title": "Enterprise ERP System for Federal Ministry",
    "client": "National IT Board",
    "sector": "IT Services",
    "deadline": "30 August 2026",
    "budget": "PKR 85 Million",
    "mandatory_requirements": [
        "MUST hold ISO 9001:2015 and ISO 27001 certifications",
        "SHALL have completed at least 3 similar ERP projects for government clients",
        "The proposed solution MUST be cloud-hosted with 99.9% uptime SLA",
        "All data MUST be stored on local servers within Pakistan",
        "The vendor MUST provide a dedicated project manager with PMP certification",
        "Source code MUST be handed over to the Ministry upon project completion",
    ]
}

print("=" * 60)
print("🔍 RUNNING COMPLIANCE CHECK...")
print("=" * 60)

compliance = check_compliance(mock_rfp["mandatory_requirements"])

for item in compliance["items"]:
    print(f"\n{item['status']} [{item['confidence']}] {item['requirement'][:70]}")
    print(f"   → {item['best_match'][:90]}")

print(f"\n📊 Compliance Rate: {compliance['passed']}/{compliance['total']} = {compliance['compliance_rate']}%")

print("\n" + "=" * 60)
print("📈 CALCULATING WIN PROBABILITY...")
print("=" * 60)

win_data = calculate_win_probability(mock_rfp, compliance)

print(f"\n🎯 Final Win Probability Score: {win_data['final_score']} / 100")
print(f"🚦 Decision: {win_data['decision']}")
print(f"💬 Rationale: {win_data['rationale']}")

print("\n📋 Score Breakdown:")
for factor, data in win_data["breakdown"].items():
    sample_str = f"  (n={data['sample']})" if "sample" in data else ""
    print(f"   {factor:28s} {data['score']:5.1f}  (weight {data['weight']}){sample_str}")

print("\n🎉 Step 4 Complete! Compliance checker and scorer are working.")