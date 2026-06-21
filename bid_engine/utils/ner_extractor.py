import re

def extract_entities(text):
    entities = {
        "deadlines": [],
        "budgets": [],
        "certifications": [],
        "organizations": [],
        "locations": []
    }

    # Dates
    date_pattern = r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b'
    entities["deadlines"] = re.findall(date_pattern, text)

    # Budgets
    budget_pattern = r'(?:\$|USD|PKR|Rs\.?)\s?[\d,]+(?:\.\d+)?(?:\s?(?:Million|Billion|M|B))?'
    entities["budgets"] = re.findall(budget_pattern, text, re.IGNORECASE)

    # Certifications
    certs = re.findall(
        r'ISO\s?\d{4,5}|CMMI|PMP|ITIL|SOC\s?2',
        text,
        re.IGNORECASE
    )
    entities["certifications"] = list(set(certs))

    # Organizations
    orgs = re.findall(
        r'\b[A-Z][A-Za-z]+\s(?:Ltd|Limited|Inc|Corporation|Authority|Agency)\b',
        text
    )
    entities["organizations"] = list(set(orgs))

    return entities