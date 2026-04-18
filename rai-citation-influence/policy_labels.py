POLICY_LABELS = {
    "brazil-ai-for-the-good-of-all": "Brazil: AI for the Good of All",
    "china-interim-measures-for-management-of-generative-ai": "China: Interim Measures for Generative AI",
    "egypt-national-guidelines-for-trustworthy-ai": "Egypt: National Guidelines for Trustworthy AI",
    "eu-ai-act": "EU: AI Act",
    "eu-communication-on-ai-innovation": "EU: Communication on AI Innovation",
    "eu-ethics-guidelines-for-trustworthy-ai": "EU: Ethics Guidelines for Trustworthy AI",
    "eu-general-data-protection-regulation": "EU: General Data Protection Regulation (GDPR)",
    "eu-guidelines-on-definition-of-ai-systems": "EU: Guidelines on Definition of AI Systems",
    "g7-hiroshima-ai-process-code-of-conduct": "G7 Hiroshima Summit: AI Process Code of Conduct",
    "gpai-code-of-practice-version-3": "GPAI: Code of Practice (v3)",
    "new-zealand-strategy-for-ai-investing-with-confidence": "New Zealand: AI Strategy — Investing with Confidence",
    "nist-ai-risk-management-framework": "NIST: AI Risk Management Framework",
    "uk-pro-innovation-approach-to-ai-regulation": "UK: Pro-Innovation Approach to AI Regulation",
    "unesco-recommendation-on-ethics-of-ai": "UNESCO: Recommendation on the Ethics of AI",
    "us-america-ai-action-plan": "US: America AI Action Plan",
    "us-blueprint-for-ai-bill-of-rights": "US: Blueprint for an AI Bill of Rights",
    "us-eo-14110": "US: Executive Order 14110 on Safe AI",
    "us-eo-14141": "US: Executive Order 14141 on AI Infrastructure",
}

def apply_label(slug):
    """Convert a file slug or full path to a human-readable policy name."""
    clean = slug.split("/")[-1].replace(".pdf", "")
    return POLICY_LABELS.get(clean, clean)  # falls back to slug if not found