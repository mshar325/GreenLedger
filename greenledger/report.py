"""Turns already-computed facts into a written report via Groq. The model never sees
training data or invents numbers — every figure in the prompt is one we already computed
or a cited source range from recommendations.py. Its only job is prose, not arithmetic.
"""
import streamlit as st
from groq import Groq

MODEL = "openai/gpt-oss-20b"

SYSTEM_PROMPT = (
    "You are writing a short sustainability report for a small business owner. "
    "You will be given the business's risk tier, model confidence, benchmark percentiles, "
    "the factors driving the prediction, and a set of pre-written, source-cited "
    "recommendations. Rephrase these into a clear, encouraging, plain-language report of "
    "3-4 short paragraphs. Do not invent, estimate, or state any number, percentage, or "
    "savings figure that is not explicitly given to you below — if you want to mention a "
    "number, use only the ones provided. Do not add new recommendations beyond the ones "
    "given. Write for a business owner, not a data scientist."
)


def _get_client():
    try:
        api_key = st.secrets.get("GROQ_API_KEY", None)
    except Exception:
        api_key = None  # no secrets.toml configured yet -- report generation is optional
    if not api_key:
        return None
    return Groq(api_key=api_key)


def generate_report(business_type: str, risk_label: str, proba: dict,
                     pct_all: float, pct_type: float | None,
                     top_drivers: list[str], recommendations: list[dict]) -> str | None:
    client = _get_client()
    if client is None:
        return None

    facts = [
        f"Business type: {business_type}",
        f"Predicted risk tier: {risk_label}",
        f"Model confidence by tier: Low {proba['Low']:.0f}%, Medium {proba['Medium']:.0f}%, High {proba['High']:.0f}%",
        f"Higher predicted risk than {pct_all:.0f}% of the 741 real small commercial buildings studied",
    ]
    if pct_type is not None:
        facts.append(f"Higher predicted risk than {pct_type:.0f}% of other {business_type} businesses specifically")
    if top_drivers:
        facts.append(f"Top factors pushing the prediction toward '{risk_label}': {', '.join(top_drivers)}")
    facts.append("Recommendations (use exactly these, with exactly these figures, no others):")
    for rec in recommendations:
        facts.append(f"- {rec['title']}: {rec['detail']} (Source: {rec['source']})")

    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "\n".join(facts)},
        ],
        temperature=0.4,
        max_tokens=500,
    )
    return resp.choices[0].message.content
