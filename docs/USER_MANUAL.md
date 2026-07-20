# GreenLedger — User Manual

*SDG GreenTech 2026 · mandatory deliverable*

GreenLedger is a web app for screening the sustainability (energy-rating) risk of small
UK commercial buildings, and a research platform on how the MEES regulation distorts the
underlying register. Live app: the deployed Streamlit Cloud URL.

---

## Who it's for
- **A small business owner / tenant** wondering whether their premises is at risk of
  failing the MEES minimum energy standard, before paying for a formal assessment.
- **A portfolio manager** (agent, landlord) with many buildings and a limited inspection
  budget, deciding which buildings to assess first.
- **A researcher / assessor / policymaker** interested in evidence of threshold effects in
  the EPC register.

---

## The five tabs

### 1. Risk Assessment
1. Fill the short form: business type, region, floor area, year, heating fuel, whether
   there's air conditioning, and the building environment (heating/ventilation type).
2. Click **Assess risk**. You get:
   - a **risk tier** (Low / Medium / High) with the model's confidence;
   - **peer benchmarking** — how your building compares to real ones like it;
   - **why** — the factors driving the prediction (SHAP);
   - **recommendations** — cited, source-backed efficiency measures.
3. **Pathway to E simulator** — toggle *Simulate retrofits*, change the heating fuel /
   environment / AC, and see (a) the model's re-scored risk and (b) **what actually
   happened to real buildings** that made that exact change (from 200k+ repeat
   certificates).
4. **AI-written report** (optional) — click *Generate written report* for a plain-English
   summary. Requires a Groq API key; nothing is generated unless you ask.

### 2. Dashboard
A live overview of the 485k-building register subset: KPI cards (certificate counts,
median rating, High-risk share, trend), a colour-coded England & Wales region map (green =
efficient, magenta = poor), region rankings, and the year-over-year efficiency trend.
Use the **period filter** (All / Since 2022 / 2025 only) to slice it.

### 3. MEES Distortion
The research centrepiece. Shows the certificate distribution vs its statistical
counterfactual around the legal E/F boundary, the excess/missing mass, and the year-by-year
strength of the bunching against placebo boundaries — evidence that the MEES letting ban
induced a strong behavioural response at exactly the regulated threshold.

### 4. Audit Triage
For portfolio holders. Set an **audit budget** (% of buildings you can inspect); the tool
ranks buildings by model uncertainty and shows the queue. It also shows whether ranking by
uncertainty beats random auditing (value-of-information curve), whether the model stays
reliable over time (conformal coverage), a **fairness audit** of who gets over-flagged,
and the green-computing model comparison with confidence intervals.

### 5. About
Method, data provenance, and a plain limitations list.

---

## How to read a result
- **Lower asset rating = more efficient.** Bands run A+ (best) to G (worst); E ends at 125,
  F begins at 126, and letting an F/G building is unlawful under MEES.
- **Risk tiers** collapse the 8 bands to three: Low (A+/A/B), Medium (C/D), High (E/F/G).
- The prediction is a **screening estimate from cheap inputs**, not a statutory EPC. It
  tells you whether a real assessment is worth booking — it does not replace one.

## Privacy & data
No personal data is collected. All predictions run locally in the app from the values you
type. The underlying data is the public UK non-domestic EPC register.

## Running it yourself
See `README.md` — `pip install -r requirements.txt`, run `export_artifacts.py` once to
generate the model, then `streamlit run app.py`.

## Troubleshooting
- *"No GROQ_API_KEY configured"* on the report — expected; the report is optional and
  everything else works without it.
- Map or dashboard slow on first load — the app caches 485k rows on first render, then it's
  instant.
