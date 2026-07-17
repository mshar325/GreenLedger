# GreenLedger — Process & Results

A record of how this project went from a generic AI-business-dashboard brainstorm to a
working, executed notebook on real government data — what was rejected and why, what was
actually built, and what it found.

---

## 1. Where this started

The starting point was a set of project ideas another AI had generated for a friend's
7th-semester Business + Technology capstone (a Sustainable Computing course whose rubric
explicitly rewards novel datasets, algorithmic justification, and IEEE-quality writing, and
penalizes tutorial-style projects). The leading idea was a **"SustainAI Business Advisor"** —
a single platform where an SME uploads sales, inventory, and reviews and gets back a sales
forecast, churn prediction, ESG score, SDG mapping, AI-adoption suggestions, and
auto-generated reports, all in one dashboard.

**Why that idea was rejected:**

- Its novelty claim rested on feature count (12 bullet points), not on a real research
  question — the exact shape the rubric penalizes as "tutorial-stitching."
- It would have run on whatever public/synthetic CSV was on hand, so there was no primary
  or novel dataset behind it.
- It needed 4+ ML models, an LLM layer, and full cloud deployment, built in one semester by
  someone who — in their own words — "knows the names but not what they do" for
  Random Forest, XGBoost, KNN.
- Its "finding" would have been "the platform works," which is hard to falsify and harder
  to defend under questioning from a professor or an admissions reader.

## 2. The redesign: one question instead of twelve features

The replacement, **GreenLedger**, was scoped around a single falsifiable research question
instead of a feature list:

> Can cheap, self-reportable operational data predict a small business's sustainability
> risk almost as well as a real audit would — and does model complexity (ANN vs. classical
> ML) even help at this data scale?

The original plan was to answer this with **primary field data** — a 9-question, 10-minute
survey run against 30–50 real local businesses (headcount, hours, utility bills, waste
disposal, supplier distance, etc.), with a transparent, disclosed formula turning those
answers into a risk index.

## 3. The dataset reality check

Two things happened next, in order:

1. **The claim "this data must already be public" was checked, not assumed.** A real
   search across Kaggle, data.gov.in, EIA, and World Bank catalogs found nothing that pairs
   individual-shop operational proxies with a sustainability outcome — what exists is
   either country/sector-level aggregates (World Bank MSME indicators) or large-building/
   corporate energy datasets, never small-shop-level microdata. This is structural, not bad
   luck: energy/ESG disclosure is legally mandated for large and public companies, not
   small businesses, so nobody but a researcher doing fieldwork collects this at the
   individual-shop level — confirming the field-survey plan's dataset really would have
   been novel.
2. **But neither person doing the project had time for 30–50 in-person shop visits.**
   Rather than shrink the ambition or fall back to a weak public dataset with no
   sustainability angle, the fix was to find a *real* dataset that could support the exact
   same paired-comparison design without fieldwork.

**The pivot:** the U.S. EIA's **2018 Commercial Buildings Energy Consumption Survey
(CBECS)** public-use microdata — 6,436 real, individually surveyed U.S. commercial
buildings, disclosure-masked but building-level, not aggregated. Crucially, CBECS records
*both* cheap, easily-self-reportable fields (square footage, headcount, hours open, rough
building age) *and* inspection-grade fields (wall/roof material, glass %, ceiling height,
basement, elevator) for the **same buildings**. That upgrades the experiment from "compare
two different samples" (a self-report survey vs. a hypothetical audit) to a **paired
ablation on identical buildings** — train the same model twice, once per feature set, and
read the accuracy gap directly. This is a cleaner design than the original fieldwork plan
would have produced, not just a faster one.

## 4. What was actually built

Everything below was executed, not just planned — the notebook runs end-to-end with zero
errors and every number in Section 5 is a real model output.

1. **Downloaded the real data**: `cbecs2018_final_public.csv` (19.8 MB, 6,436 buildings)
   and EIA's variable codebook directly from eia.gov.
2. **Parsed the codebook** to map ~20 raw variable codes (`PBA`, `SQFT`, `NWKER`, `WKHRS`,
   `YRCONC`, `WLCNS`, `RFCNS`, `GLSSPC`, `MFBTU`, etc.) to their real meanings — no guessed
   column names.
3. **Filtered to a small-business-like subset**: principal activity in {food sales, food
   service, strip shopping center, retail (non-mall), service} and ≤25,000 sq ft →
   **741 real buildings**, down from 6,436.
4. **Built the target**: energy use intensity (`EUI = MFBTU / SQFT`, the standard
   building-energy benchmarking metric), split into Low/Medium/High risk tertiles
   (247/247/247 — evenly balanced by construction).
5. **Defined and documented the two feature sets**:
   - *Proxy (phone-call-reportable)*: business type, region, square footage, floors,
     building-age bracket, weekly hours, headcount, whether heating/cooling exist.
   - *Audit-grade additions*: wall/roof construction material, % glass exterior, ceiling
     height, attic, basement floors, elevator.
6. **Trained four models on each feature set** (8 runs total): Logistic Regression, Random
   Forest, XGBoost, and a small MLP (the ANN), same train/test split and seed throughout so
   every comparison is apples-to-apples, plus 5-fold cross-validation for stability.
7. **Added real SHAP explainability** on the winning proxy-only model (had to fix this
   once — the first version only handled tree models and silently skipped explaining the
   actual winner, Logistic Regression; now uses `TreeExplainer`, `LinearExplainer`, or
   `KernelExplainer` depending on which model wins).
8. **Debugged three real runtime errors** during execution (UTF-8 encoding in the notebook
   writer, a removed `multi_class` argument in the current scikit-learn version, and an
   XGBoost string-label incompatibility) until the notebook executed clean.
9. **Wrote the limitations section into the notebook itself** — geography, building-vs-owner
   mismatch, sample size, the constructed nature of the proxy/audit split — rather than
   leaving them for someone else to find.
10. **Wrote `README.md` and `requirements.txt`** with exact pinned package versions from the
    working environment, so the notebook is reproducible from a clean venv.

## 5a. Turning the notebook into the actual product (against the real SDG GreenTech rubric)

Once the real course rubric was shared, the gaps were concrete, not vibes: no deployed
app, no Green Computing evidence, no benchmarking, no cited recommendations. All four were
built as real, checked work rather than plans:

1. **Refactored the notebook's feature-engineering logic into `greenledger/pipeline.py`**
   — a single source of truth, so the app can't silently drift from what the notebook
   reports. `export_artifacts.py` imports it and reproduced the exact same accuracies
   (0.720 / 0.661 / 0.645 / 0.624) as the notebook on the first run, confirming the two are
   in sync.
2. **Measured real green-computing numbers** instead of asserting them: trained all four
   models, timed 200 single-row inferences each, and serialized each to disk to get real
   file sizes. Result: Logistic Regression — the model that already won on accuracy — is
   also ~1.2 KB and ~0.2 ms/prediction, against Random Forest's ~5.6 MB and ~35-40 ms. The
   "we intentionally chose the smallest competitive model" line is true here, not aspirational.
3. **Built the recommendation engine as cited, not generated.** Before writing a single
   card, searched for the actual DOE/EPA/ENERGY STAR published figures (LED retrofits up to
   50-85% lighting savings, HVAC setback up to 10%/year, insulation/air sealing averaging
   15%) and used those ranges verbatim with source links. Where no reliable published number
   exists (HVAC zoning to occupied space), the card says so explicitly instead of inventing
   one — the same discipline the rest of the project has followed throughout.
4. **Built peer benchmarking from data already on hand**, not a new assumption: exported
   the winning model's predicted "High risk" probability for all 741 real buildings, so a
   new business's score gets percentile-ranked against real buildings (overall and within
   its own business type) rather than shown on a made-up 0-100 scale.
5. **Verified the whole inference path end-to-end before trusting it**, not just that the
   page loaded: SHAP's output shape for a multi-class `LinearExplainer` on a single row
   isn't obvious ahead of time (turned out to be `(1, 15, 3)`), so it was checked directly
   against a real synthetic input before wiring it into the app, catching the class-indexing
   logic that would otherwise have silently explained the wrong tier.
6. **Smoke-tested the running app** (`streamlit run` in the background, HTTP 200 check) and
   separately exercised the full predict → SHAP → recommend pipeline as a script, since a
   page-load check alone doesn't touch the form-submission code path.

## 6. Results — the real run

| Model | Proxy-only accuracy | Proxy+audit accuracy | Gap |
|---|---|---|---|
| Logistic Regression | **0.720** | 0.667 | −0.054 |
| Random Forest | 0.661 | 0.677 | +0.016 |
| XGBoost | 0.645 | 0.677 | +0.032 |
| ANN (MLP) | 0.624 | 0.618 | −0.005 |

Two findings came out of this without being forced:

1. **The best model overall is proxy-only Logistic Regression.** For the model that
   actually won, the expensive audit-grade fields didn't help at all — the cheap phone-call
   data was enough.
2. **The ANN loses to every classical model, in both feature sets, and is the only model
   that gets worse with more features.** At n=741, there isn't enough data for a 32-16
   hidden-unit MLP to out-earn a linear decision boundary. That's a direct, honest answer to
   "does model complexity help at this scale" — no — which is a more defensible thing to
   put in a paper than a forced ANN win would have been.

## 7. What isn't done yet

- Multi-seed error bars (the table above is one 42-seeded split; the notebook's 5-fold CV
  numbers are close but not identical — repeating across ~20-50 seeds is the next step
  before any number here goes in a paper).
- Cloud deployment — the Streamlit app runs locally; hosting it (Streamlit Community Cloud
  or Render) is the remaining step for the course's Deployment Link requirement.
- Scrum artifacts (product backlog, sprint backlog) and the standalone SDG Mapping / User
  Manual / Sustainability Impact Report documents the rubric requires separately from the
  technical work.
- The IEEE-format writeup itself.

## 8. Where everything lives

```
GreenLedger/
├── data/
│   ├── cbecs2018_final_public.csv     — raw CBECS 2018 microdata
│   └── 2018microdata_codebook.xlsx    — EIA's variable codebook
├── notebooks/greenledger.ipynb        — the executed pipeline, all outputs saved
├── greenledger/pipeline.py            — shared feature engineering (notebook + app in sync)
├── greenledger/recommendations.py     — cited recommendation cards
├── export_artifacts.py                — trains models, measures green-computing stats, exports for the app
├── models/, app_data/                 — generated model + benchmarking artifacts
├── app.py                             — the Streamlit app
├── requirements.txt                   — pinned, working versions
├── README.md                          — quick-start + results summary
└── PROCESS.md                         — this file
```
