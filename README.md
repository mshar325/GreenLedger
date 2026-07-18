# GreenLedger

Can a small business's energy-rating risk be predicted from a short, self-reportable
questionnaire — instead of a full building audit? And does an ANN actually help at this
data scale?

**v2**: built on the UK's real non-domestic **Energy Performance Certificate register**
(epc.opendatacommunities.org), pooled 2018-2026 — 430,942 buildings (2018-2024) for
training, 54,460 buildings assessed in 2025 held out as a genuine out-of-time test set.
This replaces the original CBECS-based v1 (741 U.S. buildings) — see
[`PROCESS.md`](PROCESS.md) for the full story of why, including a filter bug caught and
fixed mid-migration and a model-selection mistake caught before it shipped. Live app:
deployed on Streamlit Community Cloud (private).

## What's here

```
GreenLedger/
├── data/
│   ├── uk_epc/*.csv                   # raw EPC certificates by year (gitignored -- ~300MB, see below)
│   ├── uk_outcodes.csv                # UK postcode-district → lat/long lookup, for the map
│   └── cbecs2018_final_public.csv     # v1's dataset, kept for history (see PROCESS.md)
├── notebooks/greenledger.ipynb        # full pipeline: EDA -> features -> models -> SHAP
├── greenledger/
│   ├── pipeline.py                    # shared data prep / feature engineering (single source of truth)
│   ├── recommendations.py             # rule-based recs, cited to Carbon Trust / GOV.UK
│   └── report.py                      # optional LLM (Groq) report generation, strictly grounded
├── export_artifacts.py                # trains models, temporal holdout, green-computing stats, dashboard data
├── models/, app_data/                 # exported model + benchmarking + dashboard artifacts (generated)
├── app.py                             # the Streamlit app (Risk Assessment, Dashboard, Green Computing, About)
├── requirements.txt
└── venv/                              # local virtualenv (not portable — rebuild with the command below)
```

## Getting the raw data

Bulk non-domestic EPC data requires a free GOV.UK One Login account:
1. Go to [epc.opendatacommunities.org](https://epc.opendatacommunities.org/), choose
   **Non-domestic Energy Performance Certificates**.
2. Download the bulk CSV set, extract, and copy `certificates-2018.csv` through
   `certificates-2026.csv` into `data/uk_epc/`.
3. `data/uk_outcodes.csv` (postcode-district centroids, for the map) is small enough to be
   committed and is already here.

## Method, in one paragraph

Filtered the EPC register to buildings whose property type reads as an independent small
business (retail/financial/professional, restaurant/cafe, office/workshop) under 500 m².
Target: the *official assessed* `asset_rating_band` (A+ best to G worst — not a label this
project invented), collapsed to Low/Medium/High. Two feature sets: **proxy** (business
type, region, floor area, main heating fuel, air-conditioning presence, building
environment, assessment year — all self-reportable) and **proxy+audit** (adds
air-conditioning capacity rating, the one field here that plausibly needs a real
inspection). Four models — Logistic Regression, Random Forest, XGBoost, a small MLP —
trained on 2018-2024, tested on 2025 buildings none of them saw during training.

## What it found — and why picking a "winner" by accuracy would have been a mistake

| Model | Accuracy | Macro F1 | **High-risk recall** |
|---|---|---|---|
| Logistic Regression | 0.413 | 0.234 | 0.049 |
| Random Forest | 0.529 | 0.480 | **0.699** |
| XGBoost | 0.547 | 0.493 | 0.584 |
| ANN (MLP) | **0.681** | 0.491 | 0.044 |

The ANN has the highest raw accuracy of any model here — and catches essentially none of
the buildings that are genuinely High risk (4.4% recall). It wins accuracy by defaulting to
the majority "Medium" tier. A tool meant to flag at-risk businesses that mostly says
"you're fine" is not fit for purpose, however good its accuracy number looks — so the app
selects its model on **High-risk recall**, not accuracy, which picks **Random Forest**
instead (≈70% recall). This is a real methodological finding, not a footnote: see
`notebooks/greenledger.ipynb` Section 5 for the full confusion-matrix comparison.

On the metric that matters, the audit-only field barely helps (Random Forest: 0.699 vs.
0.700 proxy vs. proxy+audit) — the narrower echo of the original "cheap data gets you most
of the way there" finding, on a thinner audit-grade feature set than the original CBECS
version had access to.

## Running it

```bash
py -3.12 -m venv venv
./venv/Scripts/pip install -r requirements.txt
./venv/Scripts/python export_artifacts.py          # trains models, ~4 min on this dataset's scale
./venv/Scripts/python -m streamlit run app.py
```

Or re-execute the notebook headless end to end:

```bash
./venv/Scripts/python -m jupyter nbconvert --to notebook --execute --inplace notebooks/greenledger.ipynb
```

The app has four tabs: **Risk Assessment** (questionnaire → risk tier, SHAP-based "why,"
peer benchmarking, cited recommendations, optional LLM-written report), **Dashboard**
(KPIs, a UK hexagon-bin map, rating-band and regional breakdowns — built from the full
~485k-row pooled dataset), **Green Computing** (the real size/speed/recall comparison, and
the honest tension described below), and **About**.

## Green computing — an honest tension, not a clean win

Unlike v1 (where the most accurate model also happened to be the smallest), here the
model that actually does the job — Random Forest, on High-risk recall — is also the
**largest and slowest** in the comparison (~61 MB, ~59 ms/prediction) versus XGBoost's
~1.4 MB and ~1 ms for noticeably worse recall. The app deploys the model that works, and
says so plainly rather than claiming a free lunch that isn't there — see the Green
Computing tab and `notebooks/greenledger.ipynb` Section 5.

## Honest limitations (stated here, in the notebook, and in the app's About tab)

- EPC coverage is England and Wales only; the method transfers, the specific numbers may
  not.
- The register gives postcode-district location, not exact coordinates — the dashboard
  map jitters points for visual clarity, disclosed there.
- A real secular trend exists in ratings across years (buildings have gotten more
  efficient almost every year) — handled via the 2018-2024 → 2025 out-of-time split rather
  than a random split, which is a harder, more honest test by design.
- The audit-grade feature set here is thinner than v1's (mainly one HVAC capacity field) —
  this register doesn't capture wall/roof/insulation detail the way CBECS did.
- `MLPClassifier` has no `sample_weight` support in scikit-learn, unlike every other model
  compared here — a real, disclosed asymmetry in the ANN comparison, not a level playing
  field pretending to be one.
- Risk tiers collapse the official 8-band rating into 3 for modeling; the full A+-G
  distribution is shown as-is in the Dashboard tab.

## Not built yet

- Multi-seed / bootstrapped confidence intervals around the recall numbers.
- The IEEE-format writeup (now with a stronger "accuracy vs. recall" methodological angle).
- Cloud deployment of this v2 (v1 was deployed; this version needs the same GitHub push +
  Streamlit Cloud redeploy).
- Scrum artifacts and the three standalone SDG GreenTech documents (SDG Mapping, User
  Manual, Sustainability Impact Report).
