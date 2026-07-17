# GreenLedger

Can a small business's energy-intensity risk be predicted from only the operational
details an owner could state from memory in a phone call — instead of a full building
audit? And does an ANN actually help at that data scale?

Built on the real U.S. EIA **2018 Commercial Buildings Energy Consumption Survey (CBECS)**
public-use microdata — no fieldwork, no synthetic data. See
[`notebooks/greenledger.ipynb`](notebooks/greenledger.ipynb) for the full analysis.

## What's here

```
GreenLedger/
├── data/
│   ├── cbecs2018_final_public.csv     # raw CBECS 2018 microdata (6,436 buildings)
│   └── 2018microdata_codebook.xlsx    # EIA's variable codebook
├── notebooks/
│   └── greenledger.ipynb              # full pipeline: EDA -> features -> models -> SHAP
├── greenledger/
│   ├── pipeline.py                    # shared data prep / feature engineering (single source of truth)
│   └── recommendations.py             # rule-based recs, every number cited to DOE/EPA/ENERGY STAR
├── export_artifacts.py                # trains models, measures real green-computing stats, exports to models/ + app_data/
├── models/                            # exported model + scaler + metadata (generated, not committed)
├── app_data/                          # exported SHAP background + benchmarking distributions (generated)
├── app.py                             # the Streamlit app
├── requirements.txt
└── venv/                              # local virtualenv (not portable — rebuild with the command below)
```

## Method, in one paragraph

Filtered CBECS's 6,436 buildings down to 741 whose principal activity resembles an
independent small business (food sales, food service, strip shopping center, retail,
service) under 25,000 sq ft. Every building has both **proxy fields** (square footage,
headcount, hours open, rough building-age bracket — things an owner could state on a call)
and **audit-grade fields** (wall/roof material, % glass, ceiling height, basement,
elevator — things that need a walkthrough). Target: annual energy use intensity
(`MFBTU / SQFT`), split into Low/Medium/High risk tertiles. Four models — Logistic
Regression, Random Forest, XGBoost, a small MLP (the ANN) — trained twice each, on the
identical buildings and split: once proxy-only, once proxy+audit. The accuracy delta
between those two runs is the actual "cost of skipping the audit."

## What it found (one run, seed=42 — see Section 8 of the notebook for what's still needed to firm this up)

| Model | Proxy-only | Proxy+audit | Gap |
|---|---|---|---|
| Logistic Regression | **0.720** | 0.667 | −0.054 |
| Random Forest | 0.661 | 0.677 | +0.016 |
| XGBoost | 0.645 | 0.677 | +0.032 |
| ANN (MLP) | 0.624 | 0.618 | −0.005 |

- The best model overall is **proxy-only Logistic Regression** — the audit-grade fields
  didn't help the model that mattered most.
- The **ANN underperforms every classical model in both feature sets** and is the only
  model that gets worse with more features. At n=741, a 32-16 MLP doesn't have enough
  data to beat a linear boundary — a real, reportable answer to "does depth help here,"
  not a failed attempt at one.

## Running it

```bash
py -3.12 -m venv venv
./venv/Scripts/pip install -r requirements.txt
./venv/Scripts/jupyter notebook notebooks/greenledger.ipynb
```

Or re-execute the notebook headless end to end:

```bash
./venv/Scripts/python -m jupyter nbconvert --to notebook --execute --inplace notebooks/greenledger.ipynb
```

**The app.** `export_artifacts.py` retrains the four proxy-only models (using the exact
same `greenledger/pipeline.py` logic the notebook uses), measures real inference time and
model size for each, and exports the winner plus everything the app needs:

```bash
./venv/Scripts/python export_artifacts.py
./venv/Scripts/python -m streamlit run app.py
```

The app has three tabs: **Risk Assessment** (10-question form → risk tier, SHAP-based
"why," peer benchmarking against the real 741 buildings, and cited recommendations),
**Green Computing** (the real size/speed/accuracy comparison across all four models), and
**About** (method + limitations, same as below).

## Honest limitations (stated here, and again in the notebook and the app's About tab)

- CBECS is U.S.-only; the method transfers, the specific numbers may not.
- CBECS surveys the *building*, not the *business* — treated as a reasonable stand-in for
  a standalone small business at this size/activity filter, not a perfect match.
- The proxy vs. audit-grade split is a construction we assigned based on what's plausible
  to self-report vs. what needs inspection — documented in Section 3 of the notebook so
  it's checkable, not asserted.
- Single 80/20-ish split reported above; the notebook's cross-validated numbers (5-fold)
  are close but not identical — multi-seed error bars are the next thing to add before
  this goes in a paper.

## Green computing (real measurements, not estimates)

`export_artifacts.py` times and sizes all four models on the same machine and split. The
winning model, proxy-only Logistic Regression, is also the smallest and fastest by a wide
margin — roughly 1.2 KB and 0.2 ms per prediction versus Random Forest's ~5.6 MB and
~35-40 ms. It didn't need to be the efficient choice and the accurate choice at the same
time; it just happened to be both, which is exactly what the app's Green Computing tab
reports rather than asserts.

## Not built yet

- The recommendation engine's non-quantified card (space/HVAC zoning) intentionally has no
  cited percentage — no reliable published figure exists for it, so it stays qualitative
  rather than getting a made-up number.
- Multi-seed confidence intervals (see Limitations above).
- The IEEE-format writeup.
- Cloud deployment (currently local-only; Streamlit Community Cloud / Render are the
  straightforward next step since the app has no server-side state beyond the exported
  model files).
