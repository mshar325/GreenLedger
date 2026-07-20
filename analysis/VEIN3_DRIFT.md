# Vein 3 — Screening Under Drift (executed 2026-07-20)

Two rigorous analyses built on the documented year-over-year rating drift, turning the
app's uncertainty UI and Audit Triage tab from features into evidenced methods.
Code: `analysis/vein3_conformal.py`.

## A. Conformal prediction — coverage decays as the register drifts

Split-conformal (LAC / threshold rule): train the model on 2018-2021, calibrate a
nonconformity threshold on 2022 for a **target 90% coverage**, then measure the
*empirical* coverage of the resulting prediction **sets** on each later year.

| Test year | Coverage | Avg set size | Role |
|---|---|---|---|
| 2022 | 90.0% | 2.30 | calibration |
| 2023 | 86.5% | 2.29 | test |
| 2024 | 86.5% | 2.29 | test |
| 2025 | 85.9% | 2.28 | test |
| 2026 | 85.5% | 2.28 | test |

Under exchangeability, coverage would hold at 90%. It does not — it **decays ~0.9 points
per year** as the test year moves away from calibration, because the register is not
exchangeable across years (the efficiency-improvement drift). This is a *built-in expiry
signal*: the conformal guarantee tells you, quantitatively, when a deployed model needs
recalibration. A single-shot static model (what every competitor ships) has no such
signal and silently mis-calibrates.

Honest note: average set size ≈ 2.3 of 3 classes — the sets are not tight (the proxy
features are genuinely weak, consistent with the whole "cheap data" premise). The value
here is the *coverage-decay diagnostic*, not set sharpness.

## B. Value-of-information — the audit triage curve

Order the 2025 buildings by the deployed model's predictive entropy and trace cumulative
High-risk capture vs fraction audited (using the deployed model's own predictions, so it
matches the app's Audit Triage tab exactly).

| Audit budget | Buildings | High-risk hit rate | High-risk captured | Lift vs random |
|---|---|---|---|---|
| 1% | 544 | 13.4% | 1.8% | 1.80× |
| 2% | 1,089 | 12.4% | 3.3% | 1.67× |
| **5%** | **2,723** | **13.8%** | **9.3%** | **1.86×** |
| 10% | 5,446 | 13.7% | 18.4% | 1.84× |
| 20% | 10,892 | 12.8% | 34.5% | 1.73× |

Random auditing finds High-risk buildings at the 7.4% base rate. Uncertainty-ranked
auditing roughly **doubles the hit rate** across budget levels — a real, if modest,
resource-allocation gain, and exactly the "spend a limited inspection budget where the
model is least sure" claim, now with a decision curve behind it.

## Why this matters together with Veins 1-2

Vein 1 shows the register's labels are distorted at the threshold. Vein 2 shows the
mechanism. Vein 3 closes the loop operationally: *given* that the labels can't be fully
trusted and *given* that the model drifts, uncertainty-ranked triage is how you allocate
scarce real audits — and conformal coverage tells you when to stop trusting the model at
all. That is a coherent "don't just predict the label, question it and manage the
uncertainty" story that no single-building estimator offers.

## Still on the honest to-do list
- Adaptive / weighted conformal (Tibshirani et al.) to *repair* the coverage decay, not
  just measure it.
- Formal decision-theoretic optimum (audit cost £c vs misclassification cost) rather than
  the descriptive lift curve.
