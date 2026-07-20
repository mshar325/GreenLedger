"""Vein 3 -- Screening under drift, made rigorous.

Two analyses on the documented year-over-year rating drift:

 (A) Split-conformal prediction (LAC / threshold rule). Train the model on 2018-2021,
     calibrate a nonconformity threshold on 2022 for a target 90% coverage, then measure
     the EMPIRICAL coverage of the resulting prediction SETS on each later year
     (2023, 2024, 2025). Under exchangeability coverage should hold at ~90%; the whole
     point is that the register is NOT exchangeable across years (drift), so coverage is
     expected to decay as the test year moves away from the calibration year -- and we
     quantify that decay, plus the average set size (efficiency).

 (B) Value-of-information / audit triage curve. Order the 2025 buildings by model
     uncertainty (predictive entropy) and trace cumulative High-risk capture vs fraction
     audited, against the random-audit baseline -- formalizing the Audit Triage tab as a
     decision curve with a single lift/AUC-style summary.

Outputs: analysis/results/vein3_*.csv + app_data/conformal_coverage.csv, triage_curve.csv,
and two figures.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_sample_weight

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from greenledger.pipeline import PROXY_CAT, PROXY_NUM, LABELS, build_feature_matrix, load_pooled  # noqa: E402

OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(exist_ok=True)
APP_DATA = ROOT / "app_data"
ALPHA = 0.10           # target 90% coverage
SEED = 42

DATA, LUT = ROOT / "data" / "uk_epc", ROOT / "data" / "uk_outcodes.csv"
print("Loading years...")
by_year = {y: load_pooled(DATA, LUT, [y]) for y in range(2018, 2027)
           if (DATA / f"certificates-{y}.csv").exists()}


def feats(df, cols):
    return build_feature_matrix(df, PROXY_NUM, PROXY_CAT).reindex(columns=cols, fill_value=0)


def ycodes(df):
    return pd.Categorical(df["risk_tier"], categories=LABELS, ordered=True).codes


# ---- train on 2018-2021, calibrate on 2022 ----------------------------------------
train_df = pd.concat([by_year[y] for y in (2018, 2019, 2020, 2021)], ignore_index=True)
Xtr = build_feature_matrix(train_df, PROXY_NUM, PROXY_CAT)
cols = list(Xtr.columns)
ytr = ycodes(train_df)
rf = RandomForestClassifier(n_estimators=200, max_depth=12, n_jobs=4, random_state=SEED)
rf.fit(Xtr.values, ytr, sample_weight=compute_sample_weight("balanced", ytr))
print(f"  trained on {len(train_df):,} (2018-2021)")

cal = by_year[2022]
Xcal, ycal = feats(cal, cols).values, ycodes(cal)
p_cal = rf.predict_proba(Xcal)
# LAC nonconformity: s = 1 - p(true class). Threshold = (1-alpha) quantile w/ finite-sample correction.
scores = 1 - p_cal[np.arange(len(ycal)), ycal]
n = len(scores)
qhat = np.quantile(scores, min(1.0, np.ceil((n + 1) * (1 - ALPHA)) / n), method="higher")
print(f"  calibrated on 2022 (n={n:,}); qhat={qhat:.3f} -> include class if p >= {1-qhat:.3f}")

# ---- (A) coverage + set size by test year -----------------------------------------
rows = []
for y in range(2022, 2027):
    if y not in by_year:
        continue
    d = by_year[y]
    Xy, yy = feats(d, cols).values, ycodes(d)
    py = rf.predict_proba(Xy)
    sets = py >= (1 - qhat)                        # boolean [n,3]
    covered = sets[np.arange(len(yy)), yy]
    rows.append({"year": y, "n": len(yy), "coverage": round(covered.mean(), 3),
                 "avg_set_size": round(sets.sum(1).mean(), 3),
                 "role": "calibration" if y == 2022 else "test"})
    print(f"  {y}: coverage {covered.mean()*100:.1f}%  avg set size {sets.sum(1).mean():.2f}")
cov = pd.DataFrame(rows)
cov.to_csv(OUT / "vein3_conformal_coverage.csv", index=False)
cov.to_csv(APP_DATA / "conformal_coverage.csv", index=False)

# ---- (B) value-of-information triage curve on 2025 --------------------------------
# Uses the DEPLOYED model's exported predictions (triage_2025.csv.gz), NOT the fresh
# 2018-2021 conformal model above, so this curve matches the app's Audit Triage tab
# exactly rather than reporting a second, contradictory lift number.
tri = pd.read_csv(APP_DATA / "triage_2025.csv.gz", compression="gzip")
proba = tri[["p_low", "p_medium", "p_high"]].values
entropy = -np.nansum(proba * np.log(np.clip(proba, 1e-12, 1)), axis=1)
is_high = (tri["actual_tier"] == "High").astype(int).values
yte = is_high  # for length only below
order = np.argsort(-entropy)                      # most uncertain first
cum_high = np.cumsum(is_high[order])
total_high = is_high.sum()
fracs = np.arange(1, len(yte) + 1) / len(yte)
capture = cum_high / total_high
# summarize at audit-budget checkpoints
curve_rows = []
for b in [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25]:
    k = max(1, int(len(yte) * b))
    hit = is_high[order[:k]].mean() * 100
    curve_rows.append({"budget_frac": b, "n_audited": k,
                       "high_hit_rate_%": round(hit, 1),
                       "high_captured_%": round(100 * cum_high[k - 1] / total_high, 1),
                       "random_hit_rate_%": round(is_high.mean() * 100, 1),
                       "lift": round(hit / (is_high.mean() * 100), 2)})
curve = pd.DataFrame(curve_rows)
curve.to_csv(OUT / "vein3_triage_curve.csv", index=False)
curve.to_csv(APP_DATA / "triage_curve.csv", index=False)
print("\nTriage decision curve (2025):")
print(curve.to_string(index=False))

# ---- figures ----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 4.5))
t = cov[cov.role != "calibration"]
ax.axhline(90, color="#4c7a5a", ls="--", label="Target 90% coverage")
ax.plot(cov["year"], cov["coverage"] * 100, marker="o", color="#8f2d2d", lw=2.5,
        label="Realized coverage")
ax.axvline(2022, color="grey", ls=":", lw=1); ax.text(2022.05, ax.get_ylim()[0], "calibrated here", fontsize=8)
ax.set_xlabel("Test year"); ax.set_ylabel("Empirical coverage (%)")
ax.set_title("Conformal coverage decays as drift grows (calibrated on 2022)")
ax.legend(); plt.tight_layout(); plt.savefig(OUT / "vein3_coverage.png", dpi=150); plt.close()

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(fracs * 100, capture * 100, color="#2f5d3c", lw=2.5, label="Uncertainty-ranked audit")
ax.plot([0, 100], [0, 100], color="#3e5266", ls="--", label="Random audit")
ax.set_xlabel("% of portfolio audited"); ax.set_ylabel("% of High-risk buildings caught")
ax.set_title("Audit triage value-of-information curve (2025)")
ax.legend(); plt.tight_layout(); plt.savefig(OUT / "vein3_triage_curve.png", dpi=150); plt.close()

print("\nSaved conformal_coverage.csv, triage_curve.csv + 2 figures")
