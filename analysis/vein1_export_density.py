"""Exports app-facing CSVs for the MEES Distortion tab: pooled observed-vs-
counterfactual density around the E/F notch, and the by-year excess-mass series.
Run after vein1_bunching.py."""
import shutil
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from vein1_bunching import THRESHOLDS, bunching_estimate, load_year  # noqa: E402

APP_DATA = ROOT / "app_data"

ratings = []
for y in range(2018, 2027):
    full, _ = load_year(y)
    ratings.append(full)
pooled = pd.concat(ratings)

thr, flo, fhi, zlo, zhi = THRESHOLDS["E/F 125 (MEES)"]
res = bunching_estimate(pooled, thr, flo, fhi, zlo, zhi, n_boot=200)

pd.DataFrame({
    "rating": res["bins"], "observed": res["counts"], "counterfactual": res["counterfactual"],
}).to_csv(APP_DATA / "bunching_density.csv", index=False)

by_year = pd.read_csv(HERE / "results" / "vein1_bunching_results.csv")
by_year[by_year["sample"] == "full"][
    ["year", "threshold", "b_norm", "b_ci_lo", "b_ci_hi"]
].to_csv(APP_DATA / "bunching_by_year.csv", index=False)

print(f"pooled 2018-2026: b={res['b_norm']:.2f}, B={res['B_excess']:.0f}, M={res['M_missing']:.0f}")
print("wrote app_data/bunching_density.csv and app_data/bunching_by_year.csv")
