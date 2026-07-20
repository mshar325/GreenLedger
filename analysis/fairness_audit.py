"""Fairness / bias audit of the risk model and the audit-triage queue.

For a tool that flags "High risk" and allocates scarce inspections, the question is not
just "is it accurate?" but "who does it flag, and does it flag some groups beyond what
their real base rate justifies?" Uses the exported 2025 out-of-time predictions.

Metrics per group (region, business type):
  base_rate     -- actual share High (ground truth, 2025)
  flag_rate     -- share the model predicts High (selection rate)
  precision     -- of those flagged High, share actually High (calibration / reliability)
  recall        -- of actual High, share the model flags (equal-opportunity view)
  flag_gap      -- flag_rate - base_rate (does the model over/under-flag this group?)
  queue_share   -- group's share of the top-5% uncertainty audit queue vs its pop share

Outputs: analysis/results/fairness_*.csv + app_data/fairness_summary.csv, printed report.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(exist_ok=True)
APP_DATA = ROOT / "app_data"

t = pd.read_csv(APP_DATA / "triage_2025.csv.gz", compression="gzip")
t["uncertainty"] = (t["uncertainty_entropy"] / np.log(3)).clip(0, 1)
t["pred_high"] = (t["predicted_tier"] == "High").astype(int)
t["actual_high"] = (t["actual_tier"] == "High").astype(int)

queue = t.sort_values(["uncertainty", "p_high"], ascending=False).head(int(len(t) * 0.05))


def group_audit(col):
    rows = []
    pop = len(t)
    qn = len(queue)
    for name, g in t.groupby(col):
        if len(g) < 100:                # skip tiny/artefact groups (e.g. Scotland border rows)
            continue
        flagged = g["pred_high"].mean()
        base = g["actual_high"].mean()
        prec = g.loc[g["pred_high"] == 1, "actual_high"].mean() if g["pred_high"].sum() else np.nan
        rec = g.loc[g["actual_high"] == 1, "pred_high"].mean() if g["actual_high"].sum() else np.nan
        qshare = (queue[col] == name).mean()
        popshare = len(g) / pop
        rows.append({col: name, "n": len(g),
                     "base_rate_%": round(base * 100, 1), "flag_rate_%": round(flagged * 100, 1),
                     "flag_gap_pts": round((flagged - base) * 100, 1),
                     "precision_%": round(prec * 100, 1), "recall_%": round(rec * 100, 1),
                     "pop_share_%": round(popshare * 100, 1), "queue_share_%": round(qshare * 100, 1),
                     "queue_over_rep": round(qshare / popshare, 2) if popshare else np.nan})
    return pd.DataFrame(rows).sort_values("flag_gap_pts", ascending=False)


for col, label in [("uk_region", "region"), ("property_type_group", "business type")]:
    tab = group_audit(col)
    tab.to_csv(OUT / f"fairness_by_{label.replace(' ', '_')}.csv", index=False)
    print(f"\n=== Fairness by {label} ===")
    print(tab.to_string(index=False))

# compact summary for the app: worst over-flag and worst queue over-representation
reg = group_audit("uk_region")
typ = group_audit("property_type_group")
summ = pd.DataFrame([
    {"dimension": "region", "metric": "largest over-flag (flag_rate - base_rate)",
     "group": reg.iloc[0]["uk_region"], "value": f"+{reg.iloc[0]['flag_gap_pts']} pts"},
    {"dimension": "region", "metric": "precision spread (max - min)",
     "group": "—", "value": f"{reg['precision_%'].max()-reg['precision_%'].min():.1f} pts"},
    {"dimension": "business type", "metric": "largest over-flag",
     "group": typ.iloc[0]["property_type_group"], "value": f"+{typ.iloc[0]['flag_gap_pts']} pts"},
    {"dimension": "business type", "metric": "queue over-representation (max)",
     "group": typ.sort_values("queue_over_rep", ascending=False).iloc[0]["property_type_group"],
     "value": f"{typ['queue_over_rep'].max():.2f}x pop share"},
])
summ.to_csv(APP_DATA / "fairness_summary.csv", index=False)
print("\n=== Compact summary (app) ===")
print(summ.to_string(index=False))
print("\nSaved fairness CSVs.")
