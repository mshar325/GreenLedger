"""Vein 1 -- Counterfactual density (bunching) test at the MEES E/F threshold.

Design (Chetty et al. 2011 / Kleven & Waseem 2013, notch variant):
  Bin certificate counts c_j at integer asset ratings j inside a fit window that
  spans exactly the two bands adjacent to the threshold (no other band boundary
  inside). Fit
      c_j = sum_i beta_i * x_j^i  +  sum_{k in Z} gamma_k 1[j=k]
            + delta_5 1[j % 5 == 0, j not in Z] + delta_10 1[j % 10 == 0, j not in Z] + eps_j
  where x_j is the centered/scaled rating and Z is an excluded window around the
  threshold. The polynomial + heaping terms form the counterfactual c_hat_j; the
  gamma's measure bin-level deviations inside Z.

  Excess mass  B = sum of (c_j - c_hat_j) over the "just-compliant" side of Z
  Missing mass M = sum of (c_hat_j - c_j) over the "just-failing"  side of Z
  Normalized  b  = B / mean(c_hat_j over the just-compliant side)     (Chetty)
  Under pure relabeling across the notch, M/B ~ 1.

  SEs / CIs: parametric residual bootstrap (resample non-excluded-bin residuals,
  add to fitted values everywhere, re-estimate; percentile CI).

Thresholds:
  MEES   E/F at 125/126  (letting a rating >=126 building is unlawful: notch)
  Placebo C/D at 75/76 and B/C at 50/51 (no regulatory consequence)

Lower asset_rating = better. "Just compliant" = bins <= threshold.

Run:  ./venv/Scripts/python.exe analysis/vein1_bunching.py
Outputs: analysis/results/vein1_bunching_results.csv + figures.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from greenledger.pipeline import SMALL_BIZ_PATTERN, FLOOR_AREA_CAP_M2  # noqa: E402

DATA_DIR = ROOT / "data" / "uk_epc"
OUT_DIR = Path(__file__).resolve().parent / "results"
OUT_DIR.mkdir(exist_ok=True)

YEARS = list(range(2012, 2027))
POLY_DEGREE = 4
N_BOOT = 500
SEED = 42

# threshold: (fit_lo, fit_hi, excl_lo, excl_hi)  -- fit window spans exactly the two
# adjacent bands so no other band boundary sits inside it.
THRESHOLDS = {
    "E/F 125 (MEES)": (125, 101, 150, 121, 130),
    "C/D 75 (placebo)": (75, 51, 100, 71, 80),
    "B/C 50 (placebo)": (50, 26, 75, 46, 55),
}


def load_year(year):
    f = DATA_DIR / f"certificates-{year}.csv"
    df = pd.read_csv(f, usecols=["asset_rating", "property_type", "floor_area"])
    df = df[df["asset_rating"].notna()]
    df["asset_rating"] = df["asset_rating"].round().astype(int)
    small = df[df["property_type"].str.contains(SMALL_BIZ_PATTERN, na=False)
               & (df["floor_area"] <= FLOOR_AREA_CAP_M2)]
    return df["asset_rating"], small["asset_rating"]


def bunching_estimate(ratings, threshold, fit_lo, fit_hi, excl_lo, excl_hi,
                       poly=POLY_DEGREE, n_boot=N_BOOT, seed=SEED):
    bins = np.arange(fit_lo, fit_hi + 1)
    counts = ratings.value_counts().reindex(bins, fill_value=0).sort_index()
    c = counts.values.astype(float)
    x = (bins - threshold) / 10.0  # centered/scaled for conditioning

    in_Z = (bins >= excl_lo) & (bins <= excl_hi)
    X_cols = [x ** i for i in range(poly + 1)]
    heap5 = ((bins % 5 == 0) & ~in_Z).astype(float)
    heap10 = ((bins % 10 == 0) & ~in_Z).astype(float)
    X_cols += [heap5, heap10]
    Z_dummies = [(bins == k).astype(float) for k in range(excl_lo, excl_hi + 1)]
    X = np.column_stack(X_cols + Z_dummies)

    below = (bins >= excl_lo) & (bins <= threshold)   # just-compliant side of Z
    above = (bins > threshold) & (bins <= excl_hi)    # just-failing side of Z

    def estimate(y):
        fit = sm.OLS(y, X).fit()
        beta = fit.params
        cf = X[:, : poly + 3] @ beta[: poly + 3]      # polynomial + heaping only
        B = float(np.sum(y[below] - cf[below]))
        M = float(np.sum(cf[above] - y[above]))
        denom = float(np.mean(cf[below]))
        b = B / denom if denom > 0 else np.nan
        return fit, cf, B, M, b

    fit, cf, B, M, b = estimate(c)

    rng = np.random.default_rng(seed)
    resid = c[~in_Z] - fit.fittedvalues[~in_Z]
    boots = np.empty(n_boot)
    for r in range(n_boot):
        eps = rng.choice(resid, size=len(c), replace=True)
        y_star = np.clip(fit.fittedvalues + eps, 0, None)
        boots[r] = estimate(y_star)[4]
    lo, hi = np.percentile(boots, [2.5, 97.5])

    return {"B_excess": B, "M_missing": M, "b_norm": b, "b_se": float(np.std(boots)),
            "b_ci_lo": float(lo), "b_ci_hi": float(hi),
            "mass_ratio_M_over_B": M / B if B > 0 else np.nan,
            "n_in_window": int(c.sum()), "bins": bins, "counts": c, "counterfactual": cf}


def main():
    rows, pooled = [], {"full": {}, "smallbiz": {}}
    for year in YEARS:
        try:
            full, small = load_year(year)
        except FileNotFoundError:
            continue
        for sample, ratings in [("full", full), ("smallbiz", small)]:
            pooled[sample][year] = ratings
            for name, (thr, flo, fhi, zlo, zhi) in THRESHOLDS.items():
                res = bunching_estimate(ratings, thr, flo, fhi, zlo, zhi)
                rows.append({"year": year, "sample": sample, "threshold": name,
                             **{k: v for k, v in res.items()
                                if k not in ("bins", "counts", "counterfactual")}})
                print(f"{year} {sample:8s} {name:18s} b={res['b_norm']:7.2f} "
                      f"[{res['b_ci_lo']:6.2f},{res['b_ci_hi']:6.2f}]  M/B={res['mass_ratio_M_over_B']:5.2f}")

    results = pd.DataFrame(rows)
    results.to_csv(OUT_DIR / "vein1_bunching_results.csv", index=False)

    # Figure 1: observed vs counterfactual at the MEES threshold, pooled MEES-era full sample
    mees_era = pd.concat([pooled["full"][y] for y in pooled["full"] if y >= 2018])
    thr, flo, fhi, zlo, zhi = THRESHOLDS["E/F 125 (MEES)"]
    res = bunching_estimate(mees_era, thr, flo, fhi, zlo, zhi)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(res["bins"], res["counts"], color="#4c7a5a", width=0.9, label="Observed")
    ax.plot(res["bins"], res["counterfactual"], color="#c9683e", lw=2.5, label="Counterfactual")
    ax.axvspan(zlo - 0.5, zhi + 0.5, color="grey", alpha=0.15, label="Excluded window")
    ax.axvline(125.5, color="#8f2d2d", ls="--", lw=1.5, label="MEES E/F boundary")
    ax.set_xlabel("Asset rating (lower = better)"); ax.set_ylabel("Certificates")
    ax.set_title(f"Bunching at the MEES E/F notch, pooled 2018-2026 (b = {res['b_norm']:.1f})")
    ax.legend()
    plt.tight_layout(); plt.savefig(OUT_DIR / "vein1_pooled_mees.png", dpi=150); plt.close()

    # Figure 2: year-by-year normalized excess mass, MEES vs placebos (full sample)
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = {"E/F 125 (MEES)": "#8f2d2d", "C/D 75 (placebo)": "#4c7a5a", "B/C 50 (placebo)": "#3e5266"}
    for name in THRESHOLDS:
        sub = results[(results["sample"] == "full") & (results["threshold"] == name)]
        ax.errorbar(sub["year"], sub["b_norm"],
                     yerr=[sub["b_norm"] - sub["b_ci_lo"], sub["b_ci_hi"] - sub["b_norm"]],
                     marker="o", capsize=3, label=name, color=colors[name])
    for xv, lbl in [(2018, "MEES: new lets"), (2023, "MEES: all lets")]:
        ax.axvline(xv - 0.5, color="grey", ls=":", lw=1)
        ax.text(xv - 0.45, ax.get_ylim()[1] * 0.95, lbl, fontsize=8, color="grey")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("Certificate year"); ax.set_ylabel("Normalized excess mass b")
    ax.set_title("Excess mass at the E/F notch vs placebo boundaries, by year")
    ax.legend()
    plt.tight_layout(); plt.savefig(OUT_DIR / "vein1_by_year.png", dpi=150); plt.close()

    print("\nSaved:", OUT_DIR / "vein1_bunching_results.csv", "+ 2 figures")


if __name__ == "__main__":
    main()
