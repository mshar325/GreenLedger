"""Vein 2, causal-flavored: building fixed-effects (within) regression.

The descriptive panel says "escapes co-occur with fuel switches and floor-area changes."
A sharper question: holding the BUILDING fixed (absorbing every time-invariant thing
about it -- location, type, size class, owner), what within-building change in an
observable is associated with what change in the rating? We de-mean rating and predictors
within each building (the within / fixed-effects transformation) and OLS on the de-meaned
data with standard errors clustered by building. Still observational (unmeasured
concurrent changes remain), but it removes all cross-building confounding -- a real step
up from the raw cross-section.

Outputs: analysis/results/vein2_fixed_effects.csv, printed summary.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "uk_epc"
OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(exist_ok=True)

COLS = ["uprn", "lodgement_date", "asset_rating", "main_heating_fuel",
        "building_environment", "aircon_present", "floor_area"]
print("Loading panel...")
frames = []
for y in range(2012, 2027):
    f = DATA / f"certificates-{y}.csv"
    if f.exists():
        frames.append(pd.read_csv(f, usecols=COLS, dtype={"uprn": "string"}))
d = pd.concat(frames, ignore_index=True)
d = d[d["uprn"].notna() & (d["uprn"].str.strip() != "") & d["asset_rating"].notna()].copy()
d["lodgement_date"] = pd.to_datetime(d["lodgement_date"], errors="coerce")
d = d[d["lodgement_date"].notna()]
d["asset_rating"] = d["asset_rating"].round().astype(float)
d["year"] = d["lodgement_date"].dt.year

# keep buildings with 2+ certs (needed for within-building variation)
counts = d.groupby("uprn")["uprn"].transform("size")
d = d[counts >= 2].copy()
print(f"  {d['uprn'].nunique():,} buildings with 2+ certs, {len(d):,} certificates")

# observable predictors
d["elec"] = (d["main_heating_fuel"] == "Grid Supplied Electricity").astype(float)
d["gas"] = (d["main_heating_fuel"] == "Natural Gas").astype(float)
d["oil"] = (d["main_heating_fuel"] == "Oil").astype(float)
d["has_ac"] = (d["aircon_present"] == "Yes").astype(float)
d["log_floor"] = np.log(d["floor_area"].clip(lower=1))
year_dummies = pd.get_dummies(d["year"], prefix="yr", drop_first=True).astype(float)

predictors = ["elec", "gas", "oil", "has_ac", "log_floor"] + list(year_dummies.columns)
X = pd.concat([d[["elec", "gas", "oil", "has_ac", "log_floor"]], year_dummies], axis=1)
Xy = pd.concat([d[["uprn", "asset_rating"]].reset_index(drop=True), X.reset_index(drop=True)], axis=1)

# within transformation: subtract each building's mean from outcome and every predictor
g = Xy.groupby("uprn")
demeaned = {}
for col in ["asset_rating"] + predictors:
    demeaned[col] = Xy[col] - g[col].transform("mean")
dm = pd.DataFrame(demeaned)
# drop predictors with no within-building variance (they get absorbed to zero)
keep = [c for c in predictors if dm[c].abs().sum() > 1e-6]
print(f"  {len(keep)} predictors with within-building variation")

model = sm.OLS(dm["asset_rating"], sm.add_constant(dm[keep], has_constant="add"))
# cluster-robust SEs by building
res = model.fit(cov_type="cluster", cov_kwds={"groups": Xy["uprn"]})

coef = pd.DataFrame({
    "coef": res.params, "std_err": res.bse, "t": res.tvalues, "p": res.pvalues,
    "ci_lo": res.conf_int()[0], "ci_hi": res.conf_int()[1],
}).round(3)
# highlight the interpretable ones (lower rating = better)
focus = coef.loc[[c for c in ["elec", "gas", "oil", "has_ac", "log_floor"] if c in coef.index]]
coef.to_csv(OUT / "vein2_fixed_effects.csv")
print("\nWithin-building fixed-effects coefficients (outcome = asset rating, lower=better):")
print(focus.to_string())
print(f"\nn={int(res.nobs):,} building-certificates, "
      f"clustered SEs by building, year fixed effects absorbed.")
print("Negative coef => within the same building, this feature is associated with a "
      "LOWER (better) rating. Saved vein2_fixed_effects.csv")
