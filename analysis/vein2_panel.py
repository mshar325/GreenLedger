"""Vein 2 -- Repeat-certificate (UPRN) panel: the F->E mechanism test.

Questions:
  1. When a building certified F/G is re-certified, how often does it land at E or
     better ("escape"), and did that escape rate move with MEES enforcement?
  2. Where do escapes LAND? If disproportionately at 121-125 ("just barely E"),
     that is do-just-enough threshold targeting, not deep retrofit.
  3. How fast are escape re-certifications? Days/weeks (consistent with
     re-assessment / assessor shopping) vs months/years (consistent with physical
     retrofit)?
  4. What observable inputs changed across the escape? Heating fuel, building
     environment (HVAC servicing strategy), air-con presence, floor area (>5%
     re-measurement). NOTE: the non-domestic register publishes no glazing or
     fabric fields, so "nothing observable changed" means none of THESE changed --
     fabric work is invisible here. Stated in MECHANISM_VEIN2.md.

Outputs: analysis/results/vein2_*.csv, two figures, printed summary.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "uk_epc"
OUT_DIR = Path(__file__).resolve().parent / "results"
OUT_DIR.mkdir(exist_ok=True)

YEARS = list(range(2012, 2027))
COLS = ["uprn", "lodgement_date", "asset_rating", "asset_rating_band",
        "main_heating_fuel", "building_environment", "aircon_present",
        "floor_area", "property_type"]

print("Loading all years...")
frames = []
for y in YEARS:
    f = DATA_DIR / f"certificates-{y}.csv"
    if not f.exists():
        continue
    df = pd.read_csv(f, usecols=COLS, dtype={"uprn": "string"})
    frames.append(df)
allc = pd.concat(frames, ignore_index=True)
allc = allc[allc["uprn"].notna() & (allc["uprn"].str.strip() != "")]
allc["lodgement_date"] = pd.to_datetime(allc["lodgement_date"], errors="coerce")
allc = allc[allc["lodgement_date"].notna() & allc["asset_rating"].notna()]
allc["asset_rating"] = allc["asset_rating"].round().astype(int)
print(f"  {len(allc):,} certificates with a UPRN")

allc = allc.sort_values(["uprn", "lodgement_date"])
g = allc.groupby("uprn", sort=False)
n_certs = g.size()
print(f"  {len(n_certs):,} unique buildings; {(n_certs >= 2).sum():,} with 2+ certs; "
      f"{(n_certs >= 3).sum():,} with 3+")

# consecutive certificate pairs within building
prev = allc.groupby("uprn", sort=False).shift(1)
pairs = allc.join(prev, rsuffix="_prev")
pairs = pairs[pairs["asset_rating_prev"].notna()].copy()
pairs["gap_days"] = (pairs["lodgement_date"] - pairs["lodgement_date_prev"]).dt.days
pairs = pairs[pairs["gap_days"] > 0]
pairs["year"] = pairs["lodgement_date"].dt.year
print(f"  {len(pairs):,} consecutive same-building certificate pairs")

FAIL = pairs["asset_rating_prev"] >= 126           # prev cert F or G
pairs_fail = pairs[FAIL].copy()
pairs_fail["escape"] = pairs_fail["asset_rating"] <= 125
print(f"  {len(pairs_fail):,} pairs where the previous certificate was F/G; "
      f"{pairs_fail['escape'].mean()*100:.1f}% escape to E-or-better overall")

# ---- 1. escape rate by year of the new certificate --------------------------------
esc_by_year = pairs_fail.groupby("year")["escape"].agg(["mean", "size"])
esc_by_year.columns = ["escape_rate", "n_pairs"]
esc_by_year.to_csv(OUT_DIR / "vein2_escape_rate_by_year.csv")
print("\nEscape rate by year:")
print((esc_by_year["escape_rate"] * 100).round(1))

# ---- 2. landing distribution ------------------------------------------------------
esc = pairs_fail[pairs_fail["escape"]].copy()
landing = esc["asset_rating"].value_counts().sort_index()
landing.to_csv(OUT_DIR / "vein2_landing_distribution.csv")
just_e = esc["asset_rating"].between(121, 125).mean()
deep = (esc["asset_rating"] <= 100).mean()
print(f"\nOf {len(esc):,} escapes: {just_e*100:.1f}% land at 121-125 ('just barely E'); "
      f"{deep*100:.1f}% reach D or better (<=100)")
# baseline: share of ALL E-or-better certificates that sit at 121-125
all_e_or_better = allc[allc["asset_rating"] <= 125]
base_just_e = all_e_or_better["asset_rating"].between(121, 125).mean()
print(f"Baseline: {base_just_e*100:.1f}% of all E-or-better certificates sit at 121-125")

# ---- 3. speed of escapes ----------------------------------------------------------
esc["fast90"] = esc["gap_days"] <= 90
nonesc = pairs_fail[~pairs_fail["escape"]]
speed = pd.DataFrame({
    "escape_pairs": esc["gap_days"].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]),
    "non_escape_pairs": nonesc["gap_days"].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]),
}).round(0)
speed.to_csv(OUT_DIR / "vein2_gap_days.csv")
print("\nGap days (escape vs non-escape re-certifications):")
print(speed)
for cut, lbl in [(30, "<=30 days"), (90, "<=90 days"), (365, "<=1 year")]:
    print(f"  {lbl}: escapes {100*(esc['gap_days']<=cut).mean():.1f}%  "
          f"vs non-escapes {100*(nonesc['gap_days']<=cut).mean():.1f}%")

# ---- 4. what changed across the escape --------------------------------------------
esc["fuel_changed"] = esc["main_heating_fuel"].fillna("?") != esc["main_heating_fuel_prev"].fillna("?")
esc["env_changed"] = esc["building_environment"].fillna("?") != esc["building_environment_prev"].fillna("?")
esc["aircon_changed"] = esc["aircon_present"].fillna("?") != esc["aircon_present_prev"].fillna("?")
esc["floor_changed_5pct"] = (esc["floor_area"] - esc["floor_area_prev"]).abs() > 0.05 * esc["floor_area_prev"]
esc["nothing_observable"] = ~(esc["fuel_changed"] | esc["env_changed"] | esc["aircon_changed"] | esc["floor_changed_5pct"])

def change_profile(d):
    return pd.Series({
        "n": len(d),
        "fuel_changed_%": 100 * d["fuel_changed"].mean(),
        "env_changed_%": 100 * d["env_changed"].mean(),
        "aircon_changed_%": 100 * d["aircon_changed"].mean(),
        "floor_area_changed>5%_%": 100 * d["floor_changed_5pct"].mean(),
        "nothing_observable_%": 100 * d["nothing_observable"].mean(),
        "land_just_E_121_125_%": 100 * d["asset_rating"].between(121, 125).mean(),
    })

profile = pd.DataFrame({
    "fast (<=90d)": change_profile(esc[esc["fast90"]]),
    "slow (>90d)": change_profile(esc[~esc["fast90"]]),
    "all escapes": change_profile(esc),
}).round(1)
profile.to_csv(OUT_DIR / "vein2_change_profile.csv")
print("\nWhat changed across F/G -> E-or-better escapes:")
print(profile)

# fuel switch direction among escapes that switched fuel
sw = esc[esc["fuel_changed"]]
switches = (sw["main_heating_fuel_prev"].fillna("?") + " -> " + sw["main_heating_fuel"].fillna("?")).value_counts().head(10)
switches.to_csv(OUT_DIR / "vein2_fuel_switches.csv")
print("\nTop fuel switches among escapes:")
print(switches)

# ---- 5. transition evidence for the app's "Pathway to E" simulator ----------------
# Over ALL consecutive same-building pairs (any starting rating), what rating change
# was actually observed when a given attribute changed from value A to value B? This
# is the REAL panel evidence the simulator should cite alongside the model's re-score.
# Convention: delta = new_rating - prev_rating; NEGATIVE = improved (lower is better).
# Associational (co-occurring changes not controlled) -- labeled as such in the app.
APP_DATA = ROOT / "app_data"


def _tier(r):
    return "Low" if r <= 50 else ("Medium" if r <= 100 else "High")


pairs["delta"] = pairs["asset_rating"] - pairs["asset_rating_prev"]
pairs["tier_prev"] = pairs["asset_rating_prev"].map(_tier)
pairs["tier_new"] = pairs["asset_rating"].map(_tier)
_order = {"Low": 0, "Medium": 1, "High": 2}
pairs["improved_tier"] = pairs["tier_new"].map(_order) < pairs["tier_prev"].map(_order)


def transition_table(col):
    prev_col, cur_col = f"{col}_prev", col
    sub = pairs[pairs[prev_col].notna() & pairs[cur_col].notna()
                & (pairs[prev_col] != pairs[cur_col])].copy()
    g = sub.groupby([prev_col, cur_col])
    tab = g.agg(n=("delta", "size"), median_delta=("delta", "median"),
                mean_delta=("delta", "mean"),
                pct_improved_tier=("improved_tier", lambda s: 100 * s.mean())).reset_index()
    tab = tab[tab["n"] >= 20].sort_values("n", ascending=False)
    tab.columns = ["from", "to", "n", "median_delta", "mean_delta", "pct_improved_tier"]
    tab["median_delta"] = tab["median_delta"].round(1)
    tab["mean_delta"] = tab["mean_delta"].round(1)
    tab["pct_improved_tier"] = tab["pct_improved_tier"].round(1)
    return tab


fuel_tab = transition_table("main_heating_fuel")
env_tab = transition_table("building_environment")
fuel_tab.insert(0, "attribute", "main_heating_fuel")
env_tab.insert(0, "attribute", "building_environment")
transitions = pd.concat([fuel_tab, env_tab], ignore_index=True)
transitions.to_csv(APP_DATA / "vein2_transitions.csv", index=False)
transitions.to_csv(OUT_DIR / "vein2_transitions.csv", index=False)
print(f"\nTransition evidence: {len(transitions)} (from->to) pairs with n>=20 "
      f"(fuel {len(fuel_tab)}, env {len(env_tab)}). Top fuel switches by n:")
print(fuel_tab.head(8).to_string(index=False))

# ---- figures ----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 4.5))
bins = np.arange(0, 2200, 30)
ax.hist(nonesc["gap_days"].clip(upper=2190), bins=bins, alpha=0.55, label="Non-escape re-certs", color="#3e5266", density=True)
ax.hist(esc["gap_days"].clip(upper=2190), bins=bins, alpha=0.65, label="F/G -> E escapes", color="#8f2d2d", density=True)
ax.axvline(90, color="black", ls=":", lw=1); ax.text(95, ax.get_ylim()[1]*0.9, "90 days", fontsize=8)
ax.set_xlabel("Days between certificates (same building)"); ax.set_ylabel("Density")
ax.set_title("How quickly do failing buildings get re-certified?")
ax.legend(); plt.tight_layout()
plt.savefig(OUT_DIR / "vein2_gap_days.png", dpi=150); plt.close()

fig, ax = plt.subplots(figsize=(9, 4.5))
w = np.arange(60, 151)
counts = esc["asset_rating"].value_counts().reindex(w, fill_value=0)
colors = ["#8f2d2d" if 121 <= x <= 125 else "#4c7a5a" for x in w]
ax.bar(w, counts.values, color=colors, width=0.9)
ax.axvline(125.5, color="black", ls="--", lw=1.2)
ax.set_xlabel("New (post-escape) asset rating"); ax.set_ylabel("Buildings")
ax.set_title("Where do F/G buildings land when they escape? (red = 'just barely E')")
plt.tight_layout(); plt.savefig(OUT_DIR / "vein2_landing.png", dpi=150); plt.close()

print("\nSaved CSVs + 2 figures to", OUT_DIR)
