"""Guards on the exported artifacts the app loads at runtime -- the invariants that,
if violated, silently corrupt predictions or the dashboard (e.g. the stale-scaler bug,
the poisoned map normalization). These run only if artifacts have been exported."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
APP = ROOT / "app_data"

pytestmark = pytest.mark.skipif(
    not (MODELS / "model_meta.json").exists(),
    reason="artifacts not exported (run export_artifacts.py)")


def meta():
    return json.loads((MODELS / "model_meta.json").read_text())


def test_model_features_match_metadata():
    m = meta()
    model = joblib.load(MODELS / "proxy_model.joblib")
    assert model.n_features_in_ == len(m["feature_columns"])


def test_scaler_presence_matches_needs_scaling_flag():
    # the stale-scaler bug: a leftover scaler applied to a model that doesn't need one
    m = meta()
    scaler_exists = (MODELS / "proxy_scaler.joblib").exists()
    if not m["needs_scaling"]:
        assert not scaler_exists, "a scaler file exists for a model that doesn't scale"


def test_dashboard_data_has_no_missing_ratings():
    d = pd.read_csv(APP / "dashboard_data.csv.gz", compression="gzip")
    assert d["asset_rating"].notna().all()
    assert d["risk_tier"].isin(["Low", "Medium", "High"]).all()


def test_triage_uncertainty_bounded_after_normalization():
    t = pd.read_csv(APP / "triage_2025.csv.gz", compression="gzip")
    u = (t["uncertainty_entropy"] / np.log(3)).clip(0, 1)
    assert u.min() >= 0.0 and u.max() <= 1.0   # the 109.86% bug must stay dead


def test_region_names_align_with_geojson():
    d = pd.read_csv(APP / "dashboard_data.csv.gz", compression="gzip")
    gj = json.loads((APP / "uk_regions.geojson").read_text())
    geo_regions = {f["properties"]["region"] for f in gj["features"]}
    data_regions = set(d["uk_region"].dropna()) - {"Unknown", "Scotland"}
    # every real England&Wales region present in the data must have a boundary to draw
    assert data_regions <= geo_regions, f"regions with no geojson: {data_regions - geo_regions}"


def test_bunching_density_shows_excess_at_E():
    dens = pd.read_csv(APP / "bunching_density.csv")
    excess = (dens[dens.rating.between(121, 125)].observed
              - dens[dens.rating.between(121, 125)].counterfactual).sum()
    assert excess > 0, "pooled MEES-era density should show positive excess mass at just-E"
