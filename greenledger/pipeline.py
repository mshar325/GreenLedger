"""Single source of truth for GreenLedger's data prep and feature engineering.

v2: U.K. non-domestic EPC dataset (epc.opendatacommunities.org), pooled 2018-2026,
replacing the original CBECS-based v1 (see PROCESS.md for why). Real, officially
assessed target (asset_rating / asset_rating_band) instead of a self-constructed
proxy label. Used by export_artifacts.py (training) and app.py (inference).
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

RAW_COLS = ["postcode", "local_authority_label", "asset_rating", "asset_rating_band",
            "property_type", "lodgement_date", "main_heating_fuel", "floor_area",
            "aircon_present", "aircon_kw_rating", "estimated_aircon_kw_rating",
            "ac_inspection_commissioned", "building_environment", "building_level"]

# UK's non-domestic property_type labels changed wording in 2023 (dropped the old
# "A1/A2"/"B1" planning-use-class prefixes) -- match on substrings, not an exact list,
# so the filter doesn't silently break across the label change.
SMALL_BIZ_PATTERN = re.compile(
    r"retail|financial|professional|office|workshop|restaurant|cafe|takeaway|drinking",
    re.I,
)
FLOOR_AREA_CAP_M2 = 500

LABELS = ["Low", "Medium", "High"]
BAND_TO_TIER = {"A+": "Low", "A": "Low", "B": "Low",
                "C": "Medium", "D": "Medium",
                "E": "High", "F": "High", "G": "High"}

PROXY_NUM = ["log_floor_area", "lodgement_year"]
PROXY_CAT = ["property_type_group", "main_heating_fuel", "aircon_present",
             "building_environment", "uk_region"]

AUDIT_NUM = ["aircon_kw_rating_filled", "has_aircon_kw_rating"]
AUDIT_CAT = ["ac_inspection_commissioned"]


def _property_group(pt: str) -> str:
    pt = pt.lower()
    if "restaurant" in pt or "cafe" in pt or "takeaway" in pt or "drinking" in pt:
        return "Restaurant/Cafe"
    if "office" in pt or "workshop" in pt:
        return "Office/Workshop"
    return "Retail/Financial/Professional"


def load_outcode_lookup(path):
    lut = pd.read_csv(path)
    lut["outcode"] = lut["postcode"].str.strip().str.upper()
    return lut.set_index("outcode")[["latitude", "longitude", "uk_region", "town", "country_string"]]


def _extract_outcode(postcode: str) -> str:
    if not isinstance(postcode, str) or not postcode.strip():
        return None
    return postcode.strip().upper().split(" ")[0]


def load_year(csv_path, outcode_lut):
    df = pd.read_csv(csv_path, usecols=RAW_COLS)
    df = df[df["property_type"].str.contains(SMALL_BIZ_PATTERN, na=False)
            & (df["floor_area"] <= FLOOR_AREA_CAP_M2) & df["floor_area"].notna()
            & df["asset_rating_band"].isin(BAND_TO_TIER)].copy()

    df["property_type_group"] = df["property_type"].map(_property_group)
    df["risk_tier"] = df["asset_rating_band"].map(BAND_TO_TIER)
    df["lodgement_year"] = pd.to_datetime(df["lodgement_date"], errors="coerce").dt.year
    df["log_floor_area"] = np.log(df["floor_area"])
    df["has_aircon_kw_rating"] = df["aircon_kw_rating"].notna().astype(int)
    df["aircon_kw_rating_filled"] = df["aircon_kw_rating"].fillna(0)
    df["ac_inspection_commissioned"] = df["ac_inspection_commissioned"].fillna("Not applicable").astype(str)
    df["building_environment"] = df["building_environment"].fillna("Unknown")
    df["main_heating_fuel"] = df["main_heating_fuel"].fillna("Unknown")

    df["outcode"] = df["postcode"].map(_extract_outcode)
    df = df.join(outcode_lut, on="outcode")
    df["uk_region"] = df["uk_region"].fillna("Unknown")
    return df


def load_pooled(data_dir, outcode_csv, years):
    data_dir = Path(data_dir)
    lut = load_outcode_lookup(outcode_csv)
    frames = []
    for yr in years:
        f = data_dir / f"certificates-{yr}.csv"
        if f.exists():
            frames.append(load_year(f, lut))
    return pd.concat(frames, ignore_index=True)


def build_feature_matrix(df, num_cols, cat_cols):
    X_num = df[num_cols].astype(float)
    X_cat = pd.get_dummies(df[cat_cols].astype(str), drop_first=True)
    return pd.concat([X_num, X_cat], axis=1)


def encode_single_input(record: dict, feature_columns: list) -> pd.DataFrame:
    row = {
        "log_floor_area": np.log(record["floor_area"]),
        "lodgement_year": record["lodgement_year"],
    }
    df_row = pd.DataFrame([row])
    cat_row = pd.DataFrame([{
        "property_type_group": record["property_type_group"],
        "main_heating_fuel": record["main_heating_fuel"],
        "aircon_present": record["aircon_present"],
        "building_environment": record["building_environment"],
        "uk_region": record["uk_region"],
    }]).astype(str)
    cat_dummies = pd.get_dummies(cat_row)
    full = pd.concat([df_row, cat_dummies], axis=1)
    return full.reindex(columns=feature_columns, fill_value=0)
