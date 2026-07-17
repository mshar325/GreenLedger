"""Single source of truth for GreenLedger's data prep and feature engineering.

Mirrors notebooks/greenledger.ipynb exactly (same filter, same target, same feature
sets) so the trained models exported from here are consistent with what the notebook
reports. Used by export_artifacts.py (training) and app.py (inference on new input).
"""
import numpy as np
import pandas as pd

RAW_COLS = ["PBA", "REGION", "SQFT", "NFLOOR", "YRCONC", "WKHRS", "NWKER", "HT1", "COOL",
            "WLCNS", "RFCNS", "GLSSPC", "FLCEILHT", "ATTIC", "BASEMNT", "ELEVTR", "MONUSE", "MFBTU"]

SMALL_BIZ_PBA = {6: "Food sales", 15: "Food service", 23: "Strip shopping center",
                  25: "Retail other than mall", 26: "Service"}

REGIONS = {1: "Northeast", 2: "Midwest", 3: "South", 4: "West"}

YRCONC_LABELS = {2: "Before 1946", 3: "1946-1959", 4: "1960-1969", 5: "1970-1979",
                  6: "1980-1989", 7: "1990-1999", 8: "2000-2012", 9: "2013-2018"}

LABELS = ["Low", "Medium", "High"]

PROXY_NUM = ["log_sqft", "NFLOOR", "YRCONC", "WKHRS", "NWKER", "MONUSE", "ht1_bin", "cool_bin"]
PROXY_CAT = ["business_type", "REGION"]

AUDIT_NUM = ["flceilht_c", "attic_bin", "basemnt_n", "elevtr_bin", "GLSSPC"]
AUDIT_CAT = ["WLCNS", "RFCNS"]


def load_filtered_data(data_csv):
    raw = pd.read_csv(data_csv, usecols=RAW_COLS)
    df = raw[raw["PBA"].isin(SMALL_BIZ_PBA) & (raw["SQFT"] <= 25000) & raw["MFBTU"].notna()].copy()
    df["business_type"] = df["PBA"].map(SMALL_BIZ_PBA)

    df["ht1_bin"] = (df["HT1"] == 1).astype(int)
    df["cool_bin"] = (df["COOL"] == 1).astype(int)
    df["attic_bin"] = (df["ATTIC"] == 1).astype(int)
    df["elevtr_bin"] = (df["ELEVTR"].fillna(2) == 1).astype(int)
    df["basemnt_n"] = df["BASEMNT"].fillna(0)
    df["flceilht_c"] = df["FLCEILHT"].clip(upper=55)
    df["log_sqft"] = np.log(df["SQFT"])

    df["EUI"] = df["MFBTU"] / df["SQFT"]
    q1, q2 = df["EUI"].quantile([1 / 3, 2 / 3])
    df["risk_tier"] = df["EUI"].apply(lambda e: "Low" if e <= q1 else ("Medium" if e <= q2 else "High"))
    return df, (q1, q2)


def build_feature_matrix(df, num_cols, cat_cols):
    X_num = df[num_cols].astype(float)
    X_cat = pd.get_dummies(df[cat_cols].astype(str), drop_first=True)
    return pd.concat([X_num, X_cat], axis=1)


def encode_single_input(record: dict, feature_columns: list[str]) -> pd.DataFrame:
    """record has raw proxy fields (business_type, REGION, SQFT, NFLOOR, YRCONC, WKHRS,
    NWKER, MONUSE, HT1, COOL) -> one-row DataFrame aligned to the training column order."""
    row = {
        "log_sqft": np.log(record["SQFT"]),
        "NFLOOR": record["NFLOOR"],
        "YRCONC": record["YRCONC"],
        "WKHRS": record["WKHRS"],
        "NWKER": record["NWKER"],
        "MONUSE": record["MONUSE"],
        "ht1_bin": int(record["HT1"] == 1),
        "cool_bin": int(record["COOL"] == 1),
    }
    df_row = pd.DataFrame([row])
    cat_row = pd.DataFrame([{
        "business_type": record["business_type"],
        "REGION": str(record["REGION"]),
    }]).astype(str)
    cat_dummies = pd.get_dummies(cat_row)
    full = pd.concat([df_row, cat_dummies], axis=1)
    return full.reindex(columns=feature_columns, fill_value=0)
