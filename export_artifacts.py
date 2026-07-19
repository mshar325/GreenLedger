"""Trains proxy-only and proxy+audit models on the pooled UK EPC dataset, with a real
out-of-time holdout (train 2018-2024, test 2025 -- a building assessed after training
ended, not just a random slice), measures green-computing stats, and exports everything
app.py and the dashboard need.
"""
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from greenledger.pipeline import (AUDIT_CAT, AUDIT_NUM, LABELS, PROXY_CAT, PROXY_NUM,
                                    build_feature_matrix, load_pooled)

RANDOM_STATE = 42
ROOT = Path(__file__).parent
MODELS_DIR = ROOT / "models"
APP_DATA_DIR = ROOT / "app_data"
MODELS_DIR.mkdir(exist_ok=True)
APP_DATA_DIR.mkdir(exist_ok=True)

TRAIN_YEARS = [2018, 2019, 2020, 2021, 2022, 2023, 2024]
TEST_YEARS = [2025]

print("Loading pooled UK EPC data (this reads 8 years of CSVs, takes a minute)...")
df_train_raw = load_pooled(ROOT / "data" / "uk_epc", ROOT / "data" / "uk_outcodes.csv", TRAIN_YEARS)
df_test_raw = load_pooled(ROOT / "data" / "uk_epc", ROOT / "data" / "uk_outcodes.csv", TEST_YEARS)
print(f"  Train (2018-2024): {len(df_train_raw)} buildings | Test (2025, out-of-time): {len(df_test_raw)}")

y_train_full = pd.Categorical(df_train_raw["risk_tier"], categories=LABELS, ordered=True).codes
y_test = pd.Categorical(df_test_raw["risk_tier"], categories=LABELS, ordered=True).codes

Xp_train_full = build_feature_matrix(df_train_raw, PROXY_NUM, PROXY_CAT)
Xp_test = build_feature_matrix(df_test_raw, PROXY_NUM, PROXY_CAT).reindex(columns=Xp_train_full.columns, fill_value=0)

Xa_train_full = build_feature_matrix(df_train_raw, PROXY_NUM + AUDIT_NUM, PROXY_CAT + AUDIT_CAT)
Xa_test = build_feature_matrix(df_test_raw, PROXY_NUM + AUDIT_NUM, PROXY_CAT + AUDIT_CAT).reindex(columns=Xa_train_full.columns, fill_value=0)

proxy_feature_columns = list(Xp_train_full.columns)
audit_feature_columns = list(Xa_train_full.columns)


def make_models():
    return {
        "Logistic Regression": (LogisticRegression(max_iter=1000, n_jobs=4), True),
        "Random Forest": (RandomForestClassifier(n_estimators=200, max_depth=12, n_jobs=4, random_state=RANDOM_STATE), False),
        "XGBoost": (XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1, n_jobs=4,
                                    eval_metric="mlogloss", random_state=RANDOM_STATE), False),
        "ANN (MLP)": (MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=200,
                                     early_stopping=True, random_state=RANDOM_STATE), True),
    }


def evaluate(X_train_full, y_train_full, X_test, y_test, label):
    sample_weight = compute_sample_weight("balanced", y_train_full)
    scaler = StandardScaler().fit(X_train_full)
    X_train_s = scaler.transform(X_train_full)
    X_test_s = scaler.transform(X_test)

    rows, fitted, green = [], {}, []
    for name, (model, needs_scaling) in make_models().items():
        xtr = X_train_s if needs_scaling else X_train_full.values
        xte = X_test_s if needs_scaling else X_test.values

        t0 = time.perf_counter()
        if isinstance(model, XGBClassifier):
            model.fit(xtr, y_train_full, sample_weight=sample_weight)
        elif isinstance(model, MLPClassifier):
            model.fit(xtr, y_train_full)  # MLPClassifier has no sample_weight support
        else:
            model.set_params(class_weight="balanced") if hasattr(model, "class_weight") else None
            model.fit(xtr, y_train_full)
        train_time_s = time.perf_counter() - t0

        single = xte[:1]
        reps = 200
        t0 = time.perf_counter()
        for _ in range(reps):
            model.predict(single)
        inference_ms = (time.perf_counter() - t0) / reps * 1000

        tmp_path = MODELS_DIR / f"_tmp_{label}_{name.replace(' ', '_')}.joblib"
        joblib.dump(model, tmp_path)
        size_kb = tmp_path.stat().st_size / 1024
        tmp_path.unlink()

        pred = model.predict(xte)
        acc = accuracy_score(y_test, pred)
        f1 = f1_score(y_test, pred, average="macro")
        recall_high = recall_score(y_test, pred, labels=[2], average="macro")  # 2 = "High" -- the class that actually matters
        print(f"  [{label}] {name}: acc={acc:.3f} f1={f1:.3f} recall_high={recall_high:.3f} "
              f"train={train_time_s:.1f}s size={size_kb:.1f}KB inf={inference_ms:.3f}ms")

        rows.append({"feature_set": label, "model": name, "test_accuracy": round(acc, 3),
                     "test_macro_f1": round(f1, 3), "recall_high_risk": round(recall_high, 3),
                     "train_time_s": round(train_time_s, 2),
                     "inference_ms_per_prediction": round(inference_ms, 4), "model_size_kb": round(size_kb, 1)})
        fitted[name] = (model, scaler if needs_scaling else None)
    return pd.DataFrame(rows), fitted


print("\nTraining PROXY-ONLY models (2018-2024 -> tested on 2025)...")
results_proxy, fitted_proxy = evaluate(Xp_train_full, y_train_full, Xp_test, y_test, "proxy-only")

print("\nTraining PROXY+AUDIT models (2018-2024 -> tested on 2025)...")
results_audit, fitted_audit = evaluate(Xa_train_full, y_train_full, Xa_test, y_test, "proxy+audit")

results = pd.concat([results_proxy, results_audit], ignore_index=True)
results.to_csv(APP_DATA_DIR / "green_computing_stats.csv", index=False)

winner_name = results_proxy.sort_values(["recall_high_risk", "test_macro_f1"], ascending=False).iloc[0]["model"]
winner_model, winner_scaler = fitted_proxy[winner_name]
print(f"\nWinning proxy-only model (selected on High-risk recall, NOT accuracy -- "
      f"see PROCESS.md for why): {winner_name}")

joblib.dump(winner_model, MODELS_DIR / "proxy_model.joblib")
scaler_out_path = MODELS_DIR / "proxy_scaler.joblib"
if winner_scaler is not None:
    joblib.dump(winner_scaler, scaler_out_path)
elif scaler_out_path.exists():
    scaler_out_path.unlink()  # remove any stale scaler from a previous winner that needed one
with open(MODELS_DIR / "model_meta.json", "w") as f:
    json.dump({
        "winner_name": winner_name, "needs_scaling": winner_scaler is not None,
        "feature_columns": proxy_feature_columns, "labels": LABELS,
        "train_years": TRAIN_YEARS, "test_years": TEST_YEARS,
        "n_train": len(df_train_raw), "n_test": len(df_test_raw),
    }, f, indent=2)

background = Xp_train_full.sample(n=min(200, len(Xp_train_full)), random_state=RANDOM_STATE)
if winner_scaler is not None:
    background = pd.DataFrame(winner_scaler.transform(background), columns=proxy_feature_columns)
background.to_csv(APP_DATA_DIR / "shap_background.csv", index=False)

# Risk-score distribution (winner's predicted High-risk probability) on the pooled
# TEST year -- what a new input gets percentile-ranked against.
X_test_use = winner_scaler.transform(Xp_test) if winner_scaler is not None else Xp_test.values
high_idx = list(winner_model.classes_).index(2)
proba_test = winner_model.predict_proba(X_test_use)
scores = proba_test[:, high_idx]
pd.DataFrame({
    "property_type_group": df_test_raw["property_type_group"].values,
    "high_risk_score": scores,
}).to_csv(APP_DATA_DIR / "risk_score_distribution.csv", index=False)

# Audit-triage export: per-building predictions + uncertainty on the 2025 out-of-time
# set, so the app can rank an audit queue by model uncertainty (predictive entropy).
with np.errstate(divide="ignore", invalid="ignore"):
    entropy = -np.nansum(proba_test * np.log(np.clip(proba_test, 1e-12, 1)), axis=1)
triage = pd.DataFrame({
    "property_type_group": df_test_raw["property_type_group"].values,
    "uk_region": df_test_raw["uk_region"].values,
    "local_authority": df_test_raw["local_authority_label"].values,
    "floor_area_m2": df_test_raw["floor_area"].values,
    "main_heating_fuel": df_test_raw["main_heating_fuel"].values,
    "predicted_tier": [LABELS[i] for i in np.argmax(proba_test, axis=1)],
    "p_low": proba_test[:, 0].round(3), "p_medium": proba_test[:, 1].round(3),
    "p_high": proba_test[:, 2].round(3),
    "uncertainty_entropy": entropy.round(4),
    "actual_tier": df_test_raw["risk_tier"].values,  # known here; unknown at audit time in deployment
})
triage.to_csv(APP_DATA_DIR / "triage_2025.csv.gz", index=False, compression="gzip")

# Dashboard data: full pooled (train+test) dataset, trimmed to what the dashboard needs.
# lat/long are only known at postcode-outcode (district) precision, so many buildings
# share one exact point -- add a small seeded jitter purely so the map shows spread
# instead of a few oversized dots stacked on district centroids (disclosed in the app).
dash_cols = ["asset_rating", "asset_rating_band", "risk_tier", "property_type_group",
             "floor_area", "main_heating_fuel", "aircon_present", "lodgement_year",
             "uk_region", "latitude", "longitude", "local_authority_label"]
df_all = pd.concat([df_train_raw, df_test_raw], ignore_index=True)[dash_cols].copy()
rng = np.random.default_rng(RANDOM_STATE)
df_all["latitude"] = df_all["latitude"] + rng.normal(0, 0.03, size=len(df_all))
df_all["longitude"] = df_all["longitude"] + rng.normal(0, 0.03, size=len(df_all))
df_all.to_csv(APP_DATA_DIR / "dashboard_data.csv.gz", index=False, compression="gzip")

print("\nDone. Green computing comparison:")
print(results.to_string(index=False))
