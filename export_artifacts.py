"""Trains the proxy-only models, measures real green-computing stats (inference time,
model size), and exports everything the Streamlit app needs: winning model, scaler,
feature columns, SHAP background data, and the real EUI distribution for benchmarking.

Run after notebooks/greenledger.ipynb has been executed at least once (same pipeline,
kept in sync via greenledger/pipeline.py).
"""
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from greenledger.pipeline import PROXY_CAT, PROXY_NUM, LABELS, build_feature_matrix, load_filtered_data

RANDOM_STATE = 42
ROOT = Path(__file__).parent
MODELS_DIR = ROOT / "models"
APP_DATA_DIR = ROOT / "app_data"
MODELS_DIR.mkdir(exist_ok=True)
APP_DATA_DIR.mkdir(exist_ok=True)

print("Loading and filtering CBECS data...")
df, (q1, q2) = load_filtered_data(ROOT / "data" / "cbecs2018_final_public.csv")
print(f"  {len(df)} buildings, tertile cutoffs: Low<={q1:.1f} Medium<={q2:.1f}")

X_proxy = build_feature_matrix(df, PROXY_NUM, PROXY_CAT)
y = pd.Categorical(df["risk_tier"], categories=LABELS, ordered=True).codes
feature_columns = list(X_proxy.columns)

Xtr, Xte, ytr, yte = train_test_split(X_proxy, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE)
scaler = StandardScaler().fit(Xtr)
Xtr_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xte)

MODEL_SPECS = {
    "Logistic Regression": (LogisticRegression(max_iter=2000), True),
    "Random Forest": (RandomForestClassifier(n_estimators=400, max_depth=8, random_state=RANDOM_STATE), False),
    "XGBoost": (XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                               eval_metric="mlogloss", random_state=RANDOM_STATE), False),
    "ANN (MLP)": (MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=2000,
                                 early_stopping=True, random_state=RANDOM_STATE), True),
}

print("\nTraining models and measuring real green-computing stats...")
green_stats = []
fitted = {}
for name, (model, needs_scaling) in MODEL_SPECS.items():
    xtr_use, xte_use = (Xtr_s, Xte_s) if needs_scaling else (Xtr.values, Xte.values)

    t0 = time.perf_counter()
    model.fit(xtr_use, ytr)
    train_time_s = time.perf_counter() - t0

    # inference time: mean over 200 single-row predictions (simulates one app request)
    single = xte_use[:1]
    reps = 200
    t0 = time.perf_counter()
    for _ in range(reps):
        model.predict(single)
    inference_ms = (time.perf_counter() - t0) / reps * 1000

    tmp_path = MODELS_DIR / f"_tmp_{name.replace(' ', '_')}.joblib"
    joblib.dump(model, tmp_path)
    size_kb = tmp_path.stat().st_size / 1024
    tmp_path.unlink()

    pred = model.predict(xte_use)
    acc = accuracy_score(yte, pred)
    f1 = f1_score(yte, pred, average="macro")

    green_stats.append({
        "model": name, "test_accuracy": round(acc, 3), "test_macro_f1": round(f1, 3),
        "train_time_s": round(train_time_s, 4), "inference_ms_per_prediction": round(inference_ms, 4),
        "model_size_kb": round(size_kb, 1),
    })
    fitted[name] = (model, scaler if needs_scaling else None)
    print(f"  {name}: acc={acc:.3f}  size={size_kb:.1f}KB  inference={inference_ms:.3f}ms")

green_df = pd.DataFrame(green_stats).sort_values("test_accuracy", ascending=False)
green_df.to_csv(APP_DATA_DIR / "green_computing_stats.csv", index=False)

winner_name = green_df.iloc[0]["model"]
winner_model, winner_scaler = fitted[winner_name]
print(f"\nWinning proxy-only model: {winner_name}")

joblib.dump(winner_model, MODELS_DIR / "proxy_model.joblib")
if winner_scaler is not None:
    joblib.dump(winner_scaler, MODELS_DIR / "proxy_scaler.joblib")
with open(MODELS_DIR / "model_meta.json", "w") as f:
    json.dump({
        "winner_name": winner_name,
        "needs_scaling": winner_scaler is not None,
        "feature_columns": feature_columns,
        "labels": LABELS,
        "tertile_cutoffs": {"q1": float(q1), "q2": float(q2)},
        "n_buildings": len(df),
    }, f, indent=2)

# SHAP background (scaled if the winner needs scaling, matching what the app will pass to SHAP)
background = X_proxy.sample(n=min(200, len(X_proxy)), random_state=RANDOM_STATE)
if winner_scaler is not None:
    background = pd.DataFrame(winner_scaler.transform(background), columns=feature_columns)
background.to_csv(APP_DATA_DIR / "shap_background.csv", index=False)

# Real EUI distribution, for peer benchmarking (percentile rank) -- no per-business identifiers
df[["business_type", "EUI", "risk_tier"]].to_csv(APP_DATA_DIR / "eui_distribution.csv", index=False)

# Winning model's predicted "High risk" probability for every real building, so a new
# input's score can be percentile-ranked against real buildings instead of a made-up scale.
X_full_use = winner_scaler.transform(X_proxy) if winner_scaler is not None else X_proxy.values
high_idx = list(winner_model.classes_).index(2)  # class code 2 = "High" (see LABELS order)
scores = winner_model.predict_proba(X_full_use)[:, high_idx]
pd.DataFrame({"business_type": df["business_type"].values, "high_risk_score": scores}).to_csv(
    APP_DATA_DIR / "risk_score_distribution.csv", index=False)

print("\nExported to models/ and app_data/. Green computing comparison:")
print(green_df.to_string(index=False))
