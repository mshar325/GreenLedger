"""Confidence intervals on the headline numbers -- two honest sources of uncertainty:

 (A) Test-set sampling: for the deployed proxy-only models, bootstrap the 2025
     out-of-time test set (resample with replacement, recompute the metric) -> a CI
     on "how good is this model on unseen 2025 data of this size".
 (B) Training stochasticity: retrain each model across several random seeds -> spread
     of High-risk recall, i.e. is the model-selection conclusion stable, or a fluke of
     one seed?

The out-of-time split (2018-2024 -> 2025) is held FIXED throughout -- these CIs quantify
sampling + training noise around that split, not split choice.

Outputs: analysis/results/model_ci.csv + app_data/model_ci.csv (compact, for the app).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from greenledger.pipeline import PROXY_CAT, PROXY_NUM, LABELS, build_feature_matrix, load_pooled  # noqa: E402

OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(exist_ok=True)
APP_DATA = ROOT / "app_data"

TRAIN_YEARS = [2018, 2019, 2020, 2021, 2022, 2023, 2024]
SEEDS = [42, 0, 1, 2, 3]
N_BOOT = 2000

print("Loading pooled data...")
tr = load_pooled(ROOT / "data" / "uk_epc", ROOT / "data" / "uk_outcodes.csv", TRAIN_YEARS)
te = load_pooled(ROOT / "data" / "uk_epc", ROOT / "data" / "uk_outcodes.csv", [2025])
ytr = pd.Categorical(tr["risk_tier"], categories=LABELS, ordered=True).codes
yte = pd.Categorical(te["risk_tier"], categories=LABELS, ordered=True).codes
Xtr = build_feature_matrix(tr, PROXY_NUM, PROXY_CAT)
Xte = build_feature_matrix(te, PROXY_NUM, PROXY_CAT).reindex(columns=Xtr.columns, fill_value=0)
sw = compute_sample_weight("balanced", ytr)
scaler = StandardScaler().fit(Xtr)
Xtr_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xte)
print(f"  train {len(tr):,} / test {len(te):,}")


def build(name, seed):
    # class-balancing applied EXACTLY as in export_artifacts.py (the deployed model):
    # LR + RF via class_weight="balanced", XGB via sample_weight at fit time, MLP has
    # no sample-weight support (the disclosed asymmetry). Without this, RF/LR collapse
    # to the majority class and High-risk recall goes to ~0 -- a real consistency bug.
    if name == "Logistic Regression":
        return LogisticRegression(max_iter=1000, class_weight="balanced"), True
    if name == "Random Forest":
        return RandomForestClassifier(n_estimators=200, max_depth=12, n_jobs=4,
                                       class_weight="balanced", random_state=seed), False
    if name == "XGBoost":
        return XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1, n_jobs=4,
                              eval_metric="mlogloss", random_state=seed), False
    return MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=200, early_stopping=True, random_state=seed), True


MODELS = ["Logistic Regression", "Random Forest", "XGBoost", "ANN (MLP)"]
rng = np.random.default_rng(42)
boot_idx = [rng.integers(0, len(yte), len(yte)) for _ in range(N_BOOT)]

rows = []
for name in MODELS:
    seed_recalls = []
    primary_pred = None
    for si, seed in enumerate(SEEDS):
        model, needs_scaling = build(name, seed)
        xtr = Xtr_s if needs_scaling else Xtr.values
        xte = Xte_s if needs_scaling else Xte.values
        if isinstance(model, XGBClassifier):
            model.fit(xtr, ytr, sample_weight=sw)
        else:
            model.fit(xtr, ytr)
        pred = model.predict(xte)
        seed_recalls.append(recall_score(yte, pred, labels=[2], average="macro"))
        if si == 0:
            primary_pred = pred

    # (A) test-set bootstrap on the primary-seed predictions
    br, ba, bf = [], [], []
    for idx in boot_idx:
        yt, yp = yte[idx], primary_pred[idx]
        br.append(recall_score(yt, yp, labels=[2], average="macro"))
        ba.append(accuracy_score(yt, yp))
        bf.append(f1_score(yt, yp, average="macro"))
    br, ba, bf = np.array(br), np.array(ba), np.array(bf)

    rows.append({
        "model": name,
        "recall_high": round(seed_recalls[0], 3),
        "recall_boot_lo": round(np.percentile(br, 2.5), 3),
        "recall_boot_hi": round(np.percentile(br, 97.5), 3),
        "recall_seed_min": round(min(seed_recalls), 3),
        "recall_seed_max": round(max(seed_recalls), 3),
        "recall_seed_std": round(float(np.std(seed_recalls)), 4),
        "accuracy": round(accuracy_score(yte, primary_pred), 3),
        "acc_boot_lo": round(np.percentile(ba, 2.5), 3),
        "acc_boot_hi": round(np.percentile(ba, 97.5), 3),
        "macro_f1": round(f1_score(yte, primary_pred, average="macro"), 3),
        "f1_boot_lo": round(np.percentile(bf, 2.5), 3),
        "f1_boot_hi": round(np.percentile(bf, 97.5), 3),
    })
    print(f"  {name}: recall {rows[-1]['recall_high']} "
          f"[{rows[-1]['recall_boot_lo']},{rows[-1]['recall_boot_hi']}] boot; "
          f"seeds [{rows[-1]['recall_seed_min']},{rows[-1]['recall_seed_max']}]")

res = pd.DataFrame(rows)
res.to_csv(OUT / "model_ci.csv", index=False)
res.to_csv(APP_DATA / "model_ci.csv", index=False)

# The decisive check: does RF's recall CI sit entirely above the ANN's?
rf = res[res.model == "Random Forest"].iloc[0]
ann = res[res.model == "ANN (MLP)"].iloc[0]
print(f"\nRF recall CI [{rf.recall_boot_lo},{rf.recall_boot_hi}] vs "
      f"ANN [{ann.recall_boot_lo},{ann.recall_boot_hi}] -- "
      f"{'SEPARATED (selection is robust)' if rf.recall_boot_lo > ann.recall_boot_hi else 'OVERLAP'}")
print("Saved model_ci.csv")
