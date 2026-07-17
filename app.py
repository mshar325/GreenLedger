"""GreenLedger — AI-powered sustainability risk advisor for small businesses.

Predicts an energy-intensity risk tier from ~10 questions an owner could answer from
memory, explains the prediction (SHAP), benchmarks it against 741 real small commercial
buildings (U.S. EIA CBECS 2018), and gives cited recommendations. No fabricated numbers:
every quantified claim is either a real model output or a cited government source.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
import streamlit as st
from scipy import stats

from greenledger.pipeline import LABELS, REGIONS, YRCONC_LABELS, SMALL_BIZ_PBA, encode_single_input
from greenledger.recommendations import get_recommendations

ROOT = Path(__file__).parent
MODELS_DIR = ROOT / "models"
APP_DATA_DIR = ROOT / "app_data"

st.set_page_config(page_title="GreenLedger", page_icon="🌿", layout="wide")


@st.cache_resource
def load_artifacts():
    meta = json.loads((MODELS_DIR / "model_meta.json").read_text())
    model = joblib.load(MODELS_DIR / "proxy_model.joblib")
    scaler_path = MODELS_DIR / "proxy_scaler.joblib"
    scaler = joblib.load(scaler_path) if scaler_path.exists() else None
    background = pd.read_csv(APP_DATA_DIR / "shap_background.csv")
    green_stats = pd.read_csv(APP_DATA_DIR / "green_computing_stats.csv")
    risk_scores = pd.read_csv(APP_DATA_DIR / "risk_score_distribution.csv")

    if meta["winner_name"] in ("Random Forest", "XGBoost"):
        explainer = shap.TreeExplainer(model)
    elif meta["winner_name"] == "Logistic Regression":
        explainer = shap.LinearExplainer(model, background)
    else:
        explainer = shap.KernelExplainer(model.predict_proba, shap.sample(background, 50))

    return meta, model, scaler, background, explainer, green_stats, risk_scores


meta, model, scaler, background, explainer, green_stats, risk_scores = load_artifacts()
FEATURE_COLUMNS = meta["feature_columns"]

tab_assess, tab_green, tab_about = st.tabs(["Risk Assessment", "Green Computing", "About"])

# ---------------------------------------------------------------- Risk Assessment tab
with tab_assess:
    st.title("GreenLedger")
    st.caption("Sustainability risk, in the time it takes to answer 10 questions.")

    with st.form("intake"):
        c1, c2 = st.columns(2)
        with c1:
            business_type = st.selectbox("Business type", list(SMALL_BIZ_PBA.values()))
            region = st.selectbox("Region", list(REGIONS.keys()), format_func=lambda k: REGIONS[k])
            sqft = st.number_input("Approximate square footage", min_value=200, max_value=25000, value=2500, step=100)
            nfloor = st.number_input("Number of floors", min_value=1, max_value=10, value=1)
            yrconc = st.selectbox("Building age", list(YRCONC_LABELS.keys()),
                                   format_func=lambda k: YRCONC_LABELS[k], index=6)
        with c2:
            wkhrs = st.slider("Hours open per week", 0, 168, 60)
            nwker = st.number_input("Employees present during main shift", min_value=0, max_value=200, value=5)
            monuse = st.slider("Months in use per year", 0, 12, 12)
            ht1 = st.checkbox("Building has heating", value=True)
            cool = st.checkbox("Building has air conditioning", value=True)

        submitted = st.form_submit_button("Assess risk", type="primary")

    if submitted:
        record = {
            "business_type": business_type, "REGION": region, "SQFT": sqft, "NFLOOR": nfloor,
            "YRCONC": yrconc, "WKHRS": wkhrs, "NWKER": nwker, "MONUSE": monuse,
            "HT1": 1 if ht1 else 2, "COOL": 1 if cool else 2,
        }
        X_new = encode_single_input(record, FEATURE_COLUMNS)
        X_new_use = scaler.transform(X_new) if scaler is not None else X_new.values

        proba = model.predict_proba(X_new_use)[0]
        pred_idx = int(np.argmax(proba))
        pred_label = LABELS[pred_idx]

        st.divider()
        r1, r2 = st.columns([1, 1.4])

        with r1:
            tier_color = {"Low": "green", "Medium": "orange", "High": "red"}[pred_label]
            st.markdown(f"### Predicted risk: :{tier_color}[{pred_label}]")
            st.progress(float(proba[pred_idx]), text=f"{proba[pred_idx]*100:.0f}% model confidence")
            st.caption(f"Low {proba[0]*100:.0f}% · Medium {proba[1]*100:.0f}% · High {proba[2]*100:.0f}%")

            st.markdown("#### Peer benchmarking")
            high_idx = LABELS.index("High")
            my_score = model.predict_proba(X_new_use)[0][high_idx]
            all_scores = risk_scores["high_risk_score"].values
            pct_all = stats.percentileofscore(all_scores, my_score)
            same_type = risk_scores[risk_scores["business_type"] == business_type]["high_risk_score"].values
            pct_type = stats.percentileofscore(same_type, my_score) if len(same_type) > 5 else None

            st.write(f"Higher predicted risk than **{pct_all:.0f}%** of the 741 real small "
                     f"commercial buildings in this study.")
            if pct_type is not None:
                st.write(f"Higher predicted risk than **{pct_type:.0f}%** of real "
                         f"**{business_type.lower()}** businesses specifically (n={len(same_type)}).")
            st.caption("Benchmark population: U.S. EIA CBECS 2018, filtered to small "
                       "commercial buildings ≤25,000 sqft.")

        with r2:
            st.markdown("#### Why this prediction")
            X_shap = pd.DataFrame(X_new_use, columns=FEATURE_COLUMNS)
            sv = explainer.shap_values(X_shap)
            sv_pred_class = sv[pred_idx][0] if isinstance(sv, list) else np.asarray(sv)[0, :, pred_idx]

            contrib = pd.Series(sv_pred_class, index=FEATURE_COLUMNS).sort_values(key=abs, ascending=False).head(6)
            contrib_df = pd.DataFrame({"feature": contrib.index, "impact": contrib.values})
            st.bar_chart(contrib_df.set_index("feature"), horizontal=True)
            st.caption(f"SHAP contribution to the '{pred_label}' prediction — positive pushes "
                       "toward this tier, negative pushes away.")

            st.markdown("#### Recommendations")
            top_risk_drivers = contrib[contrib > 0].index.tolist()
            for rec in get_recommendations(top_risk_drivers):
                with st.container(border=True):
                    st.markdown(f"**{rec['title']}**")
                    st.write(rec["detail"])
                    st.caption(f"Source: [{rec['source']}]({rec['source_url']})")

# ---------------------------------------------------------------- Green Computing tab
with tab_green:
    st.header("Green computing: why this app runs on Logistic Regression")
    st.write("Every model below was trained and measured on the same machine, same data, "
             "same split. These are real measurements, not estimates.")
    show = green_stats.rename(columns={
        "test_accuracy": "Accuracy", "test_macro_f1": "Macro F1", "train_time_s": "Train time (s)",
        "inference_ms_per_prediction": "Inference (ms/prediction)", "model_size_kb": "Model size (KB)",
    }).set_index("model")
    st.dataframe(show, use_container_width=True)

    winner = meta["winner_name"]
    st.success(
        f"**{winner}** is both the most accurate model on the proxy-only feature set *and* "
        f"the smallest and fastest — {show.loc[winner, 'Model size (KB)']:.1f} KB and "
        f"{show.loc[winner, 'Inference (ms/prediction)']:.2f} ms per prediction, versus "
        f"{show['Model size (KB)'].max():.0f} KB for the largest model here. We didn't have "
        "to trade accuracy for efficiency in this case — the smallest model won outright — "
        "so the app runs entirely on CPU with no GPU or heavy runtime required."
    )
    st.caption("Model size = serialized model file on disk. Inference time = mean over 200 "
               "single-row predictions on the same machine.")

# ---------------------------------------------------------------- About tab
with tab_about:
    st.header("About GreenLedger")
    st.markdown(f"""
Built on the U.S. EIA's real 2018 Commercial Buildings Energy Consumption Survey (CBECS)
microdata — **{meta['n_buildings']} real buildings**, filtered to small-business-like
activity types (food sales/service, retail, service, strip malls) under 25,000 sq ft.

**Research question:** can cheap, self-reportable operational data predict a small
business's energy-intensity risk almost as well as inspection-grade detail would — and
does an ANN outperform classical ML at this scale?

**What we found:** proxy-only Logistic Regression was the single best model in the whole
comparison (72.0% test accuracy) — audit-grade detail didn't improve on it. The ANN
underperformed every classical model in both feature sets.

**Limitations**, stated plainly:
- CBECS is U.S.-only; the method transfers, the specific numbers may not.
- CBECS surveys the *building*, not the *business* — a reasonable stand-in at this
  size/activity filter, not a perfect match.
- Risk tiers are relative to this 741-building sample (tertiles), not an external
  certified ESG/audit standard.
- Recommendations cite published DOE/EPA/ENERGY STAR ranges, not numbers computed for
  your specific building — no formal energy audit has been done here.

Full methodology, code, and the executed analysis notebook: see `notebooks/greenledger.ipynb`,
`README.md`, and `PROCESS.md` in this project.
""")
