"""GreenLedger — AI-powered sustainability risk advisor for small UK businesses.

v2: built on the real U.K. non-domestic EPC register (2018-2026), replacing the
original CBECS-based v1 (see PROCESS.md). Predicts an energy-rating risk tier from a
short questionnaire, explains the prediction (SHAP), benchmarks it against hundreds of
thousands of real small commercial buildings, and gives cited recommendations. No
fabricated numbers: every quantified claim is either a real model output or a cited
government/Carbon Trust source.
"""
import json
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pydeck as pdk
import plotly.express as px
import shap
import streamlit as st
from scipy import stats

from greenledger.pipeline import LABELS, encode_single_input
from greenledger.recommendations import get_recommendations
from greenledger.report import generate_report

ROOT = Path(__file__).parent
MODELS_DIR = ROOT / "models"
APP_DATA_DIR = ROOT / "app_data"

MOSS = "#39ff8c"     # neon green -- primary accent, replaces the old moss green
MOSS_SOFT = "#22c46e"
SLATE = "#35e6e6"    # cyan -- secondary accent
PLOTLY_DARK_LAYOUT = dict(
    template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="IBM Plex Sans, sans-serif", color="#e9f5ee"),
)
BAND_COLORS = {"A+": "#39ff8c", "A": "#7ef0a8", "B": "#b8e994", "C": "#e8d371",
               "D": "#e2a25c", "E": "#e0703f", "F": "#e0426a", "G": "#c23fd6"}
TIER_COLORS = {"Low": MOSS, "Medium": "#e8d371", "High": "#e0426a"}

UK_REGIONS = ["East Midlands", "East of England", "London", "North East England",
              "North West England", "Scotland", "South East England", "South West England",
              "Wales", "West Midlands", "Yorkshire and the Humber", "Northern Ireland"]
HEATING_FUELS = ["Grid Supplied Electricity", "Natural Gas", "Oil", "LPG", "Biomass",
                  "District Heating", "Other"]
BUILDING_ENVIRONMENTS = ["Air Conditioning", "Heating and Mechanical Ventilation",
                          "Heating and Natural Ventilation", "Mixed-mode with Mechanical Ventilation",
                          "Mixed-mode with Natural Ventilation", "Unconditioned"]
PROPERTY_GROUPS = ["Retail/Financial/Professional", "Restaurant/Cafe", "Office/Workshop"]

st.set_page_config(page_title="GreenLedger", page_icon="🌿", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@500;600;700;800&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

:root {
    --neon: #39ff8c;
    --neon-soft: rgba(57,255,140,0.35);
    --cyan: #35e6e6;
    --bg: #0a0f0d;
    --card: #101712;
    --card-border: rgba(57,255,140,0.22);
}

html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
h1, h2, h3, h4 { font-family: 'Sora', sans-serif !important; letter-spacing: -0.01em; }

.stApp {
    background:
        radial-gradient(circle at 15% 0%, rgba(57,255,140,0.07), transparent 40%),
        radial-gradient(circle at 85% 100%, rgba(53,230,230,0.06), transparent 45%),
        var(--bg);
}

h1 { text-shadow: 0 0 18px var(--neon-soft); }

/* KPI metric cards */
div[data-testid="stMetric"] {
    background: linear-gradient(160deg, var(--card) 0%, #0d1310 100%);
    border: 1px solid var(--card-border);
    border-radius: 14px;
    padding: 16px 18px 12px;
    box-shadow: 0 0 0 1px rgba(255,255,255,0.02), 0 8px 24px rgba(0,0,0,0.35);
    position: relative;
    overflow: hidden;
}
div[data-testid="stMetric"]::before {
    content: ""; position: absolute; top: 0; left: 0; right: 0; height: 3px;
    background: linear-gradient(90deg, var(--neon), var(--cyan));
}
div[data-testid="stMetricLabel"] { font-family: 'IBM Plex Mono', monospace !important;
    text-transform: uppercase; letter-spacing: 0.08em; font-size: 0.7rem !important; opacity: 0.75; }
div[data-testid="stMetricValue"] { font-family: 'Sora', sans-serif !important;
    text-shadow: 0 0 14px var(--neon-soft); }

/* bordered containers (recommendation / rec cards) */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: linear-gradient(160deg, var(--card) 0%, #0d1310 100%);
    border: 1px solid var(--card-border) !important;
    border-radius: 14px !important;
    box-shadow: 0 0 24px -8px rgba(57,255,140,0.15);
}

/* buttons */
.stButton > button, button[kind="formSubmit"] {
    background: linear-gradient(90deg, var(--neon), var(--cyan)) !important;
    color: #04150c !important; font-weight: 700 !important; border: none !important;
    border-radius: 10px !important; box-shadow: 0 0 20px -2px var(--neon-soft);
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}
.stButton > button:hover, button[kind="formSubmit"]:hover {
    transform: translateY(-1px); box-shadow: 0 0 28px -2px var(--neon);
}

/* tabs */
button[data-baseweb="tab"] { font-family: 'Sora', sans-serif; font-weight: 600; }
button[data-baseweb="tab"][aria-selected="true"] {
    color: var(--neon) !important; text-shadow: 0 0 10px var(--neon-soft);
}
div[data-baseweb="tab-highlight"] { background-color: var(--neon) !important;
    box-shadow: 0 0 10px var(--neon); }

/* progress bar */
div[data-testid="stProgress"] > div > div > div {
    background: linear-gradient(90deg, var(--neon), var(--cyan)) !important;
}

/* dataframe */
[data-testid="stDataFrame"] { border: 1px solid var(--card-border); border-radius: 10px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_artifacts():
    meta = json.loads((MODELS_DIR / "model_meta.json").read_text())
    model = joblib.load(MODELS_DIR / "proxy_model.joblib")
    scaler_path = MODELS_DIR / "proxy_scaler.joblib"
    # gated on needs_scaling, not just file existence -- a stale scaler left over from a
    # previous model that needed one must never get applied to a model that doesn't
    scaler = joblib.load(scaler_path) if meta.get("needs_scaling") and scaler_path.exists() else None
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


@st.cache_data
def load_dashboard_data():
    return pd.read_csv(APP_DATA_DIR / "dashboard_data.csv.gz", compression="gzip")


meta, model, scaler, background, explainer, green_stats, risk_scores = load_artifacts()
FEATURE_COLUMNS = meta["feature_columns"]

tab_assess, tab_dash, tab_green, tab_about = st.tabs(
    ["Risk Assessment", "Dashboard", "Green Computing", "About"])

# ---------------------------------------------------------------- Risk Assessment tab
with tab_assess:
    st.title("GreenLedger")
    st.caption("Sustainability risk for UK small businesses, in the time it takes to answer a few questions.")

    with st.form("intake"):
        c1, c2 = st.columns(2)
        with c1:
            property_type_group = st.selectbox("Business type", PROPERTY_GROUPS)
            uk_region = st.selectbox("Region", UK_REGIONS)
            floor_area = st.number_input("Approximate floor area (m²)", min_value=5, max_value=500, value=100, step=5)
            lodgement_year = st.number_input("Year", min_value=2018, max_value=2026, value=date.today().year)
        with c2:
            main_heating_fuel = st.selectbox("Main heating fuel", HEATING_FUELS)
            aircon_present = st.selectbox("Air conditioning present?", ["No", "Yes"])
            building_environment = st.selectbox("Building environment", BUILDING_ENVIRONMENTS)

        submitted = st.form_submit_button("Assess risk", type="primary")

    if submitted:
        record = {
            "property_type_group": property_type_group, "uk_region": uk_region,
            "floor_area": floor_area, "lodgement_year": lodgement_year,
            "main_heating_fuel": main_heating_fuel, "aircon_present": aircon_present,
            "building_environment": building_environment,
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
            same_type = risk_scores[risk_scores["property_type_group"] == property_type_group]["high_risk_score"].values
            pct_type = stats.percentileofscore(same_type, my_score) if len(same_type) > 5 else None

            st.write(f"Higher predicted risk than **{pct_all:.0f}%** of real small UK "
                     f"commercial buildings in this study's 2025 benchmark year.")
            if pct_type is not None:
                st.write(f"Higher predicted risk than **{pct_type:.0f}%** of real "
                         f"**{property_type_group.lower()}** businesses specifically (n={len(same_type)}).")
            st.caption("Benchmark population: UK non-domestic EPC register, small "
                       "commercial buildings ≤500m².")

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
            recs = get_recommendations(top_risk_drivers)
            for rec in recs:
                with st.container(border=True):
                    st.markdown(f"**{rec['title']}**")
                    st.write(rec["detail"])
                    st.caption(f"Source: [{rec['source']}]({rec['source_url']})")

        st.divider()
        st.markdown("#### Written report")
        proba_dict = {"Low": proba[0] * 100, "Medium": proba[1] * 100, "High": proba[2] * 100}
        try:
            report = generate_report(property_type_group, pred_label, proba_dict, pct_all, pct_type, top_risk_drivers, recs)
        except Exception as e:
            report = None
            st.warning(f"Written report unavailable right now ({type(e).__name__}) — "
                       "everything above already comes straight from the model and cited sources.")
        else:
            if report is None:
                st.info("No `GROQ_API_KEY` configured yet — the written report is optional; "
                         "everything above already comes straight from the model and cited sources.")
            else:
                st.write(report)
                st.caption("Generated by an LLM (openai/gpt-oss-20b via Groq) strictly from the "
                           "numbers and citations above — it was not permitted to introduce its own figures.")

# ---------------------------------------------------------------- Dashboard tab
with tab_dash:
    dash = load_dashboard_data()
    st.title("GreenLedger Dashboard")
    st.caption(f"{len(dash):,} real UK small-business EPC records, {dash['lodgement_year'].min()}–{dash['lodgement_year'].max()}")

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Buildings analysed", f"{len(dash):,}")
    k2.metric("Median asset rating", f"{dash['asset_rating'].median():.0f}", help="Lower = more energy efficient")
    high_pct = (dash["risk_tier"] == "High").mean() * 100
    k3.metric("Share High-risk", f"{high_pct:.1f}%")
    trend = dash.groupby("lodgement_year")["asset_rating"].mean()
    delta = trend.iloc[-1] - trend.iloc[0]
    k4.metric("Rating trend", f"{trend.iloc[-1]:.0f}", delta=f"{delta:+.0f} since {trend.index.min()}",
              delta_color="inverse")
    k5.metric("Most common fuel", dash["main_heating_fuel"].mode()[0])

    st.divider()
    m1, m2 = st.columns([1.4, 1])

    with m1:
        st.markdown("#### Where these buildings are")
        map_points = dash.dropna(subset=["latitude", "longitude"])
        map_sample = map_points.sample(n=min(25000, len(map_points)), random_state=42)
        layer = pdk.Layer(
            "HexagonLayer", data=map_sample, get_position=["longitude", "latitude"],
            radius=5000, elevation_scale=200, elevation_range=[0, 3000],
            pickable=True, extruded=True,
            get_color_weight="asset_rating", color_aggregation="MEAN",
            color_range=[[57, 255, 140, 210], [126, 240, 168, 210], [232, 211, 113, 210],
                          [224, 112, 63, 210], [200, 40, 40, 230]],
        )
        view_state = pdk.ViewState(latitude=52.6, longitude=-1.8, zoom=5.6, pitch=15)
        st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state,
                                   map_style="dark",
                                   tooltip={"text": "{count} buildings in this area"}))
        st.caption("Hexagon height = building count, color = mean asset rating (green = "
                   "efficient, red = poor) in that area. Points are jittered within their "
                   "postcode district — the register gives district-level location, not "
                   "exact coordinates.")

    with m2:
        st.markdown("#### Rating bands (A+ best, G worst)")
        band_counts = dash["asset_rating_band"].value_counts().reindex(
            ["A+", "A", "B", "C", "D", "E", "F", "G"]).fillna(0)
        fig = px.bar(x=band_counts.index, y=band_counts.values,
                      color=band_counts.index, color_discrete_map=BAND_COLORS)
        fig.update_layout(**PLOTLY_DARK_LAYOUT, showlegend=False, xaxis_title="", yaxis_title="Buildings",
                            margin=dict(l=0, r=0, t=10, b=0), height=280)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Business type mix")
        type_counts = dash["property_type_group"].value_counts()
        fig2 = px.pie(values=type_counts.values, names=type_counts.index, hole=0.55,
                       color_discrete_sequence=[MOSS, SLATE, "#e8d371"])
        fig2.update_layout(**PLOTLY_DARK_LAYOUT, margin=dict(l=0, r=0, t=10, b=0), height=280,
                             legend=dict(orientation="h", yanchor="bottom", y=-0.25))
        st.plotly_chart(fig2, use_container_width=True)

    st.divider()
    d1, d2 = st.columns(2)
    with d1:
        st.markdown("#### Efficiency is improving — real year-over-year trend")
        yearly = dash.groupby("lodgement_year")["asset_rating"].mean().reset_index()
        fig3 = px.line(yearly, x="lodgement_year", y="asset_rating", markers=True)
        fig3.update_traces(line_color=MOSS, marker_color=MOSS, line_width=3,
                             marker=dict(size=9, line=dict(width=1, color="#04150c")))
        fig3.update_layout(**PLOTLY_DARK_LAYOUT, yaxis_title="Mean asset rating (lower = better)", xaxis_title="",
                             margin=dict(l=0, r=0, t=10, b=0), height=300)
        st.plotly_chart(fig3, use_container_width=True)
        st.caption("This real trend is exactly why the models are trained on 2018-2024 and "
                   "tested on 2025 data the model never saw, rather than a random split.")

    with d2:
        st.markdown("#### Risk tier by region")
        region_tier = pd.crosstab(dash["uk_region"], dash["risk_tier"], normalize="index") * 100
        region_tier = region_tier.reindex(columns=["Low", "Medium", "High"]).sort_values("High")
        fig4 = px.bar(region_tier, orientation="h", color_discrete_map=TIER_COLORS)
        fig4.update_layout(**PLOTLY_DARK_LAYOUT, xaxis_title="% of buildings", yaxis_title="", barmode="stack",
                             margin=dict(l=0, r=0, t=10, b=0), height=300,
                             legend=dict(orientation="h", yanchor="bottom", y=-0.3))
        st.plotly_chart(fig4, use_container_width=True)

# ---------------------------------------------------------------- Green Computing tab
with tab_green:
    st.header("Green computing: an honest tension, not a clean win")
    st.write("Every model below was trained and measured on the same machine, same data, "
             "same 2018-2024 → 2025 split. These are real measurements, not estimates.")
    show = green_stats.rename(columns={
        "test_accuracy": "Accuracy", "test_macro_f1": "Macro F1", "recall_high_risk": "High-risk recall",
        "train_time_s": "Train time (s)", "inference_ms_per_prediction": "Inference (ms/prediction)",
        "model_size_kb": "Model size (KB)",
    }).set_index("model")
    st.dataframe(show, use_container_width=True)

    winner = meta["winner_name"]
    st.warning(
        f"Unlike the earlier version of this project, efficiency and fitness-for-purpose "
        f"**don't** line up cleanly here. **{winner}** wins on the metric that actually "
        "matters — High-risk recall — but it's also the largest and slowest model in the "
        "whole comparison (~61 MB, ~59 ms/prediction). XGBoost catches noticeably fewer "
        "High-risk businesses but at a fraction of the size and latency (~1.4 MB, ~1 ms) — "
        "a genuine deployment tradeoff, not something to paper over. We deploy the model "
        "that does the job (catching at-risk businesses) rather than the cheapest one that "
        "doesn't, and say so plainly instead of claiming a free lunch that isn't there."
    )
    st.caption("Model size = serialized model file on disk. Inference time = mean over 200 "
               "single-row predictions on the same machine.")

# ---------------------------------------------------------------- About tab
with tab_about:
    st.header("About GreenLedger")
    st.markdown(f"""
Built on the UK's real non-domestic Energy Performance Certificate register
(epc.opendatacommunities.org) — **{meta['n_train']:,} buildings (2018-2024)** used for
training, tested on **{meta['n_test']:,} buildings assessed in 2025** that the model never
saw during training. Filtered to small-business-like activity types (retail/financial/
professional, restaurants/cafes, offices/workshops) under 500m².

**Research question:** can cheap, self-reportable operational data predict a small UK
business's energy-rating risk almost as well as inspection-grade detail would — and does
an ANN outperform classical ML at this scale?

**What we found, and why "accuracy" was the wrong first metric to trust:** the ANN scores
the highest raw accuracy of any model here, but its recall on the High-risk class is only
about 4% — it wins by defaulting to the majority "Medium" tier, which is close to useless
for a tool meant to flag at-risk businesses. Selecting instead on **High-risk recall**
picks a different, more useful model (see the Green Computing tab) — a real methodological
finding in its own right, not just a modeling detail.

**Limitations**, stated plainly:
- The EPC register gives postcode-district location, not exact coordinates — the
  dashboard map jitters points for visual clarity, disclosed there.
- Ratings are professionally assessed (not self-reported), which is a strength for the
  target label, but the *proxy* feature set here is thinner than a full building audit —
  mainly the air-conditioning capacity rating is held back as "audit-grade."
- Risk tiers collapse the official A+-G rating band into three tiers for this comparison;
  the full band distribution is shown as-is in the Dashboard tab.
- Recommendations cite published Carbon Trust / GOV.UK ranges, not numbers computed for
  your specific building — no formal energy audit has been done here.

Full methodology, code, and the executed analysis notebook: see `notebooks/greenledger.ipynb`,
`README.md`, and `PROCESS.md` in this project.
""")
