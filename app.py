"""GreenLedger — sustainability risk research platform for UK small businesses.

v3: adds the "Pathway to E" retrofit simulator, a MEES Distortion tab (Vein 1
bunching findings), and an Audit Triage simulator (uncertainty-ranked queue on the
2025 out-of-time test set). Built on the real E&W non-domestic EPC register.
No fabricated numbers: every figure is a real model output, a real register
statistic, or a cited government/Carbon Trust source.
"""
import hashlib
import json
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pydeck as pdk
import plotly.express as px
import plotly.graph_objects as go
import shap
import streamlit as st
from scipy import stats

from greenledger.pipeline import LABELS, encode_single_input
from greenledger.recommendations import get_recommendations
from greenledger.report import generate_report

ROOT = Path(__file__).parent
MODELS_DIR = ROOT / "models"
APP_DATA_DIR = ROOT / "app_data"

NEON = "#39ff8c"      # emerald -- Low risk / compliant
AMBER = "#e8d371"     # Medium risk
CRIMSON = "#ff4d6d"   # High risk / failing
CYAN = "#35e6e6"
SLATE = "#3e5266"
TIER_COLORS = {"Low": NEON, "Medium": AMBER, "High": CRIMSON}
BAND_COLORS = {"A+": "#39ff8c", "A": "#7ef0a8", "B": "#b8e994", "C": "#e8d371",
               "D": "#e2a25c", "E": "#e0703f", "F": "#e0426a", "G": "#c23fd6"}
PLOTLY_DARK_LAYOUT = dict(
    template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="IBM Plex Sans, sans-serif", color="#e9f5ee"),
)

UK_REGIONS = ["East Midlands", "East of England", "London", "North East England",
              "North West England", "South East England", "South West England",
              "Wales", "West Midlands", "Yorkshire and the Humber"]
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
    --neon: #39ff8c; --neon-soft: rgba(57,255,140,0.35);
    --crimson: #ff4d6d; --amber: #e8d371; --cyan: #35e6e6;
    --bg: #0a0f0d; --card: #101712; --card-border: rgba(57,255,140,0.22);
}
html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
h1, h2, h3, h4 { font-family: 'Sora', sans-serif !important; letter-spacing: -0.01em; }
.block-container { padding-top: 2.2rem; padding-bottom: 2rem; max-width: 1300px; }

.stApp {
    background:
        radial-gradient(circle at 15% 0%, rgba(57,255,140,0.07), transparent 40%),
        radial-gradient(circle at 85% 100%, rgba(53,230,230,0.06), transparent 45%),
        var(--bg);
}
h1 { text-shadow: 0 0 18px var(--neon-soft); }

div[data-testid="stMetric"] {
    background: linear-gradient(160deg, var(--card) 0%, #0d1310 100%);
    border: 1px solid var(--card-border); border-radius: 14px;
    padding: 16px 18px 12px;
    box-shadow: 0 0 0 1px rgba(255,255,255,0.02), 0 8px 24px rgba(0,0,0,0.35);
    position: relative; overflow: hidden;
}
div[data-testid="stMetric"]::before {
    content: ""; position: absolute; top: 0; left: 0; right: 0; height: 3px;
    background: linear-gradient(90deg, var(--neon), var(--cyan));
}
div[data-testid="stMetricLabel"] { font-family: 'IBM Plex Mono', monospace !important;
    text-transform: uppercase; letter-spacing: 0.08em; font-size: 0.7rem !important; opacity: 0.75; }
div[data-testid="stMetricValue"] { font-family: 'Sora', sans-serif !important;
    text-shadow: 0 0 14px var(--neon-soft); }

div[data-testid="stVerticalBlockBorderWrapper"] {
    background: linear-gradient(160deg, var(--card) 0%, #0d1310 100%);
    border: 1px solid var(--card-border) !important; border-radius: 14px !important;
    box-shadow: 0 0 24px -8px rgba(57,255,140,0.15);
}
.stButton > button, button[kind="formSubmit"] {
    background: linear-gradient(90deg, var(--neon), var(--cyan)) !important;
    color: #04150c !important; font-weight: 700 !important; border: none !important;
    border-radius: 10px !important; box-shadow: 0 0 20px -2px var(--neon-soft);
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}
.stButton > button:hover, button[kind="formSubmit"]:hover {
    transform: translateY(-1px); box-shadow: 0 0 28px -2px var(--neon);
}
button[data-baseweb="tab"] { font-family: 'Sora', sans-serif; font-weight: 600; }
button[data-baseweb="tab"][aria-selected="true"] {
    color: var(--neon) !important; text-shadow: 0 0 10px var(--neon-soft);
}
div[data-baseweb="tab-highlight"] { background-color: var(--neon) !important;
    box-shadow: 0 0 10px var(--neon); }
div[data-testid="stProgress"] > div > div > div {
    background: linear-gradient(90deg, var(--neon), var(--cyan)) !important;
}
[data-testid="stDataFrame"] { border: 1px solid var(--card-border); border-radius: 10px; overflow: hidden; }

.tier-badge { display:inline-block; padding: 4px 16px; border-radius: 999px;
    font-family:'Sora',sans-serif; font-weight:700; font-size:1.15rem; }
.tier-low    { background: rgba(57,255,140,0.12); color: var(--neon);   border: 1px solid var(--neon); }
.tier-medium { background: rgba(232,211,113,0.10); color: var(--amber); border: 1px solid var(--amber); }
.tier-high   { background: rgba(255,77,109,0.10);  color: var(--crimson); border: 1px solid var(--crimson);
               box-shadow: 0 0 16px -4px var(--crimson); }
</style>
""", unsafe_allow_html=True)


def tier_badge(tier):
    return f'<span class="tier-badge tier-{tier.lower()}">{tier}</span>'


@st.cache_resource
def load_artifacts():
    meta = json.loads((MODELS_DIR / "model_meta.json").read_text())
    model = joblib.load(MODELS_DIR / "proxy_model.joblib")
    scaler_path = MODELS_DIR / "proxy_scaler.joblib"
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


@st.cache_data
def load_map_sample():
    dash = load_dashboard_data()
    pts = dash.dropna(subset=["latitude", "longitude"])
    return pts.sample(n=min(20000, len(pts)), random_state=42)


@st.cache_data
def load_bunching():
    density = pd.read_csv(APP_DATA_DIR / "bunching_density.csv")
    by_year = pd.read_csv(APP_DATA_DIR / "bunching_by_year.csv")
    return density, by_year


@st.cache_data
def load_triage():
    return pd.read_csv(APP_DATA_DIR / "triage_2025.csv.gz", compression="gzip")


def predict(record):
    X = encode_single_input(record, FEATURE_COLUMNS)
    X_use = scaler.transform(X) if scaler is not None else X.values
    proba = model.predict_proba(X_use)[0]
    return X_use, proba


meta, model, scaler, background, explainer, green_stats, risk_scores = load_artifacts()
FEATURE_COLUMNS = meta["feature_columns"]

tab_assess, tab_dash, tab_mees, tab_triage, tab_about = st.tabs(
    ["Risk Assessment", "Dashboard", "MEES Distortion", "Audit Triage", "About"])

# ================================================================ Risk Assessment
with tab_assess:
    st.title("GreenLedger")
    st.caption("Sustainability risk for UK small businesses — assessed, explained, benchmarked, and simulated.")

    with st.form("intake"):
        c1, c2 = st.columns(2)
        with c1:
            f_ptype = st.selectbox("Business type", PROPERTY_GROUPS, key="f_ptype")
            f_region = st.selectbox("Region (England & Wales)", UK_REGIONS, key="f_region")
            f_floor = st.number_input("Approximate floor area (m²)", 5, 500, 100, 5, key="f_floor")
            f_year = st.number_input("Year", 2018, 2026, date.today().year, key="f_year")
        with c2:
            f_fuel = st.selectbox("Main heating fuel", HEATING_FUELS, key="f_fuel")
            f_ac = st.selectbox("Air conditioning present?", ["No", "Yes"], key="f_ac")
            f_env = st.selectbox("Building environment", BUILDING_ENVIRONMENTS, key="f_env")
        submitted = st.form_submit_button("Assess risk", type="primary")

    if submitted:
        st.session_state["record"] = {
            "property_type_group": f_ptype, "uk_region": f_region, "floor_area": f_floor,
            "lodgement_year": f_year, "main_heating_fuel": f_fuel,
            "aircon_present": f_ac, "building_environment": f_env,
        }
        st.session_state.pop("report_text", None)
        st.session_state.pop("report_key", None)

    if "record" in st.session_state:
        record = st.session_state["record"]
        X_use, proba = predict(record)
        pred_idx = int(np.argmax(proba))
        pred_label = LABELS[pred_idx]

        st.divider()
        r1, r2 = st.columns([1, 1.4])

        with r1:
            st.markdown(f"### Predicted risk: {tier_badge(pred_label)}", unsafe_allow_html=True)
            st.progress(float(proba[pred_idx]), text=f"{proba[pred_idx]*100:.0f}% model confidence")
            st.caption(f"Low {proba[0]*100:.0f}% · Medium {proba[1]*100:.0f}% · High {proba[2]*100:.0f}%")

            st.markdown("#### Peer benchmarking")
            high_idx = LABELS.index("High")
            my_score = proba[high_idx]
            pct_all = stats.percentileofscore(risk_scores["high_risk_score"].values, my_score)
            same_type = risk_scores[risk_scores["property_type_group"] == record["property_type_group"]]["high_risk_score"].values
            pct_type = stats.percentileofscore(same_type, my_score) if len(same_type) > 5 else None
            st.write(f"Higher predicted risk than **{pct_all:.0f}%** of real small UK commercial "
                     "buildings in the 2025 benchmark year.")
            if pct_type is not None:
                st.write(f"Higher than **{pct_type:.0f}%** of real "
                         f"**{record['property_type_group'].lower()}** businesses (n={len(same_type):,}).")
            st.caption("Benchmark: UK non-domestic EPC register, small commercial buildings ≤500m².")

        with r2:
            st.markdown("#### Why this prediction")
            X_shap = pd.DataFrame(X_use, columns=FEATURE_COLUMNS)
            sv = explainer.shap_values(X_shap)
            sv_pred = sv[pred_idx][0] if isinstance(sv, list) else np.asarray(sv)[0, :, pred_idx]
            contrib = pd.Series(sv_pred, index=FEATURE_COLUMNS).sort_values(key=abs, ascending=False).head(6)
            fig_shap = go.Figure(go.Bar(
                x=contrib.values[::-1], y=contrib.index[::-1], orientation="h",
                marker_color=[CRIMSON if v > 0 else NEON for v in contrib.values[::-1]]))
            fig_shap.update_layout(**PLOTLY_DARK_LAYOUT, height=240,
                                     margin=dict(l=0, r=0, t=5, b=0),
                                     xaxis_title=f"SHAP push toward '{pred_label}'")
            st.plotly_chart(fig_shap, use_container_width=True)

            st.markdown("#### Recommendations")
            top_risk_drivers = contrib[contrib > 0].index.tolist()
            recs = get_recommendations(top_risk_drivers)
            for rec in recs:
                with st.container(border=True):
                    st.markdown(f"**{rec['title']}**")
                    st.write(rec["detail"])
                    st.caption(f"Source: [{rec['source']}]({rec['source_url']})")

        # ---------------- Pathway to E: retrofit simulator ----------------
        st.divider()
        st.markdown("### Pathway to E — retrofit simulator")
        if st.toggle("Simulate retrofits", key="sim_on",
                      help="Change self-reportable attributes and see the model's re-scored risk."):
            s1, s2, s3 = st.columns(3)
            sim_fuel = s1.selectbox("Heating fuel", HEATING_FUELS,
                                     index=HEATING_FUELS.index(record["main_heating_fuel"])
                                     if record["main_heating_fuel"] in HEATING_FUELS else 0, key="sim_fuel")
            sim_env = s2.selectbox("Building environment", BUILDING_ENVIRONMENTS,
                                    index=BUILDING_ENVIRONMENTS.index(record["building_environment"]),
                                    key="sim_env")
            sim_ac = s3.selectbox("Air conditioning", ["No", "Yes"],
                                   index=["No", "Yes"].index(record["aircon_present"]), key="sim_ac")
            sim_record = {**record, "main_heating_fuel": sim_fuel,
                          "building_environment": sim_env, "aircon_present": sim_ac}
            _, sim_proba = predict(sim_record)
            sim_idx = int(np.argmax(sim_proba))
            sim_label = LABELS[sim_idx]

            b1, b2, b3 = st.columns([1, 1, 1.2])
            b1.markdown(f"**Current**<br>{tier_badge(pred_label)}", unsafe_allow_html=True)
            b2.markdown(f"**Simulated**<br>{tier_badge(sim_label)}", unsafe_allow_html=True)
            delta_high = (sim_proba[high_idx] - proba[high_idx]) * 100
            b3.metric("P(High risk)", f"{sim_proba[high_idx]*100:.0f}%",
                       delta=f"{delta_high:+.0f} pts", delta_color="inverse")
            if delta_high < 0:
                st.success(f"This combination lowers the model's High-risk probability by "
                           f"{abs(delta_high):.0f} points.")
            elif delta_high > 0:
                st.error("This combination *raises* the model's High-risk probability.")
            st.caption("Model-estimated change on register data — an association, not a "
                       "guaranteed outcome, and not a substitute for a real assessment. "
                       "Panel evidence on what actually moves ratings: see MEES Distortion tab.")

        # ---------------- Written report ----------------
        st.divider()
        st.markdown("#### Written report")
        rec_key = hashlib.md5(json.dumps(record, sort_keys=True).encode()).hexdigest()
        if st.session_state.get("report_key") != rec_key:
            proba_dict = {"Low": proba[0]*100, "Medium": proba[1]*100, "High": proba[2]*100}
            try:
                st.session_state["report_text"] = generate_report(
                    record["property_type_group"], pred_label, proba_dict,
                    pct_all, pct_type, top_risk_drivers, recs)
            except Exception as e:
                st.session_state["report_text"] = None
                st.warning(f"Written report unavailable right now ({type(e).__name__}).")
            st.session_state["report_key"] = rec_key
        report = st.session_state.get("report_text")
        if report:
            st.write(report)
            st.caption("Generated by an LLM (openai/gpt-oss-20b via Groq) strictly from the numbers "
                       "and citations above — it was not permitted to introduce its own figures.")
        else:
            st.info("No `GROQ_API_KEY` configured — the written report is optional; everything "
                    "above comes straight from the model and cited sources.")

# ================================================================ Dashboard
with tab_dash:
    dash = load_dashboard_data()
    st.title("Register Dashboard")
    st.caption(f"{len(dash):,} real small-business EPC records, "
               f"{dash['lodgement_year'].min()}–{dash['lodgement_year'].max()}")

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Buildings analysed", f"{len(dash):,}")
    k2.metric("Median asset rating", f"{dash['asset_rating'].median():.0f}",
               help="Lower = more energy efficient")
    k3.metric("Share High-risk", f"{(dash['risk_tier']=='High').mean()*100:.1f}%")
    trend = dash.groupby("lodgement_year")["asset_rating"].mean()
    k4.metric("Rating trend", f"{trend.iloc[-1]:.0f}",
               delta=f"{trend.iloc[-1]-trend.iloc[0]:+.0f} since {trend.index.min()}", delta_color="inverse")
    k5.metric("Most common fuel", dash["main_heating_fuel"].mode()[0])

    st.divider()
    m1, m2 = st.columns([1.4, 1])
    with m1:
        st.markdown("#### Where these buildings are")
        map_sample = load_map_sample()
        layer = pdk.Layer(
            "HexagonLayer", data=map_sample, get_position=["longitude", "latitude"],
            radius=5000, elevation_scale=200, elevation_range=[0, 3000],
            pickable=True, extruded=True,
            get_color_weight="asset_rating", color_aggregation="MEAN",
            color_range=[[57, 255, 140, 210], [126, 240, 168, 210], [232, 211, 113, 210],
                          [224, 112, 63, 210], [200, 40, 40, 230]],
        )
        st.pydeck_chart(pdk.Deck(
            layers=[layer],
            initial_view_state=pdk.ViewState(latitude=52.6, longitude=-1.8, zoom=5.6, pitch=15),
            map_style="dark", tooltip={"text": "{count} buildings in this area"}))
        st.caption("Hexagon height = building count, color = mean asset rating (green = efficient, "
                   "red = poor). Points are jittered within their postcode district — the register "
                   "gives district-level location, not exact coordinates.")
    with m2:
        st.markdown("#### Rating bands (A+ best, G worst)")
        band_counts = dash["asset_rating_band"].value_counts().reindex(
            ["A+", "A", "B", "C", "D", "E", "F", "G"]).fillna(0)
        fig = px.bar(x=band_counts.index, y=band_counts.values,
                      color=band_counts.index, color_discrete_map=BAND_COLORS)
        fig.update_layout(**PLOTLY_DARK_LAYOUT, showlegend=False, xaxis_title="",
                            yaxis_title="Buildings", margin=dict(l=0, r=0, t=10, b=0), height=260)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Business type mix")
        tc = dash["property_type_group"].value_counts()
        fig2 = px.pie(values=tc.values, names=tc.index, hole=0.55,
                       color_discrete_sequence=[NEON, CYAN, AMBER])
        fig2.update_layout(**PLOTLY_DARK_LAYOUT, margin=dict(l=0, r=0, t=10, b=0), height=260,
                             legend=dict(orientation="h", yanchor="bottom", y=-0.25))
        st.plotly_chart(fig2, use_container_width=True)

    st.divider()
    d1, d2 = st.columns(2)
    with d1:
        st.markdown("#### Efficiency is improving — real year-over-year trend")
        yearly = trend.reset_index()
        fig3 = px.line(yearly, x="lodgement_year", y="asset_rating", markers=True)
        fig3.update_traces(line_color=NEON, marker_color=NEON, line_width=3,
                             marker=dict(size=9, line=dict(width=1, color="#04150c")))
        fig3.update_layout(**PLOTLY_DARK_LAYOUT, yaxis_title="Mean asset rating (lower = better)",
                             xaxis_title="", margin=dict(l=0, r=0, t=10, b=0), height=300)
        st.plotly_chart(fig3, use_container_width=True)
        st.caption("This real trend is why models are trained on 2018-2024 and tested on 2025 "
                   "data they never saw, rather than a random split.")
    with d2:
        st.markdown("#### Risk tier by region")
        rt = pd.crosstab(dash["uk_region"], dash["risk_tier"], normalize="index") * 100
        rt = rt.reindex(columns=["Low", "Medium", "High"]).sort_values("High")
        fig4 = px.bar(rt, orientation="h", color_discrete_map=TIER_COLORS)
        fig4.update_layout(**PLOTLY_DARK_LAYOUT, xaxis_title="% of buildings", yaxis_title="",
                             barmode="stack", margin=dict(l=0, r=0, t=10, b=0), height=300,
                             legend=dict(orientation="h", yanchor="bottom", y=-0.3))
        st.plotly_chart(fig4, use_container_width=True)

# ================================================================ MEES Distortion
with tab_mees:
    st.title("MEES Distortion — bunching at the letting ban")
    st.caption("Since 2018 (new lets) and 2023 (all lets), letting an F/G-rated commercial building "
               "is unlawful in England & Wales. The register shows exactly what that incentive did.")
    density, by_year = load_bunching()

    excess = float((density[density["rating"].between(121, 125)]["observed"]
                     - density[density["rating"].between(121, 125)]["counterfactual"]).sum())
    missing = float((density[density["rating"].between(126, 130)]["counterfactual"]
                      - density[density["rating"].between(126, 130)]["observed"]).sum())
    mees_by_year = by_year[by_year["threshold"].str.contains("MEES")]
    b_now = mees_by_year[mees_by_year["year"] >= 2018]["b_norm"].mean()

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Excess certificates at 'just E'", f"{excess:,.0f}", help="Observed minus counterfactual, ratings 121-125, pooled 2018-2026")
    k2.metric("Missing certificates at 'just F'", f"{missing:,.0f}", help="Counterfactual minus observed, ratings 126-130")
    k3.metric("Mass drawn from just-F", f"{missing/excess*100:.0f}%", help="M/B — the notch relabeling signature")
    k4.metric("Avg excess mass b (MEES era)", f"{b_now:.1f}", help="vs ~0 pre-policy and at placebo boundaries")

    st.markdown("#### The certificate distribution vs its counterfactual (pooled 2018-2026)")
    figd = go.Figure()
    figd.add_bar(x=density["rating"], y=density["observed"], name="Observed",
                  marker_color="#4c7a5a", opacity=0.9)
    figd.add_trace(go.Scatter(x=density["rating"], y=density["counterfactual"],
                                name="Counterfactual", mode="lines",
                                line=dict(color=CYAN, width=3, dash="dash")))
    figd.add_vrect(x0=120.5, x1=125.5, fillcolor=CRIMSON, opacity=0.12,
                    annotation_text="Excess mass at E", annotation_position="top left",
                    annotation_font_color=CRIMSON)
    figd.add_vrect(x0=125.5, x1=130.5, fillcolor=SLATE, opacity=0.25,
                    annotation_text="Missing mass at F", annotation_position="top right",
                    annotation_font_color="#9db4cc")
    figd.add_vline(x=125.5, line_dash="dot", line_color=CRIMSON, line_width=2)
    figd.update_layout(**PLOTLY_DARK_LAYOUT, height=420, margin=dict(l=0, r=0, t=10, b=0),
                        xaxis_title="Asset rating (lower = better; E ends at 125, F begins at 126)",
                        yaxis_title="Certificates",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02))
    st.plotly_chart(figd, use_container_width=True)

    st.markdown("#### Excess mass by year — the policy signature")
    figy = go.Figure()
    colors = {"E/F 125 (MEES)": CRIMSON, "C/D 75 (placebo)": NEON, "B/C 50 (placebo)": SLATE}
    for name, color in colors.items():
        sub = by_year[by_year["threshold"] == name]
        figy.add_trace(go.Scatter(
            x=sub["year"], y=sub["b_norm"], name=name, mode="lines+markers",
            line=dict(color=color, width=3), marker=dict(size=8),
            error_y=dict(type="data", symmetric=False,
                          array=sub["b_ci_hi"] - sub["b_norm"],
                          arrayminus=sub["b_norm"] - sub["b_ci_lo"], width=2)))
    for xv, lbl in [(2018, "MEES: new lets"), (2023, "MEES: all lets")]:
        figy.add_vline(x=xv - 0.5, line_dash="dot", line_color="grey")
        figy.add_annotation(x=xv - 0.5, y=4.9, text=lbl, showarrow=False,
                             font=dict(size=11, color="#9aa8a0"), xanchor="left")
    figy.add_hline(y=0, line_color="#3a4d3e")
    figy.update_layout(**PLOTLY_DARK_LAYOUT, height=420, margin=dict(l=0, r=0, t=10, b=0),
                        xaxis_title="Certificate year", yaxis_title="Normalized excess mass b",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02))
    st.plotly_chart(figy, use_container_width=True)
    st.caption("b ≈ 0 in 2012, before the policy. It climbs through the 2015-2017 anticipation "
               "window, plateaus at ~3-3.9 from 2018 enforcement, and never overlaps the placebo "
               "boundaries — whose own drift after 2022 is consistent with anticipation of the "
               "announced future C/B minimum standards.")

    with st.expander("What the repeat-certificate panel adds (Vein 2 mechanism evidence)"):
        st.markdown("""
- **212,908 buildings** have 2+ certificates. Among 40,369 buildings whose previous
  certificate was F/G, the share escaping to E-or-better on re-certification rose from
  **56% (2012) to 96-99% every year since 2018**.
- **75% of escapes reach D or better** — consistent with real improvement, not minimal
  compliance. But escapes are **~2.2x over-represented** at "just barely E" (121-125)
  vs baseline.
- The **fast-escape segment (≤90 days, n=2,535)** is where the manipulation signature
  concentrates: **40% show no observable input change** and they land at just-barely-E
  at **~4.4x the baseline rate**.
- The most common observable channel in escapes is a **>5% change in assessed floor
  area (59.5%)** — a lever that requires no physical work. Fabric changes (insulation,
  glazing) are invisible in this register, so "no observable change" is an upper bound
  on gaming, not a measurement of it.

Full analysis: `analysis/MECHANISM_VEIN2.md` and `analysis/vein2_panel.py` in the repo.
""")

# ================================================================ Audit Triage
with tab_triage:
    st.title("Audit Triage — who should get a real inspection?")
    st.caption("The model screens cheaply; physical audits are expensive. Rank the 2025 "
               "out-of-time test buildings by model uncertainty and spend the audit budget "
               "where the model is least sure.")
    try:
        triage = load_triage()
    except FileNotFoundError:
        st.info("Triage data not exported yet — run `export_artifacts.py` first.")
        triage = None

    if triage is not None:
        budget = st.slider("Available physical-audit budget (% of building portfolio)",
                            1, 25, 5, key="triage_budget")
        k = max(1, int(len(triage) * budget / 100))
        queue = triage.sort_values("uncertainty_entropy", ascending=False).head(k)

        hit_queue = (queue["actual_tier"] == "High").mean() * 100
        hit_random = (triage["actual_tier"] == "High").mean() * 100
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Portfolio", f"{len(triage):,} buildings")
        c2.metric("Audit queue", f"{k:,} buildings")
        c3.metric("High-risk found in queue", f"{hit_queue:.1f}%")
        c4.metric("vs random auditing", f"{hit_random:.1f}%",
                   delta=f"{hit_queue-hit_random:+.1f} pts")
        st.caption("'High-risk found' uses the 2025 ground truth, known here because this is "
                   "a labeled test year — at deployment time it would be unknown, which is "
                   "exactly why the queue is ranked by model uncertainty.")

        st.markdown("#### Triage queue — most uncertain first")
        show = queue[["property_type_group", "uk_region", "local_authority", "floor_area_m2",
                       "main_heating_fuel", "predicted_tier", "p_high", "uncertainty_entropy"]]
        st.dataframe(
            show, use_container_width=True, height=430, hide_index=True,
            column_config={
                "property_type_group": "Business type",
                "uk_region": "Region",
                "local_authority": "Local authority",
                "floor_area_m2": st.column_config.NumberColumn("Floor m²", format="%.0f"),
                "main_heating_fuel": "Heating fuel",
                "predicted_tier": "Predicted",
                "p_high": st.column_config.ProgressColumn("P(High)", min_value=0, max_value=1),
                "uncertainty_entropy": st.column_config.ProgressColumn(
                    "Uncertainty", min_value=0, max_value=float(triage["uncertainty_entropy"].max())),
            })

    with st.expander("Green computing: the measured model tradeoff behind this app"):
        show_g = green_stats.rename(columns={
            "test_accuracy": "Accuracy", "test_macro_f1": "Macro F1",
            "recall_high_risk": "High-risk recall", "train_time_s": "Train (s)",
            "inference_ms_per_prediction": "Inference (ms)", "model_size_kb": "Size (KB)",
        }).set_index("model")
        st.dataframe(show_g, use_container_width=True)
        st.markdown(
            f"**{meta['winner_name']}** was selected on **High-risk recall**, not accuracy — "
            "one model here wins accuracy while catching ~4% of genuinely High-risk buildings. "
            "The winner is also the largest/slowest model in the comparison: a real "
            "fitness-for-purpose vs footprint tradeoff, disclosed rather than papered over.")

# ================================================================ About
with tab_about:
    st.header("About GreenLedger")
    st.markdown(f"""
Built on the UK's real non-domestic Energy Performance Certificate register
(epc.opendatacommunities.org) — **{meta['n_train']:,} buildings (2018-2024)** for training,
**{meta['n_test']:,} buildings assessed in 2025** as an untouched out-of-time test set,
and ~1.4M certificates (2012-2026) behind the MEES Distortion research tab. Filtered to
small-business-like activity (retail/financial/professional, restaurants/cafes,
offices/workshops) under 500m².

**Research questions:** (1) can cheap, self-reportable data predict a small business's
energy-rating risk almost as well as inspection-grade detail — and does an ANN help at
this scale? (2) does the MEES letting ban distort the certificate distribution itself?

**Findings so far, stated honestly:**
- The ANN posts the highest raw accuracy while catching ~4% of High-risk buildings —
  accuracy was the wrong selection metric; the app selects on High-risk recall instead.
- Certificate bunching at the E/F boundary is ~zero pre-policy, rises through the
  anticipation window, and plateaus at b≈3-3.9 under enforcement — while placebo
  boundaries stay flat until the announced future standards approach.
- Repeat-certificate evidence says most escapes from F/G look like real improvement;
  a small fast-return segment carries the manipulation signature, and assessed floor
  area is the most common observable lever.

**Limitations:** England & Wales coverage only; postcode-district location precision
(map points jittered, disclosed); risk tiers collapse the official 8-band scale to 3;
the register's audit-grade feature set is thin (mainly HVAC capacity); escape rates
are conditional on re-certification (selection); fabric changes are invisible in the
public register. Recommendations cite Carbon Trust / GOV.UK published ranges — no
formal energy audit happens here.

Method, code, and executed analyses: `notebooks/greenledger.ipynb`, `analysis/`
(`vein1_bunching.py`, `vein2_panel.py`, related-work and mechanism docs), `README.md`,
`PROCESS.md`, `RESEARCH_ROADMAP.md`.
""")
