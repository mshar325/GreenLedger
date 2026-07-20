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

/* ---------- dashboard hero / kpi / rankings ---------- */
.hero-card { border: 1px solid rgba(255,77,214,0.45); border-radius: 16px;
    background: linear-gradient(160deg, #101018 0%, #0c1210 100%);
    padding: 18px 24px; margin-bottom: 14px;
    box-shadow: 0 0 30px -12px rgba(255,77,214,0.5); }
.hero-title { font-family:'Sora',sans-serif; font-weight:800; font-size:1.5rem; color:#fff; margin:0; }
.hero-sub { color:#aab8b0; font-size:0.92rem; margin-top:2px; }
.hero-sub b { color:#e9f5ee; }
.chip { display:inline-block; font-family:'IBM Plex Mono',monospace; font-size:0.68rem;
    letter-spacing:0.08em; text-transform:uppercase; padding:3px 10px; border-radius:999px;
    border:1px solid rgba(255,77,214,0.6); color:#ff4dd6; margin:0 8px; }
.kpi { border-radius:16px; padding:16px 18px 12px; margin-bottom:14px; position:relative;
    background: linear-gradient(165deg, #12181a 0%, #0d1212 100%);
    border:1px solid rgba(255,255,255,0.07); overflow:hidden;
    box-shadow: 0 10px 26px rgba(0,0,0,0.4); }
.kpi::before { content:""; position:absolute; top:0; left:0; right:0; height:3px; }
.kpi-head { display:flex; align-items:center; gap:10px; margin-bottom:10px; }
.kpi-icon { width:34px; height:34px; border-radius:10px; display:flex; align-items:center;
    justify-content:center; font-size:1.05rem; }
.kpi-title { font-family:'Sora',sans-serif; font-weight:600; font-size:0.9rem; color:#dfe9e3; }
.kpi-value { font-family:'IBM Plex Mono',monospace; font-weight:500; font-size:1.9rem;
    color:#ffffff; line-height:1.1; }
.kpi-delta { font-size:0.82rem; margin-top:3px; }
.kpi-delta .up { color:#39ff8c; } .kpi-delta .down { color:#ff4d6d; }
.kpi-delta span.lbl { color:#8b988f; }
.kpi-sub { color:#8b988f; font-size:0.8rem; border-top:1px solid rgba(255,255,255,0.06);
    margin-top:10px; padding-top:8px; }
.kpi-spark { margin-top:8px; }
.panel { border-radius:16px; padding:18px 20px; background:linear-gradient(165deg,#12181a 0%,#0d1212 100%);
    border:1px solid rgba(255,255,255,0.07); box-shadow:0 10px 26px rgba(0,0,0,0.4); margin-bottom:14px; }
.panel-head { display:flex; align-items:center; gap:10px; margin-bottom:4px; }
.panel-title { font-family:'Sora',sans-serif; font-weight:700; font-size:1.05rem; color:#fff; }
.panel-sub { color:#8b988f; font-size:0.8rem; margin-bottom:10px; }
.rank-row { display:flex; align-items:center; gap:12px; padding:9px 0;
    border-bottom:1px solid rgba(255,255,255,0.05); }
.rank-chip { width:24px; height:24px; border-radius:7px; display:flex; align-items:center;
    justify-content:center; font-family:'IBM Plex Mono',monospace; font-size:0.75rem;
    background:#1a2420; color:#aab8b0; flex:none; }
.rank-chip.first { background:rgba(232,211,113,0.18); color:#e8d371; }
.rank-name { font-weight:600; color:#e9f5ee; font-size:0.92rem; width:190px; flex:none; }
.rank-track { flex:1; height:5px; border-radius:999px; background:rgba(255,255,255,0.06); overflow:hidden; }
.rank-fill { height:100%; border-radius:999px; }
.rank-val { font-family:'IBM Plex Mono',monospace; font-size:0.9rem; color:#fff; width:120px;
    text-align:right; flex:none; }
.rank-val span { color:#8b988f; font-size:0.78rem; margin-right:8px; }
.heat-scale { border-radius:12px; border:1px solid rgba(255,255,255,0.07); padding:12px 16px;
    background:#0d1212; margin-top:10px; }
.heat-bar { height:8px; border-radius:999px;
    background:linear-gradient(90deg,#39ff8c,#b8e994,#e8d371,#e0703f,#ff4d6d,#c23fd6); }
.heat-lbl { display:flex; justify-content:space-between; color:#8b988f; font-size:0.75rem; margin-top:5px; }
.heat-lbl b { font-family:'IBM Plex Mono',monospace; }
</style>
""", unsafe_allow_html=True)


def tier_badge(tier):
    return f'<span class="tier-badge tier-{tier.lower()}">{tier}</span>'


# One heat gradient, used by the map, the legend strip, AND the ranking bars, so no
# panel can drift onto its own palette again.
HEAT_STOPS = [(0.0, "#39ff8c"), (0.2, "#b8e994"), (0.4, "#e8d371"),
              (0.6, "#e0703f"), (0.8, "#ff4d6d"), (1.0, "#c23fd6")]


def heat_color(t):
    t = min(max(float(t), 0.0), 1.0)
    for (t0, c0), (t1, c1) in zip(HEAT_STOPS, HEAT_STOPS[1:]):
        if t <= t1:
            f = (t - t0) / (t1 - t0)
            rgb = [round(int(c0[i:i+2], 16) + f * (int(c1[i:i+2], 16) - int(c0[i:i+2], 16)))
                   for i in (1, 3, 5)]
            return "#{:02x}{:02x}{:02x}".format(*rgb)
    return HEAT_STOPS[-1][1]


def sparkline_svg(values, color, width=170, height=38):
    """Mini bar sparkline as inline SVG, last bar emphasized (real data only)."""
    v = np.asarray(values, dtype=float)
    if len(v) == 0 or np.nanmax(v) <= 0:
        return ""
    v = v / np.nanmax(v)
    n = len(v)
    gap = 3
    bw = max(2, (width - gap * (n - 1)) / n)
    bars = []
    for i, val in enumerate(v):
        h = max(3, val * (height - 2))
        x = i * (bw + gap)
        op = "1.0" if i == n - 1 else "0.45"
        bars.append(f'<rect x="{x:.1f}" y="{height-h:.1f}" width="{bw:.1f}" height="{h:.1f}" '
                     f'rx="2" fill="{color}" opacity="{op}"/>')
    return (f'<svg class="kpi-spark" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">{"".join(bars)}</svg>')


def kpi_card(icon, title, value, delta_html, sub, spark, accent):
    return f"""
<div class="kpi" style="border-top:3px solid {accent};">
  <div class="kpi-head">
    <div class="kpi-icon" style="background:{accent}22; border:1px solid {accent}55;">{icon}</div>
    <div class="kpi-title">{title}</div>
  </div>
  <div class="kpi-value">{value}</div>
  <div class="kpi-delta">{delta_html}</div>
  {spark}
  <div class="kpi-sub">{sub}</div>
</div>"""


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
def load_region_geo():
    gj = json.loads((APP_DATA_DIR / "uk_regions.geojson").read_text())
    centroids = pd.read_csv(APP_DATA_DIR / "uk_region_centroids.csv")
    return gj, centroids


@st.cache_data
def load_bunching():
    density = pd.read_csv(APP_DATA_DIR / "bunching_density.csv")
    by_year = pd.read_csv(APP_DATA_DIR / "bunching_by_year.csv")
    return density, by_year


@st.cache_data
def load_triage():
    return pd.read_csv(APP_DATA_DIR / "triage_2025.csv.gz", compression="gzip")


@st.cache_data
def load_transitions():
    return pd.read_csv(APP_DATA_DIR / "vein2_transitions.csv")


@st.cache_data
def load_model_ci():
    return pd.read_csv(APP_DATA_DIR / "model_ci.csv")


@st.cache_data
def load_conformal():
    return pd.read_csv(APP_DATA_DIR / "conformal_coverage.csv")


@st.cache_data
def load_fairness():
    return pd.read_csv(APP_DATA_DIR / "fairness_summary.csv")


def transition_evidence(attribute, from_val, to_val):
    """Real panel evidence for a specific attribute change, or None if too few cases."""
    try:
        t = load_transitions()
    except FileNotFoundError:
        return None
    row = t[(t["attribute"] == attribute) & (t["from"] == from_val) & (t["to"] == to_val)]
    return row.iloc[0] if len(row) else None


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
        for _k in ("report_text", "report_key", "report_err"):
            st.session_state.pop(_k, None)

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
            st.caption("Model-estimated change — an association, not a guaranteed outcome, "
                       "and not a substitute for a real assessment.")

            # Real panel evidence (Vein 2): what actually happened to buildings that made
            # this exact change, from 200k+ repeat-certificate histories.
            ev_rows = []
            if sim_fuel != record["main_heating_fuel"]:
                ev = transition_evidence("main_heating_fuel", record["main_heating_fuel"], sim_fuel)
                if ev is not None:
                    ev_rows.append(("heating fuel", record["main_heating_fuel"], sim_fuel, ev))
            if sim_env != record["building_environment"]:
                ev = transition_evidence("building_environment", record["building_environment"], sim_env)
                if ev is not None:
                    ev_rows.append(("building environment", record["building_environment"], sim_env, ev))
            if ev_rows:
                st.markdown("**What actually happened to real buildings that made this change**")
                for what, frm, to, ev in ev_rows:
                    direction = "improved" if ev["median_delta"] < 0 else "worsened"
                    st.markdown(
                        f"- **{what}: {frm} → {to}** — {int(ev['n']):,} real buildings did this. "
                        f"Their rating **{direction} by a median of {abs(ev['median_delta']):.0f} points**, "
                        f"and **{ev['pct_improved_tier']:.0f}%** moved to a better risk tier.")
                st.caption("Real repeat-certificate evidence from the UK register (Vein 2 panel) — "
                           "associational (other changes may co-occur), but these are actual "
                           "before/after outcomes, not a model estimate.")

        # ---------------- Written report (opt-in) ----------------
        st.divider()
        st.markdown("#### AI-written report")
        st.caption("Optional — turns the numbers above into a plain-English summary for a "
                   "business owner. Nothing is generated until you ask for it.")
        rec_key = hashlib.md5(json.dumps(record, sort_keys=True).encode()).hexdigest()

        if st.button("✨ Generate written report", key="gen_report"):
            proba_dict = {"Low": proba[0]*100, "Medium": proba[1]*100, "High": proba[2]*100}
            with st.spinner("Writing report…"):
                try:
                    st.session_state["report_text"] = generate_report(
                        record["property_type_group"], pred_label, proba_dict,
                        pct_all, pct_type, top_risk_drivers, recs)
                    st.session_state["report_err"] = None
                except Exception as e:
                    st.session_state["report_text"] = None
                    st.session_state["report_err"] = type(e).__name__
            st.session_state["report_key"] = rec_key

        if st.session_state.get("report_key") == rec_key:
            report = st.session_state.get("report_text")
            err = st.session_state.get("report_err")
            if report:
                st.write(report)
                st.caption("Generated by an LLM (openai/gpt-oss-20b via Groq) strictly from the "
                           "numbers and citations above — it was not permitted to introduce its "
                           "own figures.")
            elif err:
                st.warning(f"Report unavailable right now ({err}). Everything above already comes "
                           "straight from the model and cited sources.")
            else:
                st.info("No `GROQ_API_KEY` configured — add a Groq key in the app's secrets to "
                        "enable this. Everything above comes straight from the model and cited "
                        "sources regardless.")

# ================================================================ Dashboard
with tab_dash:
    dash = load_dashboard_data()
    hour = pd.Timestamp.now().hour
    daypart = "morning" if hour < 12 else ("afternoon" if hour < 18 else "evening")

    n_regions = dash[~dash["uk_region"].isin(["Unknown", "Scotland"])]["uk_region"].nunique()
    hero_col, filt_col = st.columns([3.2, 1])
    with hero_col:
        st.markdown(f"""
<div class="hero-card">
  <p class="hero-title">Overview</p>
  <p class="hero-sub">Good {daypart} — here's the current state of the England &amp; Wales
     non-domestic EPC register for small businesses.</p>
  <p class="hero-sub" style="margin-top:10px; font-family:'IBM Plex Mono',monospace; font-size:0.85rem;">
     <b>{len(dash):,}</b> certificates&ensp;·&ensp;<b>{n_regions}</b> regions&ensp;·&ensp;
     <b>{dash['lodgement_year'].min()} → {dash['lodgement_year'].max()}</b>
     <span class="chip" style="margin-left:14px;">≤ 500 m² small businesses</span></p>
</div>""", unsafe_allow_html=True)
    with filt_col:
        yr_filter = st.radio("Period", ["All years", "Since 2022", "2025 only"],
                              key="dash_period", label_visibility="collapsed")
    yr_min = {"All years": 2018, "Since 2022": 2022, "2025 only": 2025}[yr_filter]
    d = dash[dash["lodgement_year"] >= yr_min]

    yearly = dash.groupby("lodgement_year").agg(
        n=("asset_rating", "size"), med=("asset_rating", "median"),
        high=("risk_tier", lambda s: (s == "High").mean() * 100),
        cplus=("asset_rating", lambda s: (s <= 75).mean() * 100)).reset_index()

    def yoy(col, pct=False):
        if len(yearly) < 2:
            return ""
        a, b = yearly[col].iloc[-2], yearly[col].iloc[-1]
        chg = (b - a) / a * 100 if pct else b - a
        return chg

    kcol, chartcol = st.columns([1.15, 1.6])
    with kcol:
        g1, g2 = st.columns(2)
        n_chg = yoy("n", pct=True)
        g1.markdown(kpi_card(
            "📜", "Certificates", f"{len(d):,}",
            f'<span class="{ "down" if n_chg < 0 else "up" }">{"▼" if n_chg < 0 else "▲"} '
            f'{abs(n_chg):.1f}%</span> <span class="lbl">vs prior year</span>',
            f"{d['floor_area'].median():.0f} m² median floor area",
            sparkline_svg(yearly["n"], "#ff4dd6"), "#ff4dd6"), unsafe_allow_html=True)
        med_chg = yearly["med"].iloc[-1] - yearly["med"].iloc[0]
        g2.markdown(kpi_card(
            "⚡", "Median rating", f"{d['asset_rating'].median():.0f}",
            f'<span class="up">▼ {abs(med_chg):.0f} pts</span> '
            f'<span class="lbl">since {yearly["lodgement_year"].min()} (lower = better)</span>',
            "Official assessed asset rating",
            sparkline_svg(yearly["med"], "#39ff8c"), "#39ff8c"), unsafe_allow_html=True)
        g3, g4 = st.columns(2)
        high_now = (d["risk_tier"] == "High").mean() * 100
        high_chg = yearly["high"].iloc[-1] - yearly["high"].iloc[0]
        g3.markdown(kpi_card(
            "🔥", "High-risk share", f"{high_now:.1f}%",
            f'<span class="{ "up" if high_chg < 0 else "down" }">{"▼" if high_chg < 0 else "▲"} '
            f'{abs(high_chg):.1f} pts</span> <span class="lbl">since {yearly["lodgement_year"].min()}</span>',
            "Bands E-G collapsed to High tier",
            sparkline_svg(yearly["high"], "#e8a05c"), "#e8a05c"), unsafe_allow_html=True)
        cplus_now = (d["asset_rating"] <= 75).mean() * 100
        cplus_chg = yearly["cplus"].iloc[-1] - yearly["cplus"].iloc[0]
        g4.markdown(kpi_card(
            "🏅", "Rated C or better", f"{cplus_now:.1f}%",
            f'<span class="{ "up" if cplus_chg > 0 else "down" }">{"▲" if cplus_chg > 0 else "▼"} '
            f'{abs(cplus_chg):.1f} pts</span> <span class="lbl">since {yearly["lodgement_year"].min()}</span>',
            "The announced 2027-31 direction of travel",
            sparkline_svg(yearly["cplus"], "#35e6e6"), "#35e6e6"), unsafe_allow_html=True)
    with chartcol:
        st.markdown('<div class="panel"><div class="panel-head">'
                     '<div class="kpi-icon" style="background:#39ff8c22;border:1px solid #39ff8c55;">📊</div>'
                     '<div class="panel-title">Rating bands</div></div>'
                     f'<div class="panel-sub">{yr_filter} · A+ best → G worst · MEES bans letting F/G</div>',
                     unsafe_allow_html=True)
        band_counts = d["asset_rating_band"].value_counts().reindex(
            ["A+", "A", "B", "C", "D", "E", "F", "G"]).fillna(0)
        fig = go.Figure(go.Bar(
            x=band_counts.index, y=band_counts.values,
            marker=dict(color=[BAND_COLORS[b] for b in band_counts.index],
                         cornerradius=6),
        ))
        fig.update_layout(**PLOTLY_DARK_LAYOUT, showlegend=False, xaxis_title="",
                            yaxis_title="Buildings", margin=dict(l=0, r=0, t=6, b=0), height=372)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ---------- map + regional rankings ----------
    st.markdown('<div class="panel"><div class="panel-head">'
                 '<div class="kpi-icon" style="background:#ff4dd622;border:1px solid #ff4dd655;">📍</div>'
                 '<div class="panel-title">Buildings by region</div></div>'
                 f'<div class="panel-sub">{len(d):,} buildings · hover the map to explore · '
                 'color = mean asset rating in each area</div>', unsafe_allow_html=True)
    # Region stats computed ONCE, shared by the map, legend, and ranking bars --
    # excluding Unknown and the Scotland border-artifact bucket (its tiny-sample
    # mean ~26 was poisoning the min-max normalization and turning the map pink).
    reg_stats = d[d["uk_region"].notna() & ~d["uk_region"].isin(["Unknown", "Scotland"])].groupby(
        "uk_region").agg(n=("asset_rating", "size"), mean_rating=("asset_rating", "mean"),
                          high=("risk_tier", lambda s: (s == "High").mean() * 100)).reset_index()
    rmin, rmax = reg_stats["mean_rating"].min(), reg_stats["mean_rating"].max()
    reg_stats["z_norm"] = (reg_stats["mean_rating"] - rmin) / max(rmax - rmin, 1e-9)
    reg_stats["heat"] = reg_stats["z_norm"].map(heat_color)

    mcol, rcol = st.columns([1, 1.15])
    with mcol:
        geojson, centroids = load_region_geo()
        figm = go.Figure(go.Choropleth(
            geojson=geojson, locations=reg_stats["uk_region"], featureidkey="properties.region",
            z=reg_stats["z_norm"], zmin=0.0, zmax=1.0,
            colorscale=[[t, c] for t, c in HEAT_STOPS],
            marker_line_color="#0a0f0d", marker_line_width=1.4,
            showscale=False,
            customdata=reg_stats[["n", "mean_rating"]],
            hovertemplate="<b>%{location}</b><br>%{customdata[0]:,} buildings<br>"
                           "mean rating %{customdata[1]:.0f}<extra></extra>",
        ))
        lab = centroids.merge(reg_stats, left_on="region", right_on="uk_region")
        figm.add_trace(go.Scattergeo(
            lon=lab["lon"], lat=lab["lat"], mode="text",
            text=[f"{n:,}" for n in lab["n"]],
            textfont=dict(family="IBM Plex Mono", size=11, color="#ffffff"),
            hoverinfo="skip"))
        figm.update_geos(fitbounds="locations", visible=False, bgcolor="rgba(0,0,0,0)",
                          projection_type="mercator")
        figm.update_layout(**PLOTLY_DARK_LAYOUT, height=460,
                            margin=dict(l=0, r=0, t=0, b=0), showlegend=False)
        st.plotly_chart(figm, use_container_width=True)
        st.markdown(f"""
<div class="heat-scale">
  <div style="font-family:'IBM Plex Mono',monospace;font-size:0.7rem;letter-spacing:0.08em;
       text-transform:uppercase;color:#aab8b0;margin-bottom:6px;">Heat scale · mean asset rating</div>
  <div class="heat-bar"></div>
  <div class="heat-lbl"><b style="color:#39ff8c;">Efficient · {rmin:.0f}</b>
      <b style="color:#c23fd6;">Poor · {rmax:.0f}</b></div>
</div>""", unsafe_allow_html=True)
    with rcol:
        st.markdown('<div class="panel-sub" style="font-family:\'IBM Plex Mono\',monospace;'
                     'letter-spacing:0.08em;text-transform:uppercase;">Region rankings · '
                     'bar colour = same efficiency heat as the map</div>',
                     unsafe_allow_html=True)
        rank = reg_stats.set_index("uk_region").sort_values("n", ascending=False)
        total = rank["n"].sum()
        max_n = rank["n"].max()
        rows = []
        for i, (region, r) in enumerate(rank.head(10).iterrows(), start=1):
            chip_cls = "rank-chip first" if i == 1 else "rank-chip"
            c = r["heat"]
            rows.append(f"""
<div class="rank-row">
  <div class="{chip_cls}">{i}</div>
  <div class="rank-name">{region}</div>
  <div class="rank-track"><div class="rank-fill" style="width:{r['n']/max_n*100:.0f}%;
       background:linear-gradient(90deg,{c}88,{c}); box-shadow:0 0 8px -2px {c};"></div></div>
  <div class="rank-val"><span>{r['n']/total*100:.1f}%</span>{r['n']:,.0f}</div>
</div>""")
        st.markdown("".join(rows), unsafe_allow_html=True)
        worst = rank.sort_values("high", ascending=False).head(1)
        st.markdown(f'<div class="panel-sub" style="margin-top:10px;">Highest High-risk share: '
                     f'<b style="color:#ff4d6d;">{worst.index[0]}</b> at {worst["high"].iloc[0]:.1f}% · '
                     f'best mean rating: <b style="color:#39ff8c;">'
                     f'{rank.sort_values("mean_rating").index[0]}</b></div>',
                     unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    st.caption("Region shapes are real ONS boundaries (England & Wales); each region is "
               "colored by the mean assessed rating of its buildings and labeled with its "
               "building count.")

    # ---------- trend ----------
    st.markdown('<div class="panel"><div class="panel-head">'
                 '<div class="kpi-icon" style="background:#35e6e622;border:1px solid #35e6e655;">📈</div>'
                 '<div class="panel-title">Efficiency is improving — the real year-over-year trend</div></div>'
                 '<div class="panel-sub">Mean assessed rating, all years (lower = better). This trend is '
                 'why models train on 2018-2024 and are tested on 2025 data they never saw.</div>',
                 unsafe_allow_html=True)
    figt = go.Figure()
    figt.add_trace(go.Scatter(
        x=yearly["lodgement_year"], y=yearly["med"], mode="lines+markers",
        line=dict(color=NEON, width=3, shape="spline"),
        marker=dict(size=9, color=NEON, line=dict(width=1, color="#04150c")),
        fill="tozeroy", fillcolor="rgba(57,255,140,0.07)"))
    figt.update_layout(**PLOTLY_DARK_LAYOUT, yaxis_title="Median asset rating",
                        xaxis_title="", margin=dict(l=0, r=0, t=6, b=0), height=280)
    st.plotly_chart(figt, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

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
        # Normalize Shannon entropy (nats) to a bounded 0-1 uncertainty score by dividing
        # by its theoretical maximum ln(3) for 3 classes -> 1.0 == a perfectly uniform
        # 0.33/0.33/0.33 prediction (maximally uncertain), 0.0 == fully confident.
        triage = triage.copy()
        triage["uncertainty"] = (triage["uncertainty_entropy"] / np.log(3) * 100).clip(0, 100)

        budget = st.slider("Available physical-audit budget (% of building portfolio)",
                            1, 25, 5, key="triage_budget")
        k = max(1, int(len(triage) * budget / 100))
        # rank by uncertainty, breaking ties (many buildings pin at the ceiling) by the
        # predicted High-risk probability, so the queue is deterministic and sensible.
        queue = triage.sort_values(["uncertainty", "p_high"], ascending=False).head(k)

        hit_queue = (queue["actual_tier"] == "High").mean() * 100
        hit_random = (triage["actual_tier"] == "High").mean() * 100
        n_maxed = int((triage["uncertainty"] >= 99.9).sum())
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Portfolio", f"{len(triage):,} buildings")
        c2.metric("Audit queue", f"{k:,} buildings")
        c3.metric("High-risk found in queue", f"{hit_queue:.1f}%")
        c4.metric("vs random auditing", f"{hit_random:.1f}%",
                   delta=f"{hit_queue-hit_random:+.1f} pts")
        st.caption(f"Uncertainty = Shannon entropy of the model's 3-class probabilities, "
                   f"normalized to 0-100% (100% = a perfectly uniform prediction). "
                   f"{n_maxed:,} buildings sit at the ceiling — genuinely coin-flip cases — so "
                   f"within that group the queue breaks ties by predicted High-risk probability. "
                   f"'High-risk found' uses the 2025 ground truth, known only because this is a "
                   f"labeled test year; at deployment it would be unknown, which is the whole "
                   f"point of ranking by uncertainty.")

        st.markdown("#### Triage queue — most uncertain first")
        show = queue[["property_type_group", "uk_region", "local_authority", "floor_area_m2",
                       "main_heating_fuel", "predicted_tier", "p_high", "uncertainty"]]
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
                "uncertainty": st.column_config.ProgressColumn(
                    "Uncertainty", min_value=0.0, max_value=100.0, format="%.0f%%"),
            })

        # ---- Vein 3: does the ranking work, and does it stay reliable under drift? ----
        try:
            curve = pd.read_csv(APP_DATA_DIR / "triage_curve.csv")
            cov = load_conformal()
        except FileNotFoundError:
            curve = cov = None
        if curve is not None:
            v1, v2 = st.columns(2)
            with v1:
                st.markdown("#### Does uncertainty-ranking actually help?")
                figv = go.Figure()
                figv.add_trace(go.Scatter(
                    x=curve["budget_frac"] * 100, y=curve["high_captured_%"],
                    mode="lines+markers", line=dict(color=NEON, width=3), name="Uncertainty-ranked"))
                figv.add_trace(go.Scatter(x=[0, 100], y=[0, 100], mode="lines",
                                           line=dict(color=SLATE, dash="dash"), name="Random"))
                figv.update_layout(**PLOTLY_DARK_LAYOUT, height=300, margin=dict(l=0, r=0, t=6, b=0),
                                    xaxis_title="% audited", yaxis_title="% High-risk caught",
                                    legend=dict(orientation="h", y=1.02))
                st.plotly_chart(figv, use_container_width=True)
                st.caption(f"At a 5% audit budget the queue finds High-risk buildings at "
                           f"{curve[curve.budget_frac==0.05]['high_hit_rate_%'].iloc[0]:.1f}% "
                           f"vs {curve['random_hit_rate_%'].iloc[0]:.1f}% random "
                           f"({curve[curve.budget_frac==0.05]['lift'].iloc[0]:.1f}× lift).")
            with v2:
                st.markdown("#### Does the model stay reliable? (conformal coverage)")
                figc = go.Figure()
                tcov = cov[cov.role != "calibration"]
                figc.add_hline(y=90, line_dash="dash", line_color=NEON,
                                annotation_text="target 90%")
                figc.add_trace(go.Scatter(x=cov["year"], y=cov["coverage"] * 100,
                                           mode="lines+markers", line=dict(color=CRIMSON, width=3)))
                figc.update_layout(**PLOTLY_DARK_LAYOUT, height=300, margin=dict(l=0, r=0, t=6, b=0),
                                    xaxis_title="Test year", yaxis_title="Coverage %", showlegend=False)
                st.plotly_chart(figc, use_container_width=True)
                st.caption("Conformal prediction sets calibrated on 2022 for 90% coverage. "
                           "Coverage decays year by year as the register drifts — a built-in "
                           "expiry signal telling you when the model needs recalibration.")

        # ---- Fairness disclosure ----
        try:
            fair = load_fairness()
            with st.expander("⚖️ Fairness audit — who does the triage over-flag?"):
                st.markdown(
                    "A triage tool that allocates inspections must be checked for who it "
                    "sends them to. On the 2025 test set:")
                st.dataframe(fair, use_container_width=True, hide_index=True)
                st.markdown(
                    "**The honest finding:** the model materially **over-flags Office/Workshop "
                    "buildings** (flagged High far above their true base rate) and they dominate "
                    "the uncertainty queue at ~2.8× their share of the portfolio. Any real "
                    "deployment should stratify the audit budget by building type rather than "
                    "letting one category absorb the queue. Full audit: "
                    "`analysis/fairness_audit.py`.")
        except FileNotFoundError:
            pass

    with st.expander("Green computing: the measured model tradeoff behind this app"):
        show_g = green_stats.rename(columns={
            "test_accuracy": "Accuracy", "test_macro_f1": "Macro F1",
            "recall_high_risk": "High-risk recall", "train_time_s": "Train (s)",
            "inference_ms_per_prediction": "Inference (ms)", "model_size_kb": "Size (KB)",
        }).set_index("model")
        st.dataframe(show_g, use_container_width=True)
        try:
            ci = load_model_ci()
            st.markdown("**Recall with 95% confidence intervals** (bootstrap over the 2025 "
                        "test set; seed-spread across 5 retrainings):")
            ci_show = ci.assign(**{
                "High-risk recall (95% CI)": ci.apply(
                    lambda r: f"{r.recall_high:.3f} [{r.recall_boot_lo:.3f}, {r.recall_boot_hi:.3f}]", axis=1),
            })[["model", "High-risk recall (95% CI)"]]
            st.dataframe(ci_show, use_container_width=True, hide_index=True)
            st.caption("Random Forest's recall CI [0.685, 0.713] sits entirely above the ANN's "
                       "[0.038, 0.050] and XGBoost's [0.569, 0.599] — the recall-based selection "
                       "is statistically robust, not a one-seed artifact.")
        except FileNotFoundError:
            pass
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
(the dashboard maps at region level for that reason); risk tiers collapse the official 8-band scale to 3;
the register's audit-grade feature set is thin (mainly HVAC capacity); escape rates
are conditional on re-certification (selection); fabric changes are invisible in the
public register. Recommendations cite Carbon Trust / GOV.UK published ranges — no
formal energy audit happens here.

Method, code, and executed analyses: `notebooks/greenledger.ipynb`, `analysis/`
(`vein1_bunching.py`, `vein2_panel.py`, related-work and mechanism docs), `README.md`,
`PROCESS.md`, `RESEARCH_ROADMAP.md`.
""")
