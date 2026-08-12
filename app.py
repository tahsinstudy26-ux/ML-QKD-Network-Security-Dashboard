"""
Adaptive Hybrid Network Security Framework
Using Machine Learning and Quantum Assisted Key Distribution — Capstone C
3 Datasets (X-IIoTID / HQC-2026 / CIC-UNSW-NB15) x 12 Models
Run: streamlit run app.py
"""

import os
import zipfile
import pickle
import requests
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# =====================================================================
# Page config
# =====================================================================
st.set_page_config(
    page_title="ML-QKD Security Dashboard",
    page_icon="atom",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =====================================================================
# Auto-extract charts.zip (real matplotlib figures exported from Colab) if present
# =====================================================================
CHARTS_DIR = "charts"
if not os.path.isdir(CHARTS_DIR) and os.path.exists("capstoneC_charts.zip"):
    os.makedirs(CHARTS_DIR, exist_ok=True)
    with zipfile.ZipFile("capstoneC_charts.zip") as zf:
        zf.extractall(CHARTS_DIR)

def chart_path(filename):
    """Return a usable path to a chart PNG, checking both charts/ and the app root."""
    for p in (os.path.join(CHARTS_DIR, filename), filename):
        if os.path.exists(p):
            return p
    return None

# =====================================================================
# CSS - light theme, fixed sidebar, colorful cards
# =====================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Outfit:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@400;500;600;700&display=swap');

:root {
    --bg0:#ffffff; --bg1:#f8fafc; --bg2:#f1f5f9; --bg3:#e2e8f0;
    --card:#ffffff; --border:#e2e8f0; --border2:#cbd5e1;
    --p1:#0284c7; --p2:#6366f1; --p3:#059669; --p4:#d97706;
    --p5:#dc2626; --p6:#ea580c; --p7:#7c3aed; --p8:#0d9488;
    --new1:#059669; --new2:#ea580c;
    --tx:#0f172a; --muted:#475569; --muted2:#334155;
}
html,body,[class*="css"]{font-family:'Space Grotesk',sans-serif;background:var(--bg0)!important;color:var(--tx)!important;font-size:17px!important}
.stApp{background:var(--bg0)!important}
#MainMenu,footer,header{visibility:hidden}
.block-container{padding:1.1rem 1.8rem 3rem!important;max-width:1700px!important}

[data-testid="stSidebar"]{background:var(--bg1)!important;border-right:1px solid var(--border2)!important;
    min-width:260px!important;max-width:260px!important;transform:none!important;visibility:visible!important;position:relative!important}
[data-testid="stSidebar"][aria-expanded="false"]{min-width:260px!important;max-width:260px!important;margin-left:0!important;transform:none!important}
[data-testid="stSidebarCollapsedControl"]{display:none!important;visibility:hidden!important}
[data-testid="collapsedControl"]{display:none!important;visibility:hidden!important}
button[kind="header"]{display:none!important}
[data-testid="stSidebar"] *{color:var(--tx)!important;font-size:1.02rem!important}

/* ===== Sidebar navigation highlight (functional color, not background) ===== */
[data-testid="stSidebar"] .stRadio > div{gap:.2rem!important}
[data-testid="stSidebar"] .stRadio label{border-radius:10px!important;padding:.5rem .7rem!important;transition:all .15s ease!important;margin-bottom:.05rem!important}
[data-testid="stSidebar"] .stRadio label:hover{background:#eef2ff!important;cursor:pointer}
[data-testid="stSidebar"] .stRadio label:has(input:checked){background:linear-gradient(135deg,var(--p2),var(--p1))!important;box-shadow:0 3px 10px rgba(99,102,241,.4)!important}
[data-testid="stSidebar"] .stRadio label:has(input:checked) *{color:#ffffff!important;font-weight:800!important}

/* ===== General text sizing - all text black ===== */
.stMarkdown p, .stMarkdown li, .stMarkdown span{font-size:1.08rem!important;line-height:1.55!important;color:var(--tx)!important}
[data-testid="stCaptionContainer"], .stCaption, .stMarkdown small{font-size:1.05rem!important;color:var(--tx)!important;font-weight:500!important}
.stButton button{font-size:1.05rem!important;font-weight:700!important;padding:.55rem 1.2rem!important;border-radius:10px!important;
    background:linear-gradient(135deg,var(--p2),var(--p1))!important;color:#ffffff!important;border:none!important;
    box-shadow:0 3px 10px rgba(99,102,241,.4)!important;transition:transform .12s ease,box-shadow .12s ease!important}
.stButton button:hover{background:linear-gradient(135deg,var(--p1),var(--p7))!important;box-shadow:0 5px 16px rgba(124,58,237,.5)!important;transform:translateY(-1px)!important}
.stButton button p{font-size:1.05rem!important}
.stRadio label{font-size:1.05rem!important;padding:.2rem 0!important;font-weight:600!important;color:var(--tx)!important}
.stSelectbox label, .stSlider label, .stNumberInput label{color:var(--tx)!important;font-size:1rem!important;font-weight:800!important}
.stSelectbox div[data-baseweb="select"] *{font-size:1.02rem!important;color:var(--tx)!important}
.stSelectbox div[data-baseweb="select"]>div{border:2px solid var(--border2)!important;border-radius:10px!important}
.stSelectbox div[data-baseweb="select"]>div:hover{border-color:var(--p2)!important}
[data-testid="stExpander"]{border:2px solid var(--p2)!important;border-radius:12px!important;overflow:hidden!important;
    background:linear-gradient(135deg,#ffffff,#f7f7ff)!important}
[data-testid="stExpander"] summary{font-size:1.05rem!important;font-weight:700!important;color:var(--tx)!important}
.stAlert{border-radius:12px!important;border-left-width:6px!important;font-weight:600!important;box-shadow:0 2px 8px rgba(0,0,0,.07)!important}
.stAlert p{font-size:1.05rem!important;font-weight:500!important;color:var(--tx)!important}
div[data-testid="stForm"]{border:2px solid var(--p2)!important;border-radius:14px!important;padding:1rem!important;
    background:linear-gradient(135deg,#f8f7ff,#f0f9ff)!important;box-shadow:0 3px 12px rgba(99,102,241,.12)!important}

div[data-testid="stMetric"]{background:linear-gradient(135deg,#ffffff,#eef4ff);border:1px solid var(--border2);border-left:6px solid var(--p1);
    border-radius:12px;padding:1rem 1.15rem;box-shadow:0 3px 10px rgba(2,132,199,.16)}
div[data-testid="stMetricLabel"]{color:var(--p2)!important;font-size:.92rem!important;font-weight:700!important;text-transform:uppercase;letter-spacing:.02em}
div[data-testid="stMetricValue"]{font-family:'JetBrains Mono',monospace!important;color:var(--tx)!important;font-size:1.7rem!important;font-weight:800!important}

.card{background:linear-gradient(135deg,#ffffff,#f5f7ff);border:1px solid var(--border2);border-left:6px solid var(--p2);border-radius:14px;padding:1.2rem 1.4rem;margin-bottom:1rem;box-shadow:0 3px 12px rgba(99,102,241,.10)}
.pillrow{display:flex;gap:.6rem;flex-wrap:wrap;margin:.5rem 0 1.1rem}
.pill{background:linear-gradient(135deg,#e0e7ff,#f1f5f9);border:1.5px solid var(--p2);border-radius:999px;padding:.4rem 1.05rem;font-size:.95rem;font-weight:700;color:var(--tx);box-shadow:0 2px 6px rgba(0,0,0,.05)}
.pill b{color:var(--tx)}
.pill:nth-child(1){background:linear-gradient(135deg,#dbeafe,#eff6ff);border-color:var(--p1)}
.pill:nth-child(2){background:linear-gradient(135deg,#dcfce7,#f0fdf4);border-color:var(--p3)}
.pill:nth-child(3){background:linear-gradient(135deg,#ffedd5,#fff7ed);border-color:var(--p4)}
h1{font-family:'Outfit',sans-serif!important;font-size:2.3rem!important;font-weight:800!important;color:var(--tx)!important;padding-bottom:.2rem}
h2{font-family:'Outfit',sans-serif!important;font-size:1.7rem!important;font-weight:800!important;color:var(--tx)!important;
    border-bottom:3px solid var(--p4);display:inline-block;padding-bottom:.15rem}
h3,h4,h5{font-family:'Outfit',sans-serif!important;color:var(--tx)!important;font-size:1.4rem!important;font-weight:800!important}
hr{border-color:var(--border2)!important;border-width:1px!important;opacity:.7}
.small{color:var(--tx);font-size:.95rem}
.badge-new{background:linear-gradient(135deg,#059669,#34d399);color:#ffffff;font-weight:800;font-size:.78rem;padding:.18rem .6rem;border-radius:6px;margin-left:.4rem;box-shadow:0 2px 6px rgba(5,150,105,.4)}

.statcard{background:linear-gradient(135deg,#ffffff,#f6f9ff);border:1px solid var(--border2);border-radius:12px;padding:1rem 1.15rem;box-shadow:0 3px 10px rgba(0,0,0,.09)}
.statcard .lbl{color:var(--p2);font-size:.92rem;font-weight:700;margin-bottom:.2rem;text-transform:uppercase;letter-spacing:.02em}
.statcard .val{font-family:'JetBrains Mono',monospace;font-size:1.65rem;font-weight:800;color:var(--tx);white-space:normal;word-break:break-word;line-height:1.2}

/* ===== Colorful static tables (st.table) ===== */
[data-testid="stTable"] table{width:100%!important;border-collapse:separate!important;border-spacing:0!important;
    border-radius:12px!important;overflow:hidden!important;box-shadow:0 3px 12px rgba(0,0,0,.10)!important;border:1px solid var(--border2)!important}
[data-testid="stTable"] thead th{background:linear-gradient(135deg,var(--p2),var(--p1))!important;color:#ffffff!important;
    font-weight:800!important;font-size:1.02rem!important;padding:.7rem 1rem!important;text-align:left!important;
    text-transform:uppercase;letter-spacing:.02em;border:none!important}
[data-testid="stTable"] tbody td{color:var(--tx)!important;font-size:1.02rem!important;font-weight:600!important;
    padding:.6rem 1rem!important;border-bottom:1px solid var(--border)!important}
[data-testid="stTable"] tbody tr:nth-child(odd){background:#eef2ff!important}
[data-testid="stTable"] tbody tr:nth-child(even){background:#ffffff!important}
[data-testid="stTable"] tbody tr:hover{background:#dbeafe!important}
[data-testid="stTable"] tbody th{background:#f8fafc!important;color:var(--tx)!important;font-weight:700!important;padding:.6rem 1rem!important}
</style>
""", unsafe_allow_html=True)

# =====================================================================
# Dataset registry
# =====================================================================
DATASETS = {
    "XIIoTID": {"label": "X-IIoTID", "country": "Australia", "year": "2021",
                "results_file": "results_XIIoTID.pkl", "artifacts_file": "app_artifacts_XIIoTID.pkl",
                "prefix": "xiiot"},
    "HQC2026": {"label": "HQC-2026", "country": "Bangladesh", "year": "2026",
                "results_file": "results_HQC2026.pkl", "artifacts_file": "app_artifacts_HQC2026.pkl",
                "prefix": "hqc2026"},
    "CICUNSW": {"label": "CIC-UNSW-NB15", "country": "Canada", "year": "2024",
                "results_file": "results_CICUNSW.pkl", "artifacts_file": "app_artifacts_CICUNSW.pkl",
                "prefix": "cicunsw"},
}
DATASET_KEYS = list(DATASETS.keys())

# =====================================================================
# Large artifact files (.pkl) are hosted on Hugging Face Hub because
# they exceed GitHub's file size limits. If a file isn't found locally,
# it is downloaded once and cached on disk.
# =====================================================================
HF_ARTIFACT_URLS = {
    "XIIoTID": "https://huggingface.co/atya-tahsin/ml-qkd-dashboard-models/resolve/main/app_artifacts_XIIoTID.pkl",
    "HQC2026": "https://huggingface.co/atya-tahsin/ml-qkd-dashboard-models/resolve/main/app_artifacts_HQC2026.pkl",
    "CICUNSW": "https://huggingface.co/atya-tahsin/ml-qkd-dashboard-models/resolve/main/app_artifacts_CICUNSW.pkl",
}

HF_RESULTS_URLS = {
    "XIIoTID": "https://huggingface.co/atya-tahsin/ml-qkd-dashboard-models/resolve/main/results_XIIoTID.pkl",
}

def ensure_artifact_downloaded(local_path, url):
    """Download a large artifact file from Hugging Face if it isn't already
    present locally. Streams to disk in chunks to avoid high memory use."""
    if os.path.exists(local_path):
        return True
    try:
        with st.spinner(f"Downloading {local_path} from Hugging Face (first run only)..."):
            with requests.get(url, stream=True, timeout=300) as r:
                r.raise_for_status()
                tmp_path = local_path + ".part"
                with open(tmp_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8 * 1024 * 1024):
                        if chunk:
                            f.write(chunk)
                os.replace(tmp_path, local_path)
        return True
    except Exception as e:
        st.warning(f"Could not download {local_path} from Hugging Face: {e}")
        return False

CHART_FILES = {
    "XIIoTID": {
        "eda": "xiiot_eda.png", "model_comparison": "xiiot_model_comparison.png",
        "confusion": "xiiot_confusion_matrices.png", "roc": "xiiot_roc_curves.png",
        "feat_imp": "xiiot_feature_importance.png", "risk_qkd": "xiiot_risk_qkd.png",
        "shap": "xiiot_shap_dot.png", "shap_bar": "xiiot_shap_bar.png",
        "response": "xiiot_response_recommendation.png",
        "routing": "xiiot_detection_routing_pipeline.png", "bb84": "xiiot_bb84_pipeline.png",
    },
    "HQC2026": {
        "eda": "hqc2026_eda.png", "model_comparison": "hqc2026_model_comparison.png",
        "confusion": "hqc2026_confusion_matrices.png", "roc": "hqc2026_roc_curves.png",
        "feat_imp": "hqc2026_feature_importance.png", "risk_qkd": "hqc2026_risk_qkd.png",
        "shap": "hqc2026_shap_summary.png", "shap_bar": "hqc2026_shap_importance.png",
        "response": "hqc2026_response_recommendation.png",
        "routing": "hqc2026_detection_routing_pipeline.png", "bb84": "hqc2026_bb84_pipeline.png",
    },
    "CICUNSW": {
        "eda": "cicunsw_eda.png", "model_comparison": "cicunsw_model_comparison.png",
        "confusion": "cicunsw_confusion_matrices.png", "roc": "cicunsw_roc_curves.png",
        "feat_imp": "cicunsw_feature_importance.png", "risk_qkd": "cicunsw_risk_bb84.png",
        "shap": "cicunsw_shap_dot.png", "shap_bar": "cicunsw_shap_bar.png",
        "response": "cicunsw_response_recommendation.png",
        "routing": "cicunsw_detection_routing_pipeline.png", "bb84": "cicunsw_bb84_pipeline.png",
        "bb84_steps": "cicunsw_bb84_protocol_pipeline.png",
    },
}

MODEL_CLR = {
    'Random Forest': '#0284c7', 'Gradient Boosting': '#059669', 'XGBoost': '#d97706',
    'SVM': '#ea580c', 'Voting Ensemble': '#ca8a04', 'Stacking': '#7c3aed',
    'LightGBM': '#10b981', 'MLP Neural Net': '#f97316',
    'CatBoost': '#8b5cf6', 'Extra Trees': '#0d9488', 'AdaBoost': '#db2777', 'Naive Bayes': '#64748b',
}
DATASET_CLR = {"XIIoTID": "#0284c7", "HQC2026": "#059669", "CICUNSW": "#d97706"}

# Cross-dataset comparison charts (dataset-independent — generated once at the end of the notebook)
CROSS_CHARTS = {
    "model_comparison": "cross_model_comparison.png",
    "qkd_overhead": "cross_qkd_overhead.png",
    "shap_comparison": "cross_shap_comparison.png",
    "radar": "cross_radar_chart.png",
    "precision_recall": "cross_precision_recall.png",
    "confusion": "cross_confusion_matrix.png",
    "threshold": "cross_threshold_sensitivity.png",
}

PL = dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(238,244,255,.55)",
          font=dict(family="Space Grotesk", color="#0f172a", size=12),
          margin=dict(l=10, r=10, t=38, b=10))
LEG = dict(bgcolor="rgba(255,255,255,.6)", bordercolor="#6366f1", borderwidth=1, font_size=11)
AXC = dict(gridcolor="#dbe4f3", zerolinecolor="#c7d2fe", linecolor="#94a3b8")
def ax(**kw): return dict(**AXC, **kw)

def stat_card(label, value, accent="#0284c7"):
    st.markdown(f"""<div class='statcard' style='border-left:5px solid {accent};
        background:linear-gradient(135deg,{accent}22,{accent}08);box-shadow:0 3px 10px {accent}30'>
        <div class='lbl' style='color:{accent}'>{label}</div><div class='val'>{value}</div></div>""", unsafe_allow_html=True)

ACTION_SEVERITY = {
    "isolate": ("#fee2e2", "#991b1b"),
    "block": ("#ffedd5", "#9a3412"),
    "lockout": ("#ffedd5", "#9a3412"),
    "rate-limit": ("#fef3c7", "#92400e"),
    "revoke": ("#fef3c7", "#92400e"),
    "refresh": ("#fef3c7", "#92400e"),
    "log": ("#dcfce7", "#166534"),
    "monitor": ("#dcfce7", "#166534"),
    "manual review": ("#e0e7ff", "#3730a3"),
    "no action": ("#f1f5f9", "#334155"),
}
def action_color(action_text):
    t = (action_text or "").lower()
    for k, v in ACTION_SEVERITY.items():
        if k in t:
            return v
    return ("#f1f5f9", "#334155")

def bb84_animation_html():
    """A lightweight, self-contained animated walk-through of one BB84 session."""
    html = """
    <div id="bb84wrap" style="font-family:'Space Grotesk',sans-serif;background:#f8fafc;
         border:1px solid #e2e8f0;border-radius:12px;padding:1rem 1.2rem;">
      <div style="display:flex;justify-content:space-between;font-size:.82rem;color:#475569;margin-bottom:.5rem">
        <span>Alice</span><span>Quantum Channel</span><span>Bob</span>
      </div>
      <div id="bb84row" style="display:flex;gap:6px;flex-wrap:wrap;justify-content:center;min-height:110px;align-items:center"></div>
      <div id="bb84status" style="text-align:center;margin-top:.7rem;font-size:.9rem;color:#334155;min-height:24px"></div>
      <div style="text-align:center;margin-top:.4rem">
        <button onclick="bb84restart()" style="background:#0284c7;color:white;border:none;border-radius:8px;
        padding:.4rem 1rem;font-size:.82rem;cursor:pointer;font-family:'Space Grotesk',sans-serif">Replay</button>
      </div>
    </div>
    <script>
    const N = 16;
    const bases = ['+','x'];
    let aliceBits=[], aliceBases=[], bobBases=[], match=[];
    function rnd(arr){return arr[Math.floor(Math.random()*arr.length)];}
    function setup(){
      aliceBits=[]; aliceBases=[]; bobBases=[]; match=[];
      for(let i=0;i<N;i++){
        aliceBits.push(Math.round(Math.random()));
        aliceBases.push(rnd(bases));
        bobBases.push(rnd(bases));
        match.push(aliceBases[i]===bobBases[i]);
      }
    }
    function cellHTML(i, stage){
      let bit = aliceBits[i];
      let ab = aliceBases[i];
      let bb = bobBases[i];
      let symbol = ab==='+' ? (bit===0?String.fromCharCode(8593):String.fromCharCode(8594)) : (bit===0?String.fromCharCode(8599):String.fromCharCode(8600));
      let box = "background:#e2e8f0;color:#1e293b;border:1px solid #cbd5e1";
      let label = "";
      if(stage>=1){ box="background:#dbeafe;color:#1e40af;border:1px solid #93c5fd"; label=symbol; }
      if(stage>=2){ label = bb; box="background:#ede9fe;color:#5b21b6;border:1px solid #c4b5fd"; }
      if(stage>=3){
        if(match[i]){ box="background:#dcfce7;color:#166534;border:2px solid #22c55e"; label=bit; }
        else { box="background:#fee2e2;color:#991b1b;border:1px solid #fca5a5;opacity:.4"; label="x"; }
      }
      return "<div style=\\"width:34px;height:34px;border-radius:7px;display:flex;align-items:center;justify-content:center;font-weight:700;font-size:.85rem;transition:all .3s;" + box + "\\">" + label + "</div>";
    }
    function render(stage){
      let row = document.getElementById('bb84row');
      if(!row) return;
      row.innerHTML = '';
      for(let i=0;i<N;i++){ row.innerHTML += cellHTML(i, stage); }
    }
    function setStatus(t){ let s=document.getElementById('bb84status'); if(s) s.innerText = t; }
    async function play(){
      setup();
      render(0); setStatus("Alice generates random bits and encodes them using random bases (+ or x)...");
      await new Promise(r=>setTimeout(r,1400));
      render(1); setStatus("Qubits sent across the quantum channel to Bob...");
      await new Promise(r=>setTimeout(r,1400));
      render(2); setStatus("Bob measures each qubit using his own independently random basis...");
      await new Promise(r=>setTimeout(r,1600));
      render(3);
      let kept = match.filter(Boolean).length;
      setStatus("Sifting: bases compared publicly - " + kept + "/" + N + " bits kept (green), mismatched bits discarded (red).");
      await new Promise(r=>setTimeout(r,1800));
      setStatus("Sifting: bases compared publicly - " + kept + "/" + N + " bits kept. Remaining bits -> SHA-256 -> final quantum-secure key.");
    }
    function bb84restart(){ play(); }
    play();
    </script>
    """
    components.html(html, height=230)

# =====================================================================
# Data loading
# =====================================================================
@st.cache_resource(show_spinner="Loading model results...")
def load_dataset(key):
    info = DATASETS[key]
    out = {"results": None, "artifacts": None, "errors": []}

    if not os.path.exists(info["results_file"]) and key in HF_RESULTS_URLS:
        ensure_artifact_downloaded(info["results_file"], HF_RESULTS_URLS[key])

    if os.path.exists(info["results_file"]):
        try:
            with open(info["results_file"], "rb") as f:
                out["results"] = pickle.load(f)
        except Exception as e:
            out["errors"].append(f"results file: {e}")
    else:
        out["errors"].append(f"missing {info['results_file']}")

    if not os.path.exists(info["artifacts_file"]) and key in HF_ARTIFACT_URLS:
        ensure_artifact_downloaded(info["artifacts_file"], HF_ARTIFACT_URLS[key])

    if os.path.exists(info["artifacts_file"]):
        try:
            with open(info["artifacts_file"], "rb") as f:
                out["artifacts"] = pickle.load(f)
        except Exception as e:
            out["errors"].append(f"artifacts file: {e}")
    else:
        out["errors"].append(f"missing {info['artifacts_file']} (Attack Detector will be limited)")

    return out

ALL_DATA = {k: load_dataset(k) for k in DATASET_KEYS}

# =====================================================================
# Sidebar
# =====================================================================
with st.sidebar:
    st.markdown("""
    <div style='padding:.8rem 0 .6rem;'>
        <div style='font-family:Outfit,sans-serif;font-size:1.05rem;font-weight:800;
             background:linear-gradient(135deg,#0284c7,#6366f1,#059669);
             -webkit-background-clip:text;-webkit-text-fill-color:transparent;line-height:1.3;'>
            Adaptive ML-QKD<br>Security Framework
        </div>
    </div>
    <hr/>
    """, unsafe_allow_html=True)

    st.markdown("<div class='small'>ACTIVE DATASET</div>", unsafe_allow_html=True)
    ds_key = st.selectbox("", DATASET_KEYS,
                           format_func=lambda k: f"{DATASETS[k]['label']} ({DATASETS[k]['country']})",
                           label_visibility="collapsed")
    st.markdown("<hr/>", unsafe_allow_html=True)

    nav = st.radio("", [
        "Overview",
        "Dataset & EDA",
        "ML Models",
        "Feature Importance",
        "Risk Engine",
        "Multiclass & Response",
        "QKD Simulation",
        "Cross-Dataset Comparison",
        "Attack Detector",
        "Charts Gallery",
    ], label_visibility="collapsed")

    st.markdown("<hr/>", unsafe_allow_html=True)
    d = ALL_DATA[ds_key]
    ok = d["results"] is not None
    if not ok:
        st.markdown("<div class='small'>data missing - see errors below</div>", unsafe_allow_html=True)
    if d["errors"]:
        with st.expander("Load warnings", expanded=not ok):
            for e in d["errors"]:
                st.caption(e)

R = ALL_DATA[ds_key]["results"]
ART = ALL_DATA[ds_key]["artifacts"]
DS_INFO = DATASETS[ds_key]
CHARTS = CHART_FILES[ds_key]

def missing_data_notice():
    st.error(
        f"**{DS_INFO['results_file']}** not found next to app.py. "
        "Place the results_*.pkl files (produced by the notebook's Save Results cells) "
        "in the same folder as app.py, or use the uploader below."
    )
    up = st.file_uploader(f"Upload {DS_INFO['results_file']}", type="pkl", key=f"up_{ds_key}")
    if up is not None:
        with open(DS_INFO["results_file"], "wb") as f:
            f.write(up.getbuffer())
        st.success("Uploaded - reloading...")
        st.cache_resource.clear()
        st.rerun()

def show_chart(key, caption=None, width=None):
    """Show a real notebook-exported chart if available; otherwise a gentle notice."""
    path = chart_path(CHARTS.get(key, ""))
    if path:
        st.image(path, caption=caption, width=width, use_container_width=(width is None))
        return True
    else:
        st.info(
            f"Chart image not found (`{CHARTS.get(key, '?')}`). Run `notebook_zip_charts.py` in Colab "
            "and place `capstoneC_charts.zip` (or an extracted `charts/` folder) next to app.py."
        )
        return False

def show_cross_chart(key, caption=None, width=None):
    """Show a dataset-independent cross-comparison chart (from CROSS_CHARTS)."""
    path = chart_path(CROSS_CHARTS.get(key, ""))
    if path:
        st.image(path, caption=caption, width=width, use_container_width=(width is None))
        return True
    else:
        st.info(
            f"Chart image not found (`{CROSS_CHARTS.get(key, '?')}`). Run `notebook_zip_charts.py` in Colab "
            "and place `capstoneC_charts.zip` (or an extracted `charts/` folder) next to app.py."
        )
        return False

# =====================================================================
# PAGE: Overview
# =====================================================================
if nav == "Overview":
    st.markdown("""
    <div style='background:linear-gradient(120deg,#0284c7 0%,#6366f1 40%,#7c3aed 70%,#059669 100%);
         border-radius:18px;padding:1.7rem 2rem;margin-bottom:1.1rem;position:relative;overflow:hidden;
         box-shadow:0 8px 24px rgba(99,102,241,.35)'>
      <svg style='position:absolute;right:-10px;top:-20px;opacity:.18' width="220" height="220" viewBox="0 0 220 220">
        <g fill="none" stroke="white" stroke-width="2">
          <circle cx="60" cy="50" r="5"/><circle cx="160" cy="40" r="5"/><circle cx="110" cy="120" r="5"/>
          <circle cx="50" cy="160" r="5"/><circle cx="180" cy="150" r="5"/>
          <line x1="60" y1="50" x2="110" y2="120"/><line x1="160" y1="40" x2="110" y2="120"/>
          <line x1="110" y1="120" x2="50" y2="160"/><line x1="110" y1="120" x2="180" y2="150"/>
        </g>
        <path d="M150 60 l14 -14 l14 14 l0 20 a14 14 0 0 1 -28 0 z" fill="white" opacity=".25"/>
      </svg>
      <div style='color:white;font-family:Outfit,sans-serif;font-size:2.3rem;font-weight:800;position:relative'>
        Adaptive Hybrid Network Security Framework
      </div>
      <div style='color:rgba(255,255,255,.95);font-size:1.1rem;margin-top:.4rem;max-width:760px;position:relative;font-weight:500'>
        Machine-learning-guided intrusion detection with selective <b>BB84 Quantum Key Distribution</b>,
        validated across three independently collected network-traffic datasets.
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='pillrow'>"
                + "".join([f"<span class='pill'><b>{DATASETS[k]['label']}</b> - {DATASETS[k]['country']} ({DATASETS[k]['year']})</span>"
                           for k in DATASET_KEYS])
                + "</div>", unsafe_allow_html=True)

    st.markdown(f"### Currently viewing: {DS_INFO['label']} ({DS_INFO['country']})")
    if R is None:
        missing_data_notice()
    else:
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1: stat_card("🏆 Best Model", R["best_model"], "#0284c7")
        with c2: stat_card("🎯 F1-Score", f"{R['best_f1']:.4f}", "#059669")
        with c3: stat_card("📈 AUC-ROC", f"{R['best_auc']:.4f}", "#6366f1")
        with c4: stat_card("⚛️ QKD Overhead Saved", f"{R['risk_engine']['overhead_saved']:.1f}%", "#d97706")
        with c5: stat_card("🚨 HIGH-Risk Flows", f"{R['risk_engine']['n_high']:,}", "#dc2626")

        st.markdown("### All 3 Datasets at a Glance")
        rows = []
        for k in DATASET_KEYS:
            rr = ALL_DATA[k]["results"]
            if rr is None:
                rows.append([DATASETS[k]["label"], "-", "-", "-", "-"])
            else:
                rows.append([DATASETS[k]["label"], rr["best_model"], f"{rr['best_f1']:.4f}",
                             f"{rr['best_auc']:.4f}", f"{rr['risk_engine']['overhead_saved']:.1f}%"])
        df = pd.DataFrame(rows, columns=["Dataset", "Best Model", "F1", "AUC", "Overhead Saved"])
        st.table(df.style.hide(axis="index"))

        st.markdown("### Detection & Response Pipeline")
        st.caption(
            "The real routing logic from the notebook: every flow is scored by the binary model; "
            "LOW/MEDIUM-risk flows get AES-256 only, while HIGH-risk flows are passed to the multiclass "
            "attack-type model, whose prediction decides the recommended response - and only for the "
            "specific attack types whose response involves a fresh session key does BB84 QKD actually trigger."
        )
        routing_shown = show_chart("routing", caption=f"{DS_INFO['label']} - Detection & Response Routing Pipeline (from your notebook)")
        if not routing_shown:
            steps = [
                ("Network Flow", "all test flows", "#64748b"),
                ("Binary Model", "attack probability", "#0284c7"),
                ("Risk Engine", "LOW / MEDIUM / HIGH", "#6366f1"),
                ("Multiclass Model", "which attack type (HIGH only)", "#7c3aed"),
                ("Response Engine", "action per attack type", "#d97706"),
                ("BB84 QKD", "only for key-refresh actions", "#dc2626"),
            ]
            cols = st.columns(len(steps))
            for col, (title, sub, clr) in zip(cols, steps):
                with col:
                    st.markdown(f"""
                    <div style='background:#f8fafc;border:1px solid {clr}44;border-radius:10px;
                         padding:.6rem .4rem;text-align:center;border-top:3px solid {clr};min-height:70px'>
                        <div style='font-weight:700;font-size:.78rem;color:#1e293b'>{title}</div>
                        <div style='font-size:.68rem;color:#64748b;margin-top:.2rem'>{sub}</div>
                    </div>""", unsafe_allow_html=True)

# =====================================================================
# PAGE: Dataset & EDA
# =====================================================================
elif nav == "Dataset & EDA":
    st.markdown(f"## {DS_INFO['label']} - Dataset Overview")
    if R is None:
        missing_data_notice()
    else:
        c1, c2, c3 = st.columns(3)
        with c1: stat_card("Country / Year", f"{R['country']}", "#0284c7")
        n_total = R["risk_engine"]["n_total"]
        with c2: stat_card("Held-Out Test Flows", f"{n_total:,}", "#059669")
        with c3: stat_card("Features", f"{len(ART['feature_names'])}" if ART else "-", "#6366f1")

        st.markdown("### Exploratory Data Analysis")
        st.caption("The real class / attack-type distribution charts generated during preprocessing.")
        show_chart("eda", caption=f"{DS_INFO['label']} - Exploratory Data Analysis")

        if ART is not None:
            with st.expander("Feature list"):
                st.caption(f"All {len(ART['feature_names'])} features used by the model:")
                st.code(", ".join(ART["feature_names"]))

# =====================================================================
# PAGE: ML Models
# =====================================================================
elif nav == "ML Models":
    st.markdown(f"## {DS_INFO['label']} - 12-Model Comparison")
    if R is None:
        missing_data_notice()
    else:
        ml = R["ml_results"]
        names = list(ml.keys())
        f1s = [ml[n]["f1"] * 100 for n in names]
        accs = [ml[n]["accuracy"] * 100 for n in names]
        aucs = [ml[n]["auc"] * 100 for n in names]
        order = np.argsort(f1s)
        names_o = [names[i] for i in order]
        f1s_o = [f1s[i] for i in order]

        col1, col2 = st.columns([1.3, 1])
        with col1:
            st.markdown("#### F1-Score Ranking")
            colors = [MODEL_CLR.get(n, "#0284c7") for n in names_o]
            fig = go.Figure(go.Bar(
                x=f1s_o, y=names_o, orientation='h', marker_color=colors,
                text=[f"{v:.2f}%" for v in f1s_o], textposition='outside',
                textfont=dict(color='#475569', size=10, family='JetBrains Mono'),
            ))
            fig.update_layout(**PL, height=420, xaxis=ax(title="F1 (%)", range=[min(f1s_o) - 3, 101]),
                               yaxis=dict(**AXC, autorange='reversed'))
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("#### Metric Table")
            dfm = pd.DataFrame({
                "Model": names,
                "Accuracy": [f"{ml[n]['accuracy']:.4f}" for n in names],
                "F1": [f"{ml[n]['f1']:.4f}" for n in names],
                "AUC": [f"{ml[n]['auc']:.4f}" for n in names],
            }).sort_values("F1", ascending=False)
            st.table(dfm.style.hide(axis="index"))

        st.markdown("#### Accuracy vs AUC-ROC")
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(name="Accuracy", x=names, y=accs, marker_color="#0284c7"))
        fig2.add_trace(go.Bar(name="AUC-ROC", x=names, y=aucs, marker_color="#059669"))
        fig2.update_layout(**PL, height=340, barmode='group', legend=LEG,
                            xaxis=ax(tickangle=-30), yaxis=ax(title="%", range=[min(accs + aucs) - 3, 101]))
        st.plotly_chart(fig2, use_container_width=True)

        st.info(f"**Best model: {R['best_model']}** (F1 {R['best_f1']:.4f}, AUC {R['best_auc']:.4f}, "
                f"optimal threshold {R['optimal_threshold']:.2f}) - selected to drive the Adaptive Risk Engine.")

        st.markdown("#### Confusion Matrices (all 12 models)")
        show_chart("confusion", caption=f"{DS_INFO['label']} - Confusion Matrices")

        st.markdown("#### ROC Curves (all 12 models)")
        show_chart("roc", caption=f"{DS_INFO['label']} - ROC Curves")

# =====================================================================
# PAGE: Feature Importance (SHAP)
# =====================================================================
elif nav == "Feature Importance":
    st.markdown(f"## {DS_INFO['label']} - SHAP Feature Importance")
    if R is None:
        missing_data_notice()
    else:
        st.markdown(f"#### SHAP Summary - {R['best_model']}")
        st.caption(
            "SHAP values quantify each feature's average contribution to pushing the model's "
            "output toward 'attack' versus 'benign', computed via TreeExplainer on a 500-row test sample."
        )
        show_chart("shap", caption=f"{DS_INFO['label']} - SHAP Summary Plot ({R['best_model']})")

        shap_dict = R.get("shap_top10", {})
        if shap_dict:
            s = pd.Series(shap_dict).astype(float).sort_values(ascending=True).tail(10)
            fig = go.Figure(go.Bar(
                x=s.values, y=s.index, orientation='h', marker_color='#7c3aed',
                text=[f"{v:.4f}" for v in s.values], textposition='outside',
            ))
            fig.update_layout(**PL, height=380, xaxis=ax(title="Mean |SHAP value|"), yaxis=AXC)
            st.markdown("#### Top-10 Feature Ranking")
            st.plotly_chart(fig, use_container_width=True)

# =====================================================================
# PAGE: Risk Engine
# =====================================================================
elif nav == "Risk Engine":
    st.markdown(f"## {DS_INFO['label']} - Adaptive Risk Classification Engine")
    if R is None:
        missing_data_notice()
    else:
        rk = R["risk_engine"]
        c1, c2, c3, c4 = st.columns(4)
        with c1: stat_card("Total Flows", f"{rk['n_total']:,}", "#64748b")
        with c2: stat_card("LOW (AES-256 only)", f"{rk['n_low']:,} ({rk['n_low']/rk['n_total']*100:.1f}%)", "#059669")
        with c3: stat_card("MEDIUM (+monitoring)", f"{rk['n_medium']:,} ({rk['n_medium']/rk['n_total']*100:.1f}%)", "#d97706")
        with c4: stat_card("HIGH (routed onward)", f"{rk['n_high']:,} ({rk['n_high']/rk['n_total']*100:.1f}%)", "#dc2626")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Risk Tier Distribution")
            fig = go.Figure(go.Pie(
                labels=['LOW', 'MEDIUM', 'HIGH'],
                values=[rk['n_low'], rk['n_medium'], rk['n_high']],
                hole=0.65, marker=dict(colors=['#059669', '#d97706', '#dc2626'], line=dict(color='#ffffff', width=3)),
            ))
            fig.add_annotation(text=f"<b>{rk['n_total']:,}</b><br>Flows", x=.5, y=.5, showarrow=False,
                                font=dict(family='JetBrains Mono', size=15, color='#1e293b'))
            fig.update_layout(**PL, height=340, showlegend=True, legend=dict(**LEG, orientation='h', y=-.1, x=.15))
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.markdown("#### QKD Overhead Saved")
            saved = rk['overhead_saved']
            fig2 = go.Figure(go.Indicator(
                mode="gauge+number", value=saved,
                number={'suffix': "%", 'font': {'color': '#1e293b', 'family': 'JetBrains Mono'}},
                gauge={'axis': {'range': [0, 100], 'tickcolor': '#64748b'},
                       'bar': {'color': '#059669'}, 'bgcolor': '#f1f5f9',
                       'borderwidth': 1, 'bordercolor': '#cbd5e1'},
            ))
            fig2.update_layout(**PL, height=340)
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown("#### Risk Tier Definitions")
        st.table(pd.DataFrame([
            ["LOW", "P(attack) < 0.35", "Classical AES-256 encryption"],
            ["MEDIUM", "0.35 <= P(attack) < 0.70", "AES-256 + enhanced monitoring"],
            ["HIGH", "P(attack) >= 0.70", "Routed to multiclass attack-type model"],
        ], columns=["Tier", "Probability Range", "Security Response"]).style.hide(axis="index"))

        st.markdown("#### Risk Engine + Attack-Probability Distribution")
        show_chart("risk_qkd", caption=f"{DS_INFO['label']} - Risk Engine (from your notebook)")

# =====================================================================
# PAGE: Multiclass & Response
# =====================================================================
elif nav == "Multiclass & Response":
    st.markdown(f"## {DS_INFO['label']} - Attack-Type Model & Response Recommendation", unsafe_allow_html=True)
    st.caption(
        "For every HIGH-risk flow, this secondary model predicts the specific attack family, "
        "and the Response Engine maps that prediction to a concrete action - only certain "
        "high-severity actions (e.g., isolate + key refresh) actually trigger BB84 QKD."
    )
    if R is None:
        missing_data_notice()
    else:
        mc = R.get("multiclass")
        re_ = R.get("response_engine")
        if mc:
            c1, c2, c3 = st.columns(3)
            with c1: stat_card("Accuracy", f"{mc['accuracy']:.4f}", "#0284c7")
            with c2: stat_card("Macro F1", f"{mc['macro_f1']:.4f}", "#6366f1")
            with c3: stat_card("Weighted F1", f"{mc['weighted_f1']:.4f}", "#059669")
            st.caption(f"{len(mc.get('classes', []))} attack classes (including rare-class merge into 'Rare_Attack')")
        else:
            st.warning("No multiclass results found.")

        if re_:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### Predicted Attack-Type Breakdown (HIGH-risk flows)")
                tb = pd.Series(re_["attack_type_breakdown"]).sort_values(ascending=True)
                fig = go.Figure(go.Bar(x=tb.values, y=tb.index, orientation='h', marker_color='#7c3aed'))
                fig.update_layout(**PL, height=max(300, 26 * len(tb)), xaxis=ax(title="Flows"), yaxis=AXC)
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                st.markdown("#### Recommended Response Breakdown")
                ab = pd.Series(re_["action_breakdown"]).sort_values(ascending=True)
                bar_colors = [action_color(a)[1] for a in ab.index]
                fig2 = go.Figure(go.Bar(x=ab.values, y=ab.index, orientation='h', marker_color=bar_colors))
                fig2.update_layout(**PL, height=max(300, 26 * len(ab)), xaxis=ax(title="Flows"), yaxis=AXC)
                st.plotly_chart(fig2, use_container_width=True)
        else:
            st.warning("No response-engine results found.")

        show_chart("response", caption=f"{DS_INFO['label']} - Attack-Type-Aware Response Recommendation (from your notebook)")

        if ART is not None and ART.get("response_map"):
            st.markdown("#### Full Attack-Type -> Recommended-Action Mapping")
            st.caption("Row color reflects response severity - red/orange = urgent containment, amber = active mitigation, green = passive monitoring.")
            rm = ART["response_map"]
            rows_html = ""
            for atk, act in rm.items():
                bg, fg = action_color(act)
                rows_html += (
                    f"<tr><td style='padding:.5rem .8rem;border-bottom:1px solid #e2e8f0'>{atk}</td>"
                    f"<td style='padding:.5rem .8rem;border-bottom:1px solid #e2e8f0;background:{bg};color:{fg};font-weight:600'>{act}</td></tr>"
                )
            st.markdown(f"""
            <table style='width:100%;border-collapse:collapse;font-size:.88rem;font-family:"Space Grotesk",sans-serif'>
              <thead><tr>
                <th style='text-align:left;padding:.5rem .8rem;background:#f1f5f9;border-bottom:2px solid #cbd5e1'>Attack Type</th>
                <th style='text-align:left;padding:.5rem .8rem;background:#f1f5f9;border-bottom:2px solid #cbd5e1'>Recommended Action</th>
              </tr></thead>
              <tbody>{rows_html}</tbody>
            </table>
            """, unsafe_allow_html=True)

# =====================================================================
# PAGE: QKD Simulation
# =====================================================================
elif nav == "QKD Simulation":
    st.markdown(f"## {DS_INFO['label']} - BB84 QKD Simulation")
    st.caption(
        "The final step of the pipeline: for flows whose response requires a fresh secure key, "
        "a full BB84 session is simulated to produce (or reject) a 256-bit quantum-secure key."
    )
    if R is None:
        missing_data_notice()
    else:
        st.markdown("#### Six-Step Protocol")
        steps = [("1. Alice", "Random bits\n+ bases", "#0284c7"), ("2. Encode", "Qubits sent\nto Bob", "#6366f1"),
                  ("3. Bob", "Measures with\nrandom bases", "#059669"), ("4. Sifting", "Keep matching\nbases (~50%)", "#d97706"),
                  ("5. QBER Check", "Error < 11%\n= Secure", "#ea580c"), ("6. Privacy Amp", "SHA-256 ->\n256-bit key", "#7c3aed")]
        cols = st.columns(len(steps))
        for col, (t, s, c) in zip(cols, steps):
            with col:
                st.markdown(f"""<div style='background:#f8fafc;border:1px solid {c}44;border-top:3px solid {c};
                    border-radius:10px;padding:.6rem .3rem;text-align:center;min-height:75px'>
                    <div style='font-weight:700;font-size:.75rem'>{t}</div>
                    <div style='font-size:.65rem;color:#64748b;white-space:pre-line'>{s}</div></div>""",
                            unsafe_allow_html=True)

        st.markdown("#### Live Protocol Demonstration")
        st.caption("An animated walk-through of one BB84 session (16 qubits shown for clarity - real sessions use 256).")
        bb84_animation_html()

        st.markdown("#### Eavesdropping Scenarios")
        qkd = R.get("qkd_results", {})
        if qkd:
            rows = []
            for scenario, info in qkd.items():
                if isinstance(info, dict):
                    eve = info.get("eve_prob", info.get("eavesdrop_prob"))
                    qber = info.get("qber_mean", info.get("mean_qber", info.get("qber")))
                    qstd = info.get("qber_std")
                    sec = info.get("pct_secure", info.get("secure"))
                    qber_str = f"{qber:.4f} +/- {qstd:.4f}" if (qber is not None and qstd is not None) else (f"{qber:.4f}" if qber is not None else "-")
                    sec_str = f"{sec:.1f}%" if isinstance(sec, (int, float)) else str(sec)
                    rows.append([scenario, f"{eve:.2f}" if eve is not None else "-", qber_str, sec_str])
            if rows:
                st.table(pd.DataFrame(rows, columns=["Scenario", "Eavesdrop Prob.", "Mean QBER", "Secure Sessions"]).style.hide(axis="index"))
            else:
                st.warning("BB84 results found but in an unrecognized format.")
        else:
            st.warning("No BB84 simulation results found in results file.")
        st.caption("QBER (Quantum Bit Error Rate) below 11% -> channel secure, key accepted. "
                   "QBER at or above 11% -> eavesdropping detected, key discarded (Shor-Preskill security bound).")

        if "bb84_steps" in CHARTS:
            st.markdown("#### BB84 Protocol - Step-by-Step Pipeline")
            show_chart("bb84_steps", caption=f"{DS_INFO['label']} - BB84 QKD Protocol Step-by-Step Pipeline")

# =====================================================================
# PAGE: Cross-Dataset Comparison
# =====================================================================
elif nav == "Cross-Dataset Comparison":
    st.markdown("## Cross-Dataset Comparison", unsafe_allow_html=True)
    st.caption("This page ignores the sidebar dataset switcher - it always shows all three datasets together.")

    loaded = {k: ALL_DATA[k]["results"] for k in DATASET_KEYS if ALL_DATA[k]["results"] is not None}
    if len(loaded) == 0:
        st.error("No results files found. Upload results_*.pkl for at least one dataset from the other pages first.")
    else:
        rows = []
        for k, rr in loaded.items():
            rows.append([DATASETS[k]["label"], DATASETS[k]["country"], rr["best_model"], rr["best_f1"],
                         rr["best_auc"], rr["risk_engine"]["overhead_saved"]])
        dfc = pd.DataFrame(rows, columns=["Dataset", "Country", "Best Model", "F1", "AUC", "Overhead Saved (%)"])
        st.table(dfc.style.hide(axis="index"))

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Best-Model F1 by Dataset")
            fig = go.Figure(go.Bar(x=dfc["Dataset"], y=dfc["F1"] * 100,
                                    marker_color=[DATASET_CLR[k] for k in loaded],
                                    text=[f"{v:.2f}%" for v in dfc["F1"] * 100], textposition='outside'))
            fig.update_layout(**PL, height=340, yaxis=ax(title="F1 (%)", range=[0, 105]))
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.markdown("#### QKD Overhead Saved by Dataset")
            fig2 = go.Figure(go.Bar(x=dfc["Dataset"], y=dfc["Overhead Saved (%)"],
                                     marker_color=[DATASET_CLR[k] for k in loaded],
                                     text=[f"{v:.1f}%" for v in dfc["Overhead Saved (%)"]], textposition='outside'))
            fig2.update_layout(**PL, height=340, yaxis=ax(title="Overhead Saved (%)", range=[0, 100]))
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown("#### All 12 Models - F1 Side by Side")
        fig3 = go.Figure()
        for k, rr in loaded.items():
            ml = rr["ml_results"]
            names = list(ml.keys())
            fig3.add_trace(go.Bar(name=DATASETS[k]["label"], x=names, y=[ml[n]["f1"] * 100 for n in names],
                                   marker_color=DATASET_CLR[k]))
        fig3.update_layout(**PL, height=380, barmode='group', legend=LEG,
                            xaxis=ax(tickangle=-30), yaxis=ax(title="F1 (%)"))
        st.plotly_chart(fig3, use_container_width=True)

        st.markdown("---")
        st.markdown("#### Model Performance Comparison (All 12 Models x 3 Datasets)")
        show_cross_chart("model_comparison", caption="Cross-Dataset — Model Performance Comparison")

        st.markdown("#### QKD Overhead + Risk Engine Comparison")
        show_cross_chart("qkd_overhead", caption="Cross-Dataset — QKD Overhead and Risk-Engine Comparison")

        st.markdown("#### SHAP Feature Importance Comparison")
        show_cross_chart("shap_comparison", caption="Cross-Dataset — SHAP Feature Importance Comparison")

        rc1, rc2 = st.columns(2)
        with rc1:
            st.markdown("#### Radar Chart — Best Models")
            show_cross_chart("radar", caption="Cross-Dataset — Radar Chart of Best Models")
        with rc2:
            st.markdown("#### Precision-Recall Curves — Best Models")
            show_cross_chart("precision_recall", caption="Cross-Dataset — Precision-Recall Curves")

        st.markdown("#### Confusion Matrices — Best Models")
        show_cross_chart("confusion", caption="Cross-Dataset — Confusion Matrices (Best Models)")

        st.markdown("#### Threshold Sensitivity Analysis")
        show_cross_chart("threshold", caption="Cross-Dataset — Threshold Sensitivity Analysis")

        st.markdown("### Why LightGBM Won - Dataset by Dataset")
        wcol1, wcol2, wcol3 = st.columns(3)
        with wcol1:
            st.markdown("""<div class='card' style='border-top:3px solid #0284c7;background:linear-gradient(135deg,#0284c714,#ffffff)'>
            <b>X-IIoTID (largest margin)</b>
            <ul style='font-size:.85rem;color:#334155;padding-left:1.1rem;margin-top:.5rem'>
              <li>809,249 well-populated, near-balanced rows gave leaf-wise growth room to find sharp splits</li>
              <li>Histogram binning scales efficiently at this size without overfitting a thin minority class</li>
              <li>Result: F1 0.9971 - largest winning margin of the three datasets</li>
            </ul></div>""", unsafe_allow_html=True)
        with wcol2:
            st.markdown("""<div class='card' style='border-top:3px solid #059669;background:linear-gradient(135deg,#05966914,#ffffff)'>
            <b>HQC-2026 (narrowest margin)</b>
            <ul style='font-size:.85rem;color:#334155;padding-left:1.1rem;margin-top:.5rem'>
              <li>Most severe class imbalance (24.0% attack) of the three datasets</li>
              <li>GOSS preferentially retains hard, high-gradient examples - a direct edge on imbalanced data</li>
              <li>Result: F1 0.8082 - narrowest margin over XGBoost, reflecting real data difficulty</li>
            </ul></div>""", unsafe_allow_html=True)
        with wcol3:
            st.markdown("""<div class='card' style='border-top:3px solid #d97706;background:linear-gradient(135deg,#d9770614,#ffffff)'>
            <b>CIC-UNSW-NB15 (middle margin)</b>
            <ul style='font-size:.85rem;color:#334155;padding-left:1.1rem;margin-top:.5rem'>
              <li>CICFlowMeter features include several sparse, mutually-exclusive flag/count columns</li>
              <li>EFB bundles these into fewer effective dimensions, reducing redundancy other models had to learn around</li>
              <li>Result: F1 0.9580 - second-largest margin, between the other two datasets</li>
            </ul></div>""", unsafe_allow_html=True)

# =====================================================================
# PAGE: Attack Detector
# =====================================================================
elif nav == "Attack Detector":
    st.markdown(f"## {DS_INFO['label']} - Attack Detector")
    if ART is None:
        st.error(
            f"**{DS_INFO['artifacts_file']}** not found. This page needs the extra artifacts file "
            "(raw test-set sample + trained model + multiclass model) - see notebook_export_cell.py."
        )
        up = st.file_uploader(f"Upload {DS_INFO['artifacts_file']}", type="pkl", key=f"upart_{ds_key}")
        if up is not None:
            with open(DS_INFO["artifacts_file"], "wb") as f:
                f.write(up.getbuffer())
            st.success("Uploaded - reloading...")
            st.cache_resource.clear()
            st.rerun()
    else:
        best_model = ART["best_model"]
        needs_scale = ART["needs_scale"]
        scaler = ART.get("scaler")

        st.caption(
            f"All flows on this page come from **{DS_INFO['label']}'s real held-out test set** - a 2,000-row "
            "random sample that was never used to train any model, saved directly from the notebook. "
            "None of it is synthetic."
        )

        @st.cache_data(show_spinner=False)
        def curated_samples(dskey):
            y = ART["y_true_sample"]
            Xs = ART["X_sample"]
            benign_idx = y[y == 0].index[:6].tolist()
            attack_idx_all = y[y == 1].index.tolist()

            atk_types = {}
            if ART.get("mc_model") is not None and attack_idx_all:
                try:
                    preds = ART["mc_model"].predict(Xs.loc[attack_idx_all])
                    seen = {}
                    for idx, ptype in zip(attack_idx_all, preds):
                        if ptype not in seen:
                            seen[ptype] = idx
                    attack_idx = list(seen.values())[:14]
                    atk_types = {idx: ptype for ptype, idx in seen.items() if idx in attack_idx}
                except Exception:
                    attack_idx = attack_idx_all[:14]
            else:
                attack_idx = attack_idx_all[:14]

            return benign_idx + attack_idx, atk_types

        curated_idx, atk_type_map = curated_samples(ds_key)
        curated_labels = {}
        for i in curated_idx:
            curated_labels[i] = f"Flow #{i}"

        def show_prediction(prob, pred_label, feat_row=None):
            tier = "HIGH" if prob >= 0.70 else ("MEDIUM" if prob >= 0.35 else "LOW")
            tier_meta = {
                "LOW":    {"color": "#059669", "bg": "linear-gradient(135deg,#d1fae5,#ffffff)", "icon": "🟢"},
                "MEDIUM": {"color": "#d97706", "bg": "linear-gradient(135deg,#fef3c7,#ffffff)", "icon": "🟡"},
                "HIGH":   {"color": "#dc2626", "bg": "linear-gradient(135deg,#fee2e2,#ffffff)", "icon": "🔴"},
            }[tier]
            pred_icon = "🚨" if pred_label == 1 else "🛡️"
            pred_text = "Attack" if pred_label == 1 else "Benign"

            st.markdown("### 🔎 Detection Result")
            st.markdown(f"""
            <div style='display:flex;gap:1rem;flex-wrap:wrap;margin-bottom:1rem'>
              <div style='flex:1;min-width:220px;background:linear-gradient(135deg,#eff6ff,#ffffff);border:2.5px solid #0284c7;
                   border-radius:16px;padding:1.5rem 1.2rem;text-align:center;box-shadow:0 6px 20px rgba(2,132,199,.28)'>
                <div style='font-size:2.4rem;line-height:1'>{pred_icon}</div>
                <div style='color:#0284c7;font-weight:800;font-size:1rem;text-transform:uppercase;letter-spacing:.03em;margin-top:.5rem'>Predicted Class</div>
                <div style='font-family:"JetBrains Mono",monospace;font-size:2.1rem;font-weight:800;color:#0f172a;margin-top:.25rem'>{pred_text}</div>
              </div>
              <div style='flex:1;min-width:220px;background:linear-gradient(135deg,#eef2ff,#ffffff);border:2.5px solid #6366f1;
                   border-radius:16px;padding:1.5rem 1.2rem;text-align:center;box-shadow:0 6px 20px rgba(99,102,241,.28)'>
                <div style='font-size:2.4rem;line-height:1'>📊</div>
                <div style='color:#6366f1;font-weight:800;font-size:1rem;text-transform:uppercase;letter-spacing:.03em;margin-top:.5rem'>Attack Probability</div>
                <div style='font-family:"JetBrains Mono",monospace;font-size:2.1rem;font-weight:800;color:#0f172a;margin-top:.25rem'>{prob:.4f}</div>
              </div>
              <div style='flex:1;min-width:220px;background:{tier_meta["bg"]};border:2.5px solid {tier_meta["color"]};
                   border-radius:16px;padding:1.5rem 1.2rem;text-align:center;box-shadow:0 6px 20px {tier_meta["color"]}45'>
                <div style='font-size:2.4rem;line-height:1'>{tier_meta["icon"]}</div>
                <div style='color:{tier_meta["color"]};font-weight:800;font-size:1rem;text-transform:uppercase;letter-spacing:.03em;margin-top:.5rem'>Risk Tier</div>
                <div style='font-family:"JetBrains Mono",monospace;font-size:2.1rem;font-weight:800;color:{tier_meta["color"]};margin-top:.25rem'>{tier}</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            if tier == "HIGH" and ART.get("mc_model") is not None and feat_row is not None:
                try:
                    attack_type = ART["mc_model"].predict(feat_row)[0]
                    action = ART.get("response_map", {}).get(attack_type, "Isolate host + manual review")
                    bg, fg = action_color(action)
                    bb84 = ("bb84" in action.lower() or "key" in action.lower())
                    trigger_note = " → <b>BB84 QKD triggered</b> ⚛️" if bb84 else " (no key refresh needed for this action) 🔒"
                    st.markdown(f"""<div style='background:{bg};color:{fg};border-radius:14px;padding:1.15rem 1.4rem;
                        font-size:1.12rem;font-weight:600;border:2px solid {fg}44;box-shadow:0 4px 14px rgba(0,0,0,.10);margin-top:.2rem;line-height:1.6'>
                        🚨 <b>HIGH risk</b> → routed to multiclass model. Predicted attack type: <b>{attack_type}</b>
                        <br>🛠️ Recommended response: <b>{action}</b>{trigger_note}
                        </div>""", unsafe_allow_html=True)
                except Exception as e:
                    st.warning(f"Could not run multiclass attack-type model: {e}")
            elif tier == "HIGH":
                st.warning("🚨 HIGH risk detected (multiclass model unavailable for attack-type detail)")
            elif tier == "MEDIUM":
                st.info("👁️ MEDIUM risk → AES-256 + enhanced monitoring.")
            else:
                st.info("🔒 LOW risk → AES-256 only.")

        st.markdown("##### Pick a sample flow")
        pick = st.selectbox("Flow", curated_idx, format_func=lambda i: curated_labels[i], label_visibility="collapsed")

        row = ART["X_sample"].iloc[[pick]]
        true_label = int(ART["y_true_sample"].iloc[pick])
        st.caption(f"Showing flow #{pick} (this is a real row from the test set, not synthetic)")
        with st.expander("Raw feature values for this flow"):
            st.table(row.T.rename(columns={row.index[0]: "value"}))

        run = st.button("Run", type="primary")
        if run:
            Xte = scaler.transform(row) if needs_scale else row
            prob = float(best_model.predict_proba(Xte)[:, 1][0])
            pred = int(prob >= 0.5)
            show_prediction(prob, pred, feat_row=row)

# =====================================================================
# PAGE: Charts Gallery
# =====================================================================
elif nav == "Charts Gallery":
    st.markdown(f"## {DS_INFO['label']} - Charts Gallery")
    if R is None:
        missing_data_notice()
    else:
        st.markdown("#### F1 / Accuracy / AUC - All Models")
        ml = R["ml_results"]
        names = list(ml.keys())
        fig = make_subplots(rows=1, cols=3, subplot_titles=("F1", "Accuracy", "AUC"))
        for i, metric in enumerate(["f1", "accuracy", "auc"]):
            fig.add_trace(go.Bar(x=names, y=[ml[n][metric] * 100 for n in names],
                                  marker_color=[MODEL_CLR.get(n, "#0284c7") for n in names], showlegend=False),
                          row=1, col=i + 1)
        fig.update_layout(**PL, height=380)
        fig.update_xaxes(tickangle=-40)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Risk Tier + QKD Overhead")
        rk = R["risk_engine"]
        col1, col2 = st.columns(2)
        with col1:
            fig2 = go.Figure(go.Pie(labels=['LOW', 'MEDIUM', 'HIGH'],
                                     values=[rk['n_low'], rk['n_medium'], rk['n_high']], hole=.6,
                                     marker=dict(colors=['#059669', '#d97706', '#dc2626'])))
            fig2.update_layout(**PL, height=320)
            st.plotly_chart(fig2, use_container_width=True)
        with col2:
            fig3 = go.Figure(go.Indicator(mode="gauge+number", value=rk['overhead_saved'],
                              number={'suffix': "%"}, gauge={'bar': {'color': '#059669'}}))
            fig3.update_layout(**PL, height=320)
            st.plotly_chart(fig3, use_container_width=True)

st.markdown("<hr/><div class='small' style='text-align:center'>Capstone C - East West University, Dept. of CSE</div>",
            unsafe_allow_html=True)
