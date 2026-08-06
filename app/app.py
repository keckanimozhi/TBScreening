"""AI4TB-Mobile — Streamlit screening prototype (Task 11), restyled
(2026-07-26) to match the visual language of the ICMR proposal's app
mockups: a blue gradient header, colored risk badges/gauge, and
color-coded per-modality/per-factor cards. Colors come from the
dataviz skill's validated default palette (references/palette.md) --
categorical blue/orange/aqua for X-ray/Cough/Symptoms (documented as
passing all-pairs CVD validation for the first three slots), the fixed
status palette for risk bands (good/warning/critical), and the
blue<->red diverging pair for the contributing-symptoms polarity
(increases vs. decreases predicted risk).

Run from the `TB screening/` directory with:
    .venv/Scripts/streamlit.exe run app/app.py

A health worker enters symptoms and optionally uploads a chest X-ray
and/or a cough recording. Each available modality is scored by its own
trained model, then combined by the fusion model into a single risk
score. Missing modalities are imputed with their training-set mean (see
src/fusion/predict.py) and clearly marked as such in the UI.

This is a research-prototype UI over TRL-3/4 models trained on very
limited data (100 X-rays, 200 cough clips from 7 real subjects, and a
domain-knowledge-labeled symptom rule applied to what looks like
synthetic data) -- see docs/PROJECT_CHECKLIST.md for the full data
caveats. It is NOT a clinical decision tool.
"""

import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import streamlit as st
from PIL import Image

from src.common.config import MODELS_DIR
from src.dashboard.mock_data import generate_mock_events, monthly_counts
from src.fusion.components import (
    load_cough_model,
    load_xray_model,
    predict_cough,
    predict_xray,
)
from src.fusion.predict import load_fusion_model, predict_fused, risk_band
from src.symptom.labeling import SYMPTOM_COLUMNS
from src.xray.gradcam import gradcam_overlay_for_image
from src.xray.model import build_model as build_xray_model

# --- Palette (dataviz skill's validated default instance) -----------------
COLOR_XRAY = "#2a78d6"      # categorical slot 1 (blue)
COLOR_COUGH = "#eb6834"     # categorical slot 2 (orange)
COLOR_SYMPTOM = "#1baf7a"   # categorical slot 3 (aqua)
COLOR_GOOD = "#0ca30c"      # status: low risk
COLOR_WARNING = "#fab219"   # status: medium risk
COLOR_CRITICAL = "#d03b3b"  # status: high risk
COLOR_DIVERGING_UP = "#d03b3b"    # increases predicted risk (red pole)
COLOR_DIVERGING_DOWN = "#2a78d6"  # decreases predicted risk (blue pole)

BAND_COLOR = {"Low": COLOR_GOOD, "Medium": COLOR_WARNING, "High": COLOR_CRITICAL}
BAND_TEXT_COLOR = {"Low": "#ffffff", "Medium": "#3a2a00", "High": "#ffffff"}


def inject_css():
    st.markdown(
        """
        <style>
        .aitb-header {
            background: linear-gradient(135deg, #2a78d6 0%, #184f95 100%);
            color: #ffffff;
            padding: 22px 26px;
            border-radius: 14px;
            margin-bottom: 22px;
        }
        .aitb-header h1 { margin: 0; font-size: 25px; }
        .aitb-header p { margin: 6px 0 0 0; opacity: 0.92; font-size: 14px; }

        div[data-testid="stVerticalBlockBorderWrapper"]:has(.st-key-card-symptoms),
        .st-key-card-symptoms {
            border-left: 5px solid #1baf7a !important;
            border-radius: 12px !important;
        }
        div[data-testid="stVerticalBlockBorderWrapper"]:has(.st-key-card-xray),
        .st-key-card-xray {
            border-left: 5px solid #2a78d6 !important;
            border-radius: 12px !important;
        }
        div[data-testid="stVerticalBlockBorderWrapper"]:has(.st-key-card-cough),
        .st-key-card-cough {
            border-left: 5px solid #eb6834 !important;
            border-radius: 12px !important;
        }

        .risk-badge {
            display: inline-block;
            padding: 5px 16px;
            border-radius: 999px;
            font-weight: 700;
            font-size: 13px;
            letter-spacing: 0.02em;
        }

        .metric-tile {
            border-radius: 12px;
            padding: 14px 12px;
            color: #ffffff;
            text-align: center;
        }
        .metric-tile .tile-value { font-size: 24px; font-weight: 800; line-height: 1.1; }
        .metric-tile .tile-label { font-size: 12px; opacity: 0.92; margin-top: 2px; }
        .metric-tile.muted { background: #898781 !important; }

        .gauge-wrap { display: flex; align-items: center; justify-content: center; padding: 6px 0 2px 0; }
        .gauge {
            width: 152px; height: 152px; border-radius: 50%;
            display: flex; align-items: center; justify-content: center;
        }
        .gauge-inner {
            width: 116px; height: 116px; border-radius: 50%; background: #ffffff;
            display: flex; flex-direction: column; align-items: center; justify-content: center;
            box-shadow: inset 0 0 0 1px rgba(11,11,11,0.06);
        }
        .gauge-inner .gauge-pct { font-size: 27px; font-weight: 800; color: #0b0b0b; }
        .gauge-inner .gauge-band { font-size: 12px; font-weight: 700; margin-top: 2px; }

        .factor-row { display: flex; align-items: center; margin-bottom: 9px; }
        .factor-label { width: 320px; font-size: 13px; color: #0b0b0b; padding-right: 10px; }
        .factor-track { flex: 1; background: #e1e0d9; border-radius: 6px; height: 10px; overflow: hidden; }
        .factor-fill { height: 100%; border-radius: 6px; }
        .factor-weight { width: 56px; text-align: right; font-size: 12px; color: #52514e; padding-left: 8px; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header(subtitle: str):
    st.markdown(
        f"""
        <div class="aitb-header">
            <h1>🫁 AI4TB-Mobile</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def metric_tile(col, value_html: str, label: str, color: str, text_color: str = "#ffffff"):
    col.markdown(
        f"""
        <div class="metric-tile" style="background:{color}; color:{text_color};">
            <div class="tile-value">{value_html}</div>
            <div class="tile-label">{label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_gauge(pct: float, band: str):
    color = BAND_COLOR[band]
    st.markdown(
        f"""
        <div class="gauge-wrap">
            <div class="gauge" style="background: conic-gradient({color} {pct * 100:.0f}%, #e1e0d9 0);">
                <div class="gauge-inner">
                    <div class="gauge-pct">{pct:.0%}</div>
                    <div class="gauge-band" style="color:{color};">{band} risk</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def get_models():
    return {
        "xray": load_xray_model(),
        "cough": load_cough_model(),
        "symptom": joblib.load(MODELS_DIR / "symptom" / "logreg.pkl"),
        "fusion": load_fusion_model(),
    }


def render_screening_page():
    render_header("On-device-style multimodal TB screening — research prototype")

    models = get_models()

    col_symptoms, col_media = st.columns(2)

    with col_symptoms:
        with st.container(border=True, key="card-symptoms"):
            st.markdown("#### 📋 Symptoms")
            symptom_values = {}
            for col in SYMPTOM_COLUMNS:
                symptom_values[col] = st.checkbox(col.capitalize(), value=False, key=f"sym_{col}")

    with col_media:
        with st.container(border=True, key="card-xray"):
            st.markdown("#### 🩻 Chest X-ray (optional)")
            xray_file = st.file_uploader("Upload a chest X-ray", type=["png", "jpg", "jpeg"])
            if xray_file is not None:
                st.image(xray_file, caption="Uploaded X-ray", width=220)

        with st.container(border=True, key="card-cough"):
            st.markdown("#### 🎤 Cough recording (optional)")
            cough_file = st.file_uploader("Upload a cough recording (.wav)", type=["wav"])
            if cough_file is not None:
                st.audio(cough_file)

    if st.button("Run screening", type="primary"):
        symptom_row = [[int(symptom_values[col]) for col in SYMPTOM_COLUMNS]]
        p_symptom = models["symptom"].predict_proba(symptom_row)[0, 1]

        p_xray, xray_path, gradcam_img = None, None, None
        if xray_file is not None:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                tmp.write(xray_file.getvalue())
                xray_path = tmp.name
            p_xray = predict_xray(models["xray"], xray_path)
            gradcam_img = gradcam_overlay_for_image(models["xray"], xray_path)

        p_cough = None
        if cough_file is not None:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(cough_file.getvalue())
                cough_path = tmp.name
            p_cough = predict_cough(models["cough"], cough_path)

        fused = predict_fused(models["fusion"], p_xray=p_xray, p_cough=p_cough, p_symptom=p_symptom)
        band = risk_band(fused["prob"])

        st.divider()
        st.markdown("### Result")

        m1, m2, m3 = st.columns(3)
        metric_tile(m1, f"{p_xray:.0%}" if p_xray is not None else "—", "X-ray", COLOR_XRAY)
        metric_tile(m2, f"{p_cough:.0%}" if p_cough is not None else "—", "Cough", COLOR_COUGH)
        metric_tile(m3, f"{p_symptom:.0%}", "Symptoms", COLOR_SYMPTOM)

        gauge_col, info_col = st.columns([1, 2])
        with gauge_col:
            render_gauge(fused["prob"], band)
        with info_col:
            st.markdown(
                f'<span class="risk-badge" style="background:{BAND_COLOR[band]}; color:{BAND_TEXT_COLOR[band]};">{band.upper()} RISK</span>',
                unsafe_allow_html=True,
            )
            if not fused["used"]["p_xray"]:
                st.caption("X-ray not provided — fused score used the training-set average for this input.")
            if not fused["used"]["p_cough"]:
                st.caption("Cough not provided — fused score used the training-set average for this input.")

            if band == "High":
                st.error("Recommendation: refer for confirmatory testing (GeneXpert/TrueNat/sputum microscopy).")
            elif band == "Medium":
                st.info("Recommendation: clinical review recommended; consider confirmatory testing.")
            else:
                st.success("Recommendation: low presumptive risk from available inputs; routine follow-up.")

        if gradcam_img is not None:
            st.markdown("#### X-ray explainability (Grad-CAM)")
            st.image(gradcam_img, caption="Red/yellow = regions that most influenced the X-ray prediction", width=280)

        checked = [col for col, val in symptom_values.items() if val]
        if checked:
            coefs = dict(zip(SYMPTOM_COLUMNS, models["symptom"].coef_[0]))
            ranked = sorted(checked, key=lambda c: -abs(coefs[c]))
            max_abs = max(abs(coefs[c]) for c in ranked) or 1.0

            st.markdown("#### Contributing symptoms (by learned weight)")
            rows = []
            for col in ranked:
                w = coefs[col]
                width_pct = abs(w) / max_abs * 100
                color = COLOR_DIVERGING_UP if w >= 0 else COLOR_DIVERGING_DOWN
                rows.append(
                    f"""
                    <div class="factor-row">
                        <div class="factor-label">{col.capitalize()}</div>
                        <div class="factor-track"><div class="factor-fill" style="width:{width_pct:.0f}%; background:{color};"></div></div>
                        <div class="factor-weight">{w:+.2f}</div>
                    </div>
                    """
                )
            st.markdown("".join(rows), unsafe_allow_html=True)
            st.caption("Red = increases predicted risk · Blue = decreases predicted risk")


def render_dashboard_page():
    render_header("Surveillance dashboard — simulated data")
    st.warning(
        "**All data on this page is simulated (Task 12 mock data), not real "
        "surveillance data.** No real geolocated screening events exist yet — "
        "this demonstrates the dashboard the proposal describes ahead of real "
        "field deployment. See src/dashboard/mock_data.py."
    )

    events = generate_mock_events()

    k1, k2, k3, k4 = st.columns(4)
    metric_tile(k1, str(len(events)), "Total screened (mock)", "#2a78d6")
    metric_tile(k2, str(int((events["risk_band"] == "High").sum())), "High risk", COLOR_CRITICAL)
    metric_tile(k3, str(int((events["risk_band"] == "Medium").sum())), "Medium risk", COLOR_WARNING, "#3a2a00")
    metric_tile(k4, str(int((events["risk_band"] == "Low").sum())), "Low risk", COLOR_GOOD)

    st.markdown("#### 🗺️ TB risk hotspot map — Perundurai / Erode region")
    st.map(events, latitude="lat", longitude="lon", color="color", size=120)
    st.caption("Red = high risk, orange = medium, green = low (mock risk bands).")

    st.markdown("#### 📈 Monthly screening volume (mock, last 12 months)")
    counts = monthly_counts(events).set_index("month")
    st.bar_chart(counts["cases"], color=COLOR_XRAY)


def main():
    st.set_page_config(page_title="AI4TB-Mobile", page_icon="🫁", layout="wide")
    inject_css()
    st.sidebar.title("🫁 AI4TB-Mobile")
    page = st.sidebar.radio("Navigate", ["Screening", "Surveillance Dashboard"])

    if page == "Screening":
        render_screening_page()
    else:
        render_dashboard_page()


if __name__ == "__main__":
    main()
