"""
app.py
======
Streamlit UI for the Scam Message Risk Assistant.

Strict Separation of Concerns:
- Handles ONLY presentation, user interaction, and state management.
- Zero ML or OCR processing logic lives here — analysis is delegated to:
    * src.analyzer.analyze_message() for risk assessment & explainability.
    * src.ocr.extract_text_from_image() & src.ocr.is_tesseract_available() for OCR.

Privacy Guarantee:
- 100% Local Execution. No message text or image pixels leave this device.
"""

from __future__ import annotations

import time
from pathlib import Path

from PIL import Image
import streamlit as st

from src.analyzer import RiskAnalysisResult, analyze_message
from src.model import UNCERTAINTY_THRESHOLD, load_model
from src.ocr import (
    OcrResult,
    TesseractNotFoundError,
    extract_text_from_image,
    is_tesseract_available,
)

# ---------------------------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Scam Message Risk Assistant",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .badge-high {
        background-color: #ffebee;
        color: #c62828;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1.15rem;
        display: inline-block;
        border: 1px solid #ef9a9a;
    }
    .badge-mod {
        background-color: #fff8e1;
        color: #ef6c00;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1.15rem;
        display: inline-block;
        border: 1px solid #ffe082;
    }
    .badge-low {
        background-color: #e8f5e9;
        color: #2e7d32;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1.15rem;
        display: inline-block;
        border: 1px solid #a5d6a7;
    }
    .badge-unknown {
        background-color: #f5f5f5;
        color: #616161;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1.15rem;
        display: inline-block;
        border: 1px solid #e0e0e0;
    }
    .indicator-chip {
        display: inline-block;
        border-radius: 16px;
        padding: 4px 10px;
        margin: 3px;
        font-size: 0.88rem;
    }
    .indicator-risk {
        background-color: #fce8e6;
        border: 1px solid #fad2cf;
        color: #c5221f;
    }
    .indicator-safe {
        background-color: #e6f4ea;
        border: 1px solid #ceead6;
        color: #137333;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Cached Model Loader
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading trained ML model into memory...")
def get_model():
    """Load the trained scikit-learn pipeline once per session."""
    models_dir = Path(__file__).resolve().parent / "models"
    return load_model(models_dir)


# ---------------------------------------------------------------------------
# Demonstration Samples
# ---------------------------------------------------------------------------

SAMPLE_MESSAGES = {
    "Phishing Email (Account Suspension)": (
        "URGENT: Your bank account has been temporarily suspended due to suspicious activity. "
        "Please verify your identity and update payment details within 24 hours at "
        "https://secure-banking-alert.com/login to prevent permanent termination."
    ),
    "SMS Prize Scam (Lure Tactic)": (
        "Congratulations! You won a $1,000 Walmart Gift Card in our holiday sweepstakes. "
        "Claim your prize immediately at http://claim-gift-now.info before it expires."
    ),
    "Legitimate Meeting Notice (Safe)": (
        "Hi team, just a reminder that our project status meeting is scheduled for tomorrow at "
        "2:00 PM in Conference Room B. Please review the attached slide deck beforehand. Thanks!"
    ),
    "Ambiguous / Out-of-Distribution": (
        "quick reminder for order 98741. status pending review. please ping when available."
    ),
}


# ---------------------------------------------------------------------------
# Sidebar: Settings, Model Card & OCR Diagnostics
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("🛡️ System & Model Info")

    st.markdown(
        """
        > **Local Privacy Guarantee**  
        > All text processing, OCR, and ML inferences run 100% locally on your machine.  
        > **Zero bytes** leave this computer.
        """
    )

    st.divider()

    st.subheader("Model Performance")
    st.markdown(
        """
        - **Algorithm:** TF-IDF (1-2 ngrams) + LogReg ($C=10.0$)
        - **Test Accuracy:** `97.74%`
        - **Test F1 (Scam):** `0.9641` *(Gate: $\ge 0.85$)*
        - **ROC-AUC:** `0.9956`
        - **Calibration Error (ECE):** `0.0274`
        """
    )

    st.divider()

    st.subheader("Uncertainty Configuration")
    confidence_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.50,
        max_value=0.95,
        value=UNCERTAINTY_THRESHOLD,
        step=0.05,
        help="Predictions where max(P_ham, P_scam) falls below this threshold are flagged as uncertain.",
    )

    st.divider()

    st.subheader("Phase 2: OCR Status")
    tesseract_ok, tesseract_info = is_tesseract_available()
    if tesseract_ok:
        st.success("Tesseract OCR Engine Ready")
        st.caption(f"Path: `{tesseract_info}`")
    else:
        st.warning("Tesseract Binary Not Detected")
        st.caption("Screenshot OCR requires local Tesseract engine.")
        with st.expander("Install Instructions"):
            st.code("winget install UB-Mannheim.TesseractOCR", language="powershell")

    st.divider()
    st.caption("College Project • Scam Message Risk Assistant • 2026-2027")


# ---------------------------------------------------------------------------
# Main UI Header
# ---------------------------------------------------------------------------

st.title("🛡️ Scam Message Risk Assistant")
st.markdown(
    "Analyze incoming emails, SMS, and screenshot captures for social engineering, "
    "credential phishing, and scam patterns with transparent explainability."
)

tab_text, tab_ocr = st.tabs(["📝 Pasted Text Analysis", "📸 Screenshot OCR Analysis (Phase 2)"])


# ---------------------------------------------------------------------------
# Helper: Render Risk Assessment Report
# ---------------------------------------------------------------------------

def render_risk_report(result: RiskAnalysisResult, latency_ms: float):
    """Render the full structured risk analysis report."""
    st.divider()

    # 1. Top-Level Risk Banner
    risk_cols = st.columns([2, 1, 1])

    with risk_cols[0]:
        st.subheader("Risk Assessment")
        if result.risk_level == "HIGH":
            st.markdown('<span class="badge-high">🚨 HIGH RISK INDICATORS</span>', unsafe_allow_html=True)
        elif result.risk_level == "MODERATE":
            st.markdown('<span class="badge-mod">⚠️ MODERATE RISK SUSPICION</span>', unsafe_allow_html=True)
        elif result.risk_level == "LOW":
            st.markdown('<span class="badge-low">✅ LOW RISK DETECTED</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="badge-unknown">❓ UNKNOWN / INSUFFICIENT DATA</span>', unsafe_allow_html=True)

    with risk_cols[1]:
        st.metric(
            label="Scam Probability",
            value=f"{result.probability_scam:.1%}",
            delta=f"{(result.probability_scam - 0.5):+.1%} vs neutral",
            delta_color="inverse",
        )

    with risk_cols[2]:
        st.metric(
            label="Model Confidence",
            value=f"{result.confidence:.1%}",
            delta=f"{latency_ms:.1f} ms local",
            delta_color="off",
        )

    # 2. Uncertainty Alert Banner
    if result.is_uncertain:
        st.warning(
            "⚠️ **Prediction Uncertainty Flagged**\n\n"
            + "\n".join([f"- {reason}" for reason in result.uncertainty_reasons])
            + "\n\n*Because confidence is below threshold or vocabulary is unfamiliar, "
            "do not treat a low risk rating as a guarantee of safety.*"
        )

    # 3. Calibrated Risk Gauge
    st.write("**Calibrated Risk Distribution:**")
    st.progress(result.probability_scam)
    st.caption(
        f"0% (Safe) ←———————— {result.probability_scam:.1%} Scam Probability ————————→ 100% (High Scam Risk)"
    )

    st.markdown(f"**Recommended Action:** {result.recommended_action}")

    # 4. Explainability & Feature Attribution
    st.divider()
    st.subheader("🔎 Why was this message scored this way?")

    exp_col1, exp_col2 = st.columns(2)

    with exp_col1:
        st.markdown("##### 🚨 Top Scam Risk Indicators (Pulling score up)")
        if result.top_risk_indicators:
            for item in result.top_risk_indicators:
                st.markdown(
                    f"<span class='indicator-chip indicator-risk'><b>{item['term']}</b> "
                    f"(weight: +{item['weight']:.2f})</span>",
                    unsafe_allow_html=True,
                )
        else:
            st.info("No strong scam indicator phrases identified.")

    with exp_col2:
        st.markdown("##### 🛡️ Top Safe / Legitimate Markers (Pulling score down)")
        if result.top_safe_indicators:
            for item in result.top_safe_indicators:
                st.markdown(
                    f"<span class='indicator-chip indicator-safe'><b>{item['term']}</b> "
                    f"(weight: -{item['weight']:.2f})</span>",
                    unsafe_allow_html=True,
                )
        else:
            st.info("No strong safe marker phrases identified.")

    # 5. Domain & Heuristic Flags
    if result.heuristic_flags or result.found_urls:
        st.divider()
        st.subheader("🚩 Detected Patterns & Indicators")

        flag_cols = st.columns(2)
        with flag_cols[0]:
            if result.heuristic_flags:
                st.write("**Behavioral Signals Detected:**")
                for flag in result.heuristic_flags:
                    st.markdown(f"- ⚠️ {flag}")

        with flag_cols[1]:
            if result.found_urls:
                st.write("**Extracted Links:**")
                for url in result.found_urls:
                    st.code(url, language="text")
                st.caption("Never click unverified links from unknown senders.")


# ---------------------------------------------------------------------------
# TAB 1: Pasted Text Analysis
# ---------------------------------------------------------------------------

with tab_text:
    st.write("**Quick-load demonstration samples:**")
    cols = st.columns(len(SAMPLE_MESSAGES))

    if "input_text" not in st.session_state:
        st.session_state["input_text"] = ""

    for i, (name, sample_text) in enumerate(SAMPLE_MESSAGES.items()):
        if cols[i].button(name, key=f"sample_{i}", use_container_width=True):
            st.session_state["input_text"] = sample_text

    message_input = st.text_area(
        "**Paste message text below:**",
        value=st.session_state["input_text"],
        height=160,
        placeholder="Paste message contents here (e.g. email body, SMS text, unsolicited offer)...",
    )

    action_cols = st.columns([1, 1, 4])
    analyze_btn = action_cols[0].button("🔍 Analyze Risk", key="btn_analyze_text", type="primary", use_container_width=True)
    clear_btn = action_cols[1].button("🗑️ Clear", key="btn_clear_text", use_container_width=True)

    if clear_btn:
        st.session_state["input_text"] = ""
        st.rerun()

    if analyze_btn or message_input.strip():
        if not message_input.strip():
            st.warning("Please paste or type a message to begin analysis.")
        else:
            model = get_model()
            t0 = time.perf_counter()
            report = analyze_message(
                message_input,
                pipeline=model,
                threshold=confidence_threshold,
            )
            lat = (time.perf_counter() - t0) * 1000
            render_risk_report(report, lat)


# ---------------------------------------------------------------------------
# TAB 2: Screenshot OCR Analysis (Phase 2)
# ---------------------------------------------------------------------------

with tab_ocr:
    st.subheader("📸 Screenshot Text Extraction & Risk Assessment")
    st.markdown(
        "Upload a screenshot of a suspicious SMS, WhatsApp, Telegram, or email message. "
        "Text is extracted locally via Tesseract OCR and evaluated through the risk model."
    )

    tess_available, tess_msg = is_tesseract_available()
    if not tess_available:
        st.error(
            "**Tesseract OCR is not installed or discoverable on this machine.**\n\n"
            "To enable Phase 2 OCR:\n"
            "1. Open PowerShell and run: `winget install UB-Mannheim.TesseractOCR`\n"
            "2. Restart this application once installed."
        )

    uploaded_file = st.file_uploader(
        "Upload screenshot image (.png, .jpg, .jpeg, .webp):",
        type=["png", "jpg", "jpeg", "webp"],
        disabled=not tess_available,
    )

    if uploaded_file is not None:
        img_cols = st.columns([1, 1])
        with img_cols[0]:
            st.image(uploaded_file, caption="Uploaded Screenshot", use_container_width=True)

        with img_cols[1]:
            st.write("**OCR Processing:**")
            if st.button("🔎 Extract & Analyze Image", type="primary", use_container_width=True):
                with st.spinner("Extracting text via local OCR..."):
                    try:
                        file_bytes = uploaded_file.getvalue()
                        ocr_res: OcrResult = extract_text_from_image(file_bytes)
                    except TesseractNotFoundError as exc:
                        st.error(str(exc))
                        st.stop()
                    except Exception as exc:
                        st.error(f"OCR Error: {exc}")
                        st.stop()

                st.session_state["ocr_text"] = ocr_res.extracted_text
                st.session_state["ocr_conf"] = ocr_res.confidence
                st.session_state["ocr_warning"] = ocr_res.warning

        if "ocr_text" in st.session_state:
            st.divider()
            st.subheader("Extracted Text & Quality")

            meta_cols = st.columns(3)
            meta_cols[0].metric("OCR Confidence", f"{st.session_state['ocr_conf']:.1f}%")
            meta_cols[1].metric("Word Count", len(st.session_state['ocr_text'].split()))
            meta_cols[2].caption("Verify or edit text below if needed before running risk assessment.")

            if st.session_state.get("ocr_warning"):
                st.warning(f"⚠️ {st.session_state['ocr_warning']}")

            edited_text = st.text_area(
                "Extracted Text (Editable):",
                value=st.session_state["ocr_text"],
                height=120,
            )

            if st.button("⚡ Assess Risk of Extracted Text", type="primary"):
                if not edited_text.strip():
                    st.warning("Extracted text is empty.")
                else:
                    model = get_model()
                    t0 = time.perf_counter()
                    report = analyze_message(
                        edited_text,
                        pipeline=model,
                        threshold=confidence_threshold,
                    )
                    lat = (time.perf_counter() - t0) * 1000
                    render_risk_report(report, lat)
