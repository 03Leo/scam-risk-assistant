"""
src/analyzer.py
===============
Risk analysis, explainability engine, uncertainty scoring, and heuristic signals.

Strict Separation of Concerns:
- This module encapsulates all risk interpretation, n-gram contribution extraction,
  and uncertainty detection.
- app.py strictly consumes the output of analyze_message() to render the UI.

Core Engineering Principles:
1. Honest Risk Framing: Predictions are reported as calibrated risk levels
   (Low, Moderate, High) with explicit uncertainty tags rather than binary verdicts.
2. Local Explainability: Extracts the specific n-grams in the input message that
   drove the Logistic Regression model toward scam or safe classifications.
3. Out-of-Distribution (OOD) / Low-Information Detection: Flags messages that
   contain little to no recognizable vocabulary or produce low-confidence probabilities.
"""

from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np
from sklearn.pipeline import Pipeline

from src.model import UNCERTAINTY_THRESHOLD, load_model, predict

log = logging.getLogger("analyzer")

# ---------------------------------------------------------------------------
# Heuristic pattern definitions (Supplementary Domain Flags)
# ---------------------------------------------------------------------------

URGENCY_PATTERNS = [
    r"\burgent\b",
    r"\bimmediat(e|ely)\b",
    r"\bact now\b",
    r"\baction required\b",
    r"\bwithin 24 hours\b",
    r"\bsuspended\b",
    r"\bexpir(e|ing|ed)\b",
    r"\blast warning\b",
    r"\bfinal notice\b",
    r"\bdeadline\b",
]

FINANCIAL_CREDENTIAL_PATTERNS = [
    r"\bpassword\b",
    r"\bpin\b",
    r"\botp\b",
    r"\bssn\b",
    r"\bsocial security\b",
    r"\bverify your (account|identity|card|bank)\b",
    r"\bbank account\b",
    r"\bcredit card\b",
    r"\bupdate payment\b",
    r"\bbilling issue\b",
    r"\blogin to\b",
]

LURE_PATTERNS = [
    r"\bwinner\b",
    r"\bwon\b",
    r"\bprize\b",
    r"\bclaim\b",
    r"\blottery\b",
    r"\bfree gift\b",
    r"\bgift card\b",
    r"\binheritance\b",
    r"\bmillion dollars\b",
    r"\bbitcoin\b",
    r"\bcrypto\b",
    r"\bcongratulations\b",
]

URL_PATTERN = re.compile(
    r"(https?://\S+|www\.\S+|\b[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(?:/[^\s]*)?)",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Data Contract
# ---------------------------------------------------------------------------

@dataclass
class RiskAnalysisResult:
    text: str
    risk_level: str               # "LOW", "MODERATE", "HIGH"
    risk_color: str               # "green", "orange", "red"
    probability_scam: float       # P(scam) in [0.0, 1.0]
    probability_ham: float        # P(ham) in [0.0, 1.0]
    confidence: float             # max(P_scam, P_ham) in [0.5, 1.0]
    is_uncertain: bool            # True if confidence < threshold or OOD
    uncertainty_reasons: list[str]# Explanations for uncertainty
    top_risk_indicators: list[dict] # Contributing scam tokens [{'term': ..., 'weight': ...}]
    top_safe_indicators: list[dict] # Contributing safe tokens
    heuristic_flags: list[str]    # Triggered rule-based patterns
    found_urls: list[str]         # Extracted links/URLs
    is_out_of_distribution: bool  # True if too few known vocabulary terms
    vocabulary_match_count: int   # Number of in-vocabulary tokens
    recommended_action: str       # Actionable, safe advice for the user

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Explainability helper: Feature Contribution Extraction
# ---------------------------------------------------------------------------

def extract_feature_contributions(
    pipeline: Pipeline,
    text: str,
    top_k: int = 5,
) -> tuple[list[dict], list[dict], int]:
    """
    Extract the most influential tokens in *text* using the linear model coefficients.

    For Logistic Regression with TF-IDF:
    Contribution_i = tfidf_value_i * coefficient_i
    - Positive contribution => pushes prediction toward SCAM (Risk Indicator).
    - Negative contribution => pushes prediction toward HAM (Safe Indicator).

    Returns
    -------
    (risk_indicators, safe_indicators, in_vocabulary_count)
    """
    tfidf = pipeline.named_steps["tfidf"]
    clf = pipeline.named_steps["clf"]

    # Transform the single text string
    X_vec = tfidf.transform([text])
    feature_names = tfidf.get_feature_names_out()
    coef = clf.coef_[0]  # shape (n_features,)

    non_zero_indices = X_vec.nonzero()[1]
    in_vocab_count = len(non_zero_indices)

    if in_vocab_count == 0:
        return [], [], 0

    contributions = []
    for idx in non_zero_indices:
        val = X_vec[0, idx]
        weight = float(val * coef[idx])
        contributions.append({
            "term": feature_names[idx],
            "weight": round(weight, 4),
        })

    # Sort descending for scam indicators (most positive weights)
    scam_sorted = sorted(contributions, key=lambda x: x["weight"], reverse=True)
    risk_indicators = [item for item in scam_sorted if item["weight"] > 0][:top_k]

    # Sort ascending for safe indicators (most negative weights)
    safe_sorted = sorted(contributions, key=lambda x: x["weight"])
    safe_indicators = [
        {"term": item["term"], "weight": round(abs(item["weight"]), 4)}
        for item in safe_sorted if item["weight"] < 0
    ][:top_k]

    return risk_indicators, safe_indicators, in_vocab_count


# ---------------------------------------------------------------------------
# Heuristic pattern matching
# ---------------------------------------------------------------------------

def scan_heuristics(text: str) -> tuple[list[str], list[str]]:
    """Scan text for high-risk pattern markers and URLs."""
    flags = []
    text_lower = text.lower()

    if any(re.search(pat, text_lower) for pat in URGENCY_PATTERNS):
        flags.append("Urgency / Pressure tactic detected")

    if any(re.search(pat, text_lower) for pat in FINANCIAL_CREDENTIAL_PATTERNS):
        flags.append("Requests sensitive credentials or financial information")

    if any(re.search(pat, text_lower) for pat in LURE_PATTERNS):
        flags.append("Unrealistic prize, lottery, or incentive lure detected")

    raw_urls = URL_PATTERN.findall(text)
    # Filter out false positives like numbers with decimals
    urls = [u for u in raw_urls if not re.match(r"^\d+(\.\d+)+$", u)]
    if urls:
        flags.append(f"Contains {len(urls)} link(s) or web address(es)")

    return flags, urls


# ---------------------------------------------------------------------------
# Core Analysis Engine
# ---------------------------------------------------------------------------

def analyze_message(
    text: str,
    pipeline: Optional[Pipeline] = None,
    models_dir: Optional[Path] = None,
    threshold: float = UNCERTAINTY_THRESHOLD,
) -> RiskAnalysisResult:
    """
    Complete risk assessment of a single message.

    Parameters
    ----------
    text       : Raw input string to analyze.
    pipeline   : Pre-loaded sklearn Pipeline. If None, loaded from models_dir.
    models_dir : Directory containing scam_risk_model.joblib.
    threshold  : Minimum confidence threshold (default 0.70). Below this,
                 is_uncertain is set to True.

    Returns
    -------
    RiskAnalysisResult dataclass with calibrated risk indicators.
    """
    clean_str = text.strip() if text else ""

    # 1. Edge Case: Empty or whitespace-only input
    if not clean_str:
        return RiskAnalysisResult(
            text=text,
            risk_level="UNKNOWN",
            risk_color="gray",
            probability_scam=0.0,
            probability_ham=1.0,
            confidence=1.0,
            is_uncertain=True,
            uncertainty_reasons=["Empty message provided."],
            top_risk_indicators=[],
            top_safe_indicators=[],
            heuristic_flags=[],
            found_urls=[],
            is_out_of_distribution=True,
            vocabulary_match_count=0,
            recommended_action="Please provide message text to evaluate.",
        )

    # 2. Load model if not provided
    if pipeline is None:
        if models_dir is None:
            models_dir = Path(__file__).resolve().parents[1] / "models"
        pipeline = load_model(models_dir)

    # 3. Model inference
    preds = predict(pipeline, [clean_str], threshold=threshold)
    p_scam = float(preds["probability_scam"][0])
    p_ham = float(preds["probability_ham"][0])
    confidence = float(preds["confidence"][0])
    is_uncertain_pred = bool(preds["is_uncertain"][0])

    # 4. Feature attribution
    risk_indicators, safe_indicators, in_vocab_count = extract_feature_contributions(
        pipeline, clean_str, top_k=5
    )

    # 5. Out-of-Distribution / Vocabulary check
    words = clean_str.split()
    is_ood = False
    uncertainty_reasons = []

    if len(words) >= 4 and in_vocab_count == 0:
        is_ood = True
        uncertainty_reasons.append(
            "None of the words match the model's vocabulary (unusual language or gibberish)."
        )
    elif in_vocab_count < 2 and len(words) > 3:
        is_ood = True
        uncertainty_reasons.append(
            "Very few recognizable terms found. Assessment reliability is reduced."
        )

    if is_uncertain_pred:
        uncertainty_reasons.append(
            f"Model prediction confidence ({confidence:.1%}) is below the {threshold:.0%} certainty threshold."
        )

    is_uncertain = is_uncertain_pred or is_ood

    # 6. Heuristic scan
    heuristic_flags, found_urls = scan_heuristics(clean_str)

    # 7. Risk Tier Determination
    # Calibrated categorization:
    # High: P(scam) >= 0.70
    # Moderate: 0.35 <= P(scam) < 0.70
    # Low: P(scam) < 0.35
    if p_scam >= 0.70:
        risk_level = "HIGH"
        risk_color = "red"
        recommended_action = (
            "Caution: High risk indicators detected. Do not click any links, "
            "do not download attachments, and never provide credentials or payments. "
            "Verify the sender through an independent, official channel."
        )
    elif p_scam >= 0.35:
        risk_level = "MODERATE"
        risk_color = "orange"
        recommended_action = (
            "Suspicious indicators detected. Exercise caution before responding or "
            "clicking links. Check the sender's actual address or phone number."
        )
    else:
        risk_level = "LOW"
        risk_color = "green"
        recommended_action = (
            "No strong scam indicators identified. However, always remain vigilant "
            "with unexpected requests or attachments."
        )

    if is_uncertain:
        recommended_action += (
            " Note: The model flagged this analysis as uncertain; do not treat "
            "a low risk rating as a guarantee of safety."
        )

    return RiskAnalysisResult(
        text=text,
        risk_level=risk_level,
        risk_color=risk_color,
        probability_scam=round(p_scam, 4),
        probability_ham=round(p_ham, 4),
        confidence=round(confidence, 4),
        is_uncertain=is_uncertain,
        uncertainty_reasons=uncertainty_reasons,
        top_risk_indicators=risk_indicators,
        top_safe_indicators=safe_indicators,
        heuristic_flags=heuristic_flags,
        found_urls=found_urls,
        is_out_of_distribution=is_ood,
        vocabulary_match_count=in_vocab_count,
        recommended_action=recommended_action,
    )
