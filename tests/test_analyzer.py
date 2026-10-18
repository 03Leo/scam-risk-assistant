"""
tests/test_analyzer.py
======================
Unit tests for src/analyzer.py.

Covers:
- Explainability (feature contribution extraction)
- Heuristic pattern scanning
- Full message risk analysis across High, Moderate, Low, and Edge cases
- Uncertainty and Out-of-Distribution (OOD) flag correctness
"""

from __future__ import annotations

from pathlib import Path
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.analyzer import (
    RiskAnalysisResult,
    analyze_message,
    extract_feature_contributions,
    scan_heuristics,
)
from src.model import RANDOM_SEED


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def mock_pipeline():
    """A fitted pipeline with min_df=1 so mock terms are preserved in vocab."""
    train_texts = [
        "hey are we meeting tomorrow for lunch",
        "thanks for sending the report see you tomorrow for lunch",
        "please find the project notes attached for our meeting",
        "urgent verify your bank account now or it will be suspended",
        "congratulations you won a free gift card click here to claim",
        "urgent final notice update payment details immediately or account suspended",
    ]
    train_labels = [0, 0, 0, 1, 1, 1]

    tfidf = TfidfVectorizer(
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=1,
    )
    clf = LogisticRegression(class_weight="balanced", random_state=RANDOM_SEED)
    pipeline = Pipeline(steps=[("tfidf", tfidf), ("clf", clf)])
    pipeline.fit(train_texts, train_labels)
    return pipeline


# ---------------------------------------------------------------------------
# Heuristic Scanner Tests
# ---------------------------------------------------------------------------

class TestScanHeuristics:
    def test_detects_urgency_patterns(self):
        flags, _ = scan_heuristics("This is an urgent request, take action immediately.")
        assert any("Urgency" in f for f in flags)

    def test_detects_financial_patterns(self):
        flags, _ = scan_heuristics("Please verify your bank account and enter your password.")
        assert any("sensitive credentials" in f for f in flags)

    def test_detects_lure_patterns(self):
        flags, _ = scan_heuristics("Congratulations you won a lottery prize!")
        assert any("lure" in f for f in flags)

    def test_extracts_urls(self):
        flags, urls = scan_heuristics("Go to https://secure-login.suspicious.com/verify to proceed.")
        assert len(urls) >= 1
        assert "secure-login.suspicious.com" in urls[0]
        assert any("link" in f for f in flags)

    def test_clean_text_no_flags(self):
        flags, urls = scan_heuristics("Let us grab coffee on Friday.")
        assert len(flags) == 0
        assert len(urls) == 0


# ---------------------------------------------------------------------------
# Feature Contribution Extraction Tests
# ---------------------------------------------------------------------------

class TestFeatureContributions:
    def test_identifies_risk_indicators(self, mock_pipeline):
        text = "urgent verify bank account suspended"
        risk_ind, _, in_vocab = extract_feature_contributions(mock_pipeline, text, top_k=3)
        assert in_vocab > 0
        terms = [item["term"] for item in risk_ind]
        assert any(t in terms for t in ["urgent", "verify", "account", "suspended", "bank"])

    def test_identifies_safe_indicators(self, mock_pipeline):
        text = "hey meeting tomorrow for lunch thanks"
        _, safe_ind, in_vocab = extract_feature_contributions(mock_pipeline, text, top_k=3)
        assert in_vocab > 0
        terms = [item["term"] for item in safe_ind]
        assert any(t in terms for t in ["meeting", "lunch", "thanks", "tomorrow"])

    def test_handles_out_of_vocabulary(self, mock_pipeline):
        text = "xyzabc1234 qwertynop"
        risk_ind, safe_ind, in_vocab = extract_feature_contributions(mock_pipeline, text)
        assert in_vocab == 0
        assert len(risk_ind) == 0
        assert len(safe_ind) == 0


# ---------------------------------------------------------------------------
# Message Risk Analysis Tests
# ---------------------------------------------------------------------------

class TestAnalyzeMessage:
    def test_handles_empty_message(self, mock_pipeline):
        res = analyze_message("", pipeline=mock_pipeline)
        assert res.risk_level == "UNKNOWN"
        assert res.is_uncertain is True
        assert len(res.uncertainty_reasons) > 0

    def test_handles_whitespace_only(self, mock_pipeline):
        res = analyze_message("   \n\t  ", pipeline=mock_pipeline)
        assert res.risk_level == "UNKNOWN"
        assert res.is_uncertain is True

    def test_high_risk_scam_message(self, mock_pipeline):
        msg = "urgent verify your bank account now or it will be suspended immediately"
        res = analyze_message(msg, pipeline=mock_pipeline)
        assert res.risk_level in ("HIGH", "MODERATE")
        assert res.probability_scam >= 0.50
        assert len(res.top_risk_indicators) > 0
        assert len(res.heuristic_flags) > 0

    def test_low_risk_ham_message(self, mock_pipeline):
        msg = "hey are we meeting tomorrow for lunch thanks report"
        res = analyze_message(msg, pipeline=mock_pipeline)
        assert res.risk_level in ("LOW", "MODERATE")
        assert res.probability_scam < 0.50

    def test_out_of_distribution_flags_uncertainty(self, mock_pipeline):
        msg = "zxywvutsrqponmlkjihgfedcba foo bar baz"
        res = analyze_message(msg, pipeline=mock_pipeline)
        assert res.is_out_of_distribution is True
        assert res.is_uncertain is True
        assert any("vocabulary" in r.lower() for r in res.uncertainty_reasons)

    def test_result_structure_and_types(self, mock_pipeline):
        msg = "simple test message"
        res = analyze_message(msg, pipeline=mock_pipeline)
        assert isinstance(res, RiskAnalysisResult)
        d = res.to_dict()
        assert "risk_level" in d
        assert "probability_scam" in d
        assert "confidence" in d
        assert "recommended_action" in d
