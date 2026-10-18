"""
tests/test_model.py
===================
Unit tests for src/model.py.

Covers: pipeline structure, training behaviour, prediction outputs,
model persistence round-trip, and uncertainty flag logic.

Run with:
    python -m pytest tests/test_model.py -v
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from src.model import (
    MODEL_FILENAME,
    RANDOM_SEED,
    UNCERTAINTY_THRESHOLD,
    build_pipeline,
    load_model,
    predict,
    save_model,
    train,
)


# ---------------------------------------------------------------------------
# Helpers & fixtures
# ---------------------------------------------------------------------------

def _synthetic_df(n_ham: int = 100, n_scam: int = 40) -> pd.DataFrame:
    """Minimal synthetic DataFrame with realistic vocabulary contrast."""
    ham_texts  = [f"hey how are you doing today message number {i}" for i in range(n_ham)]
    scam_texts = [
        f"urgent click here verify account suspended prize winner act now {i}"
        for i in range(n_scam)
    ]
    texts  = ham_texts + scam_texts
    labels = [0] * n_ham + [1] * n_scam
    return (
        pd.DataFrame({"text": texts, "label": labels, "source": "test"})
        .sample(frac=1, random_state=RANDOM_SEED)
        .reset_index(drop=True)
    )


@pytest.fixture(scope="module")
def small_dfs():
    """Provide a small train/val split once per test module."""
    df    = _synthetic_df(n_ham=100, n_scam=40)
    split = int(0.75 * len(df))
    return df.iloc[:split].reset_index(drop=True), df.iloc[split:].reset_index(drop=True)


@pytest.fixture(scope="module")
def trained_pipeline(small_dfs):
    """Trained pipeline from small synthetic data — created once per module."""
    train_df, val_df = small_dfs
    pipeline, _ = train(train_df, val_df)
    return pipeline


# ---------------------------------------------------------------------------
# build_pipeline
# ---------------------------------------------------------------------------

class TestBuildPipeline:
    def test_has_two_steps(self):
        assert len(build_pipeline().steps) == 2

    def test_step_names(self):
        names = [name for name, _ in build_pipeline().steps]
        assert names == ["tfidf", "clf"]

    def test_class_weight_balanced(self):
        assert build_pipeline().named_steps["clf"].class_weight == "balanced"

    def test_custom_C_propagates(self):
        p = build_pipeline(C=0.05)
        assert p.named_steps["clf"].C == pytest.approx(0.05)

    def test_sublinear_tf_enabled(self):
        assert build_pipeline().named_steps["tfidf"].sublinear_tf is True

    def test_ngram_includes_bigrams(self):
        assert build_pipeline().named_steps["tfidf"].ngram_range == (1, 2)

    def test_random_state_set(self):
        assert build_pipeline().named_steps["clf"].random_state == RANDOM_SEED


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------

class TestTrain:
    def test_returns_pipeline_and_metrics(self, small_dfs):
        train_df, val_df = small_dfs
        pipeline, metrics = train(train_df, val_df)
        assert pipeline is not None
        assert isinstance(metrics, dict)

    def test_metrics_keys_present(self, small_dfs):
        train_df, val_df = small_dfs
        _, metrics = train(train_df, val_df)
        for key in ("accuracy", "precision", "recall", "f1", "n_train", "n_val"):
            assert key in metrics, f"Missing key '{key}'"

    def test_metrics_in_valid_range(self, small_dfs):
        train_df, val_df = small_dfs
        _, metrics = train(train_df, val_df)
        for key in ("accuracy", "precision", "recall", "f1"):
            assert 0.0 <= metrics[key] <= 1.0, f"{key}={metrics[key]} out of [0,1]"

    def test_row_counts_correct(self, small_dfs):
        train_df, val_df = small_dfs
        _, metrics = train(train_df, val_df)
        assert metrics["n_train"] == len(train_df)
        assert metrics["n_val"]   == len(val_df)

    def test_pipeline_can_predict_after_train(self, small_dfs):
        train_df, val_df = small_dfs
        pipeline, _ = train(train_df, val_df)
        result = pipeline.predict(["click here to win a prize"])
        assert len(result) == 1

    def test_deterministic_with_fixed_seed(self, small_dfs):
        """Training twice with the same data and seed must produce identical F1."""
        train_df, val_df = small_dfs
        _, m1 = train(train_df, val_df)
        _, m2 = train(train_df, val_df)
        assert m1["f1"] == m2["f1"]


# ---------------------------------------------------------------------------
# predict
# ---------------------------------------------------------------------------

class TestPredict:
    def test_required_output_keys(self, trained_pipeline):
        result = predict(trained_pipeline, ["hello", "click here urgent"])
        for key in ("label", "probability_ham", "probability_scam",
                    "confidence", "is_uncertain"):
            assert key in result, f"Missing key '{key}'"

    def test_output_length_matches_input(self, trained_pipeline):
        texts  = ["msg one", "msg two", "msg three"]
        result = predict(trained_pipeline, texts)
        for key in ("label", "probability_scam", "confidence", "is_uncertain"):
            assert len(result[key]) == 3, f"len mismatch for key '{key}'"

    def test_probabilities_sum_to_one(self, trained_pipeline):
        result = predict(trained_pipeline, ["test message"])
        total = result["probability_ham"][0] + result["probability_scam"][0]
        assert abs(total - 1.0) < 1e-6

    def test_probabilities_between_zero_and_one(self, trained_pipeline):
        result = predict(trained_pipeline, ["hello world", "urgent verify now"])
        assert (result["probability_scam"] >= 0).all()
        assert (result["probability_scam"] <= 1).all()

    def test_confidence_equals_max_proba(self, trained_pipeline):
        result = predict(trained_pipeline, ["some message"])
        expected = max(result["probability_ham"][0], result["probability_scam"][0])
        assert abs(result["confidence"][0] - expected) < 1e-6

    def test_uncertainty_flag_threshold_one(self, trained_pipeline):
        """With threshold=1.0 every prediction is uncertain."""
        result = predict(trained_pipeline, ["hello"], threshold=1.0)
        assert result["is_uncertain"].all()

    def test_uncertainty_flag_threshold_zero(self, trained_pipeline):
        """With threshold=0.0 no prediction is uncertain."""
        result = predict(trained_pipeline, ["hello"], threshold=0.0)
        assert not result["is_uncertain"].any()

    def test_handles_empty_string(self, trained_pipeline):
        result = predict(trained_pipeline, [""])
        assert len(result["label"]) == 1

    def test_handles_none_in_list(self, trained_pipeline):
        """None values should be coerced to empty strings, not raise."""
        result = predict(trained_pipeline, [None, "click here prize"])
        assert len(result["label"]) == 2

    def test_label_is_binary(self, trained_pipeline):
        texts  = [f"test message variant {i}" for i in range(30)]
        result = predict(trained_pipeline, texts)
        assert set(result["label"]).issubset({0, 1})

    def test_works_with_pandas_series(self, trained_pipeline):
        series = pd.Series(["hello world", "urgent click here"])
        result = predict(trained_pipeline, series)
        assert len(result["label"]) == 2


# ---------------------------------------------------------------------------
# save_model / load_model (persistence round-trip)
# ---------------------------------------------------------------------------

class TestModelPersistence:
    def test_save_creates_correct_filename(self, trained_pipeline, tmp_path):
        path = save_model(trained_pipeline, tmp_path)
        assert path.exists()
        assert path.name == MODEL_FILENAME

    def test_load_returns_callable_pipeline(self, trained_pipeline, tmp_path):
        save_model(trained_pipeline, tmp_path)
        loaded = load_model(tmp_path)
        preds  = loaded.predict(["click here to claim your prize now"])
        assert len(preds) == 1

    def test_load_raises_when_absent(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="No trained model"):
            load_model(tmp_path)

    def test_predictions_identical_after_round_trip(self, trained_pipeline, tmp_path):
        """Round-trip must produce bit-identical predictions."""
        texts = [
            "hello friend how are you",
            "urgent verify your account immediately",
            "click prize win now limited offer",
        ]
        orig_preds = trained_pipeline.predict(texts)
        save_model(trained_pipeline, tmp_path)
        loaded_preds = load_model(tmp_path).predict(texts)
        assert (orig_preds == loaded_preds).all()

    def test_save_is_idempotent(self, trained_pipeline, tmp_path):
        """Saving twice must not raise and must leave one valid model file."""
        save_model(trained_pipeline, tmp_path)
        save_model(trained_pipeline, tmp_path)
        loaded = load_model(tmp_path)
        assert loaded is not None
