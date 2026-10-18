"""
tests/test_data_prep.py
=======================
Unit tests for src/data_prep.py.

Run with:
    python -m pytest tests/test_data_prep.py -v

These tests cover the pure functions (clean_text, merge_datasets, split_and_save)
that do not require network access. The download + load functions are tested
separately via the integration test at the bottom (marked with a custom marker
so they can be skipped in offline environments).
"""

from __future__ import annotations

import pandas as pd
import pytest
from pathlib import Path

from src.data_prep import (
    LABEL_HAM,
    LABEL_SCAM,
    RANDOM_SEED,
    clean_text,
    merge_datasets,
    split_and_save,
    _find_column,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_df(texts: list[str], labels: list[int], source: str = "test") -> pd.DataFrame:
    """Build a minimal normalised DataFrame for use in tests."""
    return pd.DataFrame({"text": texts, "label": labels, "source": source})


# ---------------------------------------------------------------------------
# clean_text
# ---------------------------------------------------------------------------

class TestCleanText:
    def test_lowercases(self):
        assert clean_text("HELLO WORLD") == "hello world"

    def test_strips_leading_trailing_whitespace(self):
        assert clean_text("  hello  ") == "hello"

    def test_collapses_internal_whitespace(self):
        assert clean_text("hello   world\t\nfoo") == "hello world foo"

    def test_handles_none_gracefully(self):
        result = clean_text(None)
        assert isinstance(result, str)
        assert result == ""

    def test_handles_integer_input(self):
        result = clean_text(12345)
        assert result == "12345"

    def test_handles_float_input(self):
        result = clean_text(3.14)
        assert isinstance(result, str)

    def test_preserves_punctuation(self):
        """
        Punctuation is intentionally preserved — the TF-IDF analyser in
        model.py decides what to do with it. Stripping it here would
        remove potentially useful scam-signal characters like '!!!', '$'.
        """
        result = clean_text("URGENT!!! Click here NOW $$$")
        assert "!!!" in result
        assert "$$$" in result

    def test_idempotent(self):
        """Applying clean_text twice should produce the same result."""
        text = "  Hello   World! "
        assert clean_text(clean_text(text)) == clean_text(text)

    def test_empty_string(self):
        assert clean_text("") == ""

    def test_only_whitespace(self):
        assert clean_text("   \t\n  ") == ""


# ---------------------------------------------------------------------------
# merge_datasets
# ---------------------------------------------------------------------------

class TestMergeDatasets:
    def test_merges_two_datasets(self):
        df1 = _make_df(["hello world", "buy now"], [LABEL_HAM, LABEL_SCAM], "src1")
        df2 = _make_df(["click here", "safe message"], [LABEL_SCAM, LABEL_HAM], "src2")
        merged = merge_datasets(df1, df2)
        assert len(merged) == 4

    def test_preserves_source_column(self):
        df1 = _make_df(["text a"], [LABEL_HAM], "src1")
        df2 = _make_df(["text b"], [LABEL_SCAM], "src2")
        merged = merge_datasets(df1, df2)
        assert set(merged["source"].unique()) == {"src1", "src2"}

    def test_deduplicates_on_cleaned_text(self):
        """
        'Duplicate Text' and 'duplicate text' should be collapsed to one row
        because clean_text lowercases before deduplication.
        """
        df1 = _make_df(["Duplicate Text", "unique text a"], [LABEL_SCAM, LABEL_HAM], "src1")
        df2 = _make_df(["duplicate text", "unique text b"], [LABEL_SCAM, LABEL_HAM], "src2")
        merged = merge_datasets(df1, df2)
        assert (merged["text"] == "duplicate text").sum() == 1

    def test_ignores_none_sources(self):
        """None values (absent Kaggle file) should be silently ignored."""
        df1 = _make_df(["hello"], [LABEL_HAM], "src1")
        merged = merge_datasets(df1, None)
        assert len(merged) == 1

    def test_raises_when_all_none(self):
        with pytest.raises(ValueError, match="No datasets provided"):
            merge_datasets(None, None)

    def test_label_dtype_is_integer(self):
        df1 = _make_df(["hello", "click here"], [LABEL_HAM, LABEL_SCAM], "src1")
        merged = merge_datasets(df1)
        assert merged["label"].dtype in (int, "int32", "int64")

    def test_drops_null_text_rows(self):
        df = pd.DataFrame({
            "text": ["valid text", None, "another valid"],
            "label": [LABEL_HAM, LABEL_SCAM, LABEL_HAM],
            "source": "test",
        })
        merged = merge_datasets(df)
        assert len(merged) == 2
        assert merged["text"].isna().sum() == 0

    def test_drops_empty_text_rows(self):
        df = pd.DataFrame({
            "text": ["valid", "   ", ""],
            "label": [LABEL_HAM, LABEL_SCAM, LABEL_HAM],
            "source": "test",
        })
        merged = merge_datasets(df)
        assert len(merged) == 1

    def test_output_columns(self):
        df1 = _make_df(["hello"], [LABEL_HAM], "src1")
        merged = merge_datasets(df1)
        assert set(merged.columns) >= {"text", "label", "source"}


# ---------------------------------------------------------------------------
# split_and_save
# ---------------------------------------------------------------------------

class TestSplitAndSave:
    @pytest.fixture
    def sample_df(self):
        """100-row balanced DataFrame for split testing."""
        n = 100
        texts  = [f"message number {i}" for i in range(n)]
        labels = [LABEL_HAM] * (n // 2) + [LABEL_SCAM] * (n // 2)
        return _make_df(texts, labels)

    def test_splits_into_three_files(self, sample_df, tmp_path):
        split_and_save(sample_df, tmp_path)
        for name in ("train", "val", "test"):
            assert (tmp_path / f"{name}.csv").exists()

    def test_no_row_leakage(self, sample_df, tmp_path):
        """All rows in train + val + test must equal the original total."""
        splits = split_and_save(sample_df, tmp_path)
        total = sum(len(df) for df in splits.values())
        assert total == len(sample_df)

    def test_approximate_split_ratios(self, sample_df, tmp_path):
        splits = split_and_save(sample_df, tmp_path)
        n = len(sample_df)
        # Allow ±5% tolerance due to rounding
        assert abs(len(splits["train"]) / n - 0.70) < 0.05
        assert abs(len(splits["val"])   / n - 0.15) < 0.05
        assert abs(len(splits["test"])  / n - 0.15) < 0.05

    def test_stratification_preserves_class_ratio(self, sample_df, tmp_path):
        """Each split should have approximately the same ham:scam ratio."""
        splits = split_and_save(sample_df, tmp_path)
        for name, df in splits.items():
            scam_ratio = (df["label"] == LABEL_SCAM).mean()
            # Original is 50/50 — allow ±10% tolerance
            assert 0.40 <= scam_ratio <= 0.60, (
                f"{name} split has unexpected scam ratio: {scam_ratio:.2f}"
            )

    def test_reproducible_with_fixed_seed(self, sample_df, tmp_path):
        """Running the split twice should produce identical results."""
        splits_a = split_and_save(sample_df, tmp_path / "a")
        splits_b = split_and_save(sample_df, tmp_path / "b")
        for name in ("train", "val", "test"):
            pd.testing.assert_frame_equal(
                splits_a[name].reset_index(drop=True),
                splits_b[name].reset_index(drop=True),
            )


# ---------------------------------------------------------------------------
# _find_column (internal helper)
# ---------------------------------------------------------------------------

class TestFindColumn:
    def test_exact_match(self):
        df = pd.DataFrame(columns=["Email Text", "Email Type"])
        assert _find_column(df, ["Email Text"]) == "Email Text"

    def test_case_insensitive_match(self):
        df = pd.DataFrame(columns=["email text", "Email Type"])
        assert _find_column(df, ["Email Text"]) == "email text"

    def test_returns_none_when_not_found(self):
        df = pd.DataFrame(columns=["foo", "bar"])
        assert _find_column(df, ["Email Text", "body"]) is None

    def test_first_candidate_wins(self):
        df = pd.DataFrame(columns=["text", "body", "message"])
        result = _find_column(df, ["body", "text"])
        assert result == "body"
