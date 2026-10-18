"""
src/evaluation.py
=================
Rigorous model evaluation: classification metrics, probability calibration
analysis, and false positive / false negative inspection.

Outputs are intentionally honest and conservative:
  - Metrics are reported per class so minority (scam/phishing) performance
    is visible separately from majority (ham).
  - Calibration analysis checks whether predicted probabilities match actual
    frequencies (ECE: Expected Calibration Error).
  - FP/FN examples are surfaced explicitly to support error analysis.
  - The Milestone 2 quality gate (F1 >= 0.85 on test split) is checked.

Run as CLI
----------
    python -m src.evaluation
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.model import load_model, predict

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

N_CALIBRATION_BINS: int = 10
N_ERROR_SAMPLES:    int = 10      # FP/FN examples to surface
F1_PASS_GATE:       float = 0.85  # Milestone 2 quality gate (test split)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("evaluation")


# ---------------------------------------------------------------------------
# 1. Classification metrics
# ---------------------------------------------------------------------------

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba_scam: np.ndarray,
) -> dict:
    """Compute overall and per-class metrics."""
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    return {
        # Overall (scam = positive class)
        "accuracy":       round(float(accuracy_score(y_true, y_pred)), 4),
        "precision":      round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall":         round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1":             round(float(f1_score(y_true, y_pred, zero_division=0)), 4),

        # Class 0 — Legitimate (ham)
        "precision_ham":  round(float(precision_score(y_true, y_pred, pos_label=0, zero_division=0)), 4),
        "recall_ham":     round(float(recall_score(y_true, y_pred, pos_label=0, zero_division=0)), 4),
        "f1_ham":         round(float(f1_score(y_true, y_pred, pos_label=0, zero_division=0)), 4),

        # Class 1 — Scam / Phishing
        "precision_scam": round(float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)), 4),
        "recall_scam":    round(float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)), 4),
        "f1_scam":        round(float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)), 4),

        # Probability-based (threshold-independent)
        "roc_auc":        round(float(roc_auc_score(y_true, y_proba_scam)), 4),

        # Confusion matrix
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "confusion_matrix": cm.tolist(),
    }


# ---------------------------------------------------------------------------
# 2. Calibration analysis
# ---------------------------------------------------------------------------

def calibration_analysis(
    y_true: np.ndarray,
    y_proba_scam: np.ndarray,
    n_bins: int = N_CALIBRATION_BINS,
) -> dict:
    """Compute reliability diagram data and Expected Calibration Error (ECE)."""
    bins        = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.clip(np.digitize(y_proba_scam, bins, right=True) - 1, 0, n_bins - 1)

    mean_pred  = np.zeros(n_bins)
    frac_pos   = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins, dtype=int)

    for b in range(n_bins):
        mask = bin_indices == b
        bin_counts[b] = int(mask.sum())
        if bin_counts[b] > 0:
            mean_pred[b] = float(y_proba_scam[mask].mean())
            frac_pos[b]  = float(y_true[mask].mean())

    ece = float(np.sum(bin_counts * np.abs(mean_pred - frac_pos)) / max(len(y_true), 1))

    return {
        "bin_edges":             bins.tolist(),
        "mean_predicted_prob":   mean_pred.tolist(),
        "fraction_of_positives": frac_pos.tolist(),
        "bin_counts":            bin_counts.tolist(),
        "ece":                   round(ece, 4),
    }


# ---------------------------------------------------------------------------
# 3. False Positive / False Negative analysis
# ---------------------------------------------------------------------------

def false_analysis(
    texts: pd.Series,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba_scam: np.ndarray,
    n: int = N_ERROR_SAMPLES,
) -> dict:
    """Surface top False Positives and False Negatives."""
    texts_arr = np.array(texts.astype(str))

    def _samples(mask: np.ndarray, descending: bool) -> list[dict]:
        idxs = np.where(mask)[0]
        if len(idxs) == 0:
            return []
        order = np.argsort(y_proba_scam[idxs])
        if descending:
            order = order[::-1]
        idxs = idxs[order[:n]]
        return [
            {
                "text":       texts_arr[i][:200],
                "true_label": int(y_true[i]),
                "pred_label": int(y_pred[i]),
                "p_scam":     round(float(y_proba_scam[i]), 4),
            }
            for i in idxs
        ]

    fp_mask = (y_true == 0) & (y_pred == 1)
    fn_mask = (y_true == 1) & (y_pred == 0)

    return {
        "false_positives": _samples(fp_mask, descending=True),
        "false_negatives": _samples(fn_mask, descending=False),
        "n_fp": int(fp_mask.sum()),
        "n_fn": int(fn_mask.sum()),
    }


# ---------------------------------------------------------------------------
# 4. Full split evaluation
# ---------------------------------------------------------------------------

def evaluate_split(
    pipeline,
    df: pd.DataFrame,
    split_name: str = "test",
) -> dict:
    """Run complete evaluation suite on one data split."""
    log.info("Evaluating '%s' split (%d rows) ...", split_name, len(df))

    preds        = predict(pipeline, df["text"])
    y_true       = df["label"].astype(int).values
    y_pred       = preds["label"]
    y_proba_scam = preds["probability_scam"]
    confidence   = preds["confidence"]
    is_uncertain = preds["is_uncertain"]

    metrics     = compute_metrics(y_true, y_pred, y_proba_scam)
    calibration = calibration_analysis(y_true, y_proba_scam)
    errors      = false_analysis(df["text"], y_true, y_pred, y_proba_scam)

    uncertainty_stats = {
        "n_uncertain":     int(is_uncertain.sum()),
        "pct_uncertain":   round(float(is_uncertain.mean()) * 100, 1),
        "mean_confidence": round(float(confidence.mean()), 4),
    }

    return {
        "split":             split_name,
        "metrics":           metrics,
        "calibration":       calibration,
        "errors":            errors,
        "uncertainty_stats": uncertainty_stats,
    }


# ---------------------------------------------------------------------------
# 5. Report printer
# ---------------------------------------------------------------------------

def print_report(results: dict) -> None:
    """Print a clean, structured evaluation report."""
    split = results["split"].upper()
    m     = results["metrics"]
    cal   = results["calibration"]
    err   = results["errors"]
    unc   = results["uncertainty_stats"]

    sep       = "=" * 64
    gate_pass = m["f1"] >= F1_PASS_GATE

    print(f"\n{sep}")
    print(f"  EVALUATION REPORT  --  {split} SPLIT")
    print(sep)

    print(f"\n  {'Metric':<22} {'Value':>8}")
    print(f"  {'-'*32}")
    print(f"  {'Accuracy':<22} {m['accuracy']:>8.4f}")
    print(f"  {'ROC-AUC':<22} {m['roc_auc']:>8.4f}")

    print(f"\n  {'Class':<16} {'Precision':>10} {'Recall':>8} {'F1':>8}")
    print(f"  {'-'*46}")
    print(f"  {'Legitimate (0)':<16} {m['precision_ham']:>10.4f} {m['recall_ham']:>8.4f} {m['f1_ham']:>8.4f}")
    print(f"  {'Scam/Phish (1)':<16} {m['precision_scam']:>10.4f} {m['recall_scam']:>8.4f} {m['f1_scam']:>8.4f}")

    gate_label = "PASS" if gate_pass else "FAIL"
    print(f"\n  Milestone 2 gate  (F1 >= {F1_PASS_GATE})  :  "
          f"{gate_label}  (F1 = {m['f1']:.4f})")

    print(f"\n  Confusion Matrix")
    print(f"  {'':22} {'Pred Ham':>10} {'Pred Scam':>10}")
    print(f"  {'True Ham':<22} {m['tn']:>10} {m['fp']:>10}  <- False Positives")
    print(f"  {'True Scam':<22} {m['fn']:>10} {m['tp']:>10}  <- True Positives")
    print(f"\n  ** FN = {m['fn']} missed scams -- critical error type **")

    print(f"\n  Calibration")
    print(f"  Expected Calibration Error (ECE) : {cal['ece']:.4f}")

    print(f"\n  Uncertainty  (confidence < threshold)")
    print(f"  Uncertain predictions : {unc['n_uncertain']}  ({unc['pct_uncertain']}%)")
    print(f"  Mean confidence       : {unc['mean_confidence']:.4f}")

    n_fp_shown = min(3, len(err["false_positives"]))
    print(f"\n  False Positives  (total={err['n_fp']})  -- top {n_fp_shown} by confidence:")
    if err["false_positives"]:
        for i, ex in enumerate(err["false_positives"][:3], 1):
            snippet = ex["text"][:80].replace("\n", " ")
            print(f"  [{i}] P(scam)={ex['p_scam']:.3f}  |  {snippet!r}")
    else:
        print("  (none)")

    n_fn_shown = min(3, len(err["false_negatives"]))
    print(f"\n  False Negatives  (total={err['n_fn']})  -- top {n_fn_shown} hardest misses:")
    if err["false_negatives"]:
        for i, ex in enumerate(err["false_negatives"][:3], 1):
            snippet = ex["text"][:80].replace("\n", " ")
            print(f"  [{i}] P(scam)={ex['p_scam']:.3f}  |  {snippet!r}")
    else:
        print("  (none)")

    print(f"\n{sep}\n")


# ---------------------------------------------------------------------------
# CLI entry point:  python -m src.evaluation
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    root          = Path(__file__).resolve().parents[1]
    processed_dir = root / "data" / "processed"
    models_dir    = root / "models"

    try:
        pipeline = load_model(models_dir)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    val_path = processed_dir / "val.csv"
    if val_path.exists():
        val_results = evaluate_split(pipeline, pd.read_csv(val_path), "val")
        print_report(val_results)

    test_path = processed_dir / "test.csv"
    if not test_path.exists():
        print(f"ERROR: '{test_path}' not found. Run: python -m src.data_prep")
        sys.exit(1)

    test_results = evaluate_split(pipeline, pd.read_csv(test_path), "test")
    print_report(test_results)

    test_f1 = test_results["metrics"]["f1"]
    if test_f1 >= F1_PASS_GATE:
        print(f"OK Milestone 2 PASSED  (Test F1 = {test_f1:.4f} >= {F1_PASS_GATE})")
    else:
        print(f"Milestone 2 NOT YET PASSED  (Test F1 = {test_f1:.4f} < {F1_PASS_GATE})")
