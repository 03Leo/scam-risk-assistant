"""
src/model.py
============
TF-IDF + Logistic Regression training, prediction, and persistence.

Design decisions
----------------
TF-IDF  ngram_range=(1,2): Unigrams capture scam keywords ("urgent", "verify",
        "click"); bigrams capture phrase patterns ("click here", "account
        suspended", "prize winner"). Using both improves recall on
        phishing-style language.

sublinear_tf=True: Applies log(1+tf) scaling. Prevents aggressively repeated
        keywords (common in scam messages) from dominating the feature space.

class_weight='balanced': Adjusts weights inversely proportional to class
        frequencies. Critical for our 87.6%/12.4% ham/scam split — without
        this, the model would optimise for accuracy by predicting ham.

Logistic Regression (not SVM/tree): Natively outputs calibrated probabilities
        via predict_proba(), enabling uncertainty scoring in analyzer.py.
        Also fast to train, memory-efficient (<50MB RAM), and explainable.

Run as CLI
----------
    python -m src.model
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Union

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline

# ---------------------------------------------------------------------------
# Configuration — single source of truth
# ---------------------------------------------------------------------------

RANDOM_SEED: int = 42
MODEL_FILENAME: str = "scam_risk_model.joblib"

# Predictions where max(P(ham), P(scam)) < this value are flagged as uncertain
UNCERTAINTY_THRESHOLD: float = 0.70

# TF-IDF hyperparameters
TFIDF_PARAMS: dict = {
    "analyzer":      "word",
    "ngram_range":   (1, 2),       # unigrams + bigrams
    "max_features":  50_000,       # vocabulary cap — fits comfortably in 8 GB RAM
    "sublinear_tf":  True,         # log(1 + tf) term scaling
    "min_df":        2,            # ignore tokens appearing in only 1 doc
    "strip_accents": "unicode",
}

# Logistic Regression hyperparameters
LOGREG_PARAMS: dict = {
    "C":             1.0,          # regularisation (tuned via tune_hyperparameters)
    "class_weight":  "balanced",   # handles class imbalance automatically
    "max_iter":      1_000,        # ensures convergence
    "solver":        "lbfgs",
    "random_state":  RANDOM_SEED,
}

# Values evaluated during hyperparameter tuning
C_SWEEP: list[float] = [0.01, 0.1, 1.0, 10.0, 100.0]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("model")


# ---------------------------------------------------------------------------
# 1. Pipeline builder
# ---------------------------------------------------------------------------

def build_pipeline(C: float = LOGREG_PARAMS["C"]) -> Pipeline:
    """
    Construct a fresh, unfitted sklearn Pipeline.

    Steps
    -----
    'tfidf' : TfidfVectorizer with TFIDF_PARAMS.
    'clf'   : LogisticRegression with class_weight='balanced'.
    """
    tfidf = TfidfVectorizer(**TFIDF_PARAMS)
    clf = LogisticRegression(**{**LOGREG_PARAMS, "C": C})
    return Pipeline(steps=[("tfidf", tfidf), ("clf", clf)])


# ---------------------------------------------------------------------------
# 2. Training
# ---------------------------------------------------------------------------

def train(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    C: float = LOGREG_PARAMS["C"],
) -> tuple[Pipeline, dict]:
    """
    Fit the pipeline on train_df and evaluate on val_df.

    Returns
    -------
    (fitted_pipeline, val_metrics)
    """
    X_train = train_df["text"].astype(str)
    y_train = train_df["label"].astype(int)
    X_val   = val_df["text"].astype(str)
    y_val   = val_df["label"].astype(int)

    log.info("Training pipeline (C=%.3f) on %d samples ...", C, len(X_train))

    pipeline = build_pipeline(C=C)
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_val)

    val_metrics = {
        "C":         C,
        "n_train":   len(X_train),
        "n_val":     len(X_val),
        "accuracy":  round(float((y_pred == y_val).mean()), 4),
        "precision": round(float(precision_score(y_val, y_pred, zero_division=0)), 4),
        "recall":    round(float(recall_score(y_val, y_pred, zero_division=0)), 4),
        "f1":        round(float(f1_score(y_val, y_pred, zero_division=0)), 4),
    }

    log.info(
        "Val — Accuracy: %.4f | Precision: %.4f | Recall: %.4f | F1: %.4f",
        val_metrics["accuracy"],
        val_metrics["precision"],
        val_metrics["recall"],
        val_metrics["f1"],
    )
    return pipeline, val_metrics


# ---------------------------------------------------------------------------
# 3. Hyperparameter sweep
# ---------------------------------------------------------------------------

def tune_hyperparameters(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Sweep over C_SWEEP values, train once per value, evaluate on val_df.

    Returns
    -------
    DataFrame sorted by val F1 (descending).
    """
    log.info("Sweeping C values: %s", C_SWEEP)
    rows = []
    for c_val in C_SWEEP:
        _, metrics = train(train_df, val_df, C=c_val)
        rows.append(metrics)

    results = (
        pd.DataFrame(rows)
        .sort_values("f1", ascending=False)
        .reset_index(drop=True)
    )
    log.info(
        "Best C=%.3f  val F1=%.4f",
        results["C"].iloc[0], results["f1"].iloc[0],
    )
    return results


# ---------------------------------------------------------------------------
# 4. Inference with uncertainty scoring
# ---------------------------------------------------------------------------

def predict(
    pipeline: Pipeline,
    texts: Union[list[str], pd.Series],
    threshold: float = UNCERTAINTY_THRESHOLD,
) -> dict:
    """
    Run inference and return labels, calibrated probabilities, and uncertainty flags.

    Parameters
    ----------
    pipeline  : Fitted sklearn Pipeline.
    texts     : List or Series of raw message strings.
    threshold : Confidence threshold. Below this, is_uncertain is True.

    Returns
    -------
    dict with:
      'label'            : int array (0 = ham, 1 = scam)
      'probability_ham'  : float array P(ham)
      'probability_scam' : float array P(scam)
      'confidence'       : float array max(P(ham), P(scam))
      'is_uncertain'     : bool array True when confidence < threshold
    """
    texts_clean = pd.Series(texts).fillna("").astype(str)
    probas = pipeline.predict_proba(texts_clean)

    prob_ham  = probas[:, 0]
    prob_scam = probas[:, 1]
    confidence = probas.max(axis=1)

    return {
        "label":            pipeline.predict(texts_clean),
        "probability_ham":  prob_ham,
        "probability_scam": prob_scam,
        "confidence":       confidence,
        "is_uncertain":     confidence < threshold,
    }


# ---------------------------------------------------------------------------
# 5. Model persistence
# ---------------------------------------------------------------------------

def save_model(pipeline: Pipeline, models_dir: Path) -> Path:
    """Serialise fitted pipeline to disk via joblib."""
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / MODEL_FILENAME
    joblib.dump(pipeline, model_path)
    log.info("Model saved to '%s'.", model_path)
    return model_path


def load_model(models_dir: Path) -> Pipeline:
    """Load a previously saved pipeline from disk."""
    model_path = models_dir / MODEL_FILENAME
    if not model_path.exists():
        raise FileNotFoundError(
            f"No trained model found at '{model_path}'.\n"
            "Train the model first:  python -m src.model"
        )
    pipeline = joblib.load(model_path)
    log.info("Model loaded from '%s'.", model_path)
    return pipeline


# ---------------------------------------------------------------------------
# CLI entry point:  python -m src.model
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    root          = Path(__file__).resolve().parents[1]
    processed_dir = root / "data" / "processed"
    models_dir    = root / "models"

    def _load(name: str) -> pd.DataFrame:
        path = processed_dir / f"{name}.csv"
        if not path.exists():
            print(f"ERROR: '{path}' not found.")
            print("Run the data pipeline first:  python -m src.data_prep")
            sys.exit(1)
        return pd.read_csv(path)

    train_df = _load("train")
    val_df   = _load("val")

    # 1. Hyperparameter sweep
    print("\n--- Hyperparameter Sweep ---")
    sweep = tune_hyperparameters(train_df, val_df)
    print(sweep.to_string(index=False))

    # 2. Train final model with best C
    best_C = float(sweep["C"].iloc[0])
    print(f"\n--- Training final model (best C={best_C}) ---")
    pipeline, val_metrics = train(train_df, val_df, C=best_C)

    # 3. Save model
    save_model(pipeline, models_dir)

    print("\nOK Model training complete.")
    print(f"\n  {'Metric':<12} {'Val':>8}")
    print("  " + "-" * 22)
    for k in ("accuracy", "precision", "recall", "f1"):
        print(f"  {k:<12} {val_metrics[k]:>8.4f}")
    print(f"\n  Saved : models/{MODEL_FILENAME}")
    print("  Next  : python -m src.evaluation")
