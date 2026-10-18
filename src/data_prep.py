"""
src/data_prep.py
================
Dataset downloading, loading, cleaning, merging, and splitting pipeline.

All functions are pure Python — no Streamlit, no model training.
Can be imported in tests or run as a CLI script:
    python -m src.data_prep

Dataset strategy
----------------
AUTO  (1): UCI SMS Spam Collection
           Automatically downloaded as a ZIP from the UCI ML Repository.
           Labels: "ham" → 0  |  "spam" → 1
           ⚠ Caveat: UCI "spam" includes marketing texts, not only scams.
             The 'source' column is preserved so per-domain analysis is possible.

MANUAL(2): CLAIR Phishing Emails (Kaggle — no API key needed)
           Browser URL : https://www.kaggle.com/datasets/subhajournal/phishingemails
           Step 1 — click Download on that page (login required, account is free).
           Step 2 — extract and place the CSV at:
                    <project_root>/data/raw/phishing_email.csv
           Step 3 — re-run this pipeline.
           Labels: "Phishing Email" → 1  |  "Safe Email" → 0

The pipeline runs correctly with either or both sources present.
If the Kaggle file is absent, a WARNING is logged and only UCI SMS is used
(scam coverage is then limited to SMS-format texts only).

Domain note
-----------
UCI "spam" and CLAIR "phishing" represent different threat classes.
We merge them under a single binary label (0 = legitimate, 1 = risk) for the
baseline model, but the 'source' column in every split CSV allows downstream
evaluation to break results out per domain.
"""

from __future__ import annotations

import io
import logging
import re
import zipfile
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Configuration — single source of truth for seeds, ratios, URLs
# ---------------------------------------------------------------------------

RANDOM_SEED: int = 42

LABEL_SCAM: int = 1   # phishing / scam / suspicious
LABEL_HAM: int = 0    # legitimate / safe

# Direct ZIP download — no credentials required
UCI_SMS_URL: str = (
    "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
)
UCI_SMS_INNER_FILENAME: str = "SMSSpamCollection"

# Manual drop filename (user places this in data/raw/)
KAGGLE_PHISHING_FILENAME: str = "phishing_email.csv"

# Expected column names in the CLAIR Kaggle CSV
_KAGGLE_TEXT_COL: str = "Email Text"
_KAGGLE_LABEL_COL: str = "Email Type"
_KAGGLE_SCAM_VALUES: tuple[str, ...] = ("phishing email", "phishing", "spam", "scam", "fraud")

# Split ratios — must sum to 1.0
TRAIN_RATIO: float = 0.70
VAL_RATIO: float = 0.15
# TEST_RATIO = 0.15 (remainder)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("data_prep")


# ---------------------------------------------------------------------------
# 1. Download helpers
# ---------------------------------------------------------------------------

def download_uci_sms(raw_dir: Path) -> Path:
    """
    Download and extract the UCI SMS Spam Collection ZIP to *raw_dir*.

    Returns the path to the extracted TSV file.
    Idempotent — skips the download if the file already exists locally.

    Raises
    ------
    requests.HTTPError
        If the remote server returns a non-2xx status code.
    FileNotFoundError
        If the expected TSV cannot be located inside the downloaded ZIP.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    target = raw_dir / UCI_SMS_INNER_FILENAME

    if target.exists():
        log.info("UCI SMS already present at '%s' — skipping download.", target)
        return target

    log.info("Downloading UCI SMS Spam Collection …")
    log.info("  URL : %s", UCI_SMS_URL)

    response = requests.get(UCI_SMS_URL, timeout=30)
    response.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
        members = zf.namelist()
        log.debug("ZIP contents: %s", members)

        if UCI_SMS_INNER_FILENAME in members:
            tsv_member = UCI_SMS_INNER_FILENAME
        else:
            # Fallback: grab the first non-directory member (some mirrors differ)
            tsv_member = next((m for m in members if not m.endswith("/")), None)
            if tsv_member is None:
                raise FileNotFoundError(
                    f"Cannot locate a TSV file inside the UCI ZIP. "
                    f"Members found: {members}"
                )
            log.warning(
                "Expected '%s' inside ZIP but found '%s' — extracting anyway.",
                UCI_SMS_INNER_FILENAME, tsv_member,
            )

        zf.extract(tsv_member, raw_dir)
        extracted = raw_dir / tsv_member

        # Rename to canonical name if the member had a different name
        if extracted.resolve() != target.resolve():
            extracted.rename(target)

    log.info("UCI SMS extracted to '%s'.", target)
    return target


# ---------------------------------------------------------------------------
# 2. Loaders
#    Each loader returns a normalised DataFrame with exactly three columns:
#      text   : str   — raw message text (not yet cleaned)
#      label  : int   — LABEL_HAM (0) or LABEL_SCAM (1)
#      source : str   — identifier for downstream per-domain analysis
# ---------------------------------------------------------------------------

def load_uci_sms(raw_dir: Path) -> pd.DataFrame:
    """
    Load the UCI SMS Spam Collection TSV.

    Label mapping
    -------------
    "ham"  → 0 (legitimate)
    "spam" → 1 (scam/suspicious)

    Note: UCI "spam" includes marketing texts, prize-claim messages, etc.,
    not exclusively social-engineering scams. This is a known limitation
    logged via the 'source' column.
    """
    tsv_path = raw_dir / UCI_SMS_INNER_FILENAME
    if not tsv_path.exists():
        log.info("UCI SMS file not found locally — initiating download.")
        tsv_path = download_uci_sms(raw_dir)

    df = pd.read_csv(
        tsv_path,
        sep="\t",
        header=None,
        names=["raw_label", "text"],
        encoding="latin-1",   # UCI file uses latin-1 encoding
    )

    df["label"] = df["raw_label"].map({"ham": LABEL_HAM, "spam": LABEL_SCAM})
    df["source"] = "uci_sms"
    df = df[["text", "label", "source"]].copy()

    _log_class_counts("UCI SMS", df)
    return df


def load_kaggle_phishing(raw_dir: Path) -> Optional[pd.DataFrame]:
    """
    Load the CLAIR Phishing Email CSV dropped manually by the user.

    Expected path: ``<project_root>/data/raw/phishing_email.csv``

    Instructions
    ------------
    1. Open https://www.kaggle.com/datasets/subhajournal/phishingemails in a browser.
    2. Click the Download button (free Kaggle account required).
    3. Extract the ZIP and place ``phishing_email.csv`` at:
           data/raw/phishing_email.csv
    4. Re-run: python -m src.data_prep

    Returns
    -------
    Normalised DataFrame or None if the file is absent.
    """
    csv_path = raw_dir / KAGGLE_PHISHING_FILENAME

    if not csv_path.exists():
        log.warning(
            "\n"
            "  ┌─ Kaggle Phishing CSV not found ──────────────────────────────┐\n"
            "  │  File expected at : data/raw/phishing_email.csv              │\n"
            "  │  Download from    : https://www.kaggle.com/datasets/          │\n"
            "  │                     subhajournal/phishingemails               │\n"
            "  │  Action required  : extract the ZIP, drop the CSV, re-run    │\n"
            "  │                     python -m src.data_prep                   │\n"
            "  │                                                               │\n"
            "  │  ⚠ Proceeding with UCI SMS only — phishing signal is LIMITED │\n"
            "  └───────────────────────────────────────────────────────────────┘\n"
        )
        return None

    df = pd.read_csv(csv_path, encoding="utf-8")
    log.info("Kaggle phishing CSV loaded — columns: %s", list(df.columns))

    # Flexible column detection (handles minor naming variations)
    text_col = _find_column(df, [_KAGGLE_TEXT_COL, "text", "body", "message", "content"])
    label_col = _find_column(df, [_KAGGLE_LABEL_COL, "label", "type", "class", "category"])

    if text_col is None or label_col is None:
        raise ValueError(
            f"Cannot identify text/label columns in '{csv_path}'.\n"
            f"Available columns: {list(df.columns)}\n"
            f"Expected text column like '{_KAGGLE_TEXT_COL}' and "
            f"label column like '{_KAGGLE_LABEL_COL}'."
        )

    df = df[[text_col, label_col]].copy()
    df.columns = ["text", "raw_label"]

    # Map string labels → int (case-insensitive, flexible)
    unique_raw = df["raw_label"].dropna().unique()
    log.info("Kaggle raw label values: %s", unique_raw)

    if df["raw_label"].dtype == object:
        label_map: dict = {}
        for val in unique_raw:
            val_lower = str(val).strip().lower()
            label_map[val] = (
                LABEL_SCAM
                if any(kw in val_lower for kw in _KAGGLE_SCAM_VALUES)
                else LABEL_HAM
            )
        df["label"] = df["raw_label"].map(label_map)
        log.info("Label mapping applied: %s", label_map)
    else:
        # Already numeric (0/1)
        df["label"] = df["raw_label"].astype(int)

    df["source"] = "kaggle_phishing"
    df = df[["text", "label", "source"]].copy()

    _log_class_counts("Kaggle Phishing", df)
    return df


# ---------------------------------------------------------------------------
# 3. Text cleaning
# ---------------------------------------------------------------------------

_WHITESPACE_RE = re.compile(r"\s+")


def clean_text(text: object) -> str:
    """
    Minimal, non-destructive text normalisation.

    Operations applied (in order)
    ------------------------------
    1. Coerce to string (handles None, int, float inputs gracefully).
    2. Collapse all internal whitespace runs (spaces, tabs, newlines) to a
       single space.
    3. Strip leading / trailing whitespace.
    4. Lowercase.

    Deliberately conservative
    -------------------------
    We do NOT apply stemming, lemmatisation, or stop-word removal here.
    Those are TF-IDF / vectoriser hyperparameter decisions that belong in
    src/model.py — keeping them there lets us experiment without touching
    the data pipeline.
    """
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    text = _WHITESPACE_RE.sub(" ", text).strip().lower()
    return text


# ---------------------------------------------------------------------------
# 4. Merging and deduplication
# ---------------------------------------------------------------------------

def merge_datasets(*dfs: Optional[pd.DataFrame]) -> pd.DataFrame:
    """
    Concatenate DataFrames, clean text, deduplicate, and shuffle.

    Parameters
    ----------
    *dfs : DataFrames with columns [text, label, source].
           None values are silently ignored (handles missing Kaggle file).

    Returns
    -------
    Cleaned, shuffled DataFrame with columns [text, label, source].

    Deduplication
    -------------
    Exact-duplicate *cleaned* texts are removed — the first occurrence is kept.
    This prevents data leakage where the same message appears in both datasets.
    """
    non_null_dfs = [df for df in dfs if df is not None]
    if not non_null_dfs:
        raise ValueError("No datasets provided to merge_datasets().")

    combined = pd.concat(non_null_dfs, ignore_index=True)

    # ── Drop rows with null text or label ─────────────────────────────────
    before = len(combined)
    combined.dropna(subset=["text", "label"], inplace=True)
    combined["label"] = combined["label"].astype(int)
    dropped_null = before - len(combined)
    if dropped_null:
        log.info("Dropped %d rows with null text or label.", dropped_null)

    # ── Clean text ─────────────────────────────────────────────────────────
    combined["text"] = combined["text"].apply(clean_text)

    # ── Drop empty strings (post-cleaning) ────────────────────────────────
    combined = combined[combined["text"].str.strip() != ""].copy()

    # ── Deduplicate on cleaned text ────────────────────────────────────────
    before_dedup = len(combined)
    combined.drop_duplicates(subset=["text"], keep="first", inplace=True)
    removed_dups = before_dedup - len(combined)
    if removed_dups:
        log.info("Removed %d exact-duplicate messages.", removed_dups)

    # ── Shuffle ────────────────────────────────────────────────────────────
    combined = combined.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    n_total = len(combined)
    n_ham = (combined["label"] == LABEL_HAM).sum()
    n_scam = (combined["label"] == LABEL_SCAM).sum()
    log.info(
        "Merged dataset — %d total | ham: %d (%.1f%%) | scam/phishing: %d (%.1f%%)",
        n_total,
        n_ham, 100 * n_ham / n_total,
        n_scam, 100 * n_scam / n_total,
    )
    return combined


# ---------------------------------------------------------------------------
# 5. Stratified splitting and saving
# ---------------------------------------------------------------------------

def split_and_save(
    df: pd.DataFrame,
    processed_dir: Path,
) -> dict[str, pd.DataFrame]:
    """
    Stratified 70 / 15 / 15 train / val / test split with a fixed seed.

    Both the val and test splits are held out from training.
    The val split is used for hyperparameter tuning in Milestone 2.
    The test split is used ONLY for final evaluation — never for tuning.

    Parameters
    ----------
    df            : Cleaned, merged DataFrame with [text, label, source].
    processed_dir : Directory where train.csv, val.csv, test.csv are saved.

    Returns
    -------
    dict with keys 'train', 'val', 'test'.
    """
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Split off the 30% temp pool (val + test)
    train_df, temp_df = train_test_split(
        df,
        test_size=(1.0 - TRAIN_RATIO),   # 0.30
        stratify=df["label"],
        random_state=RANDOM_SEED,
    )

    # Step 2: Split temp pool 50/50 → val (15%) and test (15%)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        stratify=temp_df["label"],
        random_state=RANDOM_SEED,
    )

    splits: dict[str, pd.DataFrame] = {
        "train": train_df.reset_index(drop=True),
        "val":   val_df.reset_index(drop=True),
        "test":  test_df.reset_index(drop=True),
    }

    for name, split_df in splits.items():
        path = processed_dir / f"{name}.csv"
        split_df.to_csv(path, index=False)
        n_ham  = (split_df["label"] == LABEL_HAM).sum()
        n_scam = (split_df["label"] == LABEL_SCAM).sum()
        log.info(
            "Saved %s.csv — %d rows | ham: %d | scam: %d",
            name, len(split_df), n_ham, n_scam,
        )

    return splits


# ---------------------------------------------------------------------------
# 6. Orchestrator — full pipeline end-to-end
# ---------------------------------------------------------------------------

def run_pipeline(project_root: Path) -> dict[str, pd.DataFrame]:
    """
    Run the full data pipeline: download → load → merge → split → save.

    Parameters
    ----------
    project_root : Absolute path to the scam-risk-assistant project root.

    Returns
    -------
    dict with keys 'train', 'val', 'test' containing the final DataFrames.

    Raises
    ------
    RuntimeError
        If the merged dataset is too small to train on (< 500 rows).
    """
    raw_dir       = project_root / "data" / "raw"
    processed_dir = project_root / "data" / "processed"

    log.info("=" * 60)
    log.info("Scam Risk Assistant — Data Pipeline")
    log.info("=" * 60)
    log.info("Project root  : %s", project_root)

    # ── Load sources ──────────────────────────────────────────────────────
    download_uci_sms(raw_dir)           # ensures file is present
    uci_df    = load_uci_sms(raw_dir)
    kaggle_df = load_kaggle_phishing(raw_dir)   # None if absent

    sources_used = ["uci_sms"] + (["kaggle_phishing"] if kaggle_df is not None else [])
    log.info("Sources used  : %s", sources_used)

    # ── Merge ─────────────────────────────────────────────────────────────
    merged = merge_datasets(uci_df, kaggle_df)

    if len(merged) < 500:
        raise RuntimeError(
            f"Merged dataset has only {len(merged)} rows — too small for "
            "meaningful training. Check that at least the UCI SMS file loaded correctly."
        )

    # ── Split and save ────────────────────────────────────────────────────
    splits = split_and_save(merged, processed_dir)

    log.info("=" * 60)
    log.info("Pipeline complete.")
    log.info("  %-8s %7s rows", "Split", "Count")
    for name, df in splits.items():
        log.info("  %-8s %7d rows", name, len(df))
    log.info("=" * 60)

    return splits


# ---------------------------------------------------------------------------
# 7. Private helpers
# ---------------------------------------------------------------------------

def _find_column(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    """
    Return the first candidate column name found in *df.columns*.

    Tries exact match first, then case-insensitive strip match.
    Returns None if no candidate is found.
    """
    col_map = {c.strip().lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
        normalised = candidate.strip().lower()
        if normalised in col_map:
            return col_map[normalised]
    return None


def _log_class_counts(source_name: str, df: pd.DataFrame) -> None:
    """Log ham / scam row counts for a given DataFrame."""
    n_ham  = (df["label"] == LABEL_HAM).sum()
    n_scam = (df["label"] == LABEL_SCAM).sum()
    log.info(
        "%s loaded — %d total | ham: %d | scam/phishing: %d",
        source_name, len(df), n_ham, n_scam,
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
# Run the full pipeline from the project root:
#   python -m src.data_prep
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from pathlib import Path

    # Resolve project root: two levels up from this file (src/data_prep.py)
    project_root = Path(__file__).resolve().parents[1]
    splits = run_pipeline(project_root)

    print("\nOK Data pipeline finished successfully.")
    print(f"  {'Split':<8}  {'Rows':>7}  {'Ham':>7}  {'Scam':>7}")
    print("  " + "-" * 36)
    for name, df in splits.items():
        n_ham  = (df["label"] == 0).sum()
        n_scam = (df["label"] == 1).sum()
        print(f"  {name:<8}  {len(df):>7,}  {n_ham:>7,}  {n_scam:>7,}")
