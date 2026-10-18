# Scam Message Risk Assistant — Complete Project Documentation & Changelog

## 1. Executive Summary & Context
- **Project Goal:** Build a lightweight, local, explainable, and privacy-preserving Scam Message Risk Assistant for college submission (Deadline: January 15, 2027; Capacity: ~3 hours/day).
- **Core Constraints:**
  - Python-only (No Java, FastAPI, external databases, cloud services, or closed LLM APIs).
  - Hardware budget: $\le 8\text{ GB RAM}$, running purely on local CPU.
  - Zero Data Leakage: All inferences and text processing run 100% locally.
  - Honest Domain Framing: Predictions are reported as *Risk Indicators* and *Calibrated Probabilities* with explicit uncertainty flags, rather than absolute truths.

---

## 2. Architecture & Design Principles

### Decoupled Modular Structure
```
scam-risk-assistant/
├── data/
│   ├── raw/               # Downloaded/dropped raw datasets
│   └── processed/         # Stratified train (70%), val (15%), test (15%) splits
├── models/                # Serialised trained model artifacts (.joblib)
├── notebooks/
│   ├── eda.ipynb          # Exploratory Data Analysis with 6 auto-generated charts
│   └── *.png              # Exported publication-ready presentation figures
├── src/
│   ├── __init__.py        # Package initialization
│   ├── data_prep.py       # Idempotent download, cleaning, deduplication, splitting
│   ├── model.py           # TF-IDF + Logistic Regression, hyperparameter sweep, persistence
│   ├── evaluation.py      # Multi-metric evaluation, ECE calibration, FP/FN analysis
│   ├── analyzer.py        # Explainability, n-gram contribution extraction, uncertainty scoring
│   └── ocr.py             # Phase 2: Screenshot OCR, Pillow preprocessing, Tesseract discovery
├── tests/
│   ├── __init__.py
│   ├── test_data_prep.py  # 28 tests for data preparation
│   ├── test_model.py      # 29 tests for model training & inference
│   ├── test_analyzer.py   # 14 tests for risk analysis and explainability
│   └── test_ocr.py        # 11 tests for OCR discovery, image preprocessing, extraction
├── app.py                 # Streamlit UI (presentation only, decoupled from ML logic)
├── requirements.txt       # Environment dependencies
├── PROJECT_DOCUMENTATION.md # Comprehensive record of decisions and milestones
└── README.md              # Quickstart guide
```

---

## 3. Milestones & Progress Log

### Milestone 0: Scope, Dataset Feasibility & Roadmap (Completed)
- **Deliverables:**
  - Evaluated candidate datasets: Identified that standard UCI SMS spam is insufficient on its own because "marketing spam" (coupons, promotions) differs linguistically from "phishing scams" (urgency, credential harvesting, impersonation).
  - Formulated multi-source data strategy combining UCI SMS with CLAIR Phishing dataset.
  - Locked Milestone 2 quality gate at $\text{Test } F_1 \ge 0.85$.
  - Established project folder layout and 4-milestone roadmap.

### Milestone 1: Data Pipeline & Exploratory Data Analysis (Completed)
- **Deliverables:**
  - Created `src/data_prep.py`:
    - Automated download and extraction of UCI SMS collection.
    - Resilient parser for optional Kaggle CLAIR Phishing email corpus.
    - Text normalization: lowercasing, whitespace collapse, deduplication on cleaned text.
    - Stratified 70/15/15 train/validation/test split (`random_state=42`).
  - Created `notebooks/eda.ipynb`:
    - Generated 6 high-resolution charts for college presentations (`fig1` to `fig6`).
    - Quantified vocabulary overlap and class distributions.
  - Implemented `tests/test_data_prep.py` (28 unit tests, all green).

### Milestone 2: Model Training, Tuning & Rigorous Evaluation (Completed)
- **Deliverables:**
  - Created `src/model.py`:
    - Feature extraction: `TfidfVectorizer(ngram_range=(1,2), sublinear_tf=True, max_features=50000)`. Unigrams capture keywords ("urgent", "verify"); bigrams capture phrases ("account suspended", "click here").
    - Classifier: `LogisticRegression(class_weight='balanced', solver='lbfgs')` ensuring minority scam class is not penalized by majority class distribution.
    - Automated hyperparameter sweep over $C \in [0.01, 0.1, 1.0, 10.0, 100.0]$.
    - Best regularisation selected: $C = 10.0$ (Validation $F_1 = 0.9588$).
    - Model saved to `models/scam_risk_model.joblib`.
  - Created `src/evaluation.py`:
    - Comprehensive evaluation across Validation and Test splits.
    - Evaluated on 3,401 unseen test samples:
      - **Accuracy:** $97.74\%$
      - **ROC-AUC:** $0.9956$
      - **Scam Precision:** $96.90\%$
      - **Scam Recall:** $95.91\%$
      - **Scam $F_1$ Score:** $0.9641$ (Passed $F_1 \ge 0.85$ gate)
      - **Calibration (ECE):** $0.0274$ (well-calibrated probabilities)
      - **False Negative Rate:** $4.09\%$ ($44$ missed scams out of $1,077$)
      - **False Positive Rate:** $1.42\%$ ($33$ false alarms out of $2,324$)
  - Implemented `tests/test_model.py` (29 unit tests, all green).

### Milestone 3: Risk Analyzer & Streamlit UI (Completed)
- **Deliverables:**
  - Created `src/analyzer.py`:
    - **Local Explainability Engine:** Decomposes the model logit $w^T x$ to calculate exact individual n-gram contributions $w_i \cdot x_i$, highlighting which specific phrases in the message increased or decreased the risk score.
    - **Calibrated Risk Tiers:** LOW ($P < 35\%$), MODERATE ($35\% \le P < 70\%$), and HIGH ($P \ge 70\%$).
    - **Uncertainty & OOD Detection:** Flags predictions with confidence $< 0.70$ or messages containing negligible in-vocabulary terms.
    - **Heuristic Pattern Scanner:** Captures urgency triggers, credential/financial harvesting patterns, lure tactics, and web links.
  - Created `tests/test_analyzer.py`: 14 comprehensive unit tests verifying explainability, heuristic rules, and edge cases. Total test suite reached 71 passing tests.
  - Created `app.py`:
    - Fully decoupled Streamlit interface handling presentation only.
    - Interactive risk level badges, probability gauges, confidence metrics, and feature contribution chips.

### Milestone 4 (Phase 2): Screenshot OCR Integration (Completed)
- **Deliverables:**
  - Created `src/ocr.py`:
    - **Auto-Discovery:** Locates local Tesseract OCR engine across Windows directories and environment paths (`TESSERACT_CMD`, PATH, Program Files).
    - **Pillow Preprocessing:** Grayscale conversion, auto-contrast enhancement, and sharpness normalization for mobile screenshots.
    - **Quality & Confidence Assessment:** Evaluates per-word OCR confidence to detect blurred or corrupted image uploads.
    - **Graceful Diagnostics:** Clear fallback when Tesseract is missing without crashing the application.
  - Created `tests/test_ocr.py`: 11 unit tests covering binary discovery, image loading, contrast preprocessing, and mocked OCR extraction.
  - Enhanced `app.py`: Added dual-tab layout (`📝 Pasted Text Analysis` and `📸 Screenshot OCR Analysis`) with editable extracted text and quality scorecards. Total test suite is now **82 passing tests**.

---

## 4. Key Engineering & Research Decisions

1. **Why Logistic Regression instead of SVM or Random Forest?**
   - Logistic regression outputs well-calibrated posterior probabilities $P(y=1|x) = \sigma(w^T x + b)$ directly via the sigmoid function.
   - Inference latency is $< 2\text{ ms}$, with memory usage $< 50\text{ MB RAM}$.
   - Linear coefficients allow exact, transparent feature attribution: we can explain *why* a message was flagged by inspecting $w_i \cdot x_i$.

2. **Why Uncertainty & Out-of-Distribution (OOD) Scoring?**
   - Traditional spam filters return a binary "spam / not spam", giving users a false sense of security.
   - When a message has low confidence ($\max(P_0, P_1) < 0.70$) or contains insufficient in-vocabulary terms, the system explicitly warns the user that the prediction is uncertain.

3. **Strict Separation of Concerns:**
   - `app.py` has zero ML or OCR processing logic. It delegates to `src/analyzer.py` and `src/ocr.py`.
   - All modules can be trained, evaluated, and unit-tested headlessly via the command line.

---

## 5. Cloning, Setup & Evaluation Guide (For External Users)

If someone clones this repository from GitHub, here is the exact verification workflow:

### Step 1: Environment Setup
```bash
# Clone
git clone <repo-url>
cd scam-risk-assistant

# Create Virtual Environment
python -m venv .venv

# Activate
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Instant Run (Using Pre-Trained Model)
The repository includes the pre-trained weights in `models/scam_risk_model.joblib`. You can launch the app immediately:
```bash
streamlit run app.py
```

### Step 3: Verify the Unit Test Suite (82 Tests)
```bash
python -m pytest tests/ -v
```

### Step 4: Full Pipeline Reproduction (From Raw Data to Trained Weights)
```bash
# 1. Ingest, clean and split data
python -m src.data_prep

# 2. Sweep hyperparameters and train
python -m src.model

# 3. Evaluate against held-out test split
python -m src.evaluation

# 4. View EDA charts in notebooks/
jupyter notebook notebooks/eda.ipynb
```
