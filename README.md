# Scam Message Risk Assistant

> A lightweight, privacy-first scam/phishing message analyser built with Python, scikit-learn, and Streamlit.
> Designed for college project submission (January 15, 2027 deadline).

---

## ⚡ Quickstart: Clone & Run in Under 2 Minutes

The repository includes a pre-trained baseline model (`models/scam_risk_model.joblib`, 2.2 MB), so anyone can clone and run the interactive application immediately without downloading datasets or retraining.

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/scam-risk-assistant.git
cd scam-risk-assistant
```

### 2. Create virtual environment & install dependencies
- **Windows (PowerShell):**
  ```powershell
  python -m venv .venv
  .venv\Scripts\activate
  pip install -r requirements.txt
  ```
- **macOS / Linux:**
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  pip install -r requirements.txt
  ```

### 3. Launch the Web Application
```bash
streamlit run app.py
```
Open your browser at **`http://localhost:8501`**.

---

## 🧪 Verify Installation: Run Test Suite

To verify that all modules are working correctly, run the 82 unit tests:
```bash
python -m pytest tests/ -v
```
*(All 82 tests should pass in under 10 seconds on local CPU).*

---

## 🔁 Reproduce the Pipeline from Scratch (Optional)

If you want to re-download the data, re-run exploratory data analysis, or retrain the model:

### Step 1 — Ingestion & Stratified Splitting
```bash
python -m src.data_prep
```
- Automatically downloads and extracts the UCI SMS Spam Collection.
- Checks for optional Kaggle CLAIR Phishing CSV at `data/raw/phishing_email.csv`.
- Normalises text, removes exact duplicates, and creates stratified 70/15/15 splits in `data/processed/`.

### Step 2 — Hyperparameter Sweep & Training
```bash
python -m src.model
```
- Performs a hyperparameter sweep over regularisation parameter $C \in [0.01, 0.1, 1.0, 10.0, 100.0]$.
- Trains TF-IDF (1-2 ngrams, sublinear term scaling) + balanced Logistic Regression.
- Automatically saves the best model ($C=10.0$) to `models/scam_risk_model.joblib`.

### Step 3 — Multi-Metric Evaluation & Calibration Analysis
```bash
python -m src.evaluation
```
- Evaluates against the 3,401-sample held-out test split.
- Reports per-class precision/recall/F1, ROC-AUC, Expected Calibration Error (ECE), and top False Positives / False Negatives.

### Step 4 — Exploratory Data Analysis Notebook
```bash
jupyter notebook notebooks/eda.ipynb
```
- Generates 6 publication-ready charts (saved automatically to `notebooks/fig*.png`).

---

## 📸 Phase 2: Screenshot OCR Setup (Optional)

Tab 2 of the Streamlit application supports screenshot OCR. This runs 100% locally using Tesseract OCR:

- **Windows:**
  ```powershell
  winget install UB-Mannheim.TesseractOCR
  ```
  *(Or download installer from [UB-Mannheim/tesseract](https://github.com/UB-Mannheim/tesseract/wiki)).*
- **macOS:**
  ```bash
  brew install tesseract
  ```
- **Ubuntu / Debian:**
  ```bash
  sudo apt-get install tesseract-ocr
  ```

*(If Tesseract is not installed, Tab 1 text analysis remains fully functional, and Tab 2 shows clear installation guidance).*

---

## 📁 Repository Structure

```
scam-risk-assistant/
├── data/
│   ├── raw/               # Downloaded/dropped raw datasets (.gitkeep)
│   └── processed/         # train.csv, val.csv, test.csv (.gitkeep)
├── models/
│   ├── .gitkeep
│   └── scam_risk_model.joblib  # Pre-trained model artifact (2.2 MB)
├── notebooks/
│   ├── eda.ipynb          # Exploratory Data Analysis with 6 auto-saved figures
│   └── *.png              # Exported presentation figures
├── src/
│   ├── __init__.py
│   ├── data_prep.py       # Data pipeline & stratified splitting
│   ├── model.py           # TF-IDF + Logistic Regression & tuning
│   ├── evaluation.py      # Multi-metric evaluation & calibration
│   ├── analyzer.py        # Explainability, risk indicators & uncertainty scoring
│   └── ocr.py             # Phase 2: Screenshot OCR, Pillow preprocessing, Tesseract discovery
├── tests/
│   ├── test_data_prep.py  # 28 tests for data preparation
│   ├── test_model.py      # 29 tests for model training & prediction
│   ├── test_analyzer.py   # 14 tests for risk analysis & explainability
│   └── test_ocr.py        # 11 tests for OCR discovery, image preprocessing, extraction
├── app.py                 # Streamlit UI (presentation only, decoupled from ML)
├── PROJECT_DOCUMENTATION.md # Complete project log, theory & evaluation metrics
├── requirements.txt       # Minimal, pinned Python dependencies
├── .gitignore             # Gitignore keeping venv and raw cache out
└── README.md              # Setup and usage guide
```

---

## 📖 Complete Documentation & College Report Reference
For full mathematical formulations, design rationales, confusion matrices, and the milestone changelog, refer to [`PROJECT_DOCUMENTATION.md`](file:///f:/Projects/scam-risk-assistant/PROJECT_DOCUMENTATION.md).
