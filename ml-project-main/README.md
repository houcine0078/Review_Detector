# 🔍 Review_Detector
 
> **Automatic sentiment detection from customer reviews in French and English.**  
> Classify any review as *positive*, *negative*, or *neutral* — via CLI, batch CSV, or a real-time Streamlit interface.
 
---
 
## 📌 Overview
 
**Review_Detector** is a complete, production-ready ML pipeline for sentiment analysis of customer reviews. It supports both classical machine learning (TF-IDF + Logistic Regression / Linear SVM) and advanced Transformer-based models (CamemBERT / XLM-RoBERTa), with a clean Streamlit UI for real-time and batch predictions.
 
| Feature | Details |
|---|---|
| **Input** | Free-text review (manual or CSV upload) |
| **Output** | Predicted sentiment + confidence score |
| **Languages** | French 🇫🇷 & English 🇬🇧 |
| **Models** | Logistic Regression, Linear SVM, CamemBERT (optional) |
| **Interface** | Streamlit web app + CLI |
| **Use cases** | Customer support, e-reputation monitoring, content moderation |
 
---
 
## 🗂️ Project Structure
 
```
Review_Detector/
└── ml-project-main/
    ├── app/
    │   └── streamlit_app.py       # Streamlit web interface
    ├── data/
    │   └── sample_reviews.csv     # Sample dataset (review + sentiment)
    ├── models/                    # Saved trained models
    │   ├── logreg_model.joblib
    │   ├── svm_model.joblib
    │   └── best_model.txt         # Auto-selected best model
    ├── src/
    │   ├── preprocess.py          # Text cleaning & data loading
    │   ├── train.py               # Training & model selection
    │   ├── evaluate.py            # Evaluation & confusion matrices
    │   └── predict.py             # Inference (CLI & CSV)
    ├── requirements.txt
    └── README.md
```
 
---
 
## ⚙️ Tech Stack
 
| Layer | Choice | Why |
|---|---|---|
| Language | Python 3.10+ | Rich NLP ecosystem, Streamlit support |
| Classical ML | TF-IDF + LogReg / Linear SVM | Fast, interpretable, strong baseline |
| Advanced ML | CamemBERT / XLM-RoBERTa (Hugging Face) | State-of-the-art on French reviews (~97% accuracy) |
| UI | Streamlit | Modern web app in pure Python, no HTML/CSS needed |
| Pipeline | Scikit-learn + `transformers` / `datasets` | Unified APIs for training, evaluation, serialization |
 
---
 
## 🚀 Getting Started
 
### Prerequisites
 
- Python **3.10+** (tested on 3.11)
- GPU recommended for Transformer fine-tuning (optional)
- Windows users: run commands in **PowerShell**
### Installation
 
```bash
# 1. Clone the repository
git clone https://github.com/houcine0078/Review_Detector.git
cd Review_Detector/ml-project-main
 
# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate        # Linux / macOS
.\venv\Scripts\Activate.ps1     # Windows PowerShell
 
# 3. Install dependencies
pip install -r requirements.txt
```
 
> **Note:** `torch`, `transformers`, and `datasets` are included in `requirements.txt` for Transformer support, but training without a GPU will be slow. `numpy==1.26.4` and `pandas==2.3.3` are pinned to avoid DLL conflicts on Windows.
 
---
 
## 🧠 Training
 
The `train.py` script loads a CSV, cleans the text, splits into train/val/test, trains both classical models, optionally fine-tunes CamemBERT, and automatically saves the best model (by F1-score on validation).
 
### Classical models only (fast)
 
```bash
python -m src.train --data data/sample_reviews.csv --out_dir models
```
 
### With CamemBERT fine-tuning (optional, GPU recommended)
 
```bash
python -m src.train \
    --data data/sample_reviews.csv \
    --out_dir models \
    --train_bert \
    --bert_model camembert-base
```
 
---
 
## 📊 Evaluation
 
```bash
python -m src.evaluate \
    --data data/sample_reviews.csv \
    --model_dir models \
    --out_dir reports
```
 
Generates `reports/evaluation_report.txt` with **accuracy, precision, recall, and F1-score** for each model, plus **confusion matrix PNGs**.
 
---
 
## 🔮 Predictions
 
### Single review via CLI
 
```bash
python -m src.predict --model_dir models --text "Je recommande fortement ce produit !"
```
 
### Batch prediction on a CSV file
 
```bash
python -m src.predict \
    --model_dir models \
    --input_csv data/new_reviews.csv \
    --output_csv results.csv
```
 
Output CSV includes two new columns: `predicted_sentiment` and `confidence`.
 
---
 
## 🖥️ Streamlit Interface
 
```bash
streamlit run app/streamlit_app.py
```
 
The app runs at `http://localhost:8501` and offers:
 
- **Manual input** — type a review and get instant sentiment + confidence score
- **Pre-defined examples** — pick a positive, negative, or neutral sample
- **Batch upload** — upload a CSV of reviews, view results as a table and distribution chart
- **Download** — export the annotated CSV with predictions
- **Feature importance** — sidebar visualization of top words for the Logistic Regression model
---
 
## 📋 Dataset Format
 
Your training CSV must contain at least these two columns:
 
| Column | Description |
|---|---|
| `review` | Raw text of the customer review |
| `sentiment` | Label: `positive`, `negative`, or `neutral` |
 
Missing values are ignored. Extra columns are preserved but not used for training.
 
---
 
## 🔄 Processing Pipeline
 
```
CSV (review, sentiment)
   ↓  Text cleaning & normalization
   ↓  Train / Validation / Test split
   ↓  TF-IDF vectorization
   ↓  Train: Logistic Regression + Linear SVM
   ↓  Optional: CamemBERT fine-tuning
   ↓  Best model selection by F1 (validation set)
   ↓  Save models to models/
   ↓  Predict via CLI or Streamlit
```
 
---
 
## 🛠️ Troubleshooting
 
| Issue | Fix |
|---|---|
| `DLL load failed` on NumPy (Windows) | Already pinned to `numpy==1.26.4`. Re-run `pip install numpy==1.26.4` inside the venv if it reappears. |
| Port 8501 already in use | Run `streamlit run app/streamlit_app.py --server.port 8502` |
| CSV encoding errors | Convert your file to UTF-8 before loading |
| Slow Transformer training | Use a GPU or reduce dataset size for testing |
 
---
 
## 🔭 Future Improvements
 
- **Larger datasets** — integrate public corpora (e.g. Allociné: 100K+ French reviews) for better generalization
- **Hyperparameter tuning** — grid search or Bayesian optimization for classical models
- **Multilingual model** — fine-tune `xlm-roberta-base` for seamless French/English handling
- **Explainability** — add LIME / SHAP visualizations for Transformer predictions
- **Deployment** — Dockerize and deploy on Streamlit Community Cloud, Heroku, or Railway
---
 
## 📦 Pre-trained Models
 
The repository ships with two pre-trained models on `data/sample_reviews.csv`:
 
| File | Description |
|---|---|
| `models/logreg_model.joblib` | Logistic Regression (TF-IDF) |
| `models/svm_model.joblib` | Linear SVM (TF-IDF) |
| `models/best_model.txt` | Name of the best-performing model (auto-selected) |
 
The prediction scripts and Streamlit app load the model listed in `best_model.txt` by default.
 
---
 
## 📄 License
 
This project is open-source. Feel free to use, modify, and distribute with attribution.
