# Email & SMS Spam Detector (TF-IDF + Logistic Regression)

## Description

A machine learning project that classifies text messages and emails as **SPAM** or **HAM** (not spam), using TF-IDF text vectorization (unigrams + bigrams) and a Logistic Regression classifier trained from scratch — no pre-trained model, no hardcoded keyword rules.

Supports both **SMS datasets** and **Real Email CSV datasets** (such as Kaggle Email Spam or Enron Spam datasets) with auto-detection of CSV format and column headers.

---

## Project Structure

```
SPAM_EMAIL_DETECTOR/
├── .venv/                      # Python virtual environment
├── dataset/
│   ├── enron_spam_data.csv     # Enron Email Spam dataset (~33.7K emails)
│   └── spam_dataset.csv        # SMS Spam dataset (~5.5K messages)
├── models/
│   ├── spam_classifier.joblib  # Saved Logistic Regression model
│   └── tfidf_vectorizer.joblib # Saved TF-IDF vectorizer
├── src/
│   ├── train.py                # Data loading, cleaning, TF-IDF vectorization, model training, evaluation
│   └── predict.py              # Loads saved model & vectorizer to classify new emails/messages
├── Spam_Email_Detector.ipynb   # Interactive step-by-step Jupyter Notebook tutorial
├── requirements.txt            # Python dependencies (pandas, numpy, scikit-learn, joblib)
└── README.md
```

---

## Machine Learning Pipeline

```
Raw Email / Text Message
          │
  TF-IDF (unigrams + bigrams)
          │
  Logistic Regression Classifier
          │
      SPAM / HAM
```

---

## Features & Multi-Dataset Support

1. **Auto-Detect CSV & Headers**:
   - Seamlessly loads **standard CSV email datasets** (Kaggle `emails.csv`, Enron dataset, etc.) with columns like `text`, `message`, `subject`, `spam`, `label`, `label_num`.
   - Merges `Subject` and `Body` columns automatically when available into `Subject: ... \n\n ...`.
   - Automatically maps numeric labels (`1` -> `spam`, `0` -> `ham`).
   - Retains full backward compatibility with tab-separated SMS datasets (`"ham\tMessage..."`).
2. **Email-Optimized TF-IDF**:
   - Captures 1-word and 2-word combinations (`ngram_range=(1, 2)`) to detect key email spam phrases like `"click here"`, `"bank account"`, or `"wire transfer"`.
   - Expanded vocabulary size (`max_features=10000`).

---

## How to Swap in a Real Email Dataset

To train the detector on a real email dataset (e.g. Kaggle Email Spam or Enron Spam):

1. Download your chosen Email dataset CSV (e.g. `emails.csv` or `enron_spam_data.csv`).
2. Place the file in the `dataset/` directory (e.g. `dataset/emails.csv`).
3. Run the training script:
   ```bash
   .\.venv\Scripts\python.exe src/train.py
   ```
   The script will automatically detect the email CSV dataset, parse subject and body text, train the Logistic Regression model, evaluate its performance, and save the updated model and vectorizer into `models/`.

---

## Quickstart Guide

### 1. Setup Virtual Environment

```bash
# Windows
python -m venv .venv
.\.venv\Scripts\pip.exe install -r requirements.txt
```

### 2. Train Model

```bash
.\.venv\Scripts\python.exe src/train.py
```

### 3. Run Predictions

```bash
.\.venv\Scripts\python.exe src/predict.py
```

Type or paste any email text / SMS message when prompted to receive instant `SPAM` or `HAM` prediction along with probability score. Type `quit` to exit.
