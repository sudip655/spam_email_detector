"""
Spam Detector - Training Script
--------------------------------
Trains a Logistic Regression model to classify text messages
(SMS/email-style short text) as SPAM or HAM (not spam).

Pipeline:
    Raw text  ->  TF-IDF numbers  ->  Logistic Regression  ->  spam / ham
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# Reproducibility: using a fixed seed means we get the same
# train/test split and same results every time we run this script.
RANDOM_STATE = 42

BASE_DIR = Path(__file__).resolve().parent.parent
# Train on BOTH datasets combined so the model learns SMS spam patterns
# (promo codes, short texts) AND email spam patterns (phishing, scams).
DATA_PATHS = [
    BASE_DIR / "dataset" / "enron_spam_data.csv",   # ~33K real corporate emails
    BASE_DIR / "dataset" / "spam_dataset.csv",       # ~5.5K SMS messages
]
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)


def load_dataset(path: Path) -> pd.DataFrame:
    """
    Load the dataset (supports both SMS datasets and standard Email CSV datasets).

    Supports:
    1. Standard CSV files (e.g., Kaggle emails.csv, Enron spam dataset) with headers:
       - Text columns: 'text', 'message', 'body', 'Subject', etc.
       - Label columns: 'spam', 'label', 'label_num', 'spam/ham', 'category', 'class'
       - Automatically maps numeric 1/0 labels or 'spam'/'ham' strings.
       - Merges Subject + Body if they exist in separate columns.
    2. Legacy tab-separated SMS datasets where each line is enclosed in quotes ("ham\t...").
    """
    if not path.exists():
        # If Enron dataset is missing, automatically download it
        if "enron" in path.name.lower():
            print(f"Dataset '{path.name}' not found locally. Downloading Enron dataset...")
            import urllib.request, zipfile, os
            url = "https://raw.githubusercontent.com/MWiechmann/enron_spam_data/master/enron_spam_data.zip"
            zip_target = path.parent / "enron_temp.zip"
            path.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(url, zip_target)
            with zipfile.ZipFile(zip_target) as z:
                z.extractall(path.parent)
            if zip_target.exists():
                os.remove(zip_target)
            print(f"Downloaded '{path.name}' successfully!")
        else:
            csv_files = list(path.parent.glob("*.csv"))
            if csv_files:
                path = csv_files[0]
                print(f"Target dataset file not found, auto-selecting CSV in dataset/: {path.name}")
            else:
                raise FileNotFoundError(
                    f"Dataset not found at {path}. Place a CSV dataset in dataset/ or run train.py to auto-download."
                )

    # Attempt 1: Standard pandas read_csv for Email CSV datasets (Kaggle, Enron, etc.)
    try:
        df_raw = pd.read_csv(path)
        col_map = {str(c).lower().strip(): c for c in df_raw.columns}

        label_col = None
        for candidate in ["spam", "label", "label_num", "spam/ham", "category", "class", "v1", "target"]:
            if candidate in col_map:
                label_col = col_map[candidate]
                break

        has_subject = "subject" in col_map
        has_body = "body" in col_map or "message" in col_map
        text_col = None
        for candidate in ["text", "message", "body", "content", "email", "v2"]:
            if candidate in col_map:
                text_col = col_map[candidate]
                break

        if label_col is not None and (text_col is not None or (has_subject and has_body)):
            print(f"Detected standard CSV email dataset format: '{path.name}'")

            # Combine Subject and Message if both are present in separate columns
            if has_subject and "message" in col_map and col_map["subject"] != col_map["message"]:
                sub_col = col_map["subject"]
                msg_col = col_map["message"]
                messages = (
                    "Subject: " + df_raw[sub_col].fillna("").astype(str) + "\n\n" + df_raw[msg_col].fillna("").astype(str)
                )
            elif text_col is not None:
                messages = df_raw[text_col].astype(str)
            else:
                messages = df_raw[has_body].astype(str)

            labels = df_raw[label_col]

            # Standardize labels (1, '1', 'spam', 'true' -> 'spam', else -> 'ham')
            labels_clean = labels.apply(
                lambda val: "spam" if str(val).strip().lower() in ["1", "spam", "true", "1.0"] else "ham"
            )

            df = pd.DataFrame({"label": labels_clean, "message": messages})
            return df
    except Exception:
        # Fall back to custom tab-separated reader if read_csv fails
        pass

    # Attempt 2: Legacy custom line-by-line tab-separated reader for SMS dataset format
    print(f"Detected tab-separated SMS dataset format: '{path.name}'")
    labels, messages = [], []
    with open(path, encoding="utf-8", errors="replace") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")
            if not line:
                continue
            if line.startswith('"') and line.endswith('"'):
                line = line[1:-1]
            if "\t" not in line:
                continue
            label, message = line.split("\t", 1)
            labels.append(label.strip().lower())
            messages.append(message.strip())

    df = pd.DataFrame({"label": labels, "message": messages})
    return df


def inspect_dataset(df: pd.DataFrame) -> None:
    print("=" * 50)
    print("DATASET INSPECTION")
    print("=" * 50)
    print(f"Shape (rows, columns): {df.shape}")
    print(f"\nColumn names: {list(df.columns)}")
    print(f"\nFirst 5 rows:\n{df.head()}")
    print(f"\nMissing values per column:\n{df.isnull().sum()}")
    print(f"\nDuplicate rows: {df.duplicated().sum()}")
    print(f"\nClass distribution:\n{df['label'].value_counts()}")
    print(f"\nClass distribution (%):\n{df['label'].value_counts(normalize=True) * 100}")


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic cleaning:
    - Drop rows with missing message/label (a model can't learn from blanks)
    - Drop duplicate rows (repeated messages could leak from train into test
      and make our accuracy look artificially better than it really is)
    """
    before = len(df)
    df = df.dropna(subset=["label", "message"])
    df = df.drop_duplicates()
    after = len(df)
    print(f"\nCleaning: removed {before - after} rows (missing/duplicates). {after} rows remain.")
    return df


def split_dataset(df: pd.DataFrame):
    """
    Split into training data (model learns from this) and test data
    (model is evaluated on this, and NEVER trained on it).

    stratify=y keeps the same spam/ham ratio in both the train and
    test sets. Without this, a random split on an imbalanced dataset
    (87% ham / 13% spam) could accidentally put too few spam examples
    in the test set, making evaluation unreliable.
    """
    X = df["message"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print("\n" + "=" * 50)
    print("TRAIN/TEST SPLIT")
    print("=" * 50)
    print(f"Training messages: {len(X_train)}")
    print(f"Testing messages:  {len(X_test)}")
    print(f"Train class balance:\n{y_train.value_counts(normalize=True)}")
    print(f"Test class balance:\n{y_test.value_counts(normalize=True)}")

    return X_train, X_test, y_train, y_test


def vectorize_text(X_train, X_test):
    """
    Convert raw text into TF-IDF numeric features.

    - fit_transform on X_train: LEARN the vocabulary + word importance
      weights from training messages, and convert them to numbers.
    - transform on X_test: reuse that same learned vocabulary to convert
      test messages to numbers. We do NOT fit on test data -- doing so
      would leak information about the test set into the vectorizer.

    stop_words="english" removes very common words (the, is, at, ...)
    that carry little meaning for spam detection.
    """
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),  # unigrams & bigrams (e.g. 'click here', 'bank account')
        max_features=10000,  # expanded vocabulary for longer email texts
        lowercase=True,
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    print("\n" + "=" * 50)
    print("TF-IDF VECTORIZATION")
    print("=" * 50)
    print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")
    print(f"Training matrix shape: {X_train_tfidf.shape}")
    print(f"Testing matrix shape:  {X_test_tfidf.shape}")

    return X_train_tfidf, X_test_tfidf, vectorizer


def train_model(X_train_tfidf, y_train):
    """
    Train a Logistic Regression classifier.

    model.fit() is where the actual "learning" happens: the algorithm
    looks at every training message's TF-IDF numbers alongside its known
    label (spam/ham) and adjusts an internal weight for every word so
    that, in total, spam messages score toward "spam" and ham messages
    score toward "ham". This is different from a keyword rule because
    the model learns these weights from data and combines hundreds of
    weak signals, rather than us hardcoding "if it contains 'free'".
    """
    model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    model.fit(X_train_tfidf, y_train)

    print("\n" + "=" * 50)
    print("MODEL TRAINING")
    print("=" * 50)
    print("Logistic Regression trained successfully.")

    return model


def evaluate_model(model, X_test_tfidf, y_test):
    """
    Evaluate the trained model on the held-out test set (data the model
    has never seen). This tells us how well it's likely to perform on
    brand-new messages, not just the ones it memorized.

    - Accuracy: overall % of correct predictions. Can be misleading on
      imbalanced data (a model that always predicts "ham" would score
      ~87% accuracy here without learning anything useful).
    - Precision (for spam): of messages predicted spam, how many really
      were spam. Low precision = too many real messages wrongly flagged
      (false positives) -- annoying, could bury an important message.
    - Recall (for spam): of all real spam messages, how many did we catch.
      Low recall = spam slipping through (false negatives) -- annoying
      but less costly than losing a real message.
    - F1-score: balance of precision and recall in one number.
    - Confusion matrix: raw counts of correct/incorrect predictions,
      broken down by class.
    """
    y_pred = model.predict(X_test_tfidf)

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, labels=["ham", "spam"])

    print("\n" + "=" * 50)
    print("MODEL EVALUATION (on unseen test data)")
    print("=" * 50)
    print(f"Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print(f"\nClassification report:\n{report}")
    print("Confusion matrix (rows=actual, cols=predicted), order [ham, spam]:")
    print(cm)
    print(f"\n  True negatives  (ham correctly identified):  {cm[0][0]}")
    print(f"  False positives (ham wrongly flagged spam):    {cm[0][1]}")
    print(f"  False negatives (spam that slipped through):   {cm[1][0]}")
    print(f"  True positives  (spam correctly caught):       {cm[1][1]}")
    print(
        "\nNote: this is a strong result on this particular test set, but it "
        "does not mean the model is perfect or '100% accurate' in general -- "
        "real-world messages are far more varied than this dataset."
    )

    return acc


def test_custom_messages(model, vectorizer):
    """
    Run a few example messages through the trained model, using the
    same vectorizer fitted during training (never a new one), to prove
    the predictions come from the model, not hardcoded keyword rules.
    """
    samples = [
        "Congratulations! You have won a free prize. Claim now!",
        "Hey, are we meeting at college tomorrow?",
        "URGENT! Your account has been suspended. Click here to verify immediately.",
        "Can you send me the report before 5pm?",
    ]

    print("\n" + "=" * 50)
    print("CUSTOM MESSAGE PREDICTIONS")
    print("=" * 50)
    for msg in samples:
        msg_tfidf = vectorizer.transform([msg])
        pred = model.predict(msg_tfidf)[0]
        prob = model.predict_proba(msg_tfidf)[0]
        spam_prob = prob[list(model.classes_).index("spam")]
        print(f"\nMessage: {msg}")
        print(f"Prediction: {pred.upper()}  (spam probability: {spam_prob:.2%})")


def save_artifacts(model, vectorizer):
    """
    Save the trained model AND the fitted vectorizer separately.
    We need both at prediction time: the vectorizer turns new raw text
    into the same numeric format the model was trained on, then the
    model makes the prediction from those numbers.
    """
    model_path = MODELS_DIR / "spam_classifier.joblib"
    vectorizer_path = MODELS_DIR / "tfidf_vectorizer.joblib"

    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)

    print("\n" + "=" * 50)
    print("SAVED ARTIFACTS")
    print("=" * 50)
    print(f"Model saved to:      {model_path}")
    print(f"Vectorizer saved to: {vectorizer_path}")


if __name__ == "__main__":
    # Load and combine ALL datasets listed in DATA_PATHS
    all_dfs = []
    for path in DATA_PATHS:
        if path.exists():
            print(f"\nLoading: {path.name}")
            df_part = load_dataset(path)
            print(f"  -> {len(df_part)} rows loaded")
            all_dfs.append(df_part)
        else:
            print(f"\nSkipping (not found): {path.name}")

    if not all_dfs:
        raise FileNotFoundError("No datasets found! Place CSV files in the dataset/ folder.")

    # Combine all loaded datasets into one
    df = pd.concat(all_dfs, ignore_index=True)
    print(f"\nCombined dataset: {len(df)} total rows from {len(all_dfs)} dataset(s)")

    inspect_dataset(df)
    df = clean_dataset(df)
    X_train, X_test, y_train, y_test = split_dataset(df)
    X_train_tfidf, X_test_tfidf, vectorizer = vectorize_text(X_train, X_test)
    model = train_model(X_train_tfidf, y_train)
    evaluate_model(model, X_test_tfidf, y_test)
    test_custom_messages(model, vectorizer)
    save_artifacts(model, vectorizer)
