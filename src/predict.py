"""
Spam / Email Detector - Prediction Script
-------------------------------------------
Loads the already-trained model and vectorizer (run train.py first)
and classifies text/emails as SPAM or HAM.
"""

from pathlib import Path
import joblib

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "spam_classifier.joblib"
VECTORIZER_PATH = BASE_DIR / "models" / "tfidf_vectorizer.joblib"


def load_artifacts():
    """Load trained Logistic Regression model and TF-IDF vectorizer."""
    if not MODEL_PATH.exists() or not VECTORIZER_PATH.exists():
        raise FileNotFoundError(
            "Trained model or vectorizer not found. Run 'python src/train.py' first."
        )
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer


def predict_message(message: str, model, vectorizer):
    """
    Classify a message or email string.

    Transform using the SAME vectorizer fitted during training —
    never fit a new vectorizer at prediction time.
    """
    message_tfidf = vectorizer.transform([message])
    prediction = model.predict(message_tfidf)[0]
    proba = model.predict_proba(message_tfidf)[0]
    spam_prob = proba[list(model.classes_).index("spam")]
    return prediction, spam_prob


if __name__ == "__main__":
    model, vectorizer = load_artifacts()

    print("=" * 60)
    print(" Email / Message Spam Detector")
    print("=" * 60)
    print(" Type or paste an email/message to classify it.")
    print(" Type 'quit' or 'exit' to stop.\n")

    while True:
        try:
            message = input("Enter email text or message: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if message.lower() in ("quit", "exit"):
            break
        if not message:
            continue

        prediction, spam_prob = predict_message(message, model, vectorizer)
        print(f"\n-> PREDICTION: {prediction.upper()}")
        print(f"-> Spam Probability: {spam_prob:.2%}\n")
