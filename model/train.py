import json
import os

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
import joblib

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "train.csv")
MODEL_DIR = os.path.dirname(__file__)


def train():
    df = pd.read_csv(DATA_PATH)
    df["text"] = df["text"].fillna("")

    X_train, X_val, y_train, y_val = train_test_split(
        df["text"], df["target"], test_size=0.2, random_state=42, stratify=df["target"]
    )

    vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), sublinear_tf=True)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_val_vec = vectorizer.transform(X_val)

    clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000)
    clf.fit(X_train_vec, y_train)

    y_pred = clf.predict(X_val_vec)
    print(classification_report(y_val, y_pred, target_names=["not disaster", "disaster"]))

    joblib.dump(vectorizer, os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib"))
    joblib.dump(clf, os.path.join(MODEL_DIR, "logreg_classifier.joblib"))

    model_info = {
        "algorithm": "TF-IDF + Logistic Regression",
        "f1": round(f1_score(y_val, y_pred, average="macro"), 4),
        "n_train": len(X_train),
    }
    with open(os.path.join(MODEL_DIR, "model_info.json"), "w") as f:
        json.dump(model_info, f)

    print(f"Artifacts saved to {MODEL_DIR}/")


if __name__ == "__main__":
    train()
