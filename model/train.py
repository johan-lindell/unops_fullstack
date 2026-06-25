import os

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
import joblib

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "train.csv")
MODEL_DIR = os.path.dirname(__file__)
ARTIFACT_PATH = os.path.join(MODEL_DIR, "artifacts", "model.joblib")


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

    model_info = {
        "algorithm": "TF-IDF + Logistic Regression",
        "f1": round(f1_score(y_val, y_pred, average="macro"), 4),
        "n_train": len(df),
    }
    os.makedirs(os.path.dirname(ARTIFACT_PATH), exist_ok=True)
    joblib.dump({"vectorizer": vectorizer, "classifier": clf, "model_info": model_info}, ARTIFACT_PATH)
    print(f"Artifact saved to {ARTIFACT_PATH}")


if __name__ == "__main__":
    train()
