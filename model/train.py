import os

import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
import joblib

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "train.csv")
MODEL_DIR = os.path.dirname(__file__)
ARTIFACT_PATH = os.path.join(MODEL_DIR, "artifacts", "model.joblib")

ENCODER_MODEL = "all-MiniLM-L6-v2"


def train():
    df = pd.read_csv(DATA_PATH)
    df["text"] = df["text"].fillna("")

    X_train, X_val, y_train, y_val = train_test_split(
        df["text"], df["target"], test_size=0.2, random_state=42, stratify=df["target"]
    )

    print(f"Loading encoder: {ENCODER_MODEL}")
    encoder = SentenceTransformer(ENCODER_MODEL)

    print(f"Encoding {len(X_train)} training samples...")
    X_train_emb = encoder.encode(X_train.tolist(), show_progress_bar=True)

    print(f"Encoding {len(X_val)} validation samples...")
    X_val_emb = encoder.encode(X_val.tolist(), show_progress_bar=True)

    clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000)
    clf.fit(X_train_emb, y_train)

    y_pred = clf.predict(X_val_emb)
    print(classification_report(y_val, y_pred, target_names=["not disaster", "disaster"]))

    model_info = {
        "algorithm": f"Sentence-Transformer ({ENCODER_MODEL}) + Logistic Regression",
        "encoder": ENCODER_MODEL,
        "f1": round(f1_score(y_val, y_pred, average="macro"), 4),
        "n_train": len(X_train),
    }
    os.makedirs(os.path.dirname(ARTIFACT_PATH), exist_ok=True)
    joblib.dump({"classifier": clf, "model_info": model_info}, ARTIFACT_PATH)
    print(f"Artifact saved to {ARTIFACT_PATH}")


if __name__ == "__main__":
    train()
