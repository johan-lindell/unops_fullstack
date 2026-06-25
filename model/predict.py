import os

import joblib

ARTIFACT_PATH = os.path.join(os.path.dirname(__file__), "artifacts", "model.joblib")

_vectorizer = None
_clf = None


def _load():
    global _vectorizer, _clf
    if _vectorizer is None:
        if not os.path.exists(ARTIFACT_PATH):
            raise FileNotFoundError(
                f"Model artifact not found at {ARTIFACT_PATH}. "
                "Run `python model/train.py` to train the model."
            )
        bundle = joblib.load(ARTIFACT_PATH)
        _vectorizer = bundle["vectorizer"]
        _clf = bundle["classifier"]


def _load_model_info() -> dict:
    if os.path.exists(ARTIFACT_PATH):
        return joblib.load(ARTIFACT_PATH).get(
            "model_info", {"algorithm": "TF-IDF + Logistic Regression", "f1": 0.0, "n_train": 0}
        )
    return {"algorithm": "TF-IDF + Logistic Regression", "f1": 0.0, "n_train": 0}


MODEL_INFO: dict = _load_model_info()

LABEL_NAMES = {0: "Not a disaster", 1: "Real disaster"}


def predict_single(text: str) -> dict:
    _load()
    if not text or not text.strip():
        return {"label": 0, "label_name": LABEL_NAMES[0], "confidence": 1.0}
    vec = _vectorizer.transform([text.strip()])
    label = int(_clf.predict(vec)[0])
    confidence = float(_clf.predict_proba(vec)[0][label])
    return {"label": label, "label_name": LABEL_NAMES[label], "confidence": confidence}
