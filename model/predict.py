import json
import os

import joblib

_DIR = os.path.dirname(__file__)
_VECTORIZER_PATH = os.path.join(_DIR, "tfidf_vectorizer.joblib")
_CLASSIFIER_PATH = os.path.join(_DIR, "logreg_classifier.joblib")
_MODEL_INFO_PATH = os.path.join(_DIR, "model_info.json")

_vectorizer = None
_clf = None


def _load():
    global _vectorizer, _clf
    if _vectorizer is None:
        if not os.path.exists(_VECTORIZER_PATH) or not os.path.exists(_CLASSIFIER_PATH):
            raise FileNotFoundError(
                f"Model artifacts not found in {_DIR}/. "
                "Run `python model/train.py` to train the model."
            )
        _vectorizer = joblib.load(_VECTORIZER_PATH)
        _clf = joblib.load(_CLASSIFIER_PATH)


def _load_model_info() -> dict:
    if os.path.exists(_MODEL_INFO_PATH):
        with open(_MODEL_INFO_PATH) as f:
            return json.load(f)
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
