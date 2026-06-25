import os

import joblib

ARTIFACT_PATH = os.path.join(os.path.dirname(__file__), "artifacts", "model.joblib")

_encoder = None
_clf = None

LABEL_NAMES = {0: "Not a disaster", 1: "Real disaster"}

_DEFAULT_MODEL_INFO = {
    "algorithm": "Sentence-Transformer (all-MiniLM-L6-v2) + Logistic Regression",
    "f1": 0.0,
    "n_train": 0,
}


def _load():
    global _encoder, _clf
    if _encoder is None:
        if not os.path.exists(ARTIFACT_PATH):
            raise FileNotFoundError(
                f"Model artifact not found at {ARTIFACT_PATH}. "
                "Run `python model/train.py` to train the model."
            )
        bundle = joblib.load(ARTIFACT_PATH)
        _clf = bundle["classifier"]

        from sentence_transformers import SentenceTransformer
        encoder_name = bundle.get("model_info", {}).get("encoder", "all-MiniLM-L6-v2")
        _encoder = SentenceTransformer(encoder_name)


def _load_model_info() -> dict:
    if os.path.exists(ARTIFACT_PATH):
        return joblib.load(ARTIFACT_PATH).get("model_info", _DEFAULT_MODEL_INFO)
    return _DEFAULT_MODEL_INFO


MODEL_INFO: dict = _load_model_info()


def predict_single(text: str) -> dict:
    _load()
    if not text or not text.strip():
        return {"label": 0, "label_name": LABEL_NAMES[0], "confidence": 1.0}
    embedding = _encoder.encode([text.strip()])
    label = int(_clf.predict(embedding)[0])
    confidence = float(_clf.predict_proba(embedding)[0][label])
    return {"label": label, "label_name": LABEL_NAMES[label], "confidence": confidence}
