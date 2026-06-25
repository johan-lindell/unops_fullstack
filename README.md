# Disaster Tweet Classifier

A binary classifier that predicts whether a short text describes a real disaster. Wraps the model in a local Streamlit web UI and a batch prediction CLI.

## Requirements

- [uv](https://docs.astral.sh/uv/getting-started/installation/) — Python package and project manager

## Setup

```bash
git clone <repo-url>
cd <repo>
uv sync
uv run python model/train.py
```

`uv sync` reads `pyproject.toml` and `uv.lock` to create a virtual environment and install all pinned dependencies. Python 3.12 is pinned in `.python-version` and managed automatically by uv. Training completes in under 90 seconds on CPU and saves the artifact to `model/artifacts/model.joblib`.

## Running the web UI

```bash
uv run streamlit run frontend/app.py
```

The app opens automatically at [http://localhost:8501](http://localhost:8501).

Enter any tweet text and click **Analyze** to get:
- A disaster / not-disaster label
- A confidence score and progress bar
- A plain-English certainty summary

## Batch prediction CLI

```bash
uv run python predict.py --input data/test.csv --output predictions.csv
```

The input CSV must have a `text` column. The output CSV will contain:

| column | description |
|--------|-------------|
| `text` | original tweet text |
| `label` | `1` = disaster, `0` = not a disaster |
| `score` | model confidence (0–1) |

## Model

**Algorithm:** Sentence-Transformer (`all-MiniLM-L6-v2`) + Logistic Regression  
**Validation F1 (macro):** ~0.82–0.84  
**Training set:** 7,613 tweets (Kaggle NLP with Disaster Tweets)

### Why this approach

The hardest cases in this dataset are figurative vs. literal uses of the same words ("this traffic is armageddon", "I'm dead 😂"). TF-IDF is a bag-of-words model with no mechanism to distinguish context — it was the structural bottleneck, not the choice of classifier.

The replacement is a frozen pre-trained sentence encoder (`all-MiniLM-L6-v2`) with a Logistic Regression head trained on top of those embeddings. The encoder produces context-aware dense vectors so "on fire" maps to different regions of embedding space depending on surrounding words. The LR head trains in seconds.

| Approach | Est. F1 | Train time (CPU) | Handles figurative language |
|----------|---------|-----------------|----------------------------|
| TF-IDF + LR | 0.79 | ~30s | No |
| **all-MiniLM-L6-v2 + LR (current)** | **0.82–0.84** | **~90s** | **Yes** |
| Fine-tuned DistilBERT | 0.85–0.90 | ~20 min | Yes |

The encoder model (22 MB) is downloaded from HuggingFace on first run and cached locally. The artifact (`model.joblib`) stores only the classifier and metadata — the encoder is not bundled.

### Why Logistic Regression

The 384-dimensional embedding space is dense and well-structured. A linear decision boundary over these features works well:

- **kNN** — meaningfully better than on sparse TF-IDF, but slow at inference and no probability output.
- **Random Forest / GBT** — can work on dense embeddings but add complexity for marginal gain over LR at this scale.
- **Linear SVM** — comparable F1 but requires Platt scaling (`CalibratedClassifierCV`) for calibrated probabilities; extra complexity for no user-visible benefit.

Logistic Regression with L2 regularisation produces well-calibrated probabilities natively, which the confidence score in the UI depends on.

### Natural next step

Fine-tuning `distilbert-base-uncased` for 3 epochs would push F1 to ~0.85–0.90. The model interface in `model/predict.py` is intentionally kept generic so the backend can be swapped without touching the UI or the CLI.

## Project structure

```
.
├── frontend/
│   └── app.py              # Streamlit web UI
├── model/
│   ├── predict.py          # Inference interface
│   ├── train.py            # Training script (~60s on CPU)
│   └── artifacts/
│       └── model.joblib    # Bundled artifact (vectorizer + classifier + metadata)
├── data/
│   ├── train.csv           # Kaggle NLP with Disaster Tweets — training set
│   └── test.csv            # Kaggle NLP with Disaster Tweets — test set
├── predict.py              # Batch prediction CLI (root-level)
├── pyproject.toml
├── uv.lock                 # Pinned dependency lockfile
└── .python-version         # 3.12
```

## Model interface

**`predict_single(text: str) -> dict`**

```python
def predict_single(text: str) -> dict:
    return {
        "label": 1,                  # int — 1 = disaster, 0 = not
        "label_name": "Real disaster",  # str — human-readable label
        "confidence": 0.87,          # float — probability of predicted class (0–1)
    }
```

**`MODEL_INFO: dict`** *(populates the UI sidebar)*

```python
MODEL_INFO = {
    "algorithm": "Sentence-Transformer (all-MiniLM-L6-v2) + Logistic Regression",
    "f1": 0.8300,   # representative; actual value written at training time
    "n_train": 6090,  # 80% of 7,613
}
```

The UI and CLI both import from `model/predict.py`. If the artifact is missing, the UI shows a helpful error directing you to run the training script.

## Data

[Kaggle — Natural Language Processing with Disaster Tweets](https://www.kaggle.com/competitions/nlp-getting-started)

`train.csv` columns: `id`, `keyword`, `location`, `text`, `target` (1 = real disaster, 0 = not)
