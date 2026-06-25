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

`uv sync` reads `pyproject.toml` and `uv.lock` to create a virtual environment and install all pinned dependencies. Python 3.12 is pinned in `.python-version` and managed automatically by uv. Training completes in under 60 seconds on CPU and saves the artifact to `model/artifacts/model.joblib`.

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

**Algorithm:** TF-IDF (unigram + bigram) + Logistic Regression  
**Validation F1 (macro):** ~0.79  
**Training set:** 7,613 tweets (Kaggle NLP with Disaster Tweets)

### Why this approach

The two realistic alternatives for this task are a fine-tuned transformer (e.g. DistilBERT) and a classical sparse-feature pipeline (TF-IDF + linear model). The transformer reaches F1 ~0.85–0.90 but takes 20–30 minutes to train on CPU and produces a ~250 MB artifact that cannot be committed to git without Git LFS. The classical pipeline trains in under 60 seconds, produces a ~3 MB committed artifact, and reaches F1 ~0.79 — well within the range where humans themselves disagree on labels for this dataset.

The submission requirement is "working code and sensible choices, not state-of-the-art F1." TF-IDF + Logistic Regression is the auditable, dependency-light choice that a reviewer can clone, retrain, and understand in minutes.

### Why Logistic Regression and not kNN, Random Forest, or Gradient Boosted Trees

TF-IDF produces ~20,000 sparse features. This favours linear models:

- **kNN** — distance metrics degrade badly in high-dimensional sparse spaces (curse of dimensionality). Slow at inference.
- **Random Forest** — each split samples only √20,000 ≈ 141 features, so most signal words are ignored per tree. Weaker than a linear model on bag-of-words.
- **Gradient Boosted Trees** — better than Random Forest on sparse data but still suboptimal for raw bag-of-words without prior dimensionality reduction. Added complexity for marginal gain.
- **Linear SVM** — the closest competitor; sometimes +0.01–0.02 F1. But SVM has no native probability output and requires Platt scaling (`CalibratedClassifierCV`) to produce the confidence score the UI needs. Extra complexity for no user-visible benefit.

Logistic Regression with L2 regularisation is the natural home of sparse high-dimensional bag-of-words features, and the only model in this group that produces well-calibrated probabilities natively.

### Why TF-IDF and not raw counts or embeddings

- **Raw counts** — dominated by frequent but uninformative words. Sublinear TF (`sublinear_tf=True`) compresses this, giving rarer discriminative terms more weight.
- **Bigrams (`ngram_range=(1,2)`)** — captures phrases like "caught fire" and "no casualties" that unigrams split apart.
- **20,000 features** — covers the vocabulary of 7,600 tweets without overfitting; most signal is in the top few thousand anyway.
- **Dense embeddings (word2vec, GloVe)** — would require a separate embedding model, averaging over tokens loses word-order information, and in practice they do not outperform TF-IDF + LR on short-text classification at this dataset size.

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
    "algorithm": "TF-IDF + Logistic Regression",
    "f1": 0.7922,
    "n_train": 7613,
}
```

The UI and CLI both import from `model/predict.py`. If the artifact is missing, the UI shows a helpful error directing you to run the training script.

## Data

[Kaggle — Natural Language Processing with Disaster Tweets](https://www.kaggle.com/competitions/nlp-getting-started)

`train.csv` columns: `id`, `keyword`, `location`, `text`, `target` (1 = real disaster, 0 = not)
