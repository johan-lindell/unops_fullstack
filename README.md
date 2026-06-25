# Disaster Tweet Classifier

A binary classifier that predicts whether a short text describes a real disaster. Wraps any compatible model in a local web UI and a batch prediction CLI.

## Requirements

- [uv](https://docs.astral.sh/uv/getting-started/installation/) — Python package and project manager

## Setup

```bash
git clone <repo-url>
cd <repo>
uv sync
```

`uv sync` reads `pyproject.toml` and `uv.lock` to create a virtual environment and install all pinned dependencies. Python 3.12 is pinned in `.python-version` and managed automatically by uv.

Then train (or drop in) your model so that `model/artifacts/` contains the artifact and `model/predict.py` is implemented. See [Model interface](#model-interface).

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

## Project structure

```
.
├── frontend/
│   └── app.py              # Streamlit web UI
├── model/
│   ├── predict.py          # Inference interface (implement this)
│   ├── train.py            # Training script (produced by model agent)
│   └── artifacts/          # Saved model files (.joblib, .pkl, etc.)
├── data/
│   ├── train.csv           # Kaggle NLP with Disaster Tweets — training set
│   └── test.csv            # Kaggle NLP with Disaster Tweets — test set
├── predict.py              # Batch prediction CLI (root-level)
├── pyproject.toml
├── uv.lock                 # Pinned dependency lockfile
└── .python-version         # 3.12
```

## Model interface

Any model can be used by implementing two things in `model/predict.py`:

**`predict_single(text: str) -> dict`**

```python
def predict_single(text: str) -> dict:
    # Must return:
    return {
        "label": 1,                # int — 1 = disaster, 0 = not
        "label_name": "DISASTER",  # str — human-readable label
        "confidence": 0.87,        # float — probability of predicted class (0–1)
    }
```

**`MODEL_INFO: dict`** *(optional — populates the UI sidebar)*

```python
MODEL_INFO = {
    "algorithm": "TF-IDF + Logistic Regression",
    "f1": 0.793,
    "n_train": 7613,
}
```

The UI and CLI both import from `model/predict.py`. If the model artifact is missing, the UI shows a helpful error message directing you to run the training script.

## Data

[Kaggle — Natural Language Processing with Disaster Tweets](https://www.kaggle.com/competitions/nlp-getting-started)

`train.csv` columns: `id`, `keyword`, `location`, `text`, `target` (1 = real disaster, 0 = not)
