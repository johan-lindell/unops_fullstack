# Model Plan: Disaster Tweet Classifier

## Approach

**TF-IDF (unigram+bigram) + Logistic Regression**

Chosen because:
- TF-IDF produces ~20k sparse features; logistic regression is its natural home — linear models with L2 regularization thrive on sparse bag-of-words representations
- Logistic regression gives well-calibrated probabilities natively (needed for the confidence score in the UI — alternatives like SVM require extra Platt scaling)
- Trains in ~30 seconds, artifact is ~3 MB (committable to git)
- Satisfies the submission requirement: "a training script that produces [an artifact] in under 5 minutes on CPU"
- Achieves F1 ~0.79–0.82 on this dataset — solid, honest, auditable

**Why not other lightweight models:**
- **kNN**: poor on high-dim sparse data (curse of dimensionality)
- **Random Forest**: each split samples √20k ≈ 141 features, most signal words ignored per tree — weaker than LR here
- **Gradient Boosted Trees (LightGBM)**: better than RF on sparse data but still suboptimal for raw bag-of-words without dimensionality reduction; added complexity for marginal gain
- **Linear SVM**: closest competitor, sometimes +0.01–0.02 F1, but requires Platt scaling for calibrated probabilities

**Natural next step** (not in scope): fine-tuned DistilBERT would reach F1 ~0.85–0.90 in ~20 min CPU.

## Data

- 7,613 train rows, 3,699 test rows
- Columns: id, keyword (99.2% populated), location, text, target
- Balance: 43% disaster (1), 57% not — no special handling needed

## Files

```
train_model.py    # trains and saves model/ (~30s), prints val F1
predict.py        # batch CLI: python predict.py --input x.csv --output y.csv
model/
  tfidf_vectorizer.joblib
  logreg_classifier.joblib
```

## Training (`train_model.py`)

1. Load `data/train.csv`
2. TF-IDF: `max_features=20000`, `ngram_range=(1,2)`, `sublinear_tf=True`
3. Logistic regression: `C=1.0`, `class_weight='balanced'`, `max_iter=1000`
4. 80/20 stratified split for validation
5. Print precision, recall, F1
6. Save to `model/` with joblib

No extra feature engineering to start — keyword column and location ignored initially.

## Batch predict (`predict.py`)

```
python predict.py --input tweets.csv --output predictions.csv
```

- Reads `text` column, outputs `text, label, score`
- Empty/null text → label=0, score=0.0

## Dependencies

```
scikit-learn>=1.4,<2
pandas>=2.2,<3
```
