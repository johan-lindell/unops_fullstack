# Exploratory Data Analysis — Disaster Tweets

## Shape

| Split | Rows  | Columns |
|-------|-------|---------|
| Train | 7,613 | id, keyword, location, text, target |
| Test  | 3,263 | id, keyword, location, text |

## Class Balance

| Label | Count | Share |
|-------|-------|-------|
| 0 — not disaster | 4,342 | 57.0% |
| 1 — real disaster | 3,271 | 43.0% |

Moderate imbalance; no resampling required.

## Missingness

| Column   | Train missing | Test missing |
|----------|--------------|-------------|
| keyword  | 61 (0.8%)    | 26 (0.8%)   |
| location | 2,534 (33.3%) | 1,106 (33.9%) |
| text     | 0            | 0           |

`location` is missing for a third of both splits and is geographically noisy — best dropped. `keyword` is nearly complete.

## Text Characteristics

| Metric | Train | Test |
|--------|-------|------|
| Mean char length | 101.0 | 102.1 |
| Max char length  | 157   | 151   |
| Mean word count  | 14.9  | 15.0  |
| Max word count   | 31    | 31    |

Real-disaster tweets are slightly longer on average (108 vs 96 chars) — a subtle but consistent signal.

### Surface features (train)

| Feature | Count | Share |
|---------|-------|-------|
| Contains URL     | 3,971 | 52.2% |
| Contains hashtag | 1,761 | 23.1% |
| Contains mention | 2,039 | 26.8% |

## Keywords

221 unique keywords (URL-encoded, e.g. `body%20bags`). Examples by dominant class:

**Skew toward real disaster:** `derailment`, `outbreak`, `wreckage`, `debris`, `oil%20spill`, `typhoon`

**Skew toward not disaster (figurative use):** `body%20bags`, `armageddon`, `harm`, `deluge`, `ruin`, `wrecked`

Keywords add light signal on top of the raw text but are ambiguous on their own.

## Label Noise

- 110 duplicate texts in train
- 18 texts appear with contradictory labels (both 0 and 1) — inherent noise that sets a practical ceiling on achievable accuracy

## Modelling Takeaways

1. **Primary feature:** `text` — clean, complete in both splits
2. **Secondary feature:** `keyword` — useful but needs URL-decoding
3. **Drop:** `location` — too sparse and noisy
4. **Class imbalance** is mild (57/43); no special handling needed
5. **18 contradictory duplicates** are harmless but explain why perfect accuracy is unattainable
6. A fine-tuned transformer (DistilBERT / RoBERTa) or TF-IDF + Logistic Regression are both sensible starting points given the dataset size and tweet-length texts
