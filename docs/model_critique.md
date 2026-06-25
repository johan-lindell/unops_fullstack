# Model Approach Critique

## The Core Problem TF-IDF Cannot Solve

The hardest cases in this dataset are not about rare disaster words — they are about **figurative vs. literal usage of the same words**. The EDA confirms this:

| Keyword | Dominant class | Reason |
|---------|---------------|--------|
| `armageddon` | 0 — not disaster | Hyperbole ("this traffic is armageddon") |
| `harm` | 0 — not disaster | Abstract/metaphorical use |
| `deluge` | 0 — not disaster | Figurative ("a deluge of emails") |
| `wrecked` | 0 — not disaster | Slang ("I'm wrecked after that workout") |
| `body%20bags` | 0 — not disaster | Figurative/cultural references |

Contrast with:

| Tweet pattern | Correct class |
|--------------|--------------|
| "The building is on fire" | 1 — real disaster |
| "This team is on fire" | 0 — not disaster |
| "People are dead" | 1 — real disaster |
| "I'm dead 😂" | 0 — not disaster |

TF-IDF is a bag-of-words model. It counts and re-weights tokens — it has **no mechanism to distinguish context**. Bigrams help at the margins ("on fire", "no casualties") but tweets are too short and varied for bigrams to reliably capture figurative vs. literal patterns. The model sees "fire" in both classes at similar rates and is structurally unable to resolve the ambiguity.

This is the failure mode the dataset was specifically designed to test. The current justification in README correctly compares Logistic Regression against other classical models (kNN, RF, GBT, SVM) but never questions whether the **feature representation** is the bottleneck — which it is.

---

## The Missing Middle Ground

The README jumps from TF-IDF + LR directly to fine-tuned DistilBERT (dismissed as too slow/large) and skips the obvious middle path:

**Sentence-transformer embeddings + Logistic Regression**

- Frozen pre-trained encoder produces dense semantic embeddings
- LR head trains in seconds on top of those embeddings
- Encodes context: "on fire" in two different sentences maps to meaningfully different vectors
- No GPU, no fine-tuning, almost no added code complexity
- Artifact: embeddings not stored (re-encoded at inference); LR weights ~negligible

---

## Sentence-Transformer Model Options

All models below are available via the `sentence-transformers` library and run on CPU. Estimated F1 is for this dataset based on published benchmarks; actual results will vary.

### Tier 1 — Fast, small, good quality

| Model | Size | Embed dim | Speed (CPU) | Est. F1 | Notes |
|-------|------|-----------|-------------|---------|-------|
| `all-MiniLM-L6-v2` | 22 MB | 384 | ~1,500 sentences/s | 0.82–0.84 | Best speed/quality default. 6-layer MiniLM distilled from a larger model. Strong on short texts. |
| `all-MiniLM-L12-v2` | 33 MB | 384 | ~800 sentences/s | 0.83–0.85 | 12-layer version of the above. ~2× slower but consistently edges L6 on classification tasks. |
| `paraphrase-MiniLM-L6-v2` | 22 MB | 384 | ~1,500 sentences/s | 0.81–0.83 | Trained on paraphrase pairs rather than general NLI. Slightly weaker on short-text classification vs `all-MiniLM-L6-v2`. |

### Tier 2 — Better quality, moderate cost

| Model | Size | Embed dim | Speed (CPU) | Est. F1 | Notes |
|-------|------|-----------|-------------|---------|-------|
| `all-distilroberta-v1` | 82 MB | 768 | ~400 sentences/s | 0.83–0.85 | DistilRoBERTa base, trained on 1B sentence pairs. Better at nuance than MiniLM; 768-dim gives the LR classifier more to work with. |
| `all-mpnet-base-v2` | 420 MB | 768 | ~200 sentences/s | 0.84–0.86 | MPNet base, highest quality general-purpose model in the sentence-transformers library. Slower and larger but best representation quality without fine-tuning. |

### Tier 3 — Specialised

| Model | Size | Embed dim | Notes |
|-------|------|-----------|-------|
| `multi-qa-MiniLM-L6-cos-v1` | 22 MB | 384 | Trained on QA/retrieval rather than semantic similarity. Slightly weaker for classification. Not recommended here. |
| `paraphrase-multilingual-MiniLM-L12-v2` | 118 MB | 384 | Covers 50+ languages. Worth considering only if the tweet dataset contains non-English text (it largely doesn't). Overhead not justified for English-only. |

---

## Recommendation

**`all-MiniLM-L6-v2`** is the right default for this submission:

- 22 MB, fetched at first run (or committed with Git LFS)
- Trains full pipeline (encode + fit LR) in ~60–90s on CPU
- Expected F1 gain of ~3–5 points over TF-IDF — meaningful, not marginal
- No GPU, no fine-tuning, same LR classifier and same auditable linear decision boundary
- Code change is small: swap the vectorizer for an encoder, keep everything else

If quality matters more than speed, step up to **`all-mpnet-base-v2`** — it is the strongest frozen encoder available without fine-tuning and may close another 1–2 points.

---

## Where the Current TF-IDF Plan Holds Up

- The practical constraints (CPU, fast training, small artifact) are real
- F1 ~0.79 is not embarrassing — the Kaggle ceiling with transformers is ~0.85
- The code architecture (generic `predict_single` interface, swappable backend) is correctly designed to allow a model swap without touching the UI or CLI
- The argument for LR over kNN/RF/GBT/SVM on sparse features is correct — it's just that TF-IDF is the wrong starting representation for this problem

---

## Comparison Summary

| Approach | Est. F1 | Train time (CPU) | Artifact | Handles figurative language |
|----------|---------|-----------------|----------|----------------------------|
| TF-IDF + LR (current) | 0.79 | ~30s | ~3 MB | No |
| Sentence-transformer (MiniLM-L6) + LR | 0.82–0.84 | ~90s | ~negligible (LR only) | Yes |
| Sentence-transformer (mpnet-base) + LR | 0.84–0.86 | ~5 min | ~negligible | Yes |
| Fine-tuned DistilBERT | 0.85–0.90 | ~20 min | ~250 MB | Yes |
| Fine-tuned RoBERTa-base | 0.87–0.91 | ~30 min | ~500 MB | Yes |
