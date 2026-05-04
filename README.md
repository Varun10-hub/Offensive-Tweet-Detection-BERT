# Offensive Tweet Detection with BERT

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![TensorFlow](https://img.shields.io/badge/tensorflow-2.15-orange)
![License](https://img.shields.io/badge/license-MIT-green)

> Fine-tuned a 24-layer **BERT-large** transformer to detect offensive and hate-speech content in tweets, achieving **94% validation accuracy** — a **~6 percentage-point absolute lift** (and ~50% relative reduction in error rate) over a Bag-of-Words Logistic Regression baseline.

## Results

| Model | Validation Accuracy | Approach |
|---|---|---|
| **BERT-large (fine-tuned)** | **94.05%** | TF Hub `bert_en_uncased_L-24_H-1024_A-16/1` (24 encoder layers, 1024 hidden) + Dense(1, sigmoid) head |
| Logistic Regression (Bag-of-Words) | 88.18% | scikit-learn `CountVectorizer` + `LogisticRegression(max_iter=5)` |
| Majority-class baseline | 83.20% | Always predict "offensive" (the majority class on this dataset) |

## What this project does

Modern social platforms generate massive volumes of user-generated text and need automated tools to identify offensive or hate-speech content at scale. This project benchmarks a fine-tuned BERT-large transformer against a classic Bag-of-Words logistic regression on a labeled dataset of **24,783 tweets** to quantify the practical performance gain from contextualized language representations.

The repository contains both the original Jupyter notebook (the canonical training run) and a modular `src/` package that refactors the same logic into reusable Python modules — preprocessing, baseline, model, training, evaluation — suitable for code review and integration into larger pipelines.

## Methodology

### Pipeline

1. **Data loading** — 24,783 CrowdFlower-annotated tweets ([Davidson et al., 2017](https://arxiv.org/abs/1703.04009)); three source classes (hate speech / offensive / neither) collapsed into a binary label (offensive-or-hateful vs. clean).
2. **Bag-of-Words baseline** — `CountVectorizer` over an 80/20 train-test split, then `LogisticRegression(max_iter=5)` on the count matrix.
3. **BERT preprocessing** — for each tweet:
   - Subword tokenization via the BERT WordPiece tokenizer
   - Special tokens prepended/appended: `[CLS] ... [SEP]`
   - Padding to fixed length of 32 tokens
   - Attention masks generated (1 for real tokens, 0 for padding)
4. **Model** — pre-trained BERT-large loaded from TensorFlow Hub, fine-tuned end-to-end. The pooled `[CLS]` representation feeds into a single Dense(1, sigmoid) classification head.
5. **Training** — Adam optimizer (learning rate 1e-5), binary cross-entropy loss, batch size 64, 80/20 train/validation split, 2 epochs.
6. **Evaluation** — accuracy/loss curves vs. epoch; comparison against the baseline.

### Why these choices

| Choice | Rationale |
|---|---|
| BERT-large over BERT-base | Larger model has more representational capacity; the marginal training cost is minor on a free Colab T4 GPU. |
| Fine-tuning the full model | Frozen BERT + a linear head plateaus around 90% on this task; full fine-tuning unlocks the additional ~4 points. |
| 32-token max length | Tweets are short; 99%+ of tweets in this dataset fit within 32 BERT subword tokens after tokenization. |
| 2 epochs | Validation loss starts to climb after epoch 2 — classic mild overfitting. Stopping at 2 epochs is the optimal point. |

## Quick Start

### Setup

```bash
git clone https://github.com/Varun10-hub/offensive-tweet-detection-bert.git
cd offensive-tweet-detection-bert
pip install -r requirements.txt
```

### Run the notebook (recommended)

Open `notebooks/offensive_tweet_detection_with_bert.ipynb` in Jupyter or Google Colab and execute all cells. End-to-end training takes ~10 minutes on a single Tesla T4 GPU (free Colab tier).

> **Important**: The TF Hub BERT module used here predates Keras 3 and requires Keras 2 to load correctly. The first cell in the notebook sets `TF_USE_LEGACY_KERAS=1` *before* any TensorFlow imports — do not skip or reorder this cell. If you've already imported TensorFlow in a Colab session, restart the runtime before re-running with the flag set.

### Use the modular code

```python
from src.config import HUB_URL, HUB_ID, MAX_LENGTH
from src.data import load_dataset
from src.model import load_bert_layer, build_bert_model
from src.preprocessing import bert_encode
from src.train import train_bert
from src.evaluate import plot_training_curves

df = load_dataset()
bert_layer = load_bert_layer(trainable=True)
# (See notebook for full end-to-end example.)
```

## Project Structure

```
offensive-tweet-detection-bert/
├── README.md                                 ← You are here
├── LICENSE                                   ← MIT
├── requirements.txt                          ← Pinned dependencies
├── notebooks/
│   └── offensive_tweet_detection_with_bert.ipynb
├── src/
│   ├── __init__.py
│   ├── config.py                             ← Hyperparameters & constants
│   ├── data.py                               ← Dataset loading
│   ├── preprocessing.py                      ← Tokenization, [CLS]/[SEP], masks
│   ├── baseline.py                           ← Bag-of-Words + LogReg baseline
│   ├── model.py                              ← BERT model definition
│   ├── train.py                              ← Training loop
│   └── evaluate.py                           ← Metrics & plot helpers
├── data/
│   └── README.md                             ← Data source & schema
└── results/
    └── figures/                              ← Saved training plots
```

## Future Work

- **Modern transformer comparisons** — benchmark RoBERTa, DeBERTa-v3, and DistilBERT to map the accuracy/latency frontier.
- **Calibration & threshold tuning** — at deployment, the optimal decision threshold depends on the cost of false positives vs. false negatives; this project uses 0.5 throughout.
- **Class-conditional analysis** — split metrics by hate-speech vs. offensive-language sub-classes to identify which categories the model handles well and which it confuses.
- **Adversarial robustness** — evaluate against character-level perturbations and code-switched text.
- **Migration to `transformers` / `keras-nlp`** — replace the TF1-style hub module with a modern Keras-3-native loader for forward compatibility.

## References

1. Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). [*BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*](https://arxiv.org/abs/1810.04805). NAACL-HLT.
2. Davidson, T., Warmsley, D., Macy, M., & Weber, I. (2017). [*Automated Hate Speech Detection and the Problem of Offensive Language*](https://arxiv.org/abs/1703.04009). ICWSM.
3. [The Illustrated BERT](http://jalammar.github.io/illustrated-bert/) — Jay Alammar's visual walkthrough of the architecture.

## License

This project is released under the [MIT License](LICENSE).

## Author

**Varun Wadhia** — BSc Computer Science, University of Alberta (graduating May 2026)

[GitHub](https://github.com/Varun10-hub)

This project was originally developed as part of the **Inspirit AI: AI+X Research Mentorship Program**.
