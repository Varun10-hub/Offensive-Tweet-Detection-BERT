# Data

The dataset is **not committed** to this repository. It is loaded directly
from a public URL at runtime by `src/data.py` (or by the notebook).

## Source

- **URL**: a public Inspirit AI Google Cloud Storage bucket — see `DATA_URL`
  in [`src/config.py`](../src/config.py).
- **Size**: 24,783 labeled tweets.
- **Original publication**: Davidson, T., Warmsley, D., Macy, M., & Weber, I.
  (2017). [*Automated Hate Speech Detection and the Problem of Offensive
  Language*](https://arxiv.org/abs/1703.04009). ICWSM.

## Schema

| Column | Description |
|---|---|
| `count` | Number of CrowdFlower annotators who coded this tweet |
| `hate_speech` | Number who labeled it hate speech |
| `offensive_language` | Number who labeled it offensive |
| `neither` | Number who labeled it neither |
| `class` | Majority label: 0 = hate speech, 1 = offensive, 2 = neither |
| `tweet` | The tweet text |
| `label` | **(Added by `src/data.py`)** Binary label: 1 = offensive/hateful, 0 = clean |

## Class distribution

The dataset is imbalanced — roughly **83.2% of tweets carry the binary label `1`**
(offensive or hateful). A trivial classifier that always predicts the majority
class therefore achieves ~83.2% accuracy. Both the Bag-of-Words baseline (~88%)
and the BERT model (~94%) clear this floor.

## Privacy and ethics

The original dataset is sourced from public tweets and contains content that
many readers will find offensive or distressing. Use it with care. Do not
redistribute the raw text outside the constraints of the original publication.
