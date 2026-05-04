"""Configuration constants for the BERT offensive-tweet detection pipeline.

All tunable hyperparameters and external URLs live in one place so that
experiments can override them without touching the rest of the codebase.
"""

# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------

DATA_URL = (
    "https://storage.googleapis.com/inspirit-ai-data-bucket-1/Data/Deep%20Dives/"
    "Advanced%20Topics%20in%20AI/Sessions%201%20-%2010%20(Main%20Curriculum)/"
    "Session%204_%20State%20of%20the%20Art%20NLP%20-%20Transformers/labeled_data.csv"
)

# ----------------------------------------------------------------------------
# BERT (TensorFlow Hub)
# ----------------------------------------------------------------------------

# This older TF1-style hub module requires Keras 2. Set
# os.environ["TF_USE_LEGACY_KERAS"] = "1" before importing TensorFlow if you
# are on TF 2.16+ (which ships Keras 3 by default).
HUB_URL = "https://tfhub.dev/tensorflow/"
HUB_ID = "bert_en_uncased_L-24_H-1024_A-16/1"

# Helper file with FullTokenizer; the canonical notebook downloads it via wget.
TOKENIZATION_PY_URL = (
    "https://storage.googleapis.com/inspirit-ai-data-bucket-1/Data/Deep%20Dives/"
    "Advanced%20Topics%20in%20AI/Sessions%201%20-%2010%20(Main%20Curriculum)/"
    "Session%204_%20State%20of%20the%20Art%20NLP%20-%20Transformers/tokenization.py"
)

# ----------------------------------------------------------------------------
# Preprocessing
# ----------------------------------------------------------------------------

MAX_LENGTH = 32
CLS_TOKEN = "[CLS]"
SEP_TOKEN = "[SEP]"

# ----------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------

BATCH_SIZE = 64
EPOCHS = 2
LEARNING_RATE = 1e-5
VALIDATION_SPLIT = 0.2
SHUFFLE = False  # Deterministic ordering for reproducibility.

# ----------------------------------------------------------------------------
# Baseline
# ----------------------------------------------------------------------------

BASELINE_TEST_SIZE = 0.2
BASELINE_MAX_ITER = 5
BASELINE_RANDOM_STATE = 1000
