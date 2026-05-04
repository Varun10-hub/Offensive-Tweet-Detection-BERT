"""Modular implementation of the BERT offensive-tweet detection pipeline.

This package refactors the canonical Jupyter notebook into reusable modules
suitable for code review, integration into larger pipelines, and unit testing.

Modules
-------
config        : Hyperparameters, URLs, and pipeline constants.
data          : Dataset loading and binary-label construction.
preprocessing : BERT tokenization, [CLS]/[SEP] tagging, padding, attention masks.
baseline      : Bag-of-Words + Logistic Regression baseline.
model         : BERT-based binary classification model definition.
train         : Compilation and training loop.
evaluate      : Metrics and visualization helpers.
"""

__version__ = "1.0.0"
__author__ = "Varun Wadhia"
__license__ = "MIT"
