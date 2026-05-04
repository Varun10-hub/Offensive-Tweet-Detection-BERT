"""Bag-of-Words + Logistic Regression baseline.

A simple, well-understood baseline that establishes the floor BERT must
clear to justify its added complexity. On this dataset the baseline lands
at ~88% test accuracy.
"""

from typing import Tuple

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from .config import BASELINE_TEST_SIZE, BASELINE_MAX_ITER, BASELINE_RANDOM_STATE


def train_baseline(
    sentences,
    labels,
    test_size: float = BASELINE_TEST_SIZE,
    max_iter: int = BASELINE_MAX_ITER,
    random_state: int = BASELINE_RANDOM_STATE,
) -> Tuple[LogisticRegression, CountVectorizer, float, float]:
    """Train a Bag-of-Words + Logistic Regression baseline.

    The vectorizer is fit on the training split only (to avoid leakage).

    Parameters
    ----------
    sentences : array-like of str
        Input texts.
    labels : array-like of int
        Binary labels.
    test_size : float
        Fraction of the dataset held out for evaluation.
    max_iter : int
        Maximum iterations for the LogisticRegression solver.
    random_state : int
        Seed for reproducible splits.

    Returns
    -------
    Tuple[LogisticRegression, CountVectorizer, float, float]
        ``(model, vectorizer, train_accuracy, test_accuracy)``.
    """
    sentences_train, sentences_test, labels_train, labels_test = train_test_split(
        sentences, labels, test_size=test_size, random_state=random_state
    )

    vectorizer = CountVectorizer()
    vectorizer.fit(sentences_train)
    X_train = vectorizer.transform(sentences_train)
    X_test = vectorizer.transform(sentences_test)

    model = LogisticRegression(max_iter=max_iter)
    model.fit(X_train, labels_train)

    train_accuracy = model.score(X_train, labels_train)
    test_accuracy = model.score(X_test, labels_test)

    return model, vectorizer, train_accuracy, test_accuracy
