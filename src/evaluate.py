"""Evaluation utilities and visualization helpers."""

from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np

from .config import MAX_LENGTH


def plot_training_curves(
    history,
    save_path: Optional[Union[str, Path]] = None,
) -> None:
    """Plot training and validation loss/accuracy curves side by side.

    Parameters
    ----------
    history : keras.callbacks.History
        History object returned by ``model.fit()``.
    save_path : str or Path, optional
        If provided, save the figure to this path (PNG, dpi=150).
    """
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(14, 5))

    ax_loss.plot(history.history["loss"], label="Training")
    ax_loss.plot(history.history["val_loss"], label="Validation")
    ax_loss.set_title("Loss vs. Epoch")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Binary cross-entropy")
    ax_loss.legend()
    ax_loss.grid(alpha=0.3)

    ax_acc.plot(history.history["accuracy"], label="Training")
    ax_acc.plot(history.history["val_accuracy"], label="Validation")
    ax_acc.set_title("Accuracy vs. Epoch")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.legend()
    ax_acc.grid(alpha=0.3)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()


def predict_texts(
    model, texts, tokenizer, max_length: int = MAX_LENGTH
) -> np.ndarray:
    """Score arbitrary texts with the trained BERT model.

    Parameters
    ----------
    model : tf.keras.Model
        Trained BERT classifier.
    texts : iterable of str
        Texts to score.
    tokenizer
        BERT FullTokenizer instance.
    max_length : int
        Maximum sequence length (must match training).

    Returns
    -------
    np.ndarray
        Array of probabilities (offensive=1) of shape ``(len(texts), 1)``.
    """
    # Local import avoids a circular import at package load time.
    from .preprocessing import bert_encode

    encoded = bert_encode(texts, tokenizer, max_length=max_length)
    return model.predict(encoded)
