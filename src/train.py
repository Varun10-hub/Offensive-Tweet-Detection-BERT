"""Compilation and training loop for the BERT classification model."""

from tensorflow.keras.optimizers import Adam

from .config import (
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    SHUFFLE,
    VALIDATION_SPLIT,
)


def train_bert(
    model,
    train_input,
    train_labels,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = LEARNING_RATE,
    validation_split: float = VALIDATION_SPLIT,
    shuffle: bool = SHUFFLE,
):
    """Compile and fit the BERT model on prepared inputs.

    Parameters
    ----------
    model : tf.keras.Model
        Model returned by :func:`src.model.build_bert_model`.
    train_input : tuple of np.ndarray
        ``(input_word_ids, input_masks, input_type_ids)`` from
        :func:`src.preprocessing.bert_encode`.
    train_labels : np.ndarray
        Binary labels of shape ``(N, 1)``.
    epochs : int
        Number of training epochs. BERT typically converges in 2-3.
    batch_size : int
        Mini-batch size.
    learning_rate : float
        Adam learning rate. BERT fine-tuning typically uses 1e-5 to 5e-5.
    validation_split : float
        Fraction of the training data held out for validation.
    shuffle : bool
        Whether to shuffle data each epoch. False for reproducibility.

    Returns
    -------
    keras.callbacks.History
        The history object returned by ``model.fit()``.
    """
    model.compile(
        optimizer=Adam(learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    history = model.fit(
        train_input,
        train_labels,
        epochs=epochs,
        validation_split=validation_split,
        batch_size=batch_size,
        shuffle=shuffle,
    )

    return history
