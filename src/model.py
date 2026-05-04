"""BERT-based binary text classification model.

Architecture
------------
- BERT-large backbone (24 encoder layers, 1024 hidden, 16 attention heads)
  loaded from TensorFlow Hub.
- Take the pooled output of the [CLS] token (first position of the
  sequence output).
- Single Dense(1, sigmoid) classification head.

Note on Keras compatibility
---------------------------
The TF Hub module ``bert_en_uncased_L-24_H-1024_A-16/1`` is a TF1-style
SavedModel and requires Keras 2 to load. If you are on TensorFlow 2.16+
(which ships Keras 3 by default), set::

    import os
    os.environ["TF_USE_LEGACY_KERAS"] = "1"

*before* importing TensorFlow. Otherwise you will see a "KerasTensor is
symbolic" error when constructing the model.
"""

import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model

from .config import HUB_URL, HUB_ID, MAX_LENGTH


def load_bert_layer(
    hub_url: str = HUB_URL,
    hub_id: str = HUB_ID,
    trainable: bool = True,
) -> hub.KerasLayer:
    """Load a BERT KerasLayer from TensorFlow Hub.

    Parameters
    ----------
    hub_url : str
        Base URL of the TensorFlow Hub repository.
    hub_id : str
        Specific BERT model identifier within the hub.
    trainable : bool
        Whether BERT weights should be fine-tuned during training.
        Set to False to use BERT purely as a frozen feature extractor.

    Returns
    -------
    hub.KerasLayer
        The configured BERT layer, ready to be plugged into a Keras model.
    """
    return hub.KerasLayer(handle=hub_url + hub_id, trainable=trainable)


def build_bert_model(
    bert_layer: hub.KerasLayer, max_len: int = MAX_LENGTH
) -> Model:
    """Build a binary text classification model on top of BERT.

    Parameters
    ----------
    bert_layer : hub.KerasLayer
        The BERT layer returned by :func:`load_bert_layer`.
    max_len : int
        Sequence length the model expects (must match preprocessing).

    Returns
    -------
    tf.keras.Model
        A compiled-ready Keras model with three named inputs
        (``input_word_ids``, ``input_mask``, ``segment_ids``) and a
        single sigmoid output.
    """
    word_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
    input_mask = Input(shape=(max_len,), dtype=tf.int32, name="input_mask")
    segment_ids = Input(shape=(max_len,), dtype=tf.int32, name="segment_ids")

    # The pooled output is unused; we pull the [CLS] vector from the
    # full sequence output ourselves to keep the architecture explicit.
    _, sequence_output = bert_layer([word_ids, input_mask, segment_ids])

    cls_output = sequence_output[:, 0, :]
    output = Dense(1, activation="sigmoid", name="classification_head")(cls_output)

    return Model(
        inputs=[word_ids, input_mask, segment_ids],
        outputs=output,
        name="bert_offensive_tweet_classifier",
    )
