"""Tokenization and BERT input preparation.

The functions in this module convert raw text into the three-tensor input
format that BERT expects: word IDs, attention masks, and segment (type) IDs.
"""

from typing import Iterable, List, Tuple

import numpy as np
from tqdm.auto import tqdm

from .config import MAX_LENGTH, CLS_TOKEN, SEP_TOKEN


def tokenize_sequence(text: str, tokenizer, max_length: int = MAX_LENGTH) -> List[str]:
    """Tokenize text into BERT subword tokens with [CLS]/[SEP] markers.

    Truncates the inner token list so that the final sequence
    (including [CLS] and [SEP]) is at most ``max_length`` tokens long.

    Parameters
    ----------
    text : str
        Raw input text.
    tokenizer
        BERT FullTokenizer instance (BERT WordPiece tokenizer).
    max_length : int
        Maximum sequence length, *including* the special tokens.

    Returns
    -------
    List[str]
        Token sequence: [CLS] + tokens + [SEP].
    """
    token_list = tokenizer.tokenize(text)
    token_list = token_list[: (max_length - 2)]
    return [CLS_TOKEN] + token_list + [SEP_TOKEN]


def get_input_tokens(
    text: str, tokenizer, max_length: int = MAX_LENGTH
) -> Tuple[List[str], List[int], int, int]:
    """Tokenize text and convert it into BERT input IDs.

    Parameters
    ----------
    text : str
        Raw input text.
    tokenizer
        BERT FullTokenizer instance.
    max_length : int
        Maximum sequence length, *including* the special tokens.

    Returns
    -------
    Tuple[List[str], List[int], int, int]
        ``(sequence, input_ids, seq_length, pad_length)``:
          - sequence    : string tokens including [CLS]/[SEP]
          - input_ids   : numerical token IDs
          - seq_length  : length of the unpadded sequence
          - pad_length  : number of [PAD] tokens that need to be appended
    """
    sequence = tokenize_sequence(text, tokenizer, max_length)
    input_ids = tokenizer.convert_tokens_to_ids(sequence)
    seq_length = len(input_ids)
    pad_length = max_length - seq_length
    return sequence, input_ids, seq_length, pad_length


def bert_encode(
    texts: Iterable[str], tokenizer, max_length: int = MAX_LENGTH
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Encode a collection of texts as BERT input tensors.

    Returns the three tensors that ``hub.KerasLayer(bert_*)`` expects:
      - ``input_word_ids``  : padded token IDs, shape (N, max_length)
      - ``input_masks``     : attention masks (1 for real tokens, 0 for pads)
      - ``input_type_ids``  : segment IDs (all zeros for single-sentence input)

    Parameters
    ----------
    texts : iterable of str
        Texts to encode (e.g. a pandas Series or list).
    tokenizer
        BERT FullTokenizer instance.
    max_length : int
        Maximum sequence length, *including* the special tokens.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        ``(input_word_ids, input_masks, input_type_ids)``.
    """
    all_tokens, all_masks = [], []

    for text in tqdm(texts, desc="Encoding"):
        _, input_tokens, seq_length, pad_length = get_input_tokens(
            text, tokenizer, max_length
        )
        # Pad token IDs and build the corresponding attention mask.
        input_tokens = input_tokens + [0] * pad_length
        input_pad_mask = [1] * seq_length + [0] * pad_length

        all_tokens.append(input_tokens)
        all_masks.append(input_pad_mask)

    word_ids = np.array(all_tokens)
    masks = np.array(all_masks)
    segment_ids = np.zeros_like(word_ids)

    return word_ids, masks, segment_ids
