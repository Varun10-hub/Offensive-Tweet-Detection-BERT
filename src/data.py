"""Dataset loading utilities."""

import io
import requests
import numpy as np
import pandas as pd

from .config import DATA_URL


def load_dataset(url: str = DATA_URL) -> pd.DataFrame:
    """Load the labeled tweet dataset from a public URL.

    The source CSV has the following columns (CrowdFlower-annotated):
      - count               : number of annotators who coded the tweet
      - hate_speech         : number who labeled it hate speech
      - offensive_language  : number who labeled it offensive
      - neither             : number who labeled it neither
      - class               : majority label (0=hate speech, 1=offensive, 2=neither)
      - tweet               : the tweet text

    A binary `label` column is added to this function's output:
      - 1  -> tweet is offensive or hateful (class in {0, 1})
      - 0  -> tweet is clean                (class == 2)

    Parameters
    ----------
    url : str
        URL of the CSV file. Defaults to the Inspirit AI bucket.

    Returns
    -------
    pd.DataFrame
        DataFrame with all source columns plus the binary `label`.
    """
    response = requests.get(url)
    response.raise_for_status()

    df = pd.read_csv(io.StringIO(response.text))

    if "Unnamed: 0" in df.columns:
        del df["Unnamed: 0"]

    df["label"] = np.int32(df["class"] != 2)
    return df
