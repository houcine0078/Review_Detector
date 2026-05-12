"""
Preprocessing utilities for text data.

This module contains helper functions to clean textual data and prepare
it for traditional machine‑learning algorithms. For transformer based
models (e.g. CamemBERT), raw text should generally be passed
directly to the model’s tokenizer rather than aggressively cleaned.

Functions
---------
clean_text(text: str) -> str
    Apply a series of simple transformations (lowercasing,
    punctuation and digit removal, whitespace normalisation) to
    a single piece of text.

preprocess_dataframe(df, text_column: str) -> pandas.DataFrame
    Return a copy of the input DataFrame with an additional column
    ``clean_text`` containing processed text.

load_dataset(file_path: str, text_column: str = 'review', label_column: str = 'sentiment')
    Load a CSV file into a DataFrame, drop missing values, and
    return it with ``text_column`` and ``label_column`` present.
"""

from __future__ import annotations

import re
import string
import pandas as pd
from typing import Optional


def clean_text(text: str) -> str:
    """Basic cleaning of a piece of text.

    The cleaning routine is deliberately conservative so as not to
    destroy meaning. It lowers the case, removes punctuation and
    numeric characters and normalises whitespace.

    Parameters
    ----------
    text : str
        The input text to clean.

    Returns
    -------
    str
        The cleaned text.
    """
    if not isinstance(text, str):
        return ""
    # Lowercase
    text = text.lower()
    # Replace punctuation with spaces
    text = re.sub(f'[{re.escape(string.punctuation)}]', ' ', text)
    # Remove digits
    text = re.sub(r'\d+', ' ', text)
    # Collapse multiple spaces into one
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def preprocess_dataframe(df: pd.DataFrame, text_column: str) -> pd.DataFrame:
    """Return a copy of the DataFrame with an additional ``clean_text`` column.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing at least the text column.
    text_column : str
        Name of the column containing the raw text.

    Returns
    -------
    pandas.DataFrame
        A new DataFrame with an extra ``clean_text`` column that
        contains the processed text.
    """
    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in DataFrame.")
    df = df.copy()
    df['clean_text'] = df[text_column].astype(str).apply(clean_text)
    return df


def load_dataset(file_path: str,
                 text_column: str = 'review',
                 label_column: str = 'sentiment') -> pd.DataFrame:
    """Load a CSV dataset and clean missing entries.

    Parameters
    ----------
    file_path : str
        Path to the CSV file.
    text_column : str, default 'review'
        Name of the column containing the text.
    label_column : str, default 'sentiment'
        Name of the column containing the sentiment labels.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the requested columns with NaN rows removed.
    """
    df = pd.read_csv(file_path)
    # Drop rows with missing values in either column
    df = df.dropna(subset=[text_column, label_column])
    # Ensure correct column names exist
    if text_column not in df.columns or label_column not in df.columns:
        raise ValueError(
            f"Dataset must contain columns '{text_column}' and '{label_column}'"
        )
    df[text_column] = df[text_column].astype(str)
    df[label_column] = df[label_column].astype(str)
    return df