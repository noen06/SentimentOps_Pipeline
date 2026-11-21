"""
Text cleaning and label encoding utilities.
Deterministic and stateless — no side effects.
"""

import re
import unicodedata
import contractions
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib
from pathlib import Path


def clean_text(text: str) -> str:
    """
    Full preprocessing pipeline for a single raw text string.
    Order matters: expand contractions before removing punctuation.
    """
    if not isinstance(text, str):
        return ""

    # Normalize unicode (handle accented chars from non-English tweets)
    text = unicodedata.normalize("NFKC", text)

    # Lowercase
    text = text.lower()

    # Remove URLs (http, https, www variants)
    text = re.sub(r"https?://\S+|www\.\S+", "", text)

    # Remove @mentions and #hashtags
    text = re.sub(r"@\w+|#\w+", "", text)

    # Expand contractions BEFORE removing apostrophes
    # e.g. "can't" -> "cannot", "i'm" -> "i am"
    try:
        text = contractions.fix(text)
    except Exception:
        pass  # contractions lib can fail on edge cases; silent fallback

    # Remove non-alphabetical characters (digits, punctuation, symbols)
    text = re.sub(r"[^a-z\s]", "", text)

    # Collapse multiple whitespace into single space
    text = re.sub(r"\s+", " ", text).strip()

    return text


def clean_dataframe(
    df: pd.DataFrame,
    text_col: str = "text",
    label_col: str = "sentiment",
    drop_irrelevant: bool = True,
) -> pd.DataFrame:
    """
    Apply cleaning pipeline to an entire DataFrame.
    Handles: Irrelevant removal, empty-after-cleaning drops,
    duplicate resolution (conflict groups removed, consistent groups deduplicated).
    """
    df = df.copy()

    # Drop Irrelevant class if present
    if drop_irrelevant:
        df = df[df[label_col] != "Irrelevant"].reset_index(drop=True)

    # Drop rows with missing text or label
    df = df.dropna(subset=[text_col, label_col])

    # Clean text
    df[text_col] = df[text_col].map(clean_text)

    # Drop rows that became empty after cleaning
    df = df[df[text_col].str.strip().str.len() > 0].reset_index(drop=True)

    # ── Duplicate resolution (MUST happen before split) ────────────────────
    # Group by cleaned text; if a group has conflicting labels → remove entirely.
    # If label-consistent → keep one representative.
    grouped = df.groupby(text_col)[label_col]

    # Texts with conflicting labels across rows
    conflict_mask = grouped.transform(lambda x: x.nunique() > 1)
    df = df[~conflict_mask].reset_index(drop=True)

    # Now deduplicate keeping first occurrence of each (text, label) pair
    df = df.drop_duplicates(subset=[text_col]).reset_index(drop=True)

    return df


def encode_labels(
    labels: pd.Series,
    label_encoder: LabelEncoder | None = None,
    fit: bool = True,
) -> tuple[np.ndarray, LabelEncoder]:
    """
    Encode string sentiment labels to integer indices.
    Returns encoded array and the fitted encoder.
    Pass an existing encoder with fit=False for val/test sets.
    """
    if label_encoder is None:
        label_encoder = LabelEncoder()

    if fit:
        encoded = label_encoder.fit_transform(labels)
    else:
        encoded = label_encoder.transform(labels)

    return encoded.astype(np.int32), label_encoder


def save_encoder(encoder: LabelEncoder, path: Path) -> None:
    joblib.dump(encoder, path)


def load_encoder(path: Path) -> LabelEncoder:
    return joblib.load(path)