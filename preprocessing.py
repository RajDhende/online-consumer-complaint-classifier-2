"""
Text preprocessing for consumer complaint narratives.

The pipeline is deliberately explicit so it can be explained in a viva:

    lower-case -> expand contractions -> strip URLs/emails/redactions
    -> remove punctuation and digits -> tokenise -> remove stop-words
    -> drop very short tokens -> join back into a clean string

The same function is used at training time and at prediction time, which
prevents training/serving skew.
"""

from __future__ import annotations

import re
from typing import Iterable

import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

import config

# --------------------------------------------------------------------------- #
# Stop-words
# --------------------------------------------------------------------------- #
# Scikit-learn ships a solid English list. A few conversational fillers and
# CFPB redaction markers are added because they carry no category signal.
EXTRA_STOP_WORDS = {
    "xxxx",
    "xxx",
    "xx",
    "said",
    "told",
    "asked",
    "went",
    "going",
    "really",
    "even",
    "still",
    "want",
    "wanted",
    "know",
    "think",
    "made",
    "make",
    "take",
    "took",
    "give",
    "gave",
    "come",
    "came",
    "look",
    "looked",
    "see",
    "saw",
    "say",
    "says",
    "thing",
    "things",
    "lot",
    "us",
    "im",
    "ive",
    "id",
}

STOP_WORDS: frozenset[str] = frozenset(ENGLISH_STOP_WORDS | EXTRA_STOP_WORDS)

# A compact contraction map. Expanding before tokenising stops words such as
# "don't" becoming the meaningless tokens "don" and "t".
CONTRACTIONS = {
    "can't": "cannot",
    "won't": "will not",
    "n't": " not",
    "'re": " are",
    "'ve": " have",
    "'ll": " will",
    "'d": " would",
    "'m": " am",
    "it's": "it is",
    "that's": "that is",
    "there's": "there is",
    "what's": "what is",
    "i'm": "i am",
    "i've": "i have",
    "i'd": "i would",
    "i'll": "i will",
}

URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
EMAIL_RE = re.compile(r"\S+@\S+\.\S+")
REDACTION_RE = re.compile(r"\bx{2,}\b", re.IGNORECASE)
TOKEN_RE = re.compile(r"[a-z]{3,}")
NON_ALPHA_RE = re.compile(r"[^a-z\s]")
WHITESPACE_RE = re.compile(r"\s+")


def _expand_contractions(text: str) -> str:
    for contraction, expansion in CONTRACTIONS.items():
        text = text.replace(contraction, expansion)
    return text


def clean_text(text: object) -> str:
    """
    Turn one raw complaint narrative into a normalised token string.

    Returns an empty string for missing or unusable input, which lets callers
    filter those rows out safely.
    """
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    value = str(text)
    if not value.strip():
        return ""

    value = value.lower()
    value = _expand_contractions(value)
    value = URL_RE.sub(" ", value)
    value = EMAIL_RE.sub(" ", value)
    value = REDACTION_RE.sub(" ", value)
    value = NON_ALPHA_RE.sub(" ", value)

    tokens = [token for token in TOKEN_RE.findall(value) if token not in STOP_WORDS]
    return " ".join(tokens)


def tokenize(text: str) -> list[str]:
    """Return the clean tokens for an already-cleaned string."""
    return [token for token in text.split() if token]


def word_count(text: object) -> int:
    """Count alphabetic words in a raw narrative (used for quality filters)."""
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return 0
    return len(re.findall(r"[A-Za-z]+", str(text)))


def preprocess_dataframe(
    df: pd.DataFrame,
    text_column: str = config.TEXT_COLUMN,
    min_words: int = config.MIN_WORDS,
    max_words: int = config.MAX_WORDS,
) -> pd.DataFrame:
    """
    Add a ``clean_narrative`` column and drop rows that cannot be modelled.

    Rows are removed when the narrative is missing, shorter than ``min_words``
    or longer than ``max_words``. Long narratives are truncated rather than
    discarded once they exceed the upper bound.
    """
    if text_column not in df.columns:
        raise KeyError(f"Column '{text_column}' is required for preprocessing.")

    work = df.copy()
    work[text_column] = work[text_column].astype("string")
    work["_word_count"] = work[text_column].map(word_count)

    too_long = work["_word_count"] > max_words
    if too_long.any():
        work.loc[too_long, text_column] = (
            work.loc[too_long, text_column]
            .astype(str)
            .str.split()
            .str[:max_words]
            .str.join(" ")
        )

    work = work[work["_word_count"] >= min_words].copy()
    work["clean_narrative"] = work[text_column].map(clean_text)
    work = work[work["clean_narrative"].str.len() > 0].copy()
    work = work.drop(columns=["_word_count"])
    return work.reset_index(drop=True)


def preprocess_texts(texts: Iterable[object]) -> list[str]:
    """Clean an iterable of raw texts (used by the prediction page)."""
    return [clean_text(text) for text in texts]
