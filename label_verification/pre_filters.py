import re

import pandas as pd
from nltk.corpus import stopwords
from rapidfuzz import fuzz

STOP_WORDS = set(stopwords.words("english"))
NEGATION_PATTERN = re.compile(r"\b(not|without|no|except|never|none)\b")
PUNCTUATION_PATTERN = re.compile(r"[-_]")
STRIP_PUNCTUATION_PATTERN = re.compile(r"[^\w\s]")


def negation_pre_filter(query: str) -> bool:
    """
    Pre-filtering function to check for negation terms in the query. If any negation terms are found, the function returns True, indicating that the query should be flagged for downstream LLM processing.

    Parameters
    ----------
    query : str
        The shopping query text to be checked for negation terms.

    Returns
    -------
    bool
        True if any negation terms are found in the query.
    """
    return NEGATION_PATTERN.search(str(query).lower()) is not None


def fuzzy_token_set_pre_filter(query, product, threshold=100) -> bool:
    """
    Pre-filtering function that uses fuzzy string matching to compare the query and product name. If the token_set_ratio score is above the specified threshold, the function returns True, indicating that the query should NOT be flagged for downstream LLM processing.

    Parameters
    ----------
    query : str
        The shopping query text to be compared against the product name.
    product : str
        The product name text to be compared against the shopping query.
    threshold : int, optional, default=100
        The minimum token_set_ratio score required for the function to return True

    Returns
    -------
    bool
        True if the token_set_ratio score between the query and product is above the threshold, indicating that
        the query should NOT be flagged for downstream LLM processing.
    """
    # Remove stop words and lowercase for fuzzy matching
    if pd.isna(query) or pd.isna(product):
        return False

    def clean(text: str) -> str:
        text = str(text).lower()
        text = PUNCTUATION_PATTERN.sub(" ", text)  # replace - and _ with space
        text = STRIP_PUNCTUATION_PATTERN.sub("", text)  # remove , . ( ) etc
        return " ".join([w for w in text.split() if w not in STOP_WORDS])

    clean_query = clean(query)
    clean_product = clean(product)
    return fuzz.token_set_ratio(clean_query, clean_product) >= threshold
