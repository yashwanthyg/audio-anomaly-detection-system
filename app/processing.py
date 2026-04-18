"""Text processing module for audio anomaly detection."""

import re


def clean_text(raw: str) -> str:
    """Clean and normalize raw transcript text.

    Converts to lowercase, collapses all consecutive whitespace to a single
    space, and strips leading/trailing whitespace.

    Args:
        raw: The raw transcript text to clean.

    Returns:
        Cleaned, normalized text string.
    """
    return re.sub(r'\s+', ' ', raw.lower()).strip()


def extract_features(text: str, keywords: list[str]) -> list[float]:
    """Extract a 3-element feature vector from cleaned text.

    Args:
        text: Cleaned text to extract features from.
        keywords: List of keywords to count in the text.

    Returns:
        A list of exactly 3 floats: [text_length, exclamation_count, keyword_count].
    """
    keyword_count = sum(1 for w in text.split() if w in keywords)
    return [
        float(len(text)),
        float(text.count("!")),
        float(keyword_count),
    ]
