"""Detection module for audio anomaly detection."""

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import IsolationForest


@dataclass
class DetectionResult:
    """Result of anomaly detection on a transcript."""

    transcript: str
    keywords_detected: list[str]
    is_anomaly: bool


def scan_keywords(text: str, keywords: list[str]) -> list[str]:
    """Return keywords found in the text (case-insensitive).

    Args:
        text: The input text to scan.
        keywords: List of keywords to look for.

    Returns:
        List of keywords that appear in the text.
    """
    lower_text = text.lower()
    return [kw for kw in keywords if kw in lower_text]


NORMAL_SAMPLES = [
    [50.0, 0.0, 0.0],
    [80.0, 0.0, 0.0],
    [60.0, 1.0, 0.0],
    [70.0, 0.0, 0.0],
    [90.0, 1.0, 0.0],
    [55.0, 0.0, 0.0],
    [65.0, 0.0, 0.0],
    [75.0, 1.0, 0.0],
    [85.0, 0.0, 0.0],
    [45.0, 0.0, 0.0],
]


def build_model(contamination: float = 0.1) -> IsolationForest:
    """Create and fit an IsolationForest on NORMAL_SAMPLES.

    Args:
        contamination: The proportion of outliers in the data set.

    Returns:
        A fitted IsolationForest model.
    """
    model = IsolationForest(contamination=contamination, random_state=42)
    model.fit(np.array(NORMAL_SAMPLES))
    return model


def detect(
    text: str,
    features: list[float],
    keywords: list[str],
    model: IsolationForest,
) -> DetectionResult:
    """Combine keyword scan and model prediction into a DetectionResult.

    Args:
        text: The cleaned transcript text.
        features: A 3-element feature vector [text_length, exclamation_count, keyword_count].
        keywords: List of emergency keywords to scan for.
        model: A fitted IsolationForest model.

    Returns:
        A DetectionResult with combined anomaly detection results.

    Raises:
        ValueError: If features does not contain exactly 3 elements.
    """
    if len(features) != 3:
        raise ValueError("features must contain exactly 3 elements")

    keywords_found = scan_keywords(text, keywords)
    model_anomaly = model.predict(np.array([features]))[0] == -1
    is_anomaly = bool(model_anomaly) or len(keywords_found) > 0

    return DetectionResult(
        transcript=text,
        keywords_detected=keywords_found,
        is_anomaly=is_anomaly,
    )
