# Design Document: Audio Anomaly Detection (MVP)

## Overview

A lightweight Python prototype that processes audio files, transcribes them with OpenAI Whisper, and detects anomalies using basic text features and Isolation Forest. Results display in a Streamlit dashboard with keyword alerts and anomaly warnings.

The pipeline is linear: Audio Input → Transcription → Text Processing → Detection → UI. Each stage is a simple module — just functions and dataclasses wired together in a Streamlit app. The Isolation Forest is trained inline on hardcoded normal samples — no model files, no serialization.

## Components and Interfaces

### Component 1: Transcription (`app/transcription.py`)

**Purpose**: Load Whisper and transcribe audio to text.

```python
import whisper

_model = None

def load_model(size: str = "base") -> None:
    global _model
    _model = whisper.load_model(size)

def transcribe(file_path: str) -> str:
    if _model is None:
        load_model()
    result = _model.transcribe(file_path)
    return result.get("text", "").strip()
```

### Component 2: Text Processing (`app/processing.py`)

**Purpose**: Clean text and extract three numerical features: text length, exclamation count, keyword count.

```python
import re

def clean_text(raw: str) -> str:
    return re.sub(r'\s+', ' ', raw.lower()).strip()

def extract_features(text: str, keywords: list[str]) -> list[float]:
    keyword_count = sum(1 for w in text.split() if w in keywords)
    return [
        float(len(text)),
        float(text.count("!")),
        float(keyword_count),
    ]
```

The feature vector is always exactly 3 elements: `[text_length, exclamation_count, keyword_count]`.

### Component 3: Detection (`app/detector.py`)

**Purpose**: Keyword matching + inline Isolation Forest anomaly detection. No model files — trains on hardcoded normal samples at startup.

```python
from dataclasses import dataclass
import numpy as np
from sklearn.ensemble import IsolationForest

@dataclass
class DetectionResult:
    transcript: str
    keywords_detected: list[str]
    is_anomaly: bool

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
    model = IsolationForest(contamination=contamination, random_state=42)
    model.fit(np.array(NORMAL_SAMPLES))
    return model

def scan_keywords(text: str, keywords: list[str]) -> list[str]:
    return [kw for kw in keywords if kw in text]

def detect(text: str, features: list[float], keywords: list[str],
           model: IsolationForest) -> DetectionResult:
    kw_found = scan_keywords(text, keywords)
    arr = np.array([features])
    is_anomaly_model = model.predict(arr)[0] == -1
    return DetectionResult(
        transcript=text,
        keywords_detected=kw_found,
        is_anomaly=bool(is_anomaly_model) or len(kw_found) > 0,
    )
```

### Component 4: Streamlit App (`app/app.py`)

**Purpose**: Upload audio, transcribe, detect, display transcript and alerts. No history, no logs, no extra widgets.

```python
import tempfile, os
import streamlit as st
from app.transcription import transcribe, load_model
from app.processing import clean_text, extract_features
from app.detector import detect, build_model
from app.config import CONFIG

def main():
    st.title("Audio Anomaly Detection")
    load_model(CONFIG["whisper_model"])
    model = build_model(CONFIG["contamination"])

    uploaded = st.file_uploader("Upload audio", type=["wav", "mp3"])
    if uploaded:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded.name)[1]) as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name
        try:
            transcript = transcribe(tmp_path)
            if not transcript:
                st.info("No speech detected.")
            else:
                cleaned = clean_text(transcript)
                features = extract_features(cleaned, CONFIG["keywords"])
                result = detect(cleaned, features, CONFIG["keywords"], model)

                st.subheader("Transcript")
                st.write(result.transcript)

                if result.keywords_detected:
                    st.error(f"Keywords detected: {', '.join(result.keywords_detected)}")
                if result.is_anomaly:
                    st.warning("Anomaly detected!")
                else:
                    st.success("No anomalies detected.")
        finally:
            os.unlink(tmp_path)

if __name__ == "__main__":
    main()
```

### Component 5: Config (`app/config.py`)

```python
CONFIG = {
    "whisper_model": "base",
    "keywords": ["help", "emergency", "fire", "attack", "danger"],
    "contamination": 0.1,
    "supported_formats": [".wav", ".mp3"],
}
```

## Data Models

### DetectionResult

```python
@dataclass
class DetectionResult:
    transcript: str              # Cleaned transcript
    keywords_detected: list[str] # Matched keywords
    is_anomaly: bool             # Anomaly flag
```

## Example Usage

```python
from app.transcription import transcribe
from app.processing import clean_text, extract_features
from app.detector import detect, build_model
from app.config import CONFIG

model = build_model(CONFIG["contamination"])
transcript = transcribe("data/sample.wav")
cleaned = clean_text(transcript)
features = extract_features(cleaned, CONFIG["keywords"])
result = detect(cleaned, features, CONFIG["keywords"], model)
# DetectionResult(
#     transcript="help there is a fire in the building",
#     keywords_detected=["help", "fire"],
#     is_anomaly=True,
# )
```

## Correctness Properties

### Property 1: Cleaning never grows text

*For any* string `t`, `len(clean_text(t))` SHALL be less than or equal to `len(t)`.

### Property 2: Cleaning produces normalized output

*For any* string `t`, `clean_text(t)` SHALL return a lowercase string with no leading/trailing whitespace and no consecutive whitespace characters.

### Property 3: Feature vector dimension

*For any* cleaned text `t` and keyword list `K`, `extract_features(t, K)` SHALL return a list of exactly 3 float values.

### Property 4: Feature text_length consistency

*For any* cleaned text `t` and keyword list `K`, `extract_features(t, K)[0]` SHALL equal `float(len(t))`.

### Property 5: Feature non-negativity

*For any* cleaned text `t` and keyword list `K`, all elements of `extract_features(t, K)` SHALL be non-negative.

### Property 6: Keyword scan correctness

*For any* text `t` and keyword list `K`, every element in `scan_keywords(t, K)` SHALL appear in both `t` and `K`, and if no keyword from `K` appears in `t`, the result SHALL be an empty list.

### Property 7: Anomaly flag is OR of signals

*For any* detection run, `is_anomaly` SHALL be True if either keywords were detected OR the Isolation Forest predicted an anomaly.

### Property 8: Detection result transcript preservation

*For any* cleaned text `t` passed to `detect()`, `DetectionResult.transcript` SHALL equal `t`.

## Error Handling

| Scenario | Response |
|----------|----------|
| Unsupported file format | UI shows error, re-prompts upload |
| Whisper fails to load | UI shows "Transcription unavailable" |
| No speech in audio | UI shows "No speech detected" |
| Corrupted audio | UI shows error message |

## Testing Strategy

- **Unit tests** (`pytest`): One test file per module — processing and detector
- **Property tests** (`hypothesis`): `clean_text` length, feature vector dimension/bounds, `scan_keywords` subset
- **Integration**: End-to-end pipeline test with a sample audio file

## Dependencies

| Library | Purpose |
|---------|---------|
| `openai-whisper` | Speech-to-text |
| `torch` | Whisper backend |
| `scikit-learn` | Isolation Forest |
| `streamlit` | Dashboard UI |
| `numpy` | Numerical ops |
| `hypothesis` | Property-based testing |
| `pytest` | Test framework |
