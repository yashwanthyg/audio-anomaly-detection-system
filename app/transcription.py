"""Transcription module: loads Whisper and transcribes audio to text."""

import os

import whisper

from config import CONFIG

_model = None


def load_model(size: str = "base") -> None:
    """Load a Whisper model into the module-level variable."""
    global _model
    try:
        _model = whisper.load_model(size)
    except Exception as e:
        raise RuntimeError(f"Failed to load Whisper model '{size}': {e}") from e


def transcribe(file_path: str) -> str:
    """Transcribe an audio file to text.

    Auto-loads the model if it hasn't been loaded yet.
    Validates the file exists and has a supported format before transcribing.
    """
    global _model

    # Validate file exists
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    # Validate supported format
    ext = os.path.splitext(file_path)[1].lower()
    if ext not in CONFIG["supported_formats"]:
        raise ValueError(
            f"Unsupported audio format '{ext}'. "
            f"Supported formats: {CONFIG['supported_formats']}"
        )

    # Auto-load model if needed
    if _model is None:
        load_model()

    try:
        result = _model.transcribe(file_path)
        return result.get("text", "").strip()
    except Exception as e:
        raise RuntimeError(f"Transcription failed for '{file_path}': {e}") from e
