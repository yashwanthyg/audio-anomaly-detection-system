"""Tests for the transcription module."""

import sys
from unittest.mock import MagicMock

# Mock whisper before importing the transcription module
mock_whisper = MagicMock()
sys.modules["whisper"] = mock_whisper

import pytest

import app.transcription as transcription_mod
from app.transcription import load_model, transcribe


class TestLoadModel:
    """Tests for load_model function."""

    def setup_method(self):
        transcription_mod._model = None
        mock_whisper.reset_mock()

    def test_load_model_is_callable(self):
        assert callable(load_model)

    def test_load_model_sets_module_variable(self):
        mock_whisper.load_model.return_value = MagicMock()

        load_model("base")

        mock_whisper.load_model.assert_called_once_with("base")
        assert transcription_mod._model is not None

    def test_load_model_raises_runtime_error_on_failure(self):
        mock_whisper.load_model.side_effect = Exception("download failed")

        with pytest.raises(RuntimeError, match="Failed to load Whisper model"):
            load_model("base")

        mock_whisper.load_model.side_effect = None


class TestTranscribe:
    """Tests for transcribe function."""

    def setup_method(self):
        transcription_mod._model = None
        mock_whisper.reset_mock()

    def test_transcribe_is_callable(self):
        assert callable(transcribe)

    def test_transcribe_raises_file_not_found(self):
        with pytest.raises(FileNotFoundError, match="Audio file not found"):
            transcribe("/nonexistent/audio.wav")

    def test_transcribe_raises_value_error_for_unsupported_format(self, tmp_path):
        bad_file = tmp_path / "audio.txt"
        bad_file.write_text("not audio")

        with pytest.raises(ValueError, match="Unsupported audio format"):
            transcribe(str(bad_file))

    def test_transcribe_auto_loads_model(self, tmp_path):
        wav_file = tmp_path / "test.wav"
        wav_file.write_bytes(b"\x00" * 100)

        mock_model = MagicMock()
        mock_model.transcribe.return_value = {"text": " hello world "}
        mock_whisper.load_model.return_value = mock_model

        result = transcribe(str(wav_file))

        mock_whisper.load_model.assert_called_once()
        assert result == "hello world"

    def test_transcribe_returns_stripped_text(self, tmp_path):
        wav_file = tmp_path / "test.wav"
        wav_file.write_bytes(b"\x00" * 100)

        mock_model = MagicMock()
        mock_model.transcribe.return_value = {"text": "  some speech  "}
        transcription_mod._model = mock_model

        result = transcribe(str(wav_file))

        assert result == "some speech"

    def test_transcribe_returns_empty_for_no_text_key(self, tmp_path):
        wav_file = tmp_path / "test.mp3"
        wav_file.write_bytes(b"\x00" * 100)

        mock_model = MagicMock()
        mock_model.transcribe.return_value = {}
        transcription_mod._model = mock_model

        result = transcribe(str(wav_file))

        assert result == ""

    def test_transcribe_raises_runtime_error_on_failure(self, tmp_path):
        wav_file = tmp_path / "test.wav"
        wav_file.write_bytes(b"\x00" * 100)

        mock_model = MagicMock()
        mock_model.transcribe.side_effect = Exception("decode error")
        transcription_mod._model = mock_model

        with pytest.raises(RuntimeError, match="Transcription failed"):
            transcribe(str(wav_file))
