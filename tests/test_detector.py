"""Tests for the detection module."""

import numpy as np
import pytest
from sklearn.ensemble import IsolationForest

from app.detector import (
    DetectionResult,
    NORMAL_SAMPLES,
    build_model,
    detect,
    scan_keywords,
)


class TestDetectionResult:
    """Tests for DetectionResult dataclass."""

    def test_dataclass_fields(self):
        result = DetectionResult(
            transcript="hello", keywords_detected=["help"], is_anomaly=True
        )
        assert result.transcript == "hello"
        assert result.keywords_detected == ["help"]
        assert result.is_anomaly is True

    def test_dataclass_has_exactly_three_fields(self):
        fields = DetectionResult.__dataclass_fields__
        assert len(fields) == 3
        assert set(fields.keys()) == {"transcript", "keywords_detected", "is_anomaly"}


class TestScanKeywords:
    """Tests for scan_keywords function."""

    def test_basic_keyword_found(self):
        result = scan_keywords("please help me", ["help", "fire"])
        assert result == ["help"]

    def test_multiple_keywords_found(self):
        result = scan_keywords("help there is a fire", ["help", "fire", "danger"])
        assert result == ["help", "fire"]

    def test_no_keywords_found(self):
        result = scan_keywords("the weather is nice today", ["help", "fire", "danger"])
        assert result == []

    def test_empty_text(self):
        result = scan_keywords("", ["help", "fire"])
        assert result == []

    def test_empty_keywords_list(self):
        result = scan_keywords("help there is a fire", [])
        assert result == []

    def test_case_insensitive_uppercase_text(self):
        result = scan_keywords("PLEASE HELP ME", ["help"])
        assert result == ["help"]

    def test_case_insensitive_mixed_case_text(self):
        result = scan_keywords("Help Me Please", ["help"])
        assert result == ["help"]

    def test_mixed_case_text_with_multiple_keywords(self):
        result = scan_keywords("FIRE! Someone Help!", ["help", "fire", "danger"])
        assert result == ["help", "fire"]

    def test_keyword_as_substring_matches(self):
        # "fire" in "fireplace" — expected behavior per design using `kw in text`
        result = scan_keywords("the fireplace is warm", ["fire"])
        assert result == ["fire"]

    def test_all_keywords_present(self):
        keywords = ["help", "emergency", "fire", "attack", "danger"]
        text = "help emergency fire attack danger"
        result = scan_keywords(text, keywords)
        assert result == keywords

    def test_duplicate_keyword_in_text_returns_once(self):
        result = scan_keywords("help me help me help", ["help"])
        assert result == ["help"]

    def test_both_empty(self):
        result = scan_keywords("", [])
        assert result == []


class TestNormalSamples:
    def test_normal_samples_has_10_vectors(self):
        assert len(NORMAL_SAMPLES) == 10

    def test_each_vector_has_3_elements(self):
        for sample in NORMAL_SAMPLES:
            assert len(sample) == 3

    def test_all_values_are_floats(self):
        for sample in NORMAL_SAMPLES:
            assert all(isinstance(v, float) for v in sample)

    def test_all_values_non_negative(self):
        for sample in NORMAL_SAMPLES:
            assert all(v >= 0.0 for v in sample)


class TestBuildModel:
    def test_returns_isolation_forest(self):
        model = build_model()
        assert isinstance(model, IsolationForest)

    def test_model_is_fitted(self):
        model = build_model()
        assert hasattr(model, 'estimators_')

    def test_custom_contamination(self):
        model = build_model(contamination=0.2)
        assert model.contamination == 0.2

    def test_deterministic_with_random_state(self):
        model1 = build_model()
        model2 = build_model()
        test_vec = np.array([[50.0, 0.0, 0.0]])
        assert model1.predict(test_vec)[0] == model2.predict(test_vec)[0]


class TestDetect:
    @pytest.fixture
    def model(self):
        return build_model()

    def test_normal_text_no_keywords(self, model):
        result = detect("the weather is nice today", [60.0, 0.0, 0.0], ["help", "fire"], model)
        assert result.transcript == "the weather is nice today"
        assert result.keywords_detected == []
        assert result.is_anomaly is False

    def test_text_with_keywords_is_anomaly(self, model):
        result = detect("help there is a fire", [20.0, 0.0, 2.0], ["help", "fire"], model)
        assert "help" in result.keywords_detected
        assert "fire" in result.keywords_detected
        assert result.is_anomaly is True

    def test_anomalous_features_no_keywords(self, model):
        result = detect("oh no", [5.0, 10.0, 0.0], ["help", "fire"], model)
        assert result.keywords_detected == []
        assert result.is_anomaly is True

    def test_transcript_preservation(self, model):
        text = "some transcript text"
        result = detect(text, [50.0, 0.0, 0.0], [], model)
        assert result.transcript == text

    def test_result_is_detection_result(self, model):
        result = detect("hello", [5.0, 0.0, 0.0], [], model)
        assert isinstance(result, DetectionResult)

    def test_is_anomaly_is_boolean(self, model):
        result = detect("hello", [50.0, 0.0, 0.0], [], model)
        assert isinstance(result.is_anomaly, bool)

    def test_invalid_feature_vector_size(self, model):
        with pytest.raises(ValueError, match="exactly 3"):
            detect("hello", [50.0, 0.0], [], model)

    def test_invalid_feature_vector_too_long(self, model):
        with pytest.raises(ValueError, match="exactly 3"):
            detect("hello", [50.0, 0.0, 0.0, 1.0], [], model)

    def test_keywords_or_model_anomaly(self, model):
        result = detect("help me please", [60.0, 0.0, 1.0], ["help"], model)
        assert result.is_anomaly is True
