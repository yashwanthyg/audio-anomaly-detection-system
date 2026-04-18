"""Tests for the text processing module."""

import pytest

from app.processing import clean_text, extract_features


class TestCleanText:
    """Tests for clean_text function."""

    def test_basic_lowercase_conversion(self):
        assert clean_text("Hello World") == "hello world"

    def test_whitespace_collapse(self):
        assert clean_text("hello   world") == "hello world"

    def test_strip_leading_trailing_spaces(self):
        assert clean_text("  hello  ") == "hello"

    def test_tab_and_newline_handling(self):
        assert clean_text("hello\t\nworld") == "hello world"

    def test_empty_string_returns_empty(self):
        assert clean_text("") == ""

    def test_already_clean_text_unchanged(self):
        assert clean_text("already clean") == "already clean"

    def test_mixed_case_with_extra_whitespace(self):
        assert clean_text("  HeLLo   WoRLd  ") == "hello world"

    def test_multiple_whitespace_types(self):
        assert clean_text("a \t b \n c") == "a b c"


class TestExtractFeatures:
    """Tests for extract_features function."""

    def test_basic_text_no_keywords(self):
        result = extract_features("hello world", [])
        assert result == [11.0, 0.0, 0.0]

    def test_text_with_exclamation_marks(self):
        result = extract_features("help! fire!", ["help!", "fire!"])
        # len("help! fire!") = 11, exclamation count = 2
        assert result[0] == 11.0
        assert result[1] == 2.0

    def test_text_with_keywords(self):
        result = extract_features("help there is a fire", ["help", "fire"])
        assert result[2] == 2.0

    def test_empty_text(self):
        assert extract_features("", []) == [0.0, 0.0, 0.0]

    def test_empty_text_with_keywords(self):
        assert extract_features("", ["help", "fire"]) == [0.0, 0.0, 0.0]

    def test_long_text_length(self):
        text = "a" * 500
        result = extract_features(text, [])
        assert result[0] == 500.0

    def test_no_exclamation_marks(self):
        result = extract_features("hello world", [])
        assert result[1] == 0.0

    def test_multiple_keywords_matched(self):
        result = extract_features("help emergency fire", ["help", "emergency", "fire"])
        assert result[2] == 3.0

    def test_no_keywords_matched(self):
        result = extract_features("hello world", ["help", "fire"])
        assert result[2] == 0.0

    def test_vector_always_three_elements(self):
        result = extract_features("any text here", ["any"])
        assert len(result) == 3

    def test_all_values_are_floats(self):
        result = extract_features("test!", ["test!"])
        assert all(isinstance(v, float) for v in result)

    def test_all_values_non_negative(self):
        result = extract_features("hello world!", ["hello"])
        assert all(v >= 0.0 for v in result)
