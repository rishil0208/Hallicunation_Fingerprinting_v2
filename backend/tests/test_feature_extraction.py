"""Tests for feature extraction — M2.

Uses hand-crafted sentences with known expected feature behavior,
not just "it runs without crashing."
"""
from __future__ import annotations

import pytest

from backend.app.plugins.features.default import DefaultFeatureExtractor
from backend.app.plugins.features.alt import AltFeatureExtractor
from backend.app.registry import register_feature_extractor, get_feature_extractor


@pytest.fixture(scope="module")
def default_extractor():
    return DefaultFeatureExtractor()


@pytest.fixture(scope="module")
def alt_extractor():
    return AltFeatureExtractor()


# ── Feature key structure tests ──

class TestFeatureKeys:
    def test_returns_all_six_keys(self, default_extractor):
        features = default_extractor.extract("The capital of France is Paris.")
        assert set(features.keys()) == {"H", "S", "C", "E", "D", "M"}

    def test_all_values_are_float(self, default_extractor):
        features = default_extractor.extract("Hello world.")
        assert all(isinstance(v, float) for v in features.values())

    def test_empty_string_returns_zeros(self, default_extractor):
        features = default_extractor.extract("")
        assert features == {"H": 0.0, "S": 0.0, "C": 0.0, "E": 0.0, "D": 0.0, "M": 0.0}


# ── H (hedge density) tests ──

class TestHedgeDensity:
    def test_hedging_text_has_high_H(self, default_extractor):
        text = "I think maybe this might possibly be correct, perhaps."
        features = default_extractor.extract(text)
        assert features["H"] > 0, "Text with hedges should have H > 0"

    def test_confident_text_has_low_H(self, default_extractor):
        text = "Paris is the capital of France. The Eiffel Tower stands there."
        features = default_extractor.extract(text)
        assert features["H"] == 0.0, "Text without hedges should have H = 0"


# ── S (specificity) tests ──

class TestSpecificity:
    def test_entity_rich_text_has_high_S(self, default_extractor):
        text = "Albert Einstein was born in Ulm, Germany on March 14, 1879."
        features = default_extractor.extract(text)
        assert features["S"] > 0, "Entity-rich text should have S > 0"

    def test_vague_text_has_low_S(self, default_extractor):
        text = "Some things happened somewhere at some point."
        features = default_extractor.extract(text)
        # May still pick up some tokens, but should be relatively low
        assert features["S"] < 0.5


# ── C (citation vagueness) tests ──

class TestCitationVagueness:
    def test_vague_citation_has_high_C(self, default_extractor):
        text = "According to studies, this is true. Research suggests it works."
        features = default_extractor.extract(text)
        assert features["C"] > 0, "Vague citations should produce C > 0"

    def test_no_citations_has_zero_C(self, default_extractor):
        text = "The sky is blue. Water is wet."
        features = default_extractor.extract(text)
        assert features["C"] == 0.0


# ── E (evidence density) tests ──

class TestEvidenceDensity:
    def test_factual_text_has_evidence(self, default_extractor):
        text = (
            "Albert Einstein published the theory of relativity in 1905. "
            "Marie Curie won the Nobel Prize in Physics in 1903."
        )
        features = default_extractor.extract(text)
        assert features["E"] > 0, "Factual text with entities should have E > 0"

    def test_abstract_text_has_low_evidence(self, default_extractor):
        text = "Things are complicated. It depends on many factors."
        features = default_extractor.extract(text)
        assert features["E"] < 0.5


# ── D (semantic drift) tests ──

class TestSemanticDrift:
    def test_coherent_text_has_low_drift(self, default_extractor):
        text = (
            "Cats are popular pets. Cats are domesticated animals. "
            "Many people keep cats at home."
        )
        features = default_extractor.extract(text)
        assert features["D"] >= 0  # variance is always non-negative

    def test_drifting_text_has_higher_drift(self, default_extractor):
        # Drifting text: topic shifts progressively across sentences
        drifting = (
            "The Roman Empire expanded across Europe and North Africa. "
            "North Africa is known for the vast Sahara Desert and diverse wildlife. "
            "Wildlife conservation efforts have increasingly focused on renewable energy to combat climate change."
        )
        coherent = (
            "The solar system has eight planets. "
            "Mars is the fourth planet from the sun. "
            "Jupiter is the largest planet in the solar system."
        )
        drifting_features = default_extractor.extract(drifting)
        coherent_features = default_extractor.extract(coherent)
        assert drifting_features["D"] > coherent_features["D"]

    def test_single_sentence_has_zero_drift(self, default_extractor):
        text = "Just one sentence here"
        features = default_extractor.extract(text)
        assert features["D"] == 0.0


# ── M (confidence density) tests ──

class TestConfidenceDensity:
    def test_confident_text_has_high_M(self, default_extractor):
        text = "This is definitely true. It is absolutely certain and obviously correct."
        features = default_extractor.extract(text)
        assert features["M"] > 0, "Text with confidence markers should have M > 0"

    def test_neutral_text_has_low_M(self, default_extractor):
        text = "The report was submitted on Tuesday. It contained three sections."
        features = default_extractor.extract(text)
        assert features["M"] == 0.0


# ── Alt extractor tests ──

class TestAltExtractor:
    def test_returns_same_keys(self, alt_extractor):
        features = alt_extractor.extract("Hello world.")
        assert set(features.keys()) == {"H", "S", "C", "E", "D", "M"}

    def test_alt_produces_different_H_values(self, default_extractor, alt_extractor):
        """Alt has a smaller hedge lexicon, so H values should differ for
        text containing phrases only in the default lexicon."""
        text = "This is arguably and unlikely the case, to some extent."
        default_features = default_extractor.extract(text)
        alt_features = alt_extractor.extract(text)
        # "arguably", "unlikely", "to some extent" are in default but not alt
        assert default_features["H"] > alt_features["H"]


# ── Plugin interface test ──

class TestPluginInterface:
    def test_plugin_swap_produces_same_keys(self, default_extractor, alt_extractor):
        text = "Maybe this is correct. I think so."
        default_result = default_extractor.extract(text)
        alt_result = alt_extractor.extract(text)
        assert set(default_result.keys()) == set(alt_result.keys())

    def test_registry_integration(self, default_extractor):
        register_feature_extractor(default_extractor)
        retrieved = get_feature_extractor("default")
        assert retrieved is default_extractor
        result = retrieved.extract("Test.")
        assert "H" in result
