"""Default feature extractor — implements 6 features from spec Section 5.3."""
from __future__ import annotations

import math
import re
from typing import Optional

import numpy as np

# Lazy imports for spacy and sentence-transformers (heavy dependencies)
_nlp = None
_sentence_model = None

HEDGE_PHRASES = [
    "might", "maybe", "perhaps", "possibly", "could be", "I think",
    "I believe", "it seems", "apparently", "arguably", "likely",
    "unlikely", "probably", "not sure", "uncertain", "roughly",
    "approximately", "somewhat", "sort of", "kind of", "in a way",
    "to some extent", "as far as I know", "I'm not certain",
]

CONFIDENCE_PHRASES = [
    "definitely", "certainly", "absolutely", "undoubtedly", "clearly",
    "obviously", "without a doubt", "for sure", "guaranteed", "always",
    "never", "must be", "exactly", "precisely", "unquestionably",
    "indisputably", "no question", "100%", "every single",
]

CITATION_GESTURES = [
    "according to", "studies show", "research suggests",
    "it has been reported", "experts say", "sources indicate",
    "evidence suggests", "data shows", "findings indicate",
    "as reported", "reportedly", "it is said",
    "some say", "many believe", "it is known",
]


def _get_nlp():
    global _nlp
    if _nlp is None:
        import spacy
        _nlp = spacy.load("en_core_web_sm")
    return _nlp


def _get_sentence_model():
    global _sentence_model
    if _sentence_model is None:
        from sentence_transformers import SentenceTransformer
        _sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _sentence_model


def _count_words(text: str) -> int:
    return max(len(text.split()), 1)


def _hedge_density(text: str) -> float:
    """H — hedge-phrase count per 100 words."""
    text_lower = text.lower()
    count = sum(1 for phrase in HEDGE_PHRASES if phrase in text_lower)
    words = _count_words(text)
    return (count / words) * 100


def _specificity(text: str) -> float:
    """S — ratio of named entities + numbers/dates to total tokens."""
    nlp = _get_nlp()
    doc = nlp(text)
    if len(doc) == 0:
        return 0.0

    entity_tokens = set()
    for ent in doc.ents:
        for token in ent:
            entity_tokens.add(token.i)

    # Also count standalone numbers/dates not captured by NER
    number_pattern = re.compile(r'\b\d+[\d,./]*\b')
    number_matches = number_pattern.findall(text)
    number_count = len(number_matches)

    specific_count = len(entity_tokens) + number_count
    return min(specific_count / len(doc), 1.0)


def _citation_vagueness(text: str) -> float:
    """C — source-gesturing phrases not followed by a named source within N tokens."""
    nlp = _get_nlp()
    doc = nlp(text)
    text_lower = text.lower()

    entity_starts = {ent.start_char for ent in doc.ents}

    gesture_count = 0
    vague_count = 0
    window = 50  # characters to check after gesture

    for gesture in CITATION_GESTURES:
        start = 0
        while True:
            idx = text_lower.find(gesture, start)
            if idx == -1:
                break
            gesture_count += 1

            # Check if a named entity appears within window after the gesture
            end_window = idx + len(gesture) + window
            has_source = any(
                idx + len(gesture) <= es < end_window
                for es in entity_starts
            )
            if not has_source:
                vague_count += 1

            start = idx + 1

    if gesture_count == 0:
        return 0.0
    return vague_count / gesture_count


def _evidence_density(text: str) -> float:
    """E — concrete, checkable factual assertions per sentence."""
    nlp = _get_nlp()
    doc = nlp(text)

    sentences = list(doc.sents)
    if not sentences:
        return 0.0

    evidence_count = 0
    for sent in sentences:
        entities = [ent for ent in sent.ents]
        # A sentence with >=2 entities or an entity + a verb = evidence
        if len(entities) >= 2:
            evidence_count += 1
        elif len(entities) >= 1:
            has_verb = any(t.pos_ == "VERB" for t in sent)
            if has_verb:
                evidence_count += 1

    return evidence_count / len(sentences)


def _semantic_drift(text: str) -> float:
    """D — embedding-similarity variance across sentences."""
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
    if len(sentences) < 2:
        return 0.0

    model = _get_sentence_model()
    embeddings = model.encode(sentences, show_progress_bar=False)

    # Pairwise cosine similarities
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    normalized = embeddings / norms

    similarities = normalized @ normalized.T

    # Extract upper triangle (excluding diagonal)
    n = len(sentences)
    upper_indices = np.triu_indices(n, k=1)
    pairwise = similarities[upper_indices]

    if len(pairwise) == 0:
        return 0.0

    return float(np.var(pairwise))


def _confidence_density(text: str) -> float:
    """M — absolute/confidence words per 100 words."""
    text_lower = text.lower()
    count = sum(1 for phrase in CONFIDENCE_PHRASES if phrase in text_lower)
    words = _count_words(text)
    return (count / words) * 100


class DefaultFeatureExtractor:
    """Implements the 6 features from spec Section 5.3.

    H — hedge density (lexicon match)
    S — specificity (spaCy NER + regex)
    C — citation vagueness (pattern match + proximity)
    E — evidence density (entity + relation co-occurrence)
    D — semantic/entity drift (sentence embeddings)
    M — confidence-marker density (lexicon match)
    """

    name = "default"

    def extract(self, answer: str) -> dict[str, float]:
        if not answer.strip():
            return {"H": 0.0, "S": 0.0, "C": 0.0, "E": 0.0, "D": 0.0, "M": 0.0}

        return {
            "H": _hedge_density(answer),
            "S": _specificity(answer),
            "C": _citation_vagueness(answer),
            "E": _evidence_density(answer),
            "D": _semantic_drift(answer),
            "M": _confidence_density(answer),
        }
