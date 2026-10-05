"""Default feature extractor — implements 6 features from spec Section 5.3."""
from __future__ import annotations

import math
import os
import re
from typing import Optional

# Prevent Hugging Face from hanging with network retries
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

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
    "suggests", "indicates", "potential", "potentially", "hypothetically",
    "assumed", "presumed", "estimated", "purportedly", "supposedly",
    "allegedly", "tentatively", "generally", "typically", "often",
    "may be", "could have", "might have", "would seem", "appears", "seems",
]

CONFIDENCE_PHRASES = [
    "definitely", "certainly", "absolutely", "undoubtedly", "clearly",
    "obviously", "without a doubt", "for sure", "guaranteed", "always",
    "never", "must be", "exactly", "precisely", "unquestionably",
    "indisputably", "no question", "100%", "every single",
    "discovered", "confirms", "confirmed", "revealed", "established",
    "proves", "proven", "concluded", "demonstrates", "demonstrated",
    "undeniable", "completely", "entirely", "strictly", "officially",
    "unambiguously", "conclusively", "recorded", "documented",
]

CITATION_GESTURES = [
    "according to", "studies show", "research suggests",
    "it has been reported", "experts say", "sources indicate",
    "evidence suggests", "data shows", "findings indicate",
    "as reported", "reportedly", "it is said",
    "some say", "many believe", "it is known",
    "records show", "records indicate", "history shows", "historians say",
    "scientists report", "researchers found", "documents state", "treaty states",
    "reported by", "stated by", "claimed by", "noted by", "cited in",
    "referenced by", "discovered by", "published by", "announced by",
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


SOURCE_ENTITY_LABELS = {"PERSON", "ORG", "GPE", "LAW", "WORK_OF_ART"}


def _citation_vagueness(text: str) -> float:
    """C — source-gesturing phrases not followed by a named source within N tokens."""
    nlp = _get_nlp()
    doc = nlp(text)
    text_lower = text.lower()

    entity_starts = {ent.start_char for ent in doc.ents if ent.label_ in SOURCE_ENTITY_LABELS}

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

    entity_starts = {ent.start_char for ent in doc.ents if ent.label_ in SOURCE_ENTITY_LABELS}
    evidence_count = 0.0
    for sent in sentences:
        s_text = sent.text.lower()
        has_vague_gesture = False
        for g in CITATION_GESTURES:
            idx = s_text.find(g)
            if idx != -1:
                end_win = idx + len(g) + 50
                if not any(idx + len(g) <= es - sent.start_char < end_win for es in entity_starts):
                    has_vague_gesture = True
                    break

        if not has_vague_gesture:
            source_entities = [ent for ent in sent.ents if ent.label_ in SOURCE_ENTITY_LABELS]
            has_subj = any(t.dep_ in ("nsubj", "nsubjpass") for t in sent)
            has_verb = any(t.pos_ == "VERB" for t in sent)
            has_obj = any(t.dep_ in ("dobj", "pobj", "attr", "acomp") for t in sent)

            # Grounded factual assertion: requires verifiable source entity or attribution
            # combined with complete relational triple (subj + verb + obj)
            if source_entities and has_subj and has_verb and has_obj:
                evidence_count += 1.0
            elif len(sent.ents) >= 2 and any(ent.label_ in ("DATE", "CARDINAL", "TIME") for ent in sent.ents) and has_verb and has_subj:
                # Factual claim with specific metrics/dates but without grounded source attribution
                evidence_count += 0.5

    return min(evidence_count / len(sentences), 1.0)


def _semantic_drift(text: str) -> float:
    """D — embedding-similarity variance across sentences."""
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
    if len(sentences) < 2:
        return 0.0

    try:
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
        if len(pairwise) == 1:
            return float(1.0 - pairwise[0]) * 0.1

        return float(np.var(pairwise))
    except Exception:
        # Fallback for offline / un-downloaded embedding model:
        # compute word-set Jaccard/overlap variance across sentences
        token_sets = [set(re.findall(r'\w+', s.lower())) for s in sentences]
        overlaps = []
        for i in range(len(token_sets)):
            for j in range(i + 1, len(token_sets)):
                s1, s2 = token_sets[i], token_sets[j]
                if not s1 or not s2:
                    overlaps.append(0.0)
                else:
                    jaccard = len(s1 & s2) / max(len(s1 | s2), 1)
                    overlaps.append(jaccard)
        if not overlaps:
            return 0.0
        if len(overlaps) == 1:
            return float(1.0 - overlaps[0]) * 0.1
        return float(np.var(overlaps)) + float(1.0 - np.mean(overlaps)) * 0.05


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
