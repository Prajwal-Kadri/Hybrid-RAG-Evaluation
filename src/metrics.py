"""Metrics. Keep these names and signatures: tests/test_metrics.py imports them."""

from __future__ import annotations
import re
import string
from eval_generation import _distinct

_PUNCT_RE = re.compile(f"[{re.escape(string.punctuation)}]")

def normalize_answer(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace (tabs, newlines, spaces)."""
    text = text.lower()
    text = _PUNCT_RE.sub("", text)
    text = " ".join(text.split())
    return text
    # raise NotImplementedError


def exact_match(prediction: str, gold_answers: list[str]) -> float:
    """Ignore blank gold answers (unanswerable rows store [""]). Then:
    - no gold answers left: 1.0 iff the normalized prediction == "unanswerable"
      (so "Unanswerable." also counts, but have your model write it exactly)
    - otherwise: 1.0 iff some normalized gold answer appears in the normalized
      prediction as whole words ("6" matches "6 people", not "1960").
    """
    normalized_prediction = normalize_answer(prediction or "")
    gold_answers = [gold for gold in (gold_answers or []) if normalize_answer(gold)]
    if not gold_answers:
        # Unanswerable question: correct only if the system abstains.
        return 1.0 if normalized_prediction == "unanswerable" else 0.0
    padded_prediction = f" {normalized_prediction} "
    for gold in gold_answers:
        # Whole-word containment: gold "6" matches "6 people" but not "1960".
        if f" {normalize_answer(gold)} " in padded_prediction:
            return 1.0
    return 0.0
    # raise NotImplementedError


def source_precision(predicted_sources: list[str], gold_titles: list[str]) -> float:
    """Fraction of distinct predicted titles that are gold. 0.0 if none predicted."""
    predicted = _distinct(predicted_sources)
    if not predicted:
        return 0.0
    gold_set = set(_distinct(gold_titles))
    return len([title for title in predicted if title in gold_set]) / len(predicted)
   


def source_recall(predicted_sources: list[str], gold_titles: list[str]) -> float:
    """Fraction of distinct gold titles that were predicted. 0.0 if no gold titles."""
    gold = _distinct(gold_titles)
    if not gold:
        return 0.0
    gold_set = set(gold)
    return len([title for title in gold if title in predicted_sources]) / len(gold)
    # raise NotImplementedError
