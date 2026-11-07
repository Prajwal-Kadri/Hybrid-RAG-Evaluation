"""Point-wise answer re-ranking. Each function returns the candidates sorted
best first, breaking ties deterministically (stable sort). contexts are the
chunks the generator saw, not a candidate's "sources"."""

from __future__ import annotations

from llm_client import score_sequence_likelihood
from metrics import normalize_answer

def likelihood_rerank(
    candidates: list[dict],
    question: str,
    contexts: list[dict],
    prompt: str,
    length_normalize: bool = True,
    model_name: str | None = None,
) -> list[dict]:
    """Sort by score_sequence_likelihood(prompt, answer_candidate,
    length_normalize, model_name=model_name), highest first. model_name is
    config["answer_rerank"]["likelihood_model_name"], which may differ from
    the generation model."""
    return sorted(
        candidates,
        key=lambda c: score_sequence_likelihood(
            prompt, c["answer_candidate"], length_normalize, model_name=model_name
        ),
        reverse=True,
    )
    # raise NotImplementedError


def numsources_rerank(candidates: list[dict], question: str, contexts: list[dict]) -> list[dict]:
    """Sort by how many contexts contain the candidate (both normalized with
    metrics.normalize_answer), highest first. "unanswerable" or empty counts 0.
    You may use question-word overlap to break ties; say so in the report."""
    normalized_contexts = [
        normalize_answer(context["text"])
        for context in contexts
    ]
    scores = []
    for candidate in candidates:
        normalized_candidate = normalize_answer(candidate["answer_candidate"])
        if normalized_candidate == "unanswerable" or not normalized_candidate:
            scores.append(0)
        else:
            count = sum(
                1 for context in normalized_contexts
                if normalized_candidate in context
            )
            scores.append(count)
    return sorted(
        candidates,
        key=lambda c: scores[candidates.index(c)],
        reverse=True,
    )
    # raise NotImplementedError
