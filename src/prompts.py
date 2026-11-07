"""Generation prompts.

Write three substantially different prompts (structure, output format, or
citation style -- not just wording). Each must include the question and at
least one context, ask for a short answer with citations, and say to answer
exactly "unanswerable" (lowercase, no punctuation) when the contexts don't
support an answer.

After comparing, reorder PROMPT_REGISTRY so "p1" is your best prompt by EM.
"""

from __future__ import annotations


def prompt_1(question: str, contexts: list[dict]) -> str:
    return f"""You are a helpful assistant, provide the answer to the question below {question} using the contexts provided {contexts}. If you are not able to find the answers in the given context, For unsupported candidates, set "answer_candidate" to "unanswerable"
    and "sources" to []. Return 5 valid JSON object with these keys:
    - "answer_candidate": a short answer, without citations in the answer text. Match it to the answer in context.
    - "sources": a list of exact article titles supporting this answer.
    Return a JSON array containing exactly 5 objects.
    Always return a JSON array containing exactly five objects.
    For each unsupported candidate, use:
    {{"answer_candidate": "unanswerable", "sources": []}}
    Return raw JSON without Markdown code fences.
    Include only titles from the supplied contexts that you used.
    List each title at most once.
    Do not include every context title automatically."""
    # raise NotImplementedError


def prompt_2(question: str, contexts: list[dict]) -> str:
    return f"""Provide the answer to the question below {question} using the contexts provided {contexts}. 
    If you are not able to find the answers in the given context, For unsupported candidates, set "answer_candidate" to "unanswerable"
    and "sources" to [].
    Provide the date of the if the questions requests in the following format: MM/DD/YYYY.
    Short answers are returned with just one word, 
    Long answers are returned with one sentence or two.
    Return 5 valid JSON object with these keys:
    - "answer_candidate": a short answer, without citations in the answer text.
    - "sources": a list of exact article titles supporting this answer.
    Return a JSON array containing exactly 5 objects.    For each unsupported candidate, use:
    {{"answer_candidate": "unanswerable", "sources": []}}
    Return raw JSON without Markdown code fences.
    Include only titles from the supplied contexts that you used.
    List each title at most once.
    Do not include every context title automatically.
    """


def prompt_3(question: str, contexts: list[dict]) -> str:
    return f"""You are a assistant who provides the !!exact answer!! to what the user using the contexts provided !!{contexts}!!, provide the answer to the question below {question}. If you are not able to find the answers in the given context, For unsupported candidates, set "answer_candidate" to "unanswerable"
    and "sources" to [].
    Return 5 valid JSON object with these keys:
    Strictly follow the structure below:
    - !!"answer_candidate":  It needs to be a single answer or very very small answer.!! Example: If the context says \"The capital of France is Paris\", and the question is \"What is the capital of France?\", the answer_candidate should be \"Paris\".
    -  "sources": a list of exact article titles from the supplied contexts.
    Example: If a context has title "France" and text
    "The capital of France is Paris", the answer_candidate should be
    "Paris" and the sources should be ["France"].
    List each cited title only once.
    Return a JSON array containing exactly 5 objects.    For each unsupported candidate, use:
    {{"answer_candidate": "unanswerable", "sources": []}}
    Return raw JSON without Markdown code fences.
    Please follow the structure to the bone. Do not change the JSON structure.
    """


PROMPT_REGISTRY = {
    "p1": prompt_3,
    "p2": prompt_2,
    "p3": prompt_1,
}
