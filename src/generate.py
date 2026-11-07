"""Answer candidate generation. Never pass gold context/title/answers to the LLM."""

from __future__ import annotations

from llm_client import call_llm
from prompts import PROMPT_REGISTRY
import json

UNANSWERABLE = "unanswerable"


def generate_candidates(
    question: str, contexts: list[dict], prompt_id: str, num_candidates: int = 5, **llm_kwargs
) -> list[dict]:
    """Return exactly num_candidates {"answer_candidate": str, "sources": [title, ...]}.

    - contexts: the top_k_contexts chunks after re-ranking.
    - llm_kwargs (model_name, temperature, seed): pass through to call_llm.
    - Sample repeatedly or ask for several answers in one call (say which in
      the report); retry or pad if parsing yields too few.
    - One call per candidate: use seed = llm_kwargs["seed"] + candidate_idx.
      The same seed every time can give 5 identical candidates. Drop the seed
      if your API has none.
    - Short answers; if unsupported, exactly "unanswerable" (lowercase, no
      punctuation) with sources [].
    - sources: distinct titles the model actually cited (map indices back to
      titles if you cite by number), not every context it saw.
    """
    response = [None] * num_candidates
    call_kwargs = {**llm_kwargs}
    prompt_func = PROMPT_REGISTRY[prompt_id](question, contexts)
    try:
      raw = call_llm(prompt_func, **call_kwargs).strip()
      if raw.startswith("```") and raw.endswith("```"):  # gemini wrapped its json in '''json...'''
          raw = "\n".join(raw.splitlines()[1:-1]).strip()

      js = json.loads(raw)
      if not isinstance(js, list):
        raise ValueError("Expected a JSON array")
      
      for candidate in range(min(len(js), num_candidates)):
        if not isinstance(js[candidate], dict):
            continue  # Skip if not a dict
        a = js[candidate].get("answer_candidate")
        s = js[candidate].get("sources")
        if not isinstance(a, str) or not isinstance(s, list) or not all(isinstance(title, str) for title in s):
            continue  # Skip if types are not as expected
        
        s = list(dict.fromkeys(s))
        response[candidate] = {"answer_candidate": a, "sources": s}
        if not response[candidate]["answer_candidate"].strip() or not response[candidate]["sources"]:
          response[candidate] = {
              "answer_candidate": UNANSWERABLE,
              "sources": [],
          }
        if(response[candidate]["answer_candidate"].strip().lower() == UNANSWERABLE):
          response[candidate] = {
                "answer_candidate": UNANSWERABLE,
                "sources": [],
            }
    except (ValueError, TypeError):
      pass

    for candidate in range(num_candidates):
      if response[candidate] is None:
        response[candidate] = {
          "answer_candidate": UNANSWERABLE,
          "sources": [],
      }
    return response

