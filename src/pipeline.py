"""End-to-end pipeline. Don't change the flags; we may run it on hidden questions.

Per question: re-rank the pool_k RRF chunks -> keep top_k_contexts ->
generate num_candidates -> re-rank answers -> one row of --out.

The six required runs (--context-reranker / --prompt / --answer-reranker):
  bi p1 likelihood, bi p2 likelihood, bi p3 likelihood,
  bi p1 numsources, cross p1 likelihood, cross p1 numsources
Output file: runs/generation_<reranker>_<prompt>_<answer-reranker>.jsonl

Generated candidates are cached per question, keyed by a hash of the exact
rendered prompt plus generate.py, call_llm and the generation settings. So:
- likelihood and numsources runs re-rank the same candidates;
- any change to a prompt, its helpers, the chunks it sees, generate.py or the
  settings regenerates only the affected questions;
- reordering PROMPT_REGISTRY reuses existing candidates (no swapping needed).
runs/candidates/pool_<reranker>.jsonl is the append-only cache;
runs/candidates/candidates_<reranker>_<prompt>.jsonl holds exactly what the
latest run used (submit these). --regenerate ignores the cache.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import sys
from pathlib import Path

import generate as generate_module

from answer_rerank import likelihood_rerank, numsources_rerank
from context_rerank import bi_encoder_rerank, cross_encoder_rerank
from generate import generate_candidates
from llm_client import call_llm
from prompts import PROMPT_REGISTRY


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8-sig") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def code_fingerprint(gen_cfg: dict) -> str:
    """Hash of the generation code and settings (not the prompt)."""
    payload = {
        "generate_py": inspect.getsource(generate_module),
        "call_llm": inspect.getsource(call_llm),
        "settings": {key: gen_cfg.get(key) for key in ("model_name", "num_candidates", "temperature", "seed")},
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def row_fingerprint(code_fp: str, prompt_text: str) -> str:
    return hashlib.sha256((code_fp + "\n" + prompt_text).encode("utf-8")).hexdigest()[:20]


def load_pool(path: Path) -> dict:
    """{(id, fingerprint): candidates} from an append-only pool file."""
    if not path.exists():
        return {}
    return {(row["id"], row["fingerprint"]): row["candidates"] for row in load_jsonl(path)}


def check_candidates(candidates, num_candidates: int, query_id: str) -> None:
    """generate_candidates must return exactly num_candidates well-formed dicts."""
    ok = (
        isinstance(candidates, list)
        and len(candidates) == num_candidates
        and all(
            isinstance(c, dict)
            and isinstance(c.get("answer_candidate"), str)
            and isinstance(c.get("sources"), list)
            and all(isinstance(t, str) for t in c["sources"])
            for c in candidates
        )
    )
    if not ok:
        raise ValueError(
            f"generate_candidates returned an invalid result for {query_id}: expected exactly "
            f'{num_candidates} {{"answer_candidate": str, "sources": [str, ...]}} dicts, got {candidates!r}'
        )


def run_generation(
    questions_path: Path,
    rrf_run_path: Path,
    config_path: Path,
    context_reranker: str,
    prompt_id: str,
    answer_reranker: str,
    out_path: Path,
    regenerate: bool = False,
) -> None:
    questions = {row["id"]: row for row in load_jsonl(questions_path)}
    rrf_run = {row["id"]: row["contexts"] for row in load_jsonl(rrf_run_path)}
    config = json.loads(config_path.read_text(encoding="utf-8"))
    gen_cfg = config["generation"]
    top_k_contexts = gen_cfg["top_k_contexts"]
    llm_kwargs = {key: gen_cfg[key] for key in ("model_name", "temperature", "seed") if key in gen_cfg}
    rerank_cfg = config.get("answer_rerank", {})
    length_normalize = rerank_cfg.get("length_normalize", True)
    likelihood_model_name = rerank_cfg.get("likelihood_model_name")

    num_candidates = gen_cfg["num_candidates"]
    cache_dir = out_path.parent / "candidates"
    cache_dir.mkdir(parents=True, exist_ok=True)
    pool_path = cache_dir / f"pool_{context_reranker}.jsonl"
    used_path = cache_dir / f"candidates_{context_reranker}_{prompt_id}.jsonl"
    cache = {} if regenerate else load_pool(pool_path)
    code_fp = code_fingerprint(gen_cfg)
    used_rows = []
    reused = generated = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as handle, pool_path.open("a", encoding="utf-8") as pool_handle:
        for query_id, row in questions.items():
            question = row["question"]
            pool = rrf_run[query_id]

            if context_reranker == "bi":
                ranked = bi_encoder_rerank(question, pool, config["bi_encoder"]["model_name"])
            elif context_reranker == "cross":
                ranked = cross_encoder_rerank(question, pool, config["cross_encoder"]["model_name"])
            else:
                raise ValueError(f"unknown context reranker {context_reranker}")
            contexts = ranked[:top_k_contexts]

            prompt_text = PROMPT_REGISTRY[prompt_id](question, contexts)
            fingerprint = row_fingerprint(code_fp, prompt_text)
            candidates = cache.get((query_id, fingerprint))
            if candidates is not None:
                reused += 1
            else:
                candidates = generate_candidates(
                    question, contexts, prompt_id, num_candidates=num_candidates, **llm_kwargs
                )
                check_candidates(candidates, num_candidates, query_id)
                pool_handle.write(json.dumps({"id": query_id, "fingerprint": fingerprint, "candidates": candidates}) + "\n")
                pool_handle.flush()
                cache[(query_id, fingerprint)] = candidates
                generated += 1
            used_rows.append({"id": query_id, "fingerprint": fingerprint, "candidates": candidates})

            if answer_reranker == "likelihood":
                candidates = likelihood_rerank(
                    candidates,
                    question,
                    contexts,
                    prompt_text,
                    length_normalize=length_normalize,
                    model_name=likelihood_model_name,
                )
            elif answer_reranker == "numsources":
                candidates = numsources_rerank(candidates, question, contexts)
            else:
                raise ValueError(f"unknown answer reranker {answer_reranker}")

            handle.write(
                json.dumps(
                    {
                        "answers": candidates,
                        "id": query_id,
                        "question": question,
                        "type": row["type"],
                        "context_titles": [context["title"] for context in contexts],
                    }
                )
                + "\n"
            )

    tmp_path = used_path.with_suffix(".jsonl.tmp")
    with tmp_path.open("w", encoding="utf-8") as used_handle:
        for used in used_rows:
            used_handle.write(json.dumps(used) + "\n")
    os.replace(tmp_path, used_path)
    print(f"candidates: {reused} reused from cache, {generated} generated", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the full generation pipeline for one method.")
    parser.add_argument("--questions", required=True, type=Path)
    parser.add_argument("--rrf-run", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--context-reranker", required=True, choices=("bi", "cross"))
    parser.add_argument("--prompt", required=True, choices=("p1", "p2", "p3"))
    parser.add_argument("--answer-reranker", required=True, choices=("likelihood", "numsources"))
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--regenerate", action="store_true", help="ignore cached candidates and regenerate")
    args = parser.parse_args(argv)
    run_generation(
        args.questions,
        args.rrf_run,
        args.config,
        args.context_reranker,
        args.prompt,
        args.answer_reranker,
        args.out,
        regenerate=args.regenerate,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
