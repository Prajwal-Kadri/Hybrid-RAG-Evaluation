"""Re-rank a preserved candidate batch without invoking answer generation.

Run from the project root. Writes a separate recovery directory first;
publishing to the submission paths is a separate, verified step.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def read_rows(path):
    with path.open(encoding="utf-8-sig") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    indexed = {row["id"]: row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError(f"Duplicate question IDs in {path}")
    return indexed


def signature(candidates):
    return Counter(
        (c["answer_candidate"], tuple(c["sources"])) for c in candidates
    )


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--questions", type=Path, default=Path("data/question_bank.jsonl"))
    parser.add_argument("--rrf-run", type=Path, default=Path("runs/rrf.jsonl"))
    parser.add_argument("--candidates", type=Path, default=Path("runs/candidates/candidates_bi_p1.jsonl"))
    parser.add_argument("--paired-run", type=Path, default=Path("runs/generation_bi_p1_likelihood.jsonl"))
    parser.add_argument("--reference-run", type=Path, help="Optional unchanged likelihood-run copy to verify")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/recovery"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))

    # Cached weights only; this recovery must never call a generation API.
    os.environ.pop("CMU_AI_GATEWAY_API_KEY", None)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch
    import llm_client

    def no_generation(*_args, **_kwargs):
        raise RuntimeError("Generation is disabled during cached recovery")

    llm_client.call_llm = no_generation
    llm_client.get_gateway_client = no_generation
    import context_rerank
    from answer_rerank import numsources_rerank

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA, but no allocated CUDA GPU is visible")
    context_rerank.DEVICE = torch.device(args.device)
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))

    config = json.loads((root / "config.json").read_text())
    questions = {
        key: {field: row[field] for field in ("id", "question", "type")}
        for key, row in read_rows(root / args.questions).items()
    }
    rrf = read_rows(root / args.rrf_run)
    source = root / args.candidates
    paired_path = root / args.paired_run
    saved = read_rows(source)
    paired = read_rows(paired_path)
    if args.reference_run is not None and sha256(paired_path) != sha256(root / args.reference_run):
        raise ValueError("The likelihood file does not match the reference copy")
    if not (questions.keys() == saved.keys() == paired.keys()):
        raise ValueError("Question IDs differ between the preserved input files")
    for key, row in saved.items():
        candidates = row["candidates"]
        if len(candidates) != config["generation"]["num_candidates"]:
            raise ValueError(f"Wrong candidate count for {key}")
        if signature(candidates) != signature(paired[key]["answers"]):
            raise ValueError(f"Preserved candidates differ from likelihood: {key}")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    work = root / args.output_dir / f"bi_p1_original_{stamp}"
    work.mkdir(parents=True, exist_ok=False)
    output = work / "generation_bi_p1_numsources.jsonl"
    context_path = work / "reconstructed_contexts.jsonl"
    started = time.monotonic()
    with output.open("w") as out, context_path.open("w") as context_out:
        for done, (key, row) in enumerate(questions.items(), start=1):
            contexts = context_rerank.bi_encoder_rerank(
                row["question"], rrf[key]["contexts"],
                config["bi_encoder"]["model_name"],
            )[:config["generation"]["top_k_contexts"]]
            titles = [c["title"] for c in contexts]
            if titles != paired[key]["context_titles"]:
                raise ValueError(
                    f"Reconstructed context titles differ for {key}; "
                    "submission files have not been changed"
                )
            answers = numsources_rerank(
                saved[key]["candidates"], row["question"], contexts,
            )
            if signature(answers) != signature(paired[key]["answers"]):
                raise AssertionError(f"Candidate batch changed during sorting: {key}")
            out.write(json.dumps({
                "answers": answers, **row, "context_titles": titles,
            }) + "\n")
            context_out.write(json.dumps({"id": key, "contexts": contexts}) + "\n")
            if done % 250 == 0 or done == len(questions):
                out.flush()
                context_out.flush()
                print(f"Recovered {done}/{len(questions)}; "
                      f"elapsed={time.monotonic()-started:.1f}s", flush=True)

    shutil.copy2(source, work / "candidates_bi_p1.jsonl")
    result = subprocess.run(
        [sys.executable, str(root / "src/eval_generation.py"),
         str(root / args.questions), str(output)],
        check=True, capture_output=True, text=True,
    )
    (work / "evaluation.txt").write_text(result.stdout)
    manifest = {
        "source_candidates": str(source),
        "source_candidates_sha256": sha256(source),
        "paired_likelihood_sha256": sha256(paired_path),
        "output_sha256": sha256(output),
        "questions": len(questions), "generated": 0,
        "device": args.device,
        "context_verification": (
            "Recomputed chunks with the unchanged bi-encoder function. "
            "Ordered titles match the original likelihood run for every question. "
            "Original exact chunk IDs and prompt text were not saved, so "
            "chunk-for-chunk identity cannot be independently certified."
        ),
    }
    (work / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(result.stdout, end="", flush=True)
    print(f"RECOVERY_DIR={work}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
