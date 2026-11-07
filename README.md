# HW3: Retrieval-Augmented Generation

Prajwal Kadri (`pkadri`)

This submission retrieves Wikipedia chunks, reranks the contexts, generates five answer candidates, and reranks those candidates. All commands below run from the directory containing this README, `config.json`, `src/`, and `runs/`.

## Contents

- `Report.pdf`: design, results, comparisons, and error analysis.
- `requirements.txt` and `config.json`: dependencies, model names, and settings.
- `src/`: implementation and the supplied pipeline and official scorer.
- `scripts/recover_bi_numsources.py`: reranks a saved bi-encoder candidate batch without generating answers.
- `data/question_bank.jsonl`: the unchanged question bank.
- `tests/`: the supplied tests and mini fixtures.
- `runs/rrf.jsonl` and `runs/mini_rrf.jsonl`: full and mini retrieval outputs.
- `runs/generation_*.jsonl`: the six full generation outputs.
- `runs/candidates/candidates_*.jsonl`: the four saved candidate batches.
- `AI_USE.md`: the assignment declaration.

The corpus dump, model weights, append-only candidate pools, temporary files, and cluster logs are not included in the ZIP. The corpus can be downloaded with the command below.

## Environment

Use Python 3.12. I tested the submission with Python 3.12.10, PyTorch 2.15.0, Transformers 5.22.1, Sentence Transformers 6.2.0, OpenAI 3.44.0, Datasets 5.1.0, Accelerate 1.15.0, and pytest 9.2.0.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Testing and scoring the supplied results need no API key and do not load models. Regeneration needs network access for Hugging Face downloads and CMU AI Gateway access. Qwen likelihood scoring is much faster with a CUDA GPU; the full runs used an NVIDIA L40S. The current retrieval and context-reranking implementations use MPS on a supported Mac and CPU otherwise. Qwen uses `device_map="auto"`.

## Check the submitted results

Run the supplied tests:

```bash
python -m pytest tests/ -q -rs
```

Expected result: **21 passed, no skipped tests**.

Validate and score each of the six submitted runs:

```bash
for run_path in \
  runs/generation_bi_p1_likelihood.jsonl \
  runs/generation_bi_p2_likelihood.jsonl \
  runs/generation_bi_p3_likelihood.jsonl \
  runs/generation_bi_p1_numsources.jsonl \
  runs/generation_cross_p1_likelihood.jsonl \
  runs/generation_cross_p1_numsources.jsonl
do
  echo "$run_path"
  python src/eval_generation.py data/question_bank.jsonl "$run_path" --validate
  python src/eval_generation.py data/question_bank.jsonl "$run_path"
done
```

These are the overall official scores for the submitted files:

| Context reranker | Prompt | Answer reranker | EM | Mean precision | Mean recall |
| --- | --- | --- | ---: | ---: | ---: |
| Bi-encoder | p1 | Likelihood | 0.4349 | 0.7796 | 0.6958 |
| Bi-encoder | p2 | Likelihood | 0.4236 | 0.7905 | 0.7026 |
| Bi-encoder | p3 | Likelihood | 0.3892 | 0.8605 | 0.7648 |
| Bi-encoder | p1 | Supporting contexts | 0.4875 | 0.8563 | 0.7627 |
| Cross-encoder | p1 | Likelihood | 0.4388 | 0.8054 | 0.7140 |
| Cross-encoder | p1 | Supporting contexts | 0.5037 | 0.8943 | 0.7901 |

Each run contains 4,681 questions with five candidates each. `answers[0]` is evaluated. Source precision and recall are averaged over the 3,497 answerable questions. The scorer also prints the four question-type breakdowns.

## Configuration and prompt labels

- Chunk size: 100 words, without overlap.
- Retrieval: BM25 and `sentence-transformers/all-MiniLM-L6-v2`; take 100 candidates from each and fuse using RRF with `k=60` and one-based ranks.
- RRF pool: 20 chunks per question.
- Context reranking: MiniLM bi-encoder cosine similarity or `cross-encoder/ms-marco-MiniLM-L-6-v2`; retain the top 10 chunks.
- Generation: `gemini/gemini-2.5-flash-lite`, temperature 0.7, five candidates requested in one call.
- Likelihood scoring: `Qwen/Qwen2.5-3B-Instruct`, mean log-probability over candidate tokens only.
- Supporting-context scoring: count the 10 chunks containing the normalized candidate text. Empty and `unanswerable` candidates score zero.
- Exactly tied answer scores preserve the original candidate order.

The final prompt labels are already set in `PROMPT_REGISTRY`: `p1` calls `prompt_3` (explicit instructions and an example), `p2` calls `prompt_2` (format constraints), and `p3` calls `prompt_1` (basic instructions). These labels match the report and filenames.

## Regenerate all run files

The following sequence recreates the retrieval outputs, all six generation outputs, and all four candidate snapshots. It writes to `runs/reproduced/` so the submitted results remain available for comparison. Set `HW3_RUN_DIR="runs"` instead to overwrite the submission paths.

### 1. Set the gateway key

The client reads `CMU_AI_GATEWAY_API_KEY` and uses the base URL `https://ai-gateway.andrew.cmu.edu`. Enter the key at the hidden prompt:

```bash
export CMU_AI_GATEWAY_API_KEY="$(python -c 'import getpass; print(getpass.getpass("CMU gateway API key: "))')"
```

Do not put the key in the repository or ZIP. The local Hugging Face likelihood model does not use the gateway key.

### 2. Download the corpus

```bash
export HW3_RUN_DIR="runs/reproduced"
mkdir -p data "$HW3_RUN_DIR/candidates"

python - <<'PY'
from datasets import load_dataset
corpus = load_dataset("bbasavar/wiki-corpus-v1", split="train")
corpus.to_json("data/wiki_corpus.jsonl")
PY
```

Use the corpus as supplied, without additional cleaning. The question bank's gold `context`, `title`, and `answers` fields are evaluation labels and are not generation inputs.

### 3. Rebuild mini and full retrieval

```bash
python src/retrieve_rrf.py \
  --questions tests/fixtures/mini_question_bank.jsonl \
  --articles tests/fixtures/mini_wiki.jsonl \
  --config config.json \
  --out "$HW3_RUN_DIR/mini_rrf.jsonl"

python src/retrieve_rrf.py \
  --questions data/question_bank.jsonl \
  --articles data/wiki_corpus.jsonl \
  --config config.json \
  --out "$HW3_RUN_DIR/rrf.jsonl"
```

### 4. Run all six combinations

Run the combinations sequentially in one Python process. This keeps the context models and numerical backend consistent when the two answer rerankers share a candidate pool. It also avoids concurrent writes to the append-only pool files.

```bash
python - <<'PY'
from pathlib import Path
import os
import sys

sys.path.insert(0, str(Path("src").resolve()))
from pipeline import run_generation

out_dir = Path(os.environ["HW3_RUN_DIR"])
methods = [
    ("bi", "p1", "likelihood"),
    ("bi", "p2", "likelihood"),
    ("bi", "p3", "likelihood"),
    ("bi", "p1", "numsources"),
    ("cross", "p1", "likelihood"),
    ("cross", "p1", "numsources"),
]
for context_method, prompt_id, answer_method in methods:
    output = out_dir / f"generation_{context_method}_{prompt_id}_{answer_method}.jsonl"
    run_generation(
        questions_path=Path("data/question_bank.jsonl"),
        rrf_run_path=out_dir / "rrf.jsonl",
        config_path=Path("config.json"),
        context_reranker=context_method,
        prompt_id=prompt_id,
        answer_reranker=answer_method,
        out_path=output,
        regenerate=False,
    )
    print(f"Finished {output}", flush=True)
PY
```

The likelihood runs generate candidates when their exact prompt fingerprint is not cached. The supporting-context runs should report 4,681 reused candidates and zero generated. Candidate snapshots are written to `$HW3_RUN_DIR/candidates/`.

The prompts currently include numerical context scores. Moving between machines or changing the numerical backend can change those strings and cause cache misses. Keep paired runs together. The pipeline saves candidates incrementally but does not checkpoint completed likelihood scores, so allow enough uninterrupted time for regeneration.

### 5. Validate the reproduced outputs

```bash
for run_path in "$HW3_RUN_DIR"/generation_*.jsonl
do
  python src/eval_generation.py data/question_bank.jsonl "$run_path" --validate
  python src/eval_generation.py data/question_bank.jsonl "$run_path"
done

python - <<'PY'
from pathlib import Path
from collections import Counter
import json
import os

root = Path(os.environ["HW3_RUN_DIR"])
def load(path):
    with path.open() as handle:
        return {row["id"]: row for line in handle if line.strip()
                for row in [json.loads(line)]}
def signature(candidates):
    return Counter(json.dumps(c, sort_keys=True) for c in candidates)
for method in ("bi", "cross"):
    likelihood = load(root / f"generation_{method}_p1_likelihood.jsonl")
    counts = load(root / f"generation_{method}_p1_numsources.jsonl")
    saved = load(root / "candidates" / f"candidates_{method}_p1.jsonl")
    assert likelihood.keys() == counts.keys() == saved.keys()
    assert all(
        signature(likelihood[key]["answers"])
        == signature(counts[key]["answers"])
        == signature(saved[key]["candidates"])
        for key in likelihood
    )
    print(f"{method}: paired runs use identical candidate batches")
PY
```

Generation is stochastic. The configured seed is not passed to the gateway by the current client, so newly generated answers and metrics need not be identical to the submitted files. The report numbers can be reproduced exactly by scoring the submitted run files above.

## Rerank the saved bi-encoder candidates without generation

The submitted bi-encoder supporting-context result was rebuilt from the preserved p1 candidate batch after a cache mismatch. The utility below repeats that reranking using the included candidate snapshot and paired likelihood output. It makes no generation API calls and leaves the submitted results unchanged, writing a new directory under `runs/recovery/`.

Run this after the MiniLM weights have been downloaded by retrieval; the utility uses cached model weights in offline mode.

```bash
python scripts/recover_bi_numsources.py \
  --device cpu \
  --questions data/question_bank.jsonl \
  --rrf-run runs/rrf.jsonl \
  --candidates runs/candidates/candidates_bi_p1.jsonl \
  --paired-run runs/generation_bi_p1_likelihood.jsonl
```

Use `--device cuda` inside an allocated CUDA GPU session to perform this recovery on GPU. The helper verifies that the saved candidates match the paired likelihood run and that the reconstructed context-title order matches. The original exact chunk IDs and prompt text were not retained for this historical recovery, so full historical chunk identity cannot be independently certified; the candidate batches themselves are verified to match.
