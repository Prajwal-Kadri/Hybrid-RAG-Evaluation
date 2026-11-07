"""generation_[method].jsonl schema check.

Once you've produced the six required run files under runs/, this test
checks each one's schema (not its content/quality -- that's eval_generation.py's
job). Any file not yet present is skipped rather than failed, so you can run
this incrementally.

It does not import your pipeline code.
"""

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = ROOT / "runs"
QUESTIONS = ROOT / "data" / "question_bank.jsonl"
NUM_CANDIDATES = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))["generation"]["num_candidates"]

METHODS = (
    "generation_bi_p1_likelihood.jsonl",
    "generation_bi_p2_likelihood.jsonl",
    "generation_bi_p3_likelihood.jsonl",
    "generation_bi_p1_numsources.jsonl",
    "generation_cross_p1_likelihood.jsonl",
    "generation_cross_p1_numsources.jsonl",
)


def _load(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


@pytest.fixture(scope="module")
def question_ids():
    if not QUESTIONS.exists():
        pytest.skip("data/question_bank.jsonl is missing")
    return {row["id"] for row in _load(QUESTIONS)}


@pytest.mark.parametrize("filename", METHODS)
def test_generation_file_schema(filename, question_ids):
    path = RUNS_DIR / filename
    if not path.exists():
        pytest.skip(f"runs/{filename} not written yet")
    rows = _load(path)
    got_ids = [row["id"] for row in rows]
    assert len(got_ids) == len(set(got_ids))
    assert set(got_ids) == question_ids

    for row in rows:
        assert row["type"] in ("factoid", "complex", "multi_hop", "unanswerable")
        assert isinstance(row["question"], str) and row["question"]
        answers = row["answers"]
        assert isinstance(answers, list) and len(answers) == NUM_CANDIDATES, (
            f"{row['id']}: expected exactly {NUM_CANDIDATES} candidates, got {len(answers)}"
        )
        for candidate in answers:
            assert isinstance(candidate["answer_candidate"], str)
            sources = candidate["sources"]
            assert isinstance(sources, list)
            assert all(isinstance(source, str) for source in sources)
