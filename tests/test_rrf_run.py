"""RRF run schema check.

Run your RRF retriever (src/retrieve_rrf.py) on tests/fixtures/mini_wiki.jsonl
for the questions in tests/fixtures/mini_question_bank.jsonl and write
runs/mini_rrf.jsonl in the same schema as the full run.

This test only checks shape, not article order -- it does not import your
retrieval code.
"""

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs" / "mini_rrf.jsonl"
QUESTIONS = Path(__file__).parent / "fixtures" / "mini_question_bank.jsonl"


def _load(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


@pytest.fixture(scope="module")
def rrf_run():
    if not RUN.exists():
        pytest.skip(
            "Write runs/mini_rrf.jsonl from your RRF retriever on tests/fixtures/mini_wiki.jsonl. "
            "A skip means that file is missing."
        )
    return _load(RUN)


def test_rrf_run_covers_every_question(rrf_run):
    expected_ids = {row["id"] for row in _load(QUESTIONS)}
    got = [row["id"] for row in rrf_run]
    assert len(got) == len(set(got))
    assert set(got) == expected_ids


def test_rrf_run_contexts_are_well_formed(rrf_run):
    for row in rrf_run:
        contexts = row["contexts"]
        assert isinstance(contexts, list) and contexts
        for context in contexts:
            assert isinstance(context.get("title"), str) and context["title"]
            assert isinstance(context.get("text"), str) and context["text"]
