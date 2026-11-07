import json
from pathlib import Path

from eval_generation import evaluate, main, validate_run


def test_evaluate_uses_only_the_top_ranked_candidate():
    gold = [{"id": "a", "type": "factoid", "title": ["France"], "answers": {"text": ["Paris"]}}]
    run = [
        {
            "id": "a",
            "answers": [
                {"answer_candidate": "Paris is the capital", "sources": ["France"]},
                {"answer_candidate": "wrong", "sources": ["Lyon"]},
            ],
        }
    ]
    metrics = evaluate(gold, run)
    assert metrics["overall"]["exact_match"] == 1.0
    assert metrics["overall"]["mean_precision"] == 1.0
    assert metrics["overall"]["mean_recall"] == 1.0
    assert metrics["overall"]["n"] == 1


def test_evaluate_averages_across_questions_and_by_type():
    gold = [
        {"id": "a", "type": "factoid", "title": ["France"], "answers": {"text": ["Paris"]}},
        {"id": "b", "type": "multi_hop", "title": ["France", "Census City"], "answers": {"text": ["two million"]}},
    ]
    run = [
        {"id": "a", "answers": [{"answer_candidate": "Paris", "sources": ["France"]}]},
        {"id": "b", "answers": [{"answer_candidate": "nope", "sources": ["France"]}]},
    ]
    metrics = evaluate(gold, run)
    assert metrics["overall"]["exact_match"] == 0.5
    assert metrics["by_type"]["factoid"]["exact_match"] == 1.0
    assert metrics["by_type"]["multi_hop"]["exact_match"] == 0.0
    assert metrics["by_type"]["multi_hop"]["mean_precision"] == 1.0
    assert metrics["by_type"]["multi_hop"]["mean_recall"] == 0.5


def test_unanswerable_scored_by_abstention_and_excluded_from_source_metrics():
    gold = [
        {"id": "a", "type": "factoid", "title": ["France"], "answers": {"text": ["Paris"]}},
        {"id": "u", "type": "unanswerable", "title": ["Lyon"], "answers": {"text": []}},
    ]
    run = [
        {"id": "a", "answers": [{"answer_candidate": "Paris", "sources": ["France"]}]},
        {"id": "u", "answers": [{"answer_candidate": "Unanswerable", "sources": ["Lyon"]}]},
    ]
    metrics = evaluate(gold, run)
    assert metrics["overall"]["exact_match"] == 1.0
    assert metrics["overall"]["n"] == 2
    assert metrics["overall"]["n_sources"] == 1
    assert metrics["overall"]["mean_precision"] == 1.0
    assert metrics["by_type"]["unanswerable"]["exact_match"] == 1.0
    assert metrics["by_type"]["unanswerable"]["mean_precision"] is None

    run[1]["answers"][0]["answer_candidate"] = "Lyon is on the Rhone"
    assert evaluate(gold, run)["by_type"]["unanswerable"]["exact_match"] == 0.0


def test_real_unanswerable_format_is_not_free_credit():
    # The real data stores unanswerable rows as text [""], answer_start [-1].
    gold = [
        {"id": "a", "type": "factoid", "title": ["France"], "answers": {"text": ["Paris"]}},
        {"id": "u", "type": "unanswerable", "title": ["Lyon"], "answers": {"answer_start": [-1], "text": [""]}},
    ]
    banana = [
        {"id": "a", "answers": [{"answer_candidate": "banana", "sources": ["Lyon"]}]},
        {"id": "u", "answers": [{"answer_candidate": "banana", "sources": ["Lyon"]}]},
    ]
    metrics = evaluate(gold, banana)
    assert metrics["overall"]["exact_match"] == 0.0
    assert metrics["overall"]["n_sources"] == 1
    assert metrics["by_type"]["unanswerable"]["mean_precision"] is None

    abstain = [banana[0], {"id": "u", "answers": [{"answer_candidate": "unanswerable", "sources": []}]}]
    assert evaluate(gold, abstain)["by_type"]["unanswerable"]["exact_match"] == 1.0


def test_answerable_type_without_gold_answers_warns():
    gold = [{"id": "c", "type": "complex", "title": ["X"], "answers": {"text": []}}]
    run = [{"id": "c", "answers": [{"answer_candidate": "unanswerable", "sources": []}]}]
    assert any("has no gold answers" in w for w in evaluate(gold, run)["warnings"])


def test_validate_run_rejects_duplicate_unknown_ids_and_malformed_answers():
    gold = [{"id": "a", "type": "factoid", "title": ["France"], "answers": {"text": ["Paris"]}}]
    duplicate_id = validate_run(gold, [{"id": "a", "answers": [{"answer_candidate": "x", "sources": []}]}, {"id": "a", "answers": [{"answer_candidate": "y", "sources": []}]}])
    assert any("duplicate query id" in error for error in duplicate_id)
    unknown = validate_run(gold, [{"id": "missing", "answers": [{"answer_candidate": "x", "sources": []}]}])
    assert any("unknown query id" in error for error in unknown)
    assert any("missing 1 query ids" in error for error in unknown)
    empty_answers = validate_run(gold, [{"id": "a", "answers": []}])
    assert any("answers must be a non-empty list" in error for error in empty_answers)
    bad_sources = validate_run(gold, [{"id": "a", "answers": [{"answer_candidate": "x", "sources": "not-a-list"}]}])
    assert any("sources must be a list of strings" in error for error in bad_sources)


def test_cli_validate(tmp_path: Path):
    gold = tmp_path / "gold.jsonl"
    run = tmp_path / "run.jsonl"
    gold.write_text(
        json.dumps({"id": "a", "type": "factoid", "title": ["France"], "answers": {"text": ["Paris"]}}) + "\n",
        encoding="utf-8",
    )
    run.write_text(
        json.dumps({"id": "a", "answers": [{"answer_candidate": "Paris", "sources": ["France"]}]}) + "\n",
        encoding="utf-8",
    )
    assert main([str(gold), str(run), "--validate"]) == 0
