"""Official HW3 scorer; report numbers are checked against it. Self-contained,
so bugs in your metrics.py can't change your scores."""

from __future__ import annotations

import argparse
import json
import re
import string
import sys
from pathlib import Path

MAIN_TYPES = ("factoid", "complex", "multi_hop", "unanswerable")
UNANSWERABLE = "unanswerable"


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path} line {line_number} is not JSON") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path} line {line_number} is not an object")
            rows.append(row)
    return rows


_PUNCT_RE = re.compile(f"[{re.escape(string.punctuation)}]")
_WS_RE = re.compile(r"[\t\n\r]+")


def normalize_answer(text: str) -> str:
    text = text.lower()
    text = _PUNCT_RE.sub("", text)
    text = _WS_RE.sub(" ", text)
    return " ".join(text.split())


def gold_answer_list(texts) -> list[str]:
    """Distinct non-blank gold answers. Empty result = unanswerable
    (the data stores those as [""])."""
    answers = []
    for text in _distinct(texts):
        if normalize_answer(text) and text not in answers:
            answers.append(text)
    return answers


def exact_match(prediction: str, gold_answers: list[str]) -> float:
    normalized_prediction = normalize_answer(prediction or "")
    gold_answers = [gold for gold in (gold_answers or []) if normalize_answer(gold)]
    if not gold_answers:
        # Unanswerable question: correct only if the system abstains.
        return 1.0 if normalized_prediction == UNANSWERABLE else 0.0
    padded_prediction = f" {normalized_prediction} "
    for gold in gold_answers:
        # Whole-word containment: gold "6" matches "6 people" but not "1960".
        if f" {normalize_answer(gold)} " in padded_prediction:
            return 1.0
    return 0.0


def _distinct(values) -> list[str]:
    if not isinstance(values, list):
        return []
    seen = []
    for value in values:
        if isinstance(value, str) and value not in seen:
            seen.append(value)
    return seen


def source_precision(predicted_sources: list[str], gold_titles: list[str]) -> float:
    predicted = _distinct(predicted_sources)
    if not predicted:
        return 0.0
    gold_set = set(_distinct(gold_titles))
    return len([title for title in predicted if title in gold_set]) / len(predicted)


def source_recall(predicted_sources: list[str], gold_titles: list[str]) -> float:
    gold = _distinct(gold_titles)
    if not gold:
        return 0.0
    predicted_set = set(_distinct(predicted_sources))
    return len([title for title in gold if title in predicted_set]) / len(gold)


def validate_run(gold_rows: list[dict], run_rows: list[dict]) -> list[str]:
    errors = []
    gold_ids = [row.get("id") for row in gold_rows]
    gold_id_set = set(gold_ids)
    if len(gold_id_set) != len(gold_ids):
        errors.append("gold file has duplicate ids")
    seen = set()
    for index, row in enumerate(run_rows, start=1):
        query_id = row.get("id")
        if not isinstance(query_id, str) or not query_id:
            errors.append(f"run row {index} is missing id")
            continue
        if query_id in seen:
            errors.append(f"duplicate query id {query_id}")
        seen.add(query_id)
        if query_id not in gold_id_set:
            errors.append(f"unknown query id {query_id}")
        answers = row.get("answers")
        if not isinstance(answers, list) or not answers:
            errors.append(f"run row {query_id} answers must be a non-empty list")
            continue
        for position, candidate in enumerate(answers):
            if not isinstance(candidate, dict):
                errors.append(f"run row {query_id} answers[{position}] must be an object")
                continue
            if not isinstance(candidate.get("answer_candidate"), str):
                errors.append(f"run row {query_id} answers[{position}] missing answer_candidate string")
            sources = candidate.get("sources")
            if not isinstance(sources, list) or not all(isinstance(source, str) for source in sources):
                errors.append(f"run row {query_id} answers[{position}] sources must be a list of strings")
    missing = [query_id for query_id in gold_ids if query_id not in seen]
    if missing:
        errors.append(f"missing {len(missing)} query ids, first {missing[0]}")
    return errors


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def evaluate(gold_rows: list[dict], run_rows: list[dict]) -> dict:
    by_id = {row["id"]: row for row in run_rows if isinstance(row.get("id"), str)}

    overall_em: list[float] = []
    overall_precision: list[float] = []
    overall_recall: list[float] = []
    by_type_em = {qtype: [] for qtype in MAIN_TYPES}
    by_type_precision = {qtype: [] for qtype in MAIN_TYPES}
    by_type_recall = {qtype: [] for qtype in MAIN_TYPES}
    warnings = []

    for gold in gold_rows:
        query_id = gold.get("id")
        qtype = gold.get("type")
        run_row = by_id.get(query_id)
        if run_row is None:
            warnings.append(f"missing run row for {query_id}")
            continue
        answers = run_row.get("answers") or []
        top = answers[0] if answers else {}
        prediction = top.get("answer_candidate", "")
        predicted_sources = top.get("sources", [])

        gold_answers = gold_answer_list((gold.get("answers") or {}).get("text"))
        if not gold_answers and qtype != "unanswerable":
            warnings.append(f"{query_id} is typed {qtype} but has no gold answers; scored as unanswerable")
        gold_titles = _distinct(gold.get("title"))

        em = exact_match(prediction, gold_answers)
        overall_em.append(em)
        if qtype in MAIN_TYPES:
            by_type_em[qtype].append(em)
        else:
            warnings.append(f"skipping unknown type for {query_id}")

        # Source precision/recall are only defined for answerable questions:
        # an unanswerable question's "title" is a near-miss article, so citing
        # it must not be rewarded.
        if gold_answers:
            precision = source_precision(predicted_sources, gold_titles)
            recall = source_recall(predicted_sources, gold_titles)
            overall_precision.append(precision)
            overall_recall.append(recall)
            if qtype in MAIN_TYPES:
                by_type_precision[qtype].append(precision)
                by_type_recall[qtype].append(recall)

    return {
        "overall": {
            "exact_match": _mean(overall_em),
            "mean_precision": _mean(overall_precision),
            "mean_recall": _mean(overall_recall),
            "n": len(overall_em),
            "n_sources": len(overall_precision),
        },
        "by_type": {
            qtype: {
                "exact_match": _mean(by_type_em[qtype]),
                "mean_precision": _mean(by_type_precision[qtype]),
                "mean_recall": _mean(by_type_recall[qtype]),
                "n": len(by_type_em[qtype]),
                "n_sources": len(by_type_precision[qtype]),
            }
            for qtype in MAIN_TYPES
        },
        "warnings": warnings,
    }


def _fmt(value) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def format_report(metrics: dict) -> str:
    lines = []
    overall = metrics["overall"]
    lines.append(
        "overall  "
        + "  ".join(f"{name}={_fmt(overall[name])}" for name in ("exact_match", "mean_precision", "mean_recall"))
        + f"  n={overall['n']}  n_sources={overall['n_sources']}"
    )
    for qtype, block in metrics["by_type"].items():
        lines.append(
            f"{qtype:12} "
            + "  ".join(f"{name}={_fmt(block[name])}" for name in ("exact_match", "mean_precision", "mean_recall"))
            + f"  n={block['n']}  n_sources={block['n_sources']}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score a generation_[method].jsonl run against question_bank.jsonl.")
    parser.add_argument("gold", type=Path)
    parser.add_argument("run", type=Path)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args(argv)
    gold_rows = load_jsonl(args.gold)
    run_rows = load_jsonl(args.run)
    errors = validate_run(gold_rows, run_rows)
    if args.validate:
        if errors:
            print("\n".join(errors), file=sys.stderr)
            return 1
        print("valid")
        return 0
    if errors:
        print("\n".join(errors), file=sys.stderr)
    metrics = evaluate(gold_rows, run_rows)
    print(format_report(metrics))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
