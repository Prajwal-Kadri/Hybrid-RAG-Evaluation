import pytest

from metrics import exact_match, normalize_answer, source_precision, source_recall


def test_normalize_answer_strips_punctuation_case_and_whitespace():
    assert normalize_answer("Paris, France!") == "paris france"
    assert normalize_answer("  Two\tMillion\n") == "two million"
    assert normalize_answer("ALLCAPS") == "allcaps"


def test_exact_match_is_a_fuzzy_substring_match_against_any_gold_answer():
    assert exact_match("The answer is Paris.", ["Paris"]) == 1.0
    assert exact_match("the answer is paris", ["Paris, France"]) == 0.0
    assert exact_match("Paris, France is correct", ["Paris, France"]) == 1.0
    assert exact_match("Lyon", ["Paris", "Nice"]) == 0.0
    assert exact_match("", ["Paris"]) == 0.0


def test_exact_match_requires_whole_word_matches():
    assert exact_match("6 people", ["6"]) == 1.0
    assert exact_match("1960", ["6"]) == 0.0
    assert exact_match("in new york city", ["New York"]) == 1.0
    assert exact_match("newyorker", ["New York"]) == 0.0


def test_exact_match_on_unanswerable_requires_abstaining():
    # question_bank.jsonl stores unanswerable gold answers as [""], not [].
    assert exact_match("unanswerable", [""]) == 1.0
    assert exact_match("banana", [""]) == 0.0
    assert exact_match("Unanswerable.", []) == 1.0
    assert exact_match("unanswerable", []) == 1.0
    assert exact_match("anything", []) == 0.0
    assert exact_match("", []) == 0.0
    # Abstaining on an answerable question is wrong.
    assert exact_match("unanswerable", ["Paris"]) == 0.0


def test_source_precision_is_proportion_of_predicted_sources_that_are_valid():
    assert source_precision(["France"], ["France", "Lyon"]) == 1.0
    assert source_precision(["France", "Nice"], ["France", "Lyon"]) == pytest.approx(0.5)
    assert source_precision(["Nice"], ["France", "Lyon"]) == 0.0
    assert source_precision([], ["France"]) == 0.0
    assert source_precision(["France", "France"], ["France"]) == 1.0


def test_source_recall_is_proportion_of_gold_titles_that_were_predicted():
    assert source_recall(["France", "Lyon"], ["France", "Lyon"]) == 1.0
    assert source_recall(["France"], ["France", "Lyon"]) == pytest.approx(0.5)
    assert source_recall(["Nice"], ["France", "Lyon"]) == 0.0
    assert source_recall(["France"], []) == 0.0
    assert source_recall(["France"], ["France", "France"]) == 1.0
