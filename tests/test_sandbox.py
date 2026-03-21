"""Tests for sandbox utilities (parsing only — no subprocess execution)."""

from agentsciml.sandbox import best_score_from_results, parse_result_lines


def test_parse_result_lines():
    lines = [
        "RESULT|SK|complete|N=10|T=0.500|Gamma=1.000|method=pimc|E=-3.07|E_exact=-3.07|advantage=0.042|q_EA=0.05|time=2.80s|desc",
        "RESULT|SK|complete|N=10|T=0.300|Gamma=0.700|method=pimc|E=-3.07|E_exact=-3.07|advantage=0.015|q_EA=0.03|time=2.50s|desc2",
    ]
    parsed = parse_result_lines(lines)
    assert len(parsed) == 2
    assert parsed[0]["N"] == "10"
    assert parsed[0]["advantage"] == "0.042"
    assert parsed[1]["advantage"] == "0.015"


def test_best_score_from_results():
    parsed = [
        {"advantage": "0.042", "method": "pimc"},
        {"advantage": "0.015", "method": "pimc"},
        {"advantage": "-0.001", "method": "pimc"},
    ]
    assert best_score_from_results(parsed, "advantage") == 0.042


def test_best_score_empty():
    assert best_score_from_results([], "advantage") == float("-inf")


def test_best_score_missing_key():
    parsed = [{"method": "pimc"}]
    assert best_score_from_results(parsed, "advantage") == float("-inf")


def test_parse_result_lines_positional():
    """Fields without = are stored as field_N."""
    lines = ["RESULT|SK|complete|some_value"]
    parsed = parse_result_lines(lines)
    assert parsed[0]["field_0"] == "SK"
    assert parsed[0]["field_1"] == "complete"
    assert parsed[0]["field_2"] == "some_value"
