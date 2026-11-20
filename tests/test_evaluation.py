"""Unit tests for evaluation metrics and expected-fact checks."""

import pytest

from scripts import evaluate as evaluation


CASES = evaluation.load_cases()


@pytest.mark.parametrize("rank", [1, 2, 3, 4, 5, None])
def test_retrieval_cutoffs(rank):
    case = {"expected_chunk_ids": ["expected"]}
    hits = [{"id": f"other-{i}", "metadata": {}} for i in range(5)]
    if rank is not None:
        hits[rank - 1]["id"] = "expected"
    score = evaluation.score_retrieval(case, hits)
    assert score["rank"] == (rank if rank in (1, 2, 3, 4) else None)
    assert score["hit_at_3"] == int(rank in (1, 2, 3))
    assert score["recall_at_4"] == int(rank in (1, 2, 3, 4))


@pytest.mark.parametrize("metadata,expected", [
    ({}, 0),
    ({"table_id": "EXP-T01", "content_type": "text"}, 0),
    ({"table_id": "EXP-T02", "content_type": "table"}, 0),
    ({"table_id": "EXP-T01", "content_type": "table"}, 1),
])
def test_table_evidence_requires_the_correct_table(metadata, expected):
    hit = {"id": CASES[0]["expected_chunk_ids"][0], "metadata": metadata}
    assert evaluation.score_retrieval(CASES[0], [hit])["recall_at_4"] == expected


@pytest.mark.parametrize("identifier", CASES[-1]["expected_chunk_ids"])
def test_unchanged_remote_rule_accepts_either_version(identifier):
    score = evaluation.score_retrieval(CASES[-1], [{"id": identifier, "metadata": {}}])
    assert score["recall_at_4"] == 1


@pytest.mark.parametrize("case,wrong_answer", list(zip(CASES, [
    "$750 per employee per calendar day.",
    "Within 300 calendar days.",
    "100 hours per completed month.",
    "Up to 400 hours.",
    "10 business days of notice, with manager approval only.",
    "Report immediately to your manager.",
    "Fifty minutes.",
    "Five calendar days before the start date.",
], strict=True)), ids=[case["id"] for case in CASES])
def test_answer_checks_accept_expected_facts_and_reject_wrong_or_missing_facts(case, wrong_answer):
    assert evaluation.score_answer(case, f"**{case['expected_answer']}**")["passed"]
    assert not evaluation.score_answer(case, wrong_answer)["passed"]
    assert not evaluation.score_answer(case, "Insufficient information.")["passed"]
