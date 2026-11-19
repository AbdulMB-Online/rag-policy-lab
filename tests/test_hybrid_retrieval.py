"""Focused keyword/fusion checks with controlled data; no model or LLM calls."""

from copy import deepcopy
from unittest.mock import Mock

import pytest

from rag import hybrid_retrieval as hybrid
from rag import keyword_search as lexical
from rag.vector_store import get_collection, store_chunks


@pytest.fixture(autouse=True)
def forbid_model_calls(monkeypatch):
    def fail(*args, **kwargs):
        pytest.fail("Focused tests must not load models or generate embeddings")

    monkeypatch.setattr("rag.embeddings.load_model", fail)
    monkeypatch.setattr("rag.embeddings.embed_texts", fail)
    monkeypatch.setattr("rag.retrieval.embed_texts", fail)


@pytest.fixture
def keyword_store(tmp_path):
    records = [
        ("a-reference", "Consult PTO-T01 for notice requirements and advance approval.", {"section_id": "PTO-04"}),
        ("z-owner", "PTO-T01: Notice requirements.", {"section_id": "PTO-03.2", "table_id": "PTO-T01"}),
        ("prefix", "PTO-03.20: Notice.", {"section_id": "PTO-03.20"}),
        ("section", "PTO-03.2: Notice.", {"section_id": "PTO-03.2"}),
        ("b-tie", "Refund refund refund.", {"section_id": "EXP-04"}),
        ("a-tie", "Refund.", {"section_id": "EXP-04"}),
    ]
    chunks = [{"id": identifier, "text": text, "metadata": metadata}
              for identifier, text, metadata in records]
    path = tmp_path / "chroma"
    # Synthetic vectors support storage only; these tests never query them.
    store_chunks(chunks, [[1.0, 0.0] for _ in chunks], path)
    return path


@pytest.fixture
def searches(monkeypatch):
    calls = Mock()
    calls.vector.return_value = []
    calls.keyword.return_value = []
    monkeypatch.setattr(hybrid, "retrieve", calls.vector)
    monkeypatch.setattr(hybrid, "keyword_search", calls.keyword)
    return calls


def hit(identifier, distance=None):
    result = {"id": identifier, "text": identifier, "metadata": {"document_id": identifier}}
    if distance is None:
        result["keyword_score"] = 1
    else:
        result["distance"] = distance
    return result


def test_keyword_prefers_id_owner_over_more_generic_matches(keyword_store):
    hits = lexical.keyword_search("pto-t01 notice requirements advance approval", path=keyword_store)
    assert [item["id"] for item in hits[:2]] == ["z-owner", "a-reference"]
    assert hits[0]["keyword_score"] < hits[1]["keyword_score"]
    assert hits[0]["metadata"]["table_id"] == "PTO-T01"


@pytest.mark.parametrize("question,expected", [("PTO-03.2", ["section"]), ("PTO-03.200", [])])
def test_keyword_requires_complete_identifiers(keyword_store, question, expected):
    assert [item["id"] for item in lexical.keyword_search(question, path=keyword_store)] == expected


@pytest.mark.parametrize("question", ["the and what is", "unmatchedwordqzxv"])
def test_keyword_can_return_no_matches(keyword_store, question):
    assert lexical.keyword_search(question, path=keyword_store) == []


def test_keyword_repetition_does_not_inflate_scores_and_ties_are_stable(keyword_store):
    hits = lexical.keyword_search("REFUND refund", path=keyword_store)
    assert hits == lexical.keyword_search("refund", path=keyword_store)
    assert [item["id"] for item in hits] == ["a-tie", "b-tie"]
    assert all(item["keyword_score"] == 1 for item in hits)


def test_keyword_limits_positive_matches_without_padding(keyword_store):
    assert len(lexical.keyword_search("PTO-T01", top_k=1, path=keyword_store)) == 1
    assert len(lexical.keyword_search("PTO-T01", top_k=100, path=keyword_store)) == 2


@pytest.mark.parametrize("search", [lexical.keyword_search, hybrid.hybrid_retrieve])
@pytest.mark.parametrize("question", ["", " ", None, 123])
def test_invalid_questions_fail_before_model_calls(search, question, tmp_path):
    with pytest.raises(ValueError, match="nonempty question"):
        search(question, path=tmp_path / "missing")


@pytest.mark.parametrize("search,parameter", [
    (lexical.keyword_search, "top_k"),
    (hybrid.hybrid_retrieve, "top_k"),
    (hybrid.hybrid_retrieve, "candidate_k"),
])
@pytest.mark.parametrize("value", [0, -1, True, 1.5, "4"])
def test_invalid_result_counts_fail(search, parameter, value, tmp_path):
    with pytest.raises(ValueError, match=parameter):
        search("question", path=tmp_path / "missing", **{parameter: value})


@pytest.mark.parametrize("search", [lexical.keyword_search, hybrid.hybrid_retrieve])
@pytest.mark.parametrize("empty", [False, True])
def test_unindexed_databases_fail(search, empty, tmp_path):
    path = tmp_path / "chroma"
    if empty:
        get_collection(path)
    with pytest.raises(ValueError, match="Collection is empty" if empty else "Database is missing"):
        search("question", path=path)
    if not empty:
        assert not path.exists()


def test_fusion_rewards_consensus_and_preserves_inputs(searches):
    vectors = [hit("vector-only", 0.001), hit("shared", 0.5)]
    keywords = [hit("keyword-only"), hit("shared")]
    keywords[0]["keyword_score"] = 1000000
    original = deepcopy((vectors, keywords))
    searches.vector.return_value, searches.keyword.return_value = vectors, keywords

    results = hybrid.hybrid_retrieve("question", top_k=3)

    assert [item["id"] for item in results] == ["shared", "keyword-only", "vector-only"]
    assert results[0]["vector_rank"] == 2 and results[0]["keyword_rank"] == 2
    assert results[1]["distance"] is None and results[1]["vector_rank"] is None
    assert results[2]["distance"] == 0.001 and results[2]["keyword_rank"] is None
    assert (vectors, keywords) == original
    assert [call[0] for call in searches.method_calls] == ["vector", "keyword"]


def test_fusion_keeps_distinct_policy_versions(searches):
    old, current = hit("remote-work-v1.0", 0.1), hit("remote-work-v2.0", 0.2)
    for item, version in ((old, "1.0"), (current, "2.0")):
        item["metadata"].update(policy_id="remote-work", version=version)
    searches.vector.return_value = [old, current]
    searches.keyword.return_value = [current, old]
    results = hybrid.hybrid_retrieve("question")
    assert len(results) == 2
    assert {item["metadata"]["version"] for item in results} == {"1.0", "2.0"}


def test_empty_keyword_results_preserve_vector_order(searches):
    searches.vector.return_value = [hit("z", 0.1), hit("a", 0.2)]
    assert [item["id"] for item in hybrid.hybrid_retrieve("question")] == ["z", "a"]


def test_equal_fusion_scores_use_stable_ids(searches):
    searches.vector.return_value = [hit("z", 0.1)]
    searches.keyword.return_value = [hit("a")]
    assert [item["id"] for item in hybrid.hybrid_retrieve("question")] == ["a", "z"]


@pytest.mark.parametrize("top_k,candidate_k,requested", [(1, 12, 12), (20, 2, 20)])
def test_candidate_budget_and_final_limit(searches, tmp_path, top_k, candidate_k, requested):
    searches.vector.return_value = [hit("a", 0.1), hit("b", 0.2)]
    results = hybrid.hybrid_retrieve("question", top_k=top_k, candidate_k=candidate_k, path=tmp_path)
    searches.vector.assert_called_once_with("question", top_k=requested, path=tmp_path)
    searches.keyword.assert_called_once_with("question", top_k=requested, path=tmp_path)
    assert len(results) == min(top_k, 2)
