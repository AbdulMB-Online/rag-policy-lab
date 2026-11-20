"""Reciprocal Rank Fusion and hybrid retrieval integration checks.

Uses small temporary Chroma databases with hand-made 3-number vectors, so no
embedding model is loaded and the real index under .venv/chroma is untouched.
"""

import pytest

from rag import retrieval
from rag.retrieval import RRF_CONSTANT, hybrid_retrieve
from rag.indexing import store_chunks


@pytest.fixture
def db(tmp_path, monkeypatch):
    """Four real stored chunks; fixed query vectors put the table fourth."""
    records = [
        ("pto-v1.0:PTO-01:0", "## PTO-01: Purpose and scope\n\nPaid time off policy.", "PTO-01", 0.05),
        ("pto-v1.0:PTO-03.20:0", "### PTO-03.20: Exceptions\n\nNOTICE rules.", "PTO-03.20", 0.1),
        ("pto-v1.0:PTO-04:0", "## PTO-04: Procedures\n\nCheck notice requirements in PTO-T01.", "PTO-04", 0.2),
        ("pto-v1.0:PTO-03.2:0", "### PTO-03.2: Requesting leave\n\nPTO-T01: Notice by absence length",
         "PTO-03.2", 0.3),
    ]
    chunks = [{"id": identifier, "text": text, "metadata": {"section_id": section}}
              for identifier, text, section, _ in records]
    chunks[-1]["metadata"]["table_id"] = "PTO-T01"
    store_chunks(chunks, [[1.0, offset, 0.0] for _, _, _, offset in records], path=tmp_path)
    monkeypatch.setattr(retrieval, "embed_texts", lambda texts, is_query=False: [[1.0, 0.0, 0.0]])
    return tmp_path


def ids(hits):
    return [hit["id"] for hit in hits]


# ---------- fusion ----------


def hit(identifier, distance=None):
    record = {"id": identifier, "text": f"text {identifier}", "metadata": {"section_id": "X"}}
    if distance is not None:
        record["distance"] = distance
    return record


@pytest.fixture
def fake_lists(monkeypatch):
    """Replace both searches; set .vector/.keyword lists and inspect .calls."""
    class Lists:
        vector, keyword, calls = [], [], []

    def fake_retrieve(question, top_k, path):
        Lists.calls.append(("vector", top_k))
        return Lists.vector[:top_k]

    def fake_keyword(question, top_k, path):
        Lists.calls.append(("keyword", top_k))
        return Lists.keyword[:top_k]

    Lists.calls = []
    monkeypatch.setattr(retrieval, "retrieve", fake_retrieve)
    monkeypatch.setattr(retrieval, "keyword_search", fake_keyword)
    return Lists


def test_overlapping_hit_is_returned_once_with_summed_score(fake_lists):
    fake_lists.vector = [hit("a", 0.1), hit("b", 0.2)]
    fake_lists.keyword = [hit("b"), hit("c")]
    results = {r["id"]: r for r in hybrid_retrieve("q", top_k=10)}
    assert sorted(results) == ["a", "b", "c"]
    b = results["b"]
    assert b["rrf_score"] == pytest.approx(1 / (RRF_CONSTANT + 2) + 1 / (RRF_CONSTANT + 1))
    assert (b["vector_rank"], b["keyword_rank"], b["distance"]) == (2, 1, 0.2)


def test_keyword_only_hit_has_no_distance_or_vector_rank(fake_lists):
    fake_lists.vector = [hit("a", 0.1)]
    fake_lists.keyword = [hit("c")]
    c = next(r for r in hybrid_retrieve("q", top_k=10) if r["id"] == "c")
    assert (c["distance"], c["vector_rank"], c["keyword_rank"]) == (None, None, 1)


def test_consensus_beats_single_first_place(fake_lists):
    fake_lists.vector = [hit("vector-top", 0.1), hit("shared", 0.2)]
    fake_lists.keyword = [hit("keyword-top"), hit("shared")]
    assert ids(hybrid_retrieve("q", top_k=1)) == ["shared"]


def test_empty_keyword_list_preserves_vector_order(fake_lists):
    fake_lists.vector = [hit("c", 0.1), hit("a", 0.2), hit("b", 0.3)]
    fake_lists.keyword = []
    results = hybrid_retrieve("q", top_k=3)
    assert ids(results) == ["c", "a", "b"]
    assert all(r["keyword_rank"] is None for r in results)


def test_document_versions_stay_distinct(fake_lists):
    v1, v2 = "remote-work-v1.0:RW-03.1:0", "remote-work-v2.0:RW-03.1:0"
    fake_lists.vector = [hit(v1, 0.1), hit(v2, 0.1)]
    fake_lists.keyword = [hit(v2), hit(v1)]
    results = hybrid_retrieve("q", top_k=4)
    assert sorted(ids(results)) == [v1, v2]
    # Equal fused scores: the documented tie-break is chunk ID, which favors v1.0.
    assert results[0]["rrf_score"] == pytest.approx(results[1]["rrf_score"])
    assert ids(results) == [v1, v2]


def test_candidate_count_is_at_least_top_k(fake_lists):
    fake_lists.vector = [hit(str(n), n / 100) for n in range(20)]
    hybrid_retrieve("q", top_k=4)
    hybrid_retrieve("q", top_k=15, candidate_k=12)
    assert fake_lists.calls == [("vector", 12), ("keyword", 12), ("vector", 15), ("keyword", 15)]


def test_results_are_capped_at_top_k(fake_lists):
    fake_lists.vector = [hit(str(n), n / 100) for n in range(12)]
    assert len(hybrid_retrieve("q", top_k=4)) == 4


@pytest.mark.parametrize("kwargs", [
    {"top_k": 0}, {"top_k": -1}, {"top_k": True}, {"candidate_k": 0}, {"candidate_k": 1.5},
])
def test_hybrid_rejects_invalid_counts(fake_lists, kwargs):
    with pytest.raises(ValueError, match="positive integer"):
        hybrid_retrieve("q", **kwargs)
    assert fake_lists.calls == []


# ---------- real components on the temporary database ----------

def test_hybrid_promotes_identifier_owner_over_vector_ranking(db):
    question = "PTO-T01"
    vector_ids = ids(retrieval.retrieve(question, top_k=6, path=db))
    assert vector_ids.index("pto-v1.0:PTO-03.2:0") == 3  # vector rank 4, as in the real index
    results = hybrid_retrieve(question, top_k=4, path=db)
    assert ids(results)[:2] == ["pto-v1.0:PTO-03.2:0", "pto-v1.0:PTO-04:0"]
    table, procedure = results[0], results[1]
    assert (table["vector_rank"], table["keyword_rank"]) == (4, 1)
    assert (procedure["vector_rank"], procedure["keyword_rank"]) == (3, 2)
    # The win is narrow: 1/64 + 1/61 versus 1/63 + 1/62.
    assert table["rrf_score"] - procedure["rrf_score"] == pytest.approx(
        (1 / 64 + 1 / 61) - (1 / 63 + 1 / 62))
