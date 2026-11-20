"""Keyword matching and identifier ranking; no embedding or LLM calls."""

import pytest

from rag.keyword_search import keyword_search
from rag.vector_store import store_chunks


def chunk(identifier, text, section_id, table_id=None):
    metadata = {"section_id": section_id}
    if table_id:
        metadata["table_id"] = table_id
    return {"id": identifier, "text": text, "metadata": metadata}


@pytest.fixture
def db(tmp_path):
    records = [
        chunk("pto-v1.0:PTO-01:0", "## PTO-01: Purpose and scope\n\nPaid time off policy.", "PTO-01"),
        chunk("pto-v1.0:PTO-03.20:0", "### PTO-03.20: Exceptions\n\nNOTICE NOTICE NOTICE rules.", "PTO-03.20"),
        chunk("pto-v1.0:PTO-04:0", "## PTO-04: Procedures\n\nCheck notice requirements in PTO-T01.", "PTO-04"),
        chunk("pto-v1.0:PTO-03.2:0",
              "### PTO-03.2: Requesting leave\n\nPTO-T01: Notice by absence length\n\n| Days | Notice |",
              "PTO-03.2", table_id="PTO-T01"),
        chunk("remote-work-v1.0:RW-04:0", "## RW-04: Procedures\n\nSubmit a recurring remote request.", "RW-04"),
        chunk("remote-work-v2.0:RW-04:0", "## RW-04: Procedures\n\nSubmit a recurring remote request.", "RW-04"),
    ]
    # Chroma requires vectors for storage; keyword search does not use them.
    store_chunks(records, [[1.0, 0.0, 0.0]] * len(records), path=tmp_path)
    return tmp_path


def ids(hits):
    return [hit["id"] for hit in hits]


def test_keyword_matching_ignores_case(db):
    assert "pto-v1.0:PTO-03.2:0" in ids(keyword_search("NOTICE", path=db))


def test_exact_identifier_owner_ranks_first(db):
    # The procedure matches more terms, but the table owns the requested ID.
    hits = keyword_search("What are the notice requirements in PTO-T01?", path=db)
    assert ids(hits)[:2] == ["pto-v1.0:PTO-03.2:0", "pto-v1.0:PTO-04:0"]
    assert hits[1]["keyword_score"] > hits[0]["keyword_score"]


def test_identifier_does_not_match_longer_identifier(db):
    hits = keyword_search("PTO-03.2", path=db)
    assert "pto-v1.0:PTO-03.20:0" not in ids(hits)
    assert ids(hits) == ["pto-v1.0:PTO-03.2:0"]


def test_owner_needs_a_text_match_to_be_returned(db):
    hits = keyword_search("PTO-03.2", path=db)
    assert all(hit["keyword_score"] > 0 for hit in hits)


def test_repeated_terms_count_once(db):
    hits = {hit["id"]: hit["keyword_score"] for hit in keyword_search("notice notice", path=db)}
    assert hits["pto-v1.0:PTO-03.20:0"] == 1
    assert hits["pto-v1.0:PTO-03.2:0"] == 1


def test_distinct_terms_raise_score_and_ties_sort_by_id(db):
    hits = keyword_search("submit recurring remote request procedures", path=db)
    assert ids(hits)[:2] == ["remote-work-v1.0:RW-04:0", "remote-work-v2.0:RW-04:0"]
    assert hits[0]["keyword_score"] == hits[1]["keyword_score"] == 5
    assert hits[2]["id"] == "pto-v1.0:PTO-04:0" and hits[2]["keyword_score"] == 1


@pytest.mark.parametrize("question", ["How many of the", "zebra quantum"])
def test_no_usable_terms_or_matches_return_empty(db, question):
    assert keyword_search(question, path=db) == []


def test_keyword_top_k_limits_results(db):
    assert len(keyword_search("notice submit", top_k=2, path=db)) == 2


@pytest.mark.parametrize("question,top_k,message", [
    ("", 12, "nonempty question"),
    ("   ", 12, "nonempty question"),
    (5, 12, "nonempty question"),
    ("notice", 0, "positive integer"),
    ("notice", True, "positive integer"),
    ("notice", 2.0, "positive integer"),
])
def test_keyword_rejects_invalid_input(db, question, top_k, message):
    with pytest.raises(ValueError, match=message):
        keyword_search(question, top_k=top_k, path=db)


def test_keyword_missing_database_is_not_created(tmp_path):
    missing = tmp_path / "missing"
    with pytest.raises(ValueError, match="Database is missing"):
        keyword_search("notice", path=missing)
    assert not missing.exists()
