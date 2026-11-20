"""Vector retrieval against temporary Chroma storage and a fixed query vector."""

from unittest.mock import Mock

import pytest

from rag import retrieval
from rag.indexing import get_collection, store_chunks


@pytest.fixture
def db(tmp_path, monkeypatch):
    records = [
        {"id": "remote", "text": "Two remote days", "metadata": {"section_id": "RW-03.1"}},
        {"id": "meals", "text": "$75 daily meals", "metadata": {"section_id": "EXP-03.1"}},
    ]
    store_chunks(records, [[1.0, 0.0], [0.0, 1.0]], tmp_path)
    monkeypatch.setattr(retrieval, "embed_texts", Mock(return_value=[[1.0, 0.0]]))
    return tmp_path


def test_top_result_preserves_source_text_metadata_and_distance(db):
    hits = retrieval.retrieve("  Remote days?  ", top_k=1, path=db)
    assert len(hits) == 1
    assert hits[0] == {
        "id": "remote", "text": "Two remote days", "metadata": {"section_id": "RW-03.1"},
        "distance": pytest.approx(0.0, abs=1e-6),
    }
    retrieval.embed_texts.assert_called_once_with(["Remote days?"], is_query=True)


def test_top_k_is_capped_at_available_chunks(db):
    hits = retrieval.retrieve("Remote days?", top_k=10, path=db)
    assert [hit["id"] for hit in hits] == ["remote", "meals"]
    assert hits[0]["distance"] < hits[1]["distance"]


@pytest.mark.parametrize("empty", [False, True], ids=["missing", "empty"])
def test_missing_or_empty_index_is_reported(tmp_path, empty):
    path = tmp_path / "database"
    if empty:
        get_collection(path)
    with pytest.raises(ValueError, match="Collection is empty" if empty else "Database is missing"):
        retrieval.retrieve("Remote days?", path=path)
    if not empty:
        assert not path.exists()
