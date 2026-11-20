"""Storage and index-refresh checks with temporary Chroma and fixed vectors."""

from unittest.mock import Mock

import pytest

from rag import indexing
from rag.indexing import get_collection, store_chunks


@pytest.fixture
def chunks():
    return [
        {"id": "remote-v1", "text": "Three remote days weekly.",
         "metadata": {"version": "1.0", "status": "superseded", "supersedes": None}},
        {"id": "remote-v2", "text": "Two remote days weekly.",
         "metadata": {"version": "2.0", "status": "current", "supersedes": "remote-v1"}},
    ]


def test_stored_text_vectors_and_metadata_can_be_read_back(tmp_path, chunks):
    vectors = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    assert store_chunks(chunks, vectors, tmp_path) == 2
    assert (tmp_path / "chroma.sqlite3").is_file()
    records = get_collection(tmp_path).get(include=["documents", "metadatas", "embeddings"])
    for index, identifier in enumerate(records["ids"]):
        source_index = next(i for i, chunk in enumerate(chunks) if chunk["id"] == identifier)
        source = chunks[source_index]
        assert records["documents"][index] == source["text"]
        assert records["embeddings"][index].tolist() == vectors[source_index]
        assert records["metadatas"][index] == {
            key: value for key, value in source["metadata"].items() if value is not None
        }
    assert chunks[0]["metadata"]["supersedes"] is None


def test_repeat_writes_update_existing_ids_without_duplicates_or_deleting_other_chunks(tmp_path, chunks):
    vectors = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    store_chunks(chunks, vectors, tmp_path)
    updated = {**chunks[1], "text": "Updated current policy text."}
    assert store_chunks([updated], [[0.0, 0.0, 1.0]], tmp_path) == 2
    records = get_collection(tmp_path).get(include=["documents", "embeddings"])
    by_id = {identifier: index for index, identifier in enumerate(records["ids"])}
    assert set(by_id) == {"remote-v1", "remote-v2"}
    assert records["documents"][by_id["remote-v1"]] == chunks[0]["text"]
    assert records["documents"][by_id["remote-v2"]] == updated["text"]
    assert records["embeddings"][by_id["remote-v2"]].tolist() == [0.0, 0.0, 1.0]


@pytest.mark.parametrize("chunk_count,vector_count", [(0, 0), (2, 1), (1, 2)])
def test_empty_or_mismatched_inputs_fail_before_creating_database(tmp_path, chunks, chunk_count, vector_count):
    path = tmp_path / "unused-db"
    with pytest.raises(ValueError, match="nonempty chunks and one embedding per chunk"):
        store_chunks(chunks[:chunk_count], [[1.0, 0.0, 0.0]] * vector_count, path)
    assert not path.exists()


def test_duplicate_ids_fail_before_creating_database(tmp_path, chunks):
    path = tmp_path / "unused-db"
    with pytest.raises(ValueError, match="Chunk IDs must be unique"):
        store_chunks([chunks[0], chunks[0]], [[1.0, 0.0, 0.0]] * 2, path)
    assert not path.exists()


def test_collection_from_another_embedding_model_is_rejected(tmp_path):
    get_collection(tmp_path).modify(metadata={"embedding_model": "another-model"})
    with pytest.raises(ValueError, match="different embedding model"):
        get_collection(tmp_path)


@pytest.fixture
def index_inputs(monkeypatch, chunks):
    """Keep Chroma real; supply fixed chunking and embedding results."""
    monkeypatch.setattr(indexing, "load_documents", Mock(return_value=[{"id": "source"}]))
    monkeypatch.setattr(indexing, "load_model", Mock(return_value=Mock(tokenizer="test-tokenizer")))
    monkeypatch.setattr(indexing, "chunk_documents", Mock(return_value=chunks[:2]))
    monkeypatch.setattr(indexing, "embed_texts", Mock(return_value=[[1.0, 0.0], [0.0, 1.0]]))
    return chunks[:2]


def test_index_refresh_removes_stale_chunks_and_repeated_runs_keep_both_versions(tmp_path, index_inputs):
    stale = {"id": "obsolete", "text": "Old chunk layout", "metadata": {"version": "0.0"}}
    store_chunks([stale], [[1.0, 0.0]], tmp_path)
    for _ in range(2):
        assert indexing.index_corpus(tmp_path) == 2
        assert set(get_collection(tmp_path).get()["ids"]) == {"remote-v1", "remote-v2"}
    indexing.chunk_documents.assert_called_with([{"id": "source"}], "test-tokenizer")
    indexing.embed_texts.assert_called_with([chunk["text"] for chunk in index_inputs])


def test_failed_index_write_does_not_delete_existing_chunks(tmp_path, monkeypatch, index_inputs):
    stale = {"id": "obsolete", "text": "Existing evidence", "metadata": {"version": "0.0"}}
    store_chunks([stale], [[1.0, 0.0]], tmp_path)
    monkeypatch.setattr(indexing, "store_chunks", Mock(side_effect=RuntimeError("Write failed")))
    with pytest.raises(RuntimeError, match="Write failed"):
        indexing.index_corpus(tmp_path)
    assert get_collection(tmp_path).get()["ids"] == ["obsolete"]
