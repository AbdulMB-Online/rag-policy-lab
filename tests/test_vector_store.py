"""Storage checks with temporary Chroma databases and fixed vectors."""

import pytest

from rag.vector_store import get_collection, store_chunks


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
