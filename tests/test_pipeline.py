"""Pipeline orchestration and indexing checks without model or API calls."""

from unittest.mock import Mock

import pytest

from rag import pipeline
from rag.vector_store import get_collection, store_chunks


@pytest.fixture
def chunks():
    return [
        {"id": "remote-v1", "text": "Three remote days", "metadata": {"version": "1.0"}},
        {"id": "remote-v2", "text": "Two remote days", "metadata": {"version": "2.0"}},
        {"id": "expense", "text": "Meal reimbursement", "metadata": {}, "rrf_score": 0.02},
    ]


@pytest.fixture
def stages(monkeypatch, chunks):
    calls = Mock()
    calls.hybrid.return_value = calls.vector.return_value = chunks
    calls.rerank.return_value = chunks[1:2]
    calls.generate.return_value = "Policy answer"
    for attribute, replacement in (
        ("hybrid_retrieve", calls.hybrid), ("retrieve", calls.vector),
        ("rerank", calls.rerank), ("generate_answer", calls.generate),
    ):
        monkeypatch.setattr(pipeline, attribute, replacement)
    return calls


@pytest.mark.parametrize("options,method,limit,top_k,enabled", [
    ([], "hybrid", 12, 4, True),
    (["--retrieval", "vector", "--candidate-k", "20", "--top-k", "3"], "vector", 20, 3, True),
    (["--candidate-k", "2", "--top-k", "5"], "hybrid", 5, 5, True),
    (["--no-rerank"], "hybrid", 4, 4, False),
    (["--retrieval", "vector", "--no-rerank", "--top-k", "2"], "vector", 2, 2, False),
])
def test_cli_selects_stages_and_passes_final_context(
    monkeypatch, capsys, stages, options, method, limit, top_k, enabled,
):
    monkeypatch.setattr("sys.argv", ["pipeline", "ask", "question", *options])
    pipeline.main()
    getattr(stages, method).assert_called_once_with("question", top_k=limit)
    if enabled:
        stages.rerank.assert_called_once_with("question", stages.hybrid.return_value, top_k=top_k)
    else:
        stages.rerank.assert_not_called()
    context = stages.rerank.return_value if enabled else stages.hybrid.return_value
    stages.generate.assert_called_once_with("question", context)
    assert [call[0] for call in stages.method_calls] == (
        [method, "rerank", "generate"] if enabled else [method, "generate"]
    )
    assert capsys.readouterr().out.strip() == "Policy answer"


@pytest.mark.parametrize("options", [["--top-k", "0"], ["--candidate-k", "-1"]])
def test_cli_rejects_invalid_counts_before_retrieval(monkeypatch, capsys, stages, options):
    monkeypatch.setattr("sys.argv", ["pipeline", "ask", "question", *options])
    with pytest.raises(SystemExit) as error:
        pipeline.main()
    assert error.value.code == 2 and stages.method_calls == []
    assert "must be positive integers" in capsys.readouterr().err


def test_index_command_does_not_rerank(monkeypatch, capsys, stages):
    index = Mock(return_value=49)
    monkeypatch.setattr(pipeline, "index_corpus", index)
    monkeypatch.setattr("sys.argv", ["pipeline", "index"])
    pipeline.main()
    index.assert_called_once_with()
    assert stages.method_calls == []
    assert "Indexed 49 chunks" in capsys.readouterr().out


@pytest.fixture
def index_inputs(monkeypatch, chunks):
    """Keep Chroma real; supply fixed chunking and embedding results."""
    monkeypatch.setattr(pipeline, "load_documents", Mock(return_value=[{"id": "source"}]))
    monkeypatch.setattr(pipeline, "load_model", Mock(return_value=Mock(tokenizer="test-tokenizer")))
    monkeypatch.setattr(pipeline, "chunk_documents", Mock(return_value=chunks[:2]))
    monkeypatch.setattr(pipeline, "embed_texts", Mock(return_value=[[1.0, 0.0], [0.0, 1.0]]))
    return chunks[:2]


def test_index_refresh_removes_stale_chunks_and_repeated_runs_keep_both_versions(tmp_path, index_inputs):
    stale = {"id": "obsolete", "text": "Old chunk layout", "metadata": {"version": "0.0"}}
    store_chunks([stale], [[1.0, 0.0]], tmp_path)
    for _ in range(2):
        assert pipeline.index_corpus(tmp_path) == 2
        assert set(get_collection(tmp_path).get()["ids"]) == {"remote-v1", "remote-v2"}
    pipeline.chunk_documents.assert_called_with([{"id": "source"}], "test-tokenizer")
    pipeline.embed_texts.assert_called_with([chunk["text"] for chunk in index_inputs])


def test_failed_index_write_does_not_delete_existing_chunks(tmp_path, monkeypatch, index_inputs):
    stale = {"id": "obsolete", "text": "Existing evidence", "metadata": {"version": "0.0"}}
    store_chunks([stale], [[1.0, 0.0]], tmp_path)
    monkeypatch.setattr(pipeline, "store_chunks", Mock(side_effect=RuntimeError("Write failed")))
    with pytest.raises(RuntimeError, match="Write failed"):
        pipeline.index_corpus(tmp_path)
    assert get_collection(tmp_path).get()["ids"] == ["obsolete"]
