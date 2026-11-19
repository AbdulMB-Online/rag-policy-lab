"""Controlled reranking and CLI checks; no model downloads or LLM calls."""

from copy import deepcopy
from unittest.mock import Mock

import pytest
from torch.nn import Identity

from rag import pipeline, reranking


@pytest.fixture(autouse=True)
def forbid_model_downloads(monkeypatch):
    monkeypatch.setattr(reranking, "CrossEncoder", Mock(
        side_effect=AssertionError("Tests must not load a pretrained model"),
    ))


@pytest.fixture
def model(monkeypatch):
    fake = Mock(max_length=512)
    fake.tokenizer.encode.return_value = [1, 2, 3]
    monkeypatch.setattr(reranking, "load_model", Mock(return_value=fake))
    return fake


@pytest.fixture
def chunks():
    return [
        {"id": "remote-v1", "text": "Three remote days", "metadata": {"version": "1.0"}},
        {"id": "remote-v2", "text": "Two remote days", "metadata": {"version": "2.0"}},
        {"id": "expense", "text": "Meal reimbursement", "metadata": {}, "rrf_score": 0.02},
    ]


def test_model_is_cached_and_uses_raw_scores(monkeypatch):
    constructor = Mock()
    monkeypatch.setattr(reranking, "CrossEncoder", constructor)
    reranking.load_model.cache_clear()
    try:
        assert reranking.load_model() is reranking.load_model()
        constructor.assert_called_once()
        assert constructor.call_args.args == ("cross-encoder/ms-marco-MiniLM-L6-v2",)
        options = constructor.call_args.kwargs
        assert options["device"] == "cpu" and options["max_length"] == 512
        assert isinstance(options["activation_fn"], Identity)
    finally:
        reranking.load_model.cache_clear()


def test_ranking_preserves_inputs_metadata_and_tied_retrieval_order(model, chunks):
    original = deepcopy(chunks)
    model.predict.return_value = [-8.0, 2.0, 2.0]
    hits = reranking.rerank("  remote days?  ", chunks, top_k=2)
    assert [hit["id"] for hit in hits] == ["remote-v2", "expense"]
    assert hits == [{**chunks[1], "rerank_score": 2.0}, {**chunks[2], "rerank_score": 2.0}]
    assert chunks == original and hits[0] is not chunks[1]
    model.predict.assert_called_once_with(
        [("remote days?", chunk["text"]) for chunk in chunks], show_progress_bar=False,
    )


def test_large_top_k_keeps_distinct_versions_and_negative_scores(model, chunks):
    model.predict.return_value = [-1.0, -2.0, -3.0]
    hits = reranking.rerank("question", chunks, top_k=20)
    assert [hit["id"] for hit in hits] == [chunk["id"] for chunk in chunks]
    assert [hit["metadata"]["version"] for hit in hits[:2]] == ["1.0", "2.0"]
    assert hits[0]["rerank_score"] == -1.0


def test_empty_candidates_skip_model_loading(model):
    assert reranking.rerank("question", []) == []
    reranking.load_model.assert_not_called()


@pytest.mark.parametrize("question", ["", " ", None, 123])
def test_invalid_questions_fail_before_loading(model, question):
    with pytest.raises(ValueError, match="nonempty question"):
        reranking.rerank(question, [])
    reranking.load_model.assert_not_called()


@pytest.mark.parametrize("top_k", [0, -1, True, 1.5, "4"])
def test_invalid_limits_fail_before_loading(model, top_k):
    with pytest.raises(ValueError, match="top_k"):
        reranking.rerank("question", [], top_k=top_k)
    reranking.load_model.assert_not_called()


@pytest.mark.parametrize("items", [None, {}, [None], [{}], [{"text": " "}], [{"text": 123}]])
def test_invalid_chunks_fail_before_loading(model, items):
    with pytest.raises(ValueError, match="nonempty text"):
        reranking.rerank("question", items)
    reranking.load_model.assert_not_called()


@pytest.mark.parametrize("tokens", [512, 513])
def test_limit_checks_complete_pair_without_truncation(model, chunks, tokens):
    model.tokenizer.encode.return_value = [1] * tokens
    model.predict.return_value = [0.5]
    if tokens == 512:
        assert len(reranking.rerank("question", chunks[:1])) == 1
    else:
        with pytest.raises(ValueError, match="model limit is 512"):
            reranking.rerank("question", chunks[:1])
        model.predict.assert_not_called()
    model.tokenizer.encode.assert_called_once_with(
        "question", chunks[0]["text"], truncation=False, verbose=False,
    )


def test_score_count_mismatch_is_rejected(model, chunks):
    model.predict.return_value = [1.0]
    with pytest.raises(ValueError):
        reranking.rerank("question", chunks)


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
