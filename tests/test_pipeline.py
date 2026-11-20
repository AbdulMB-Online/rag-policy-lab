"""Pipeline orchestration checks without model or API calls."""

from unittest.mock import Mock

import pytest

from rag import pipeline


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
