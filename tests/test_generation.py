"""Foundry request construction and failures with a mocked API client."""

from unittest.mock import MagicMock

import pytest
from httpx import Request
from openai import APIConnectionError

from rag import generation


@pytest.fixture
def api(monkeypatch):
    # Never read real credentials or contact Foundry in these unit tests.
    monkeypatch.setattr(generation, "load_dotenv", lambda path: None)
    for name, value in (("OPENAI_API_KEY", "test-key"),
                        ("OPENAI_BASE_URL", "https://example.invalid/openai/v1/"),
                        ("OPENAI_MODEL", "test-deployment")):
        monkeypatch.setenv(name, value)
    constructor = MagicMock()
    monkeypatch.setattr(generation, "OpenAI", constructor)
    return constructor, constructor.return_value.__enter__.return_value


def test_foundry_request_contains_question_and_retrieved_evidence(api):
    constructor, client = api
    client.chat.completions.create.return_value.choices[0].message.content = " Two remote days. "
    chunks = [{"id": "remote-v2:RW-03.1:0", "text": "Two remote days weekly."}]
    assert generation.generate_answer("  Remote days?  ", chunks) == "Two remote days."
    constructor.assert_called_once_with(
        api_key="test-key", base_url="https://example.invalid/openai/v1/", timeout=30, max_retries=0,
    )
    client.chat.completions.create.assert_called_once_with(
        model="test-deployment", messages=[
            {"role": "system", "content": generation.SYSTEM_PROMPT},
            {"role": "user", "content": "Policy excerpts:\n[remote-v2:RW-03.1:0]\n"
             "Two remote days weekly.\n\nQuestion: Remote days?"},
        ],
    )


@pytest.mark.parametrize("section_metadata,expected_section", [
    ({"section_path": "Remote Work Policy > RW-03.1: Weekly allowance", "section_id": "RW-03.1"},
     "Remote Work Policy > RW-03.1: Weekly allowance"),
    ({"section_id": "RW-03.1"}, "RW-03.1"),
])
def test_request_includes_source_metadata_and_attribution_instructions(api, section_metadata, expected_section):
    _, client = api
    answer = "Two remote days.\n\nSources: Remote Work Policy v2.0, section RW-03.1."
    client.chat.completions.create.return_value.choices[0].message.content = answer
    chunks = [{
        "id": "remote-v2:RW-03.1:0", "text": "Two remote days weekly.",
        "metadata": {"title": "Remote Work Policy", "version": "2.0",
                     "source": "data/policies/remote-work-v2.0.md", **section_metadata},
    }]
    assert generation.generate_answer("Remote days?", chunks) == answer
    messages = client.chat.completions.create.call_args.kwargs["messages"]
    context = messages[1]["content"]
    for expected in ("[remote-v2:RW-03.1:0]", "Document: Remote Work Policy", "Version: 2.0",
                     f"Section: {expected_section}", "File: data/policies/remote-work-v2.0.md",
                     "Two remote days weekly.", "Question: Remote days?"):
        assert expected in context
    assert "Sources:" in messages[0]["content"]
    assert "sections you used" in messages[0]["content"]
    assert "without inventing a source" in messages[0]["content"]


def test_context_keeps_policy_versions_and_their_evidence_separate():
    chunks = [{
        "id": f"remote-v{version}:RW-03.1:0", "text": f"Up to {days} remote days weekly.",
        "metadata": {"title": "Remote Work Policy", "version": version, "section_id": "RW-03.1",
                     "source": f"data/policies/remote-work-v{version}.md"},
    } for version, days in (("1.0", 3), ("2.0", 2))]
    blocks = generation.format_context(chunks).split("\n\n")
    assert len(blocks) == 2
    for block, chunk in zip(blocks, chunks, strict=True):
        assert block.startswith(f"[{chunk['id']}]\n")
        assert f"Version: {chunk['metadata']['version']}\n" in block
        assert f"File: {chunk['metadata']['source']}\n" in block
        assert block.endswith(chunk["text"])


def test_missing_configuration_fails_before_opening_the_client(api, monkeypatch):
    constructor, _ = api
    monkeypatch.delenv("OPENAI_MODEL")
    with pytest.raises(ValueError, match="Missing settings: OPENAI_MODEL"):
        generation.generate_answer("Question", [{"id": "source", "text": "Evidence"}])
    constructor.assert_not_called()


def test_api_failures_propagate_instead_of_returning_an_answer(api):
    _, client = api
    client.chat.completions.create.side_effect = APIConnectionError(
        request=Request("POST", "https://example.invalid"),
    )
    with pytest.raises(APIConnectionError):
        generation.generate_answer("Question", [{"id": "source", "text": "Evidence"}])


@pytest.mark.parametrize("content", [None, " "])
def test_empty_model_answers_are_rejected(api, content):
    _, client = api
    client.chat.completions.create.return_value.choices[0].message.content = content
    with pytest.raises(ValueError, match="model returned no answer"):
        generation.generate_answer("Question", [{"id": "source", "text": "Evidence"}])
