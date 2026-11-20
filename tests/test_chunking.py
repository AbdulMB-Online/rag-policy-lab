"""Chunking unit tests with one token per word and two special tokens.

No model downloads; real BGE tokenization is exercised by corpus integration.
"""

import pytest

from rag.ingestion import chunk_documents


class WordTokenizer:
    def encode(self, text, add_special_tokens=True, **kwargs):
        return text.split() + (["[CLS]", "[SEP]"] if add_special_tokens else [])


def document(text):
    return {
        "id": "test-v2.0", "text": text,
        "metadata": {"version": "2.0", "status": "current", "source": "data/policies/test.md"},
    }


def test_chunks_preserve_heading_context_source_metadata_and_stable_ids():
    policy = document(
        "# Test policy\n\n## TEST-01: Rules\n\n### TEST-01.1: Days\n\nTwo days weekly.\n\n"
        "## TEST-02: Requests\n\nAsk your manager."
    )
    chunks = chunk_documents([policy], WordTokenizer())
    assert chunks == chunk_documents([policy], WordTokenizer())
    assert [chunk["id"] for chunk in chunks] == ["test-v2.0:TEST-01.1:0", "test-v2.0:TEST-02:0"]
    assert chunks[0]["text"] == (
        "# Test policy\n## TEST-01: Rules\n### TEST-01.1: Days\n\nTwo days weekly."
    )
    assert chunks[1]["text"] == "# Test policy\n## TEST-02: Requests\n\nAsk your manager."
    for chunk in chunks:
        assert all(chunk["metadata"][key] == value for key, value in policy["metadata"].items())
    assert chunks[0]["metadata"]["section_path"] == "Test policy > TEST-01: Rules > TEST-01.1: Days"


@pytest.mark.parametrize("overlap", [0, 5])
def test_long_prose_respects_budget_overlap_and_preserves_every_word(overlap):
    words = [f"word{i}" for i in range(80)]
    policy = document("# Test policy\n\n## TEST-01: Rules\n\n" + " ".join(words))
    chunks = chunk_documents([policy], WordTokenizer(), max_tokens=30, overlap=overlap)
    assert len(chunks) > 1
    assert len({chunk["id"] for chunk in chunks}) == len(chunks)
    recovered = []
    previous_end = 0
    for index, chunk in enumerate(chunks):
        actual = chunk["text"].split("\n\n", 1)[1].split()
        start, end = words.index(actual[0]), words.index(actual[-1]) + 1
        assert actual == words[start:end]
        assert len(chunk["text"].split()) + 2 <= 30
        assert chunk["metadata"]["token_count"] == len(chunk["text"].split()) + 2
        shared = previous_end - start
        assert 0 <= shared <= overlap
        if index and overlap:
            assert shared > 0
        recovered.extend(actual[shared:])
        previous_end = end
    assert recovered == words


def test_caption_and_table_stay_together_in_their_own_chunk():
    table = "TEST-T01: Limits\n\n| Item | Limit |\n| --- | --- |\n| Meal | $75 |"
    policy = document("# Test policy\n\n## TEST-01: Rules\n\nRead the limits.\n\n" + table)
    chunks = chunk_documents([policy], WordTokenizer())
    assert len(chunks) == 2
    assert chunks[0]["metadata"]["content_type"] == "text"
    assert chunks[1]["text"] == "# Test policy\n## TEST-01: Rules\n\n" + table
    assert chunks[1]["metadata"]["content_type"] == "table"
    assert chunks[1]["metadata"]["table_id"] == "TEST-T01"


def test_oversized_table_is_rejected_instead_of_split():
    table = "TEST-T01: Limits\n\n| Item | Limit |\n| --- | --- |\n" + "| Meal | $75 |\n" * 10
    policy = document("# Test policy\n\n## TEST-01: Rules\n\n" + table)
    with pytest.raises(ValueError, match="table exceeds 30 tokens"):
        chunk_documents([policy], WordTokenizer(), max_tokens=30, overlap=5)


@pytest.mark.parametrize("max_tokens,overlap", [(513, 80), (30, 30), (30, -1)])
def test_invalid_budgets_are_rejected(max_tokens, overlap):
    with pytest.raises(ValueError, match="0 <= overlap < max_tokens <= 512"):
        chunk_documents([], WordTokenizer(), max_tokens=max_tokens, overlap=overlap)
