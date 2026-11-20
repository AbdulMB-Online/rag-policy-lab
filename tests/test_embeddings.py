"""Embedding input handling with a mocked model; no downloads."""

from unittest.mock import Mock

import numpy as np
import pytest

from rag import indexing


@pytest.fixture
def model(monkeypatch):
    fake = Mock(max_seq_length=512)
    fake.tokenizer.encode.return_value = [1, 2, 3]
    fake.encode.return_value = np.array([[1.0, 0.0, 0.0]])
    monkeypatch.setattr(indexing, "load_model", Mock(return_value=fake))
    return fake


@pytest.mark.parametrize("is_query", [False, True])
def test_instruction_is_added_only_to_queries_and_normalization_is_requested(model, is_query):
    texts = ["How many remote days?"]
    assert indexing.embed_texts(texts, is_query=is_query) == [[1.0, 0.0, 0.0]]
    expected = (indexing.QUERY_INSTRUCTION if is_query else "") + texts[0]
    model.encode.assert_called_once_with([expected], normalize_embeddings=True, show_progress_bar=False)
    assert texts == ["How many remote days?"]


@pytest.mark.parametrize("tokens", [512, 513])
def test_model_limit_is_checked_without_silent_truncation(model, tokens):
    model.tokenizer.encode.return_value = [1] * tokens
    if tokens == 512:
        assert indexing.embed_texts(["Policy text"])
    else:
        with pytest.raises(ValueError, match="model limit is 512"):
            indexing.embed_texts(["Policy text"])
        model.encode.assert_not_called()
    model.tokenizer.encode.assert_called_once_with("Policy text", truncation=False, verbose=False)


@pytest.mark.parametrize("texts", [[], [" "], ["valid", None]])
def test_invalid_texts_are_rejected_before_loading_the_model(model, texts):
    with pytest.raises(ValueError, match="nonempty list of nonempty strings"):
        indexing.embed_texts(texts)
    indexing.load_model.assert_not_called()
