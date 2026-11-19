"""Score retrieved question–chunk pairs and select the most relevant chunks."""

from functools import cache

from sentence_transformers import CrossEncoder
from torch.nn import Identity


MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"


@cache
def load_model():
    """Load the reranker once per process; return raw relevance scores."""
    return CrossEncoder(MODEL, device="cpu", max_length=512, activation_fn=Identity())


def rerank(question: str, chunks: list[dict], top_k: int = 4) -> list[dict]:
    """Return scored copies, highest first; ties preserve the retrieval order."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Provide a nonempty question")
    if type(top_k) is not int or top_k < 1:
        raise ValueError("top_k must be a positive integer")
    if not isinstance(chunks, list) or any(
        not isinstance(chunk, dict) or not isinstance(chunk.get("text"), str)
        or not chunk["text"].strip() for chunk in chunks
    ):
        raise ValueError("Provide a list of chunks with nonempty text")
    if not chunks:
        return []

    model = load_model()
    pairs = [(question.strip(), chunk["text"]) for chunk in chunks]
    for index, (query, text) in enumerate(pairs):
        count = len(model.tokenizer.encode(query, text, truncation=False, verbose=False))
        if count > model.max_length:
            raise ValueError(
                f"Question and chunk {index} have {count} tokens; model limit is {model.max_length}"
            )
    scores = model.predict(pairs, show_progress_bar=False)
    ranked = [
        {**chunk, "rerank_score": float(score)}
        for chunk, score in zip(chunks, scores, strict=True)
    ]
    return sorted(ranked, key=lambda chunk: chunk["rerank_score"], reverse=True)[:top_k]
