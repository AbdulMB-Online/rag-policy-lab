"""Generate normalized vectors for policy chunks and questions using one model."""

from functools import cache

from sentence_transformers import SentenceTransformer


MODEL = "BAAI/bge-small-en-v1.5"
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


@cache
def load_model():
    """Load the embedding model once per Python process."""
    return SentenceTransformer(MODEL, device="cpu")


def embed_texts(texts: list[str], *, is_query: bool = False) -> list[list[float]]:
    """Embed text; add BGE's retrieval instruction only when is_query=True."""
    if not isinstance(texts, list) or not texts or any(
        not isinstance(text, str) or not text.strip() for text in texts
    ):
        raise ValueError("Provide a nonempty list of nonempty strings")
    if is_query:
        texts = [QUERY_INSTRUCTION + text for text in texts]
    model = load_model()
    for index, text in enumerate(texts):
        count = len(model.tokenizer.encode(text, truncation=False, verbose=False))
        if count > model.max_seq_length:
            raise ValueError(f"Text {index} has {count} tokens; model limit is {model.max_seq_length}")
    return model.encode(texts, normalize_embeddings=True, show_progress_bar=False).tolist()


if __name__ == "__main__":
    from rag.chunking import chunk_documents
    from rag.ingestion import load_documents

    chunks = chunk_documents(load_documents(), load_model().tokenizer)
    vectors = embed_texts([chunk["text"] for chunk in chunks])
    print(f"Model: {MODEL}")
    print(f"Model input limit: {load_model().max_seq_length} tokens")
    print(f"Chunk embeddings: ({len(vectors)}, {len(vectors[0])})")
    query = embed_texts(["How many days can I work from home each week?"], is_query=True)
    print(f"Question embedding: ({len(query)}, {len(query[0])})")
