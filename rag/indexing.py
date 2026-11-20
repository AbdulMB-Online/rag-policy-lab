"""Embed policy text, persist it in Chroma, and refresh the corpus index."""

from functools import cache
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from rag.ingestion import chunk_documents, load_documents


MODEL = "BAAI/bge-small-en-v1.5"
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
DB_PATH = Path(__file__).resolve().parents[1] / ".venv" / "chroma"
COLLECTION_NAME = "policy-chunks"


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


def get_collection(path: Path = DB_PATH):
    """Open the persistent collection, creating it on first use."""
    client = chromadb.PersistentClient(path=str(path))
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=None,
        metadata={"embedding_model": MODEL},
        configuration={"hnsw": {"space": "cosine"}},
    )
    if (collection.metadata or {}).get("embedding_model") != MODEL:
        raise ValueError("Collection uses a different embedding model; use a fresh database")
    if collection.configuration["hnsw"]["space"] != "cosine":
        raise ValueError("Collection must use cosine distance")
    return collection


def store_chunks(chunks: list[dict], embeddings: list[list[float]], path: Path = DB_PATH) -> int:
    """Upsert supplied chunks and return the total count; existing other IDs remain."""
    if not chunks or len(chunks) != len(embeddings):
        raise ValueError("Provide nonempty chunks and one embedding per chunk")
    ids = [chunk["id"] for chunk in chunks]
    if len(ids) != len(set(ids)):
        raise ValueError("Chunk IDs must be unique")
    collection = get_collection(path)
    collection.upsert(
        ids=ids,
        documents=[chunk["text"] for chunk in chunks],
        embeddings=embeddings,
        metadatas=[
            {key: value for key, value in chunk["metadata"].items() if value is not None}
            for chunk in chunks
        ],
    )
    return collection.count()


def index_corpus(path: Path = DB_PATH) -> int:
    """Store the current corpus and remove obsolete chunk IDs after a successful upsert."""
    chunks = chunk_documents(load_documents(), load_model().tokenizer)
    vectors = embed_texts([chunk["text"] for chunk in chunks])
    store_chunks(chunks, vectors, path)
    collection = get_collection(path)
    current_ids = {chunk["id"] for chunk in chunks}
    stale_ids = set(collection.get(include=[])["ids"]) - current_ids
    if stale_ids:
        collection.delete(ids=sorted(stale_ids))
    return collection.count()


if __name__ == "__main__":
    print(f"Indexed {index_corpus()} chunks in {DB_PATH}")
