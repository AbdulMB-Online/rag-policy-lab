"""Persist policy chunks and explicit embeddings in a local Chroma collection."""

from pathlib import Path

import chromadb

from rag.embeddings import MODEL


DB_PATH = Path(__file__).resolve().parents[1] / ".venv" / "chroma"
COLLECTION_NAME = "policy-chunks"


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


if __name__ == "__main__":
    from rag.chunking import chunk_documents
    from rag.embeddings import embed_texts, load_model
    from rag.ingestion import load_documents

    chunks = chunk_documents(load_documents(), load_model().tokenizer)
    vectors = embed_texts([chunk["text"] for chunk in chunks])
    count = store_chunks(chunks, vectors)
    print(f"Database: {DB_PATH}")
    print(f"Collection: {COLLECTION_NAME}")
    print(f"Upserted {len(chunks)} chunks; collection count: {count}")
