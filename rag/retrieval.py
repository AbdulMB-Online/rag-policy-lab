"""Retrieve policy chunks for a natural-language question from persistent Chroma."""

import argparse
from pathlib import Path

from rag.embeddings import embed_texts
from rag.vector_store import DB_PATH, get_collection


def retrieve(question: str, top_k: int = 4, path: Path = DB_PATH) -> list[dict]:
    """Return ranked IDs, text, metadata, and cosine distances; lower is closer."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Provide a nonempty question")
    if type(top_k) is not int or top_k < 1:
        raise ValueError("top_k must be a positive integer")
    if not (Path(path) / "chroma.sqlite3").is_file():
        raise ValueError("Database is missing; run python -m rag.vector_store first")
    collection = get_collection(path)
    count = collection.count()
    if not count:
        raise ValueError("Collection is empty; run python -m rag.vector_store first")
    query = embed_texts([question.strip()], is_query=True)
    results = collection.query(
        query_embeddings=query,
        n_results=min(top_k, count),
        include=["documents", "metadatas", "distances"],
    )
    return [
        {"id": identifier, "text": text, "metadata": metadata, "distance": float(distance)}
        for identifier, text, metadata, distance in zip(
            results["ids"][0], results["documents"][0],
            results["metadatas"][0], results["distances"][0], strict=True,
        )
    ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", help="Natural-language question about company policy")
    parser.add_argument("--top-k", type=int, default=4, help="Number of chunks to retrieve (default: 4)")
    args = parser.parse_args()
    try:
        hits = retrieve(args.question, args.top_k)
    except ValueError as error:
        parser.error(str(error))
    for rank, hit in enumerate(hits, start=1):
        metadata = hit["metadata"]
        print(f"\n{rank}. {hit['id']} | cosine distance: {hit['distance']:.4f}")
        print(f"Source: {metadata['source']} | {metadata['section_path']} | {metadata['status']}")
        print(hit["text"])
