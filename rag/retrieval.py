"""Retrieve policy chunks with vector search, keyword matching, or hybrid fusion."""

import argparse
import re
from pathlib import Path

from rag.indexing import DB_PATH, embed_texts, get_collection


TOKEN = re.compile(r"[a-z0-9]+(?:[.-][a-z0-9]+)*")
IDENTIFIER = re.compile(r"(?:rw|exp|pto|sec)-(?:t\d{2}|\d{2}(?:\.\d+)*)")
STOP_WORDS = set(
    "a an and are as at be by can do does for from how i in is it many me my "
    "of on or per please s show t tell that the this to what when where which with".split()
)
RRF_CONSTANT = 60


def retrieve(question: str, top_k: int = 4, path: Path = DB_PATH) -> list[dict]:
    """Return ranked IDs, text, metadata, and cosine distances; lower is closer."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Provide a nonempty question")
    if type(top_k) is not int or top_k < 1:
        raise ValueError("top_k must be a positive integer")
    if not (Path(path) / "chroma.sqlite3").is_file():
        raise ValueError("Database is missing; run python -m rag.pipeline index first")
    collection = get_collection(path)
    count = collection.count()
    if not count:
        raise ValueError("Collection is empty; run python -m rag.pipeline index first")
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


def keyword_search(question: str, top_k: int = 12, path: Path = DB_PATH) -> list[dict]:
    """Search all stored text; prefer exact ID owners, then distinct term matches."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Provide a nonempty question")
    if type(top_k) is not int or top_k < 1:
        raise ValueError("top_k must be a positive integer")
    if not (Path(path) / "chroma.sqlite3").is_file():
        raise ValueError("Database is missing; run python -m rag.pipeline index first")
    collection = get_collection(path)
    if not collection.count():
        raise ValueError("Collection is empty; run python -m rag.pipeline index first")

    terms = set(TOKEN.findall(question.casefold())) - STOP_WORDS
    if not terms:
        return []
    query_ids = {term for term in terms if IDENTIFIER.fullmatch(term)}
    records = collection.get(include=["documents", "metadatas"])
    hits = []
    for identifier, text, metadata in zip(
        records["ids"], records["documents"], records["metadatas"], strict=True,
    ):
        score = len(terms & set(TOKEN.findall(text.casefold())))
        if score:
            hits.append({
                "id": identifier, "text": text, "metadata": metadata or {},
                "keyword_score": score,
            })

    def sort_key(hit):
        owns_id = any(
            (hit["metadata"].get(field) or "").casefold() in query_ids
            for field in ("section_id", "table_id")
        )
        return (not owns_id, -hit["keyword_score"], hit["id"])

    return sorted(hits, key=sort_key)[:top_k]


def hybrid_retrieve(
    question: str, top_k: int = 4, path: Path = DB_PATH, candidate_k: int = 12,
) -> list[dict]:
    """Run vector search, then keyword search, and return the highest fused scores."""
    if type(top_k) is not int or top_k < 1:
        raise ValueError("top_k must be a positive integer")
    if type(candidate_k) is not int or candidate_k < 1:
        raise ValueError("candidate_k must be a positive integer")
    limit = max(candidate_k, top_k)
    vector_hits = retrieve(question, top_k=limit, path=path)
    keyword_hits = keyword_search(question, top_k=limit, path=path)

    candidates = {}
    for method, hits in (("vector", vector_hits), ("keyword", keyword_hits)):
        for rank, hit in enumerate(hits, start=1):
            candidate = candidates.setdefault(hit["id"], {
                "id": hit["id"], "text": hit["text"], "metadata": hit["metadata"],
                "distance": None, "rrf_score": 0.0,
                "vector_rank": None, "keyword_rank": None,
            })
            candidate["rrf_score"] += 1 / (RRF_CONSTANT + rank)
            candidate[f"{method}_rank"] = rank
            if method == "vector":
                candidate["distance"] = hit["distance"]

    return sorted(
        candidates.values(), key=lambda hit: (-hit["rrf_score"], hit["id"]),
    )[:top_k]


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
