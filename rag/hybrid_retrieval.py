"""Combine vector and keyword rankings with Reciprocal Rank Fusion (RRF)."""

from pathlib import Path

from rag.keyword_search import keyword_search
from rag.retrieval import retrieve
from rag.vector_store import DB_PATH


RRF_CONSTANT = 60


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
