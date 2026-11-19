"""Rank stored policy chunks by exact query terms and policy identifiers."""

from pathlib import Path
import re

from rag.vector_store import DB_PATH, get_collection


TOKEN = re.compile(r"[a-z0-9]+(?:[.-][a-z0-9]+)*")
IDENTIFIER = re.compile(r"(?:rw|exp|pto|sec)-(?:t\d{2}|\d{2}(?:\.\d+)*)")
STOP_WORDS = set(
    "a an and are as at be by can do does for from how i in is it many me my "
    "of on or per please s show t tell that the this to what when where which with".split()
)


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
