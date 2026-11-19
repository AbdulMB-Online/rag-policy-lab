"""Index company policies or ask a question using the stored corpus and LLM."""

import argparse
from pathlib import Path

from openai import APIError

from rag.chunking import chunk_documents
from rag.embeddings import embed_texts, load_model
from rag.generation import generate_answer
from rag.ingestion import load_documents
from rag.retrieval import retrieve
from rag.vector_store import DB_PATH, get_collection, store_chunks


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("index", help="Refresh the stored corpus from policy sources")
    ask = commands.add_parser("ask", help="Answer a question using retrieved policy chunks")
    ask.add_argument("question", help="Natural-language question about company policy")
    ask.add_argument("--top-k", type=int, default=4, help="Number of context chunks (default: 4)")
    args = parser.parse_args()
    try:
        if args.command == "index":
            count = index_corpus()
            print(f"Indexed {count} chunks in {DB_PATH}")
        else:
            chunks = retrieve(args.question, top_k=args.top_k)
            print(generate_answer(args.question, chunks))
    except (ValueError, OSError) as error:
        parser.error(str(error))
    except APIError as error:
        parser.exit(1, f"LLM request failed ({type(error).__name__}); check Foundry settings and connectivity.\n")


if __name__ == "__main__":
    main()
