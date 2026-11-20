"""Index company policies or ask a question using the stored corpus and LLM."""

import argparse

from openai import APIError

from rag.generation import generate_answer
from rag.indexing import DB_PATH, index_corpus
from rag.reranking import rerank
from rag.retrieval import hybrid_retrieve, retrieve


class RAGPipeline:
    """Coordinate retrieval, optional reranking, and generation with supplied components."""

    def __init__(
        self, retriever=None, reranker=None, generator=None, *,
        top_k: int = 4, candidate_k: int = 12, use_reranking: bool = True,
    ):
        if any(type(count) is not int or count < 1 for count in (top_k, candidate_k)):
            raise ValueError("top-k and candidate-k must be positive integers")
        self.retriever = hybrid_retrieve if retriever is None else retriever
        self.reranker = rerank if reranker is None else reranker
        self.generator = generate_answer if generator is None else generator
        self.top_k = top_k
        self.candidate_k = max(candidate_k, top_k)
        self.use_reranking = use_reranking

    def retrieve_context(self, question: str) -> list[dict]:
        """Return the final context without calling the LLM."""
        if not isinstance(question, str) or not question.strip():
            raise ValueError("Provide a nonempty question")
        limit = self.candidate_k if self.use_reranking else self.top_k
        chunks = self.retriever(question, top_k=limit)
        if self.use_reranking:
            chunks = self.reranker(question, chunks, top_k=self.top_k)
        return chunks

    def ask(self, question: str) -> str:
        """Generate an answer from the selected policy context."""
        return self.generator(question, self.retrieve_context(question))



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("index", help="Refresh the stored corpus from policy sources")
    ask = commands.add_parser("ask", help="Answer a question using retrieved policy chunks")
    ask.add_argument("question", help="Natural-language question about company policy")
    ask.add_argument("--top-k", type=int, default=4, help="Number of context chunks (default: 4)")
    ask.add_argument(
        "--candidate-k", type=int, default=12,
        help="Chunks to retrieve before reranking; at least top-k (default: 12)",
    )
    ask.add_argument("--no-rerank", action="store_true", help="Use retrieval order without reranking")
    ask.add_argument(
        "--retrieval", choices=("hybrid", "vector"), default="hybrid",
        help="Retrieval method (default: hybrid)",
    )
    args = parser.parse_args()
    try:
        if args.command == "index":
            count = index_corpus()
            print(f"Indexed {count} chunks in {DB_PATH}")
        else:
            pipeline = RAGPipeline(
                retriever=hybrid_retrieve if args.retrieval == "hybrid" else retrieve,
                top_k=args.top_k, candidate_k=args.candidate_k,
                use_reranking=not args.no_rerank,
            )
            print(pipeline.ask(args.question))
    except (ValueError, OSError) as error:
        parser.error(str(error))
    except APIError as error:
        parser.exit(1, f"LLM request failed ({type(error).__name__}); check Foundry settings and connectivity.\n")


if __name__ == "__main__":
    main()
