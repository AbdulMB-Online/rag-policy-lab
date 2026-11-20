"""Evaluate fixed policy questions; add --live to check eight generated answers."""

import argparse
import json
import re
import sys
from pathlib import Path

# Support both python -m scripts.evaluate and python scripts/evaluate.py.
if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from openai import APIError

from rag.generation import generate_answer
from rag.retrieval import hybrid_retrieve
from rag.reranking import rerank
from rag.retrieval import retrieve
from rag.indexing import DB_PATH


CASES_PATH = Path(__file__).resolve().parents[1] / "data" / "evaluation.json"
MODES = ("vector", "hybrid", "reranked")


def load_cases(path: Path = CASES_PATH) -> list[dict]:
    """Load at least eight unique cases and validate their evidence and checks."""
    cases = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(cases, list) or len(cases) < 8:
        raise ValueError("Provide at least eight evaluation cases")
    for case in cases:
        if not isinstance(case, dict) or any(
            not isinstance(case.get(key), str) or not case[key].strip()
            for key in ("id", "question", "expected_answer")
        ):
            raise ValueError("Each case needs an ID, question, and expected answer")
        ids = case.get("expected_chunk_ids")
        checks = case.get("answer_checks")
        if not isinstance(ids, list) or not ids or any(not isinstance(i, str) or not i.strip() for i in ids):
            raise ValueError(f"{case['id']}: provide acceptable chunk IDs")
        if not isinstance(checks, dict) or not checks:
            raise ValueError(f"{case['id']}: provide answer checks")
        for label, patterns in checks.items():
            if not label.strip() or not isinstance(patterns, list) or not patterns:
                raise ValueError(f"{case['id']}: each check needs a label and patterns")
            for pattern in patterns:
                if not isinstance(pattern, str) or not pattern.strip():
                    raise ValueError(f"{case['id']}: patterns must be nonempty strings")
                re.compile(pattern, re.IGNORECASE)
        table = case.get("expected_table_id")
        if table is not None and (not isinstance(table, str) or not table.strip()):
            raise ValueError(f"{case['id']}: invalid table ID")
        if not score_answer(case, case["expected_answer"])["passed"]:
            raise ValueError(f"{case['id']}: expected answer fails its own checks")
    if any(len({case[key] for case in cases}) != len(cases) for key in ("id", "question")):
        raise ValueError("Case IDs and questions must be unique")
    if not any(case.get("expected_table_id") for case in cases):
        raise ValueError("Include at least one table question")
    return cases


def score_retrieval(case: dict, chunks: list[dict]) -> dict:
    """Score one evidence requirement; listed chunk IDs are alternatives."""
    table = case.get("expected_table_id")
    hits = chunks[:4]
    rank = next((index for index, hit in enumerate(hits, 1)
                 if hit["id"] in case["expected_chunk_ids"]
                 and (not table or (hit["metadata"].get("table_id") == table
                                    and hit["metadata"].get("content_type") == "table"))), None)
    return {
        "rank": rank,
        "hit_at_3": int(rank is not None and rank <= 3),
        "recall_at_4": int(rank is not None),
        "retrieved_ids": [hit["id"] for hit in hits],
    }


def score_answer(case: dict, answer: str) -> dict:
    """Require every fact, accepting any pattern within each fact's alternatives."""
    if not isinstance(answer, str):
        raise ValueError("Answer must be text")
    text = re.sub(r"[*_`]", "", answer)
    missing = [label for label, patterns in case["answer_checks"].items()
               if not any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns)]
    return {"passed": not missing, "missing_checks": missing}


def retrieve_context(question: str, mode: str, path: Path = DB_PATH) -> list[dict]:
    """Use the production components with four final chunks and 12 candidates."""
    if mode == "vector":
        return retrieve(question, top_k=4, path=path)
    if mode == "hybrid":
        return hybrid_retrieve(question, top_k=4, path=path)
    if mode == "reranked":
        return rerank(question, hybrid_retrieve(question, top_k=12, path=path), top_k=4)
    raise ValueError(f"Unknown retrieval mode: {mode}")


def evaluate(cases: list[dict] | None = None, *, live: bool = False, path: Path = DB_PATH) -> dict:
    """Return real retrieval metrics and optional live-answer results; never rebuild the index."""
    cases = load_cases() if cases is None else cases
    if not cases:
        raise ValueError("Provide evaluation cases")
    report = {"retrieval": {}, "answers": None}
    contexts = {}
    for mode in MODES:
        rows = []
        for case in cases:
            hits = retrieve_context(case["question"], mode, path)
            rows.append({"id": case["id"], "question": case["question"], **score_retrieval(case, hits)})
            if mode == "reranked":
                contexts[case["id"]] = hits
        report["retrieval"][mode] = {
            "hit_at_3": sum(row["hit_at_3"] for row in rows) / len(rows),
            "recall_at_4": sum(row["recall_at_4"] for row in rows) / len(rows),
            "cases": rows,
        }
    if live:
        rows = []
        for case in cases:
            answer = generate_answer(case["question"], contexts[case["id"]])
            rows.append({"id": case["id"], "answer": answer, **score_answer(case, answer)})
        report["answers"] = {
            "pass_rate": sum(row["passed"] for row in rows) / len(rows), "cases": rows,
        }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="Generate and check answers using Foundry")
    args = parser.parse_args()
    try:
        report = evaluate(live=args.live)
    except APIError as error:
        parser.exit(1, f"Live generation failed ({type(error).__name__}); evaluation is incomplete.\n")
    except (ValueError, OSError, re.error) as error:
        parser.error(str(error))
    print(json.dumps(report, indent=2))
    if report["retrieval"]["reranked"]["recall_at_4"] < 1:
        parser.exit(1, "Final retrieval missed expected evidence; see results above.\n")
    if args.live and report["answers"]["pass_rate"] < 1:
        parser.exit(1, "Some answers failed key-information checks; see results above.\n")


if __name__ == "__main__":
    main()
