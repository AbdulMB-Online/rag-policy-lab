"""Eight-question real-corpus evaluation and credentialed Foundry answers.

Use pytest -s for metrics; -k 'not live' excludes real LLM requests.
"""

import json
import os
import re

import pytest
from dotenv import load_dotenv
from openai import APIError

from scripts import evaluate as evaluation
from rag.generation import ROOT
from rag.indexing import index_corpus
from rag.indexing import get_collection


CASES = evaluation.load_cases()


@pytest.fixture(scope="module")
def corpus_db(tmp_path_factory):
    """Index real policies once, independently of the working database."""
    path = tmp_path_factory.mktemp("evaluation")
    index_corpus(path)
    return path


@pytest.fixture(scope="module")
def retrieval_report(corpus_db):
    report = evaluation.evaluate(path=corpus_db)
    print("\nRetrieval evaluation:", json.dumps(report["retrieval"], indent=2))
    return report


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["id"])
def test_expected_evidence_exists_in_real_corpus(case, corpus_db):
    records = get_collection(corpus_db).get(ids=case["expected_chunk_ids"])
    assert set(records["ids"]) == set(case["expected_chunk_ids"])
    if case.get("expected_table_id"):
        assert all(metadata.get("table_id") == case["expected_table_id"]
                   and metadata.get("content_type") == "table"
                   for metadata in records["metadatas"])


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["id"])
def test_final_retrieval_recovers_expected_evidence(case, retrieval_report):
    rows = retrieval_report["retrieval"]["reranked"]["cases"]
    row = next(row for row in rows if row["id"] == case["id"])
    assert row["recall_at_4"] == 1, row


def test_live_answers(corpus_db):
    """Check eight real answers and their sources; missing credentials skip."""
    load_dotenv(ROOT / ".env")
    missing = [name for name in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_MODEL")
               if not os.getenv(name, "").strip()]
    if missing:
        pytest.skip("Live evaluation requires " + ", ".join(missing))
    try:
        report = evaluation.evaluate(live=True, path=corpus_db)
    except APIError as error:
        pytest.fail(f"Foundry live evaluation failed ({type(error).__name__})", pytrace=False)
    answers = report["answers"]
    print("\nLive answer evaluation:", json.dumps(answers, indent=2))
    failures = {row["id"]: row["missing_checks"] for row in answers["cases"] if not row["passed"]}
    assert not failures, failures
    assert answers["pass_rate"] == 1

    records = get_collection(corpus_db).get(include=["metadatas"])
    metadata = dict(zip(records["ids"], records["metadatas"], strict=True))
    contexts = {row["id"]: set(row["retrieved_ids"])
                for row in report["retrieval"]["reranked"]["cases"]}
    cases = {case["id"]: case for case in CASES}
    for row in answers["cases"]:
        case = cases[row["id"]]
        parts = re.split(r"(?im)^\s*(?:\*\*)?Sources:(?:\*\*)?\s*", row["answer"], maxsplit=1)
        assert len(parts) == 2, f"{row['id']}: missing Sources line"
        body, sources = parts
        # Source labels must not supply facts missing from the answer itself.
        assert evaluation.score_answer(case, body)["passed"], f"{row['id']}: answer body misses required facts"
        sources = re.sub(r"[*_`]", "", sources).casefold()
        supported = any(
            source["title"].casefold() in sources
            and re.search(rf"\b{re.escape(source['section_id'])}(?![\w.])", sources, re.IGNORECASE)
            and re.search(rf"(?<![\w.])(?:v(?:ersion)?\s*)?{re.escape(source['version'])}(?![\w.])", sources, re.IGNORECASE)
            for identifier in case["expected_chunk_ids"] if identifier in contexts[row["id"]]
            for source in [metadata[identifier]]
        )
        assert supported, f"{row['id']}: Sources must name the expected retrieved policy, version, and section"
    print(f"Source attribution: {len(answers['cases'])}/{len(CASES)} answers passed")
