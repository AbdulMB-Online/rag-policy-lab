# Part 5: Evaluation harness

Status: Dataset, runner, and pytest harness implemented. Real retrieval passed
all eight questions in every mode. The user confirmed a successful Python live
test with eight real Foundry answers; terminal evidence was shared on
2026-11-20. CI integration and hosted verification remain pending.

## Objective

Evaluate retrieval against eight fixed policy questions and check that real
generated answers contain the expected key information. At least one question
must retrieve its answer from an actual Markdown table chunk.

## Scope

Compare vector search, hybrid search, and hybrid search followed by reranking.
Generate eight live answers per evaluation run using the final hybrid-plus-
reranking pipeline. Reuse the existing corpus, index, models, and generation
component. Source-version diagnosis remains Part 6; answer citations remain
Part 7.

## Decisions and rationale

| Decision | Rationale |
| --- | --- |
| Freeze the eight questions and expected evidence before measuring retrieval. | Makes repeated comparisons consistent and avoids changing questions to hide failures. |
| Evaluate all three retrieval modes with four final chunks. | Compares the same context budget; reranking selects from 12 hybrid candidates. |
| Report Hit@3, Recall@4, answer key-information pass rate, and table-specific success. | Hit@3 replaces Hit@1 at the user's request; Recall@4 matches the actual four-chunk context. |
| Generate answers only for the final pipeline by default. | Provides answer evaluation with eight LLM requests instead of 24 per run. |
| Include two table questions and verify table metadata. | Requires retrieval of the relevant table chunk, rather than accepting a nearby prose reference. |
| Use explicit key-information checks with alternative wording. | Keeps scoring explainable without another LLM judge or dependency. |
| Build a temporary real-corpus index for pytest evaluation. | Avoids stale local data and leaves the working database untouched. The runner itself does not rebuild indexes. |
| Plan CI retrieval evaluation and credentialed live answer checks. | Keeps missing credentials distinguishable from passing answer tests; workflow integration remains pending. |
| Preserve the planted Remote Work conflict. | The scheduling-notice question has the same answer in both versions; conflicting weekly allowances remain for Part 6. |
| Review one file at a time. | Follows the agreed implementation workflow. |

The user authorized editing multiple files for the test reorganization below.
The one-file review workflow resumes after this refactor.

## Fixed questions and ground truth

| ID | Question | Expected key information | Expected evidence |
| --- | --- | --- | --- |
| E01 | What is the daily meal reimbursement limit during business travel? | $75 per employee per calendar day. | Expense Reimbursement, `EXP-03.1`, table `EXP-T01`. |
| E02 | How soon must I submit an expense claim? | Within 30 calendar days after the expense date. | Expense Reimbursement, `EXP-04`. |
| E03 | How much PTO do I earn each completed month? | 10 hours per completed month. | PTO, `PTO-03.1`. |
| E04 | How many unused PTO hours can carry into the next year? | Up to 40 hours. | PTO, `PTO-03.1`. |
| E05 | For six consecutive scheduled workdays of PTO, what notice and approvals are required? | At least 10 business days; approval from the manager and People Operations. | PTO, `PTO-03.2`, table `PTO-T01`. |
| E06 | When and where should I report a lost company laptop? | Immediately, through the IT Security incident portal. | Information Security, `SEC-04`. |
| E07 | After how much inactivity must my company laptop automatically lock? | Five minutes. | Information Security, `SEC-03.2`. |
| E08 | How far ahead must I request a recurring remote-work arrangement? | At least five business days before the proposed start date. | Remote Work, `RW-04`; either version is acceptable for this unchanged rule. |

Ground truth was checked against the policy Markdown. E01 and E05 require the
table itself: a prose chunk with the same section ID is insufficient. The
dataset identifies acceptable evidence chunks explicitly and requires the
matching `table_id` and `content_type="table"` for these two cases.

## Files and review sequence

| Order | File | Responsibility |
| --- | --- | --- |
| 1 | `docs/05-evaluation.md` | Decisions, ground truth, results, and limitations. |
| 2 | `data/evaluation.json` | Fixed questions, acceptable evidence, expected answers, and answer checks. |
| 3 | `rag/evaluation.py` | Load cases, compare retrieval modes, score answers, and print results. |
| 4 | `tests/test_evaluation.py` | Unit tests for retrieval metrics and answer scoring. |
| 5 | `tests/test_rag_integration.py` | Real corpus retrieval and live eight-answer evaluation. |
| 6 | `.github/workflows/ci.yml` | Pending: provide credentials and verify hosted live checks. |

No new package, configuration file, or separate report file is planned.

## Interfaces and data structures

Each JSON case has a stable ID, question, readable expected answer,
acceptable source chunks, and required answer checks. Equivalent evidence from
the two Remote Work versions is one evidence requirement, not two required
hits. Each answer check may accept alternative expressions, such as `five`
or `5`; every required fact must match, with numeric boundaries and units.

The runner returns per-question retrieved IDs and scores, then prints aggregate
metrics. Live mode also returns the generated answer and failed key-information
checks. These are evaluation results; no
changes to the production pipeline or stored chunk metadata are planned.

Metrics:

- **Hit@3:** fraction of questions with acceptable evidence in the first three
  results. Table questions also require matching table metadata.
- **Recall@4:** fraction of required evidence recovered in the top four,
  averaged across questions. Each current case has one evidence requirement,
  so this equals Hit@4 for this dataset.
- **Answer key-information pass rate:** fraction of live answers satisfying
  every required check. Also report each question's missing information.
- **Table retrieval success@4:** fraction of table questions whose expected
  table chunk appears in the final four, with matching ID and table metadata.
- **Table answer pass rate:** fraction of table-question answers satisfying
  every required check. These table metrics are a subset of the eight cases.

The JSON report contains Hit@3, Recall@4, and optional answer results. The
table-specific rates below are calculated from the E01 and E05 case results.

Comparative baseline misses must be recorded. The final pipeline's expected
retrieval and answer checks are the regression targets; a weaker vector
baseline must not be relabeled as successful or silently excluded.

## Validation and evidence

The harness has 36 non-live cases and one live test that generates all eight
answers, now split between scoring unit tests and real RAG integration. It
covers retrieval cutoffs (including evidence
at rank four and beyond), table identity/type, equivalent Remote Work versions,
wrong or missing answer facts, expected evidence in the real corpus, and final
reranked retrieval for every question. All three retrieval modes are measured;
only the final reranked mode's evidence recovery is a regression requirement.

The dataset was validated against the stored corpus: all expected chunk IDs
exist, both table cases have the correct `table_id` and `content_type="table"`,
and either listed Remote Work version supports E08. All expected answers and
sample paraphrases passed their checks. Sample incorrect answers, wrong
numeric values, and missing facts were rejected.

Controlled runner checks passed for source matching, table metadata, the
four-result cutoff, alternative versions, retrieval routing, eight optional
generation calls, API-error propagation, and CLI exit codes. Stubbed answers
in those checks were not used as live evaluation evidence.

Real retrieval results on the fixed eight questions:

| Mode | Hit@3 | Recall@4 |
| --- | --- | --- |
| Vector | 8/8 (100%) | 8/8 (100%) |
| Hybrid | 8/8 (100%) | 8/8 (100%) |
| Hybrid plus cross-encoder reranking | 8/8 (100%) | 8/8 (100%) |

Eight real answers from Foundry model `gpt-6.1-terra-2026-10-26` passed all
required checks: **answer key-information pass rate = 8/8 (100%)**. No answer
had a missing required check. This measures the stated facts, not complete
semantic accuracy.

| Case | Required table | Retrieval within final four | Live answer result |
| --- | --- | --- | --- |
| E01 | `EXP-T01` | Present in all three modes | Passed: $75 per employee per calendar day. |
| E05 | `PTO-T01` | Present in all three modes | Passed: 10 business days, manager and People Operations approval. |

Table retrieval success@4 was **2/2 (100%)** in every mode. Table answer pass
rate was **2/2 (100%)** for the final pipeline.

The focused non-live pytest run passed **36 tests**, with the live test
deselected. The full agent-environment run recorded **125 passed, 1 failed**:
the live test failed with `APIConnectionError` because Python could not resolve
the Foundry host. A public control host also failed Python DNS resolution.
Direct `curl` reached Foundry with HTTP 200 and produced a real answer that
passed the E01 checks, confirming endpoint and credential validity independently
of the agent's Python networking. Temporary request files were removed; no
production code or `.env` settings were changed.

The user then ran the actual Python pytest live test in their normal terminal:

```sh
HF_HOME="$PWD/.venv/huggingface" HF_HUB_OFFLINE=1 \
  .venv/bin/python -B -m pytest \
  tests/test_evaluation.py::test_live_answers -q -s
```

The supplied terminal output records **1 passed, 138 warnings in 14.78 seconds**.
All eight generated answers had `passed: true` and empty `missing_checks`;
the aggregate `pass_rate` was **1.0**. This verifies the real Python retrieval,
reranking, Foundry SDK generation, and answer-scoring path. It is one pytest
test covering eight API answers, not eight separate pytest tests. The full
suite has not yet been rerun successfully with live access in that terminal.

That successful run preceded the test reorganization. The unchanged live-test
function now resides in `tests/test_rag_integration.py`. Its current command is:

```sh
HF_HOME="$PWD/.venv/huggingface" HF_HUB_OFFLINE=1 \
  .venv/bin/python -B -m pytest \
  tests/test_rag_integration.py::test_live_answers -q -s -p no:cacheprovider
```

The warnings were Chroma's legacy embedding-configuration deprecation and the
reranker's deprecated `max_length` property. Cached model loading succeeded;
`HF_HUB_OFFLINE=1` does not disable Foundry calls.

For retrieval and scoring checks without LLM requests, run:

```sh
HF_HOME="$PWD/.venv/huggingface" HF_HUB_OFFLINE=1 \
  .venv/bin/python -B -m pytest tests/test_evaluation.py tests/test_rag_integration.py \
  -q -s -k 'not live' -p no:cacheprovider
```

## Component test organization

Tests follow component responsibilities. Keyword checks were removed from the
hybrid test file after extraction, and pipeline CLI checks were moved out of the
reranker test file. Evaluation scoring and real RAG integration are separate.
Every original test function was preserved without changing its assertions;
duplicate test functions were removed. Production code was unchanged.

| Test file | Focus |
| --- | --- |
| `tests/test_ingestion.py` | Exact text/metadata, both versions, duplicate IDs, missing files, invalid paths. |
| `tests/test_chunking.py` | Heading context, stable IDs, token budget, overlap, text coverage, intact tables. |
| `tests/test_embeddings.py` | Query-only instruction, normalization request, token limits, invalid inputs. |
| `tests/test_vector_store.py` | Stored records/vectors, metadata, upserts, invalid inputs, model compatibility. |
| `tests/test_retrieval.py` | Vector ordering, returned evidence, query encoding, result limits, missing/empty indexes. |
| `tests/test_keyword_search.py` | Keyword matching, exact identifiers, term scoring, limits, invalid inputs. |
| `tests/test_hybrid_retrieval.py` | Fusion scoring, deduplication, ties, candidate limits, hybrid integration. |
| `tests/test_reranking.py` | Controlled cross-encoder scoring, selection, validation, token limits. |
| `tests/test_generation.py` | Foundry request construction, missing configuration, API errors, empty responses. |
| `tests/test_pipeline.py` | CLI orchestration, repeat indexing, stale-chunk cleanup, failed-write preservation. |
| `tests/test_evaluation.py` | Hit@3/Recall@4 boundaries, table evidence, equivalent versions, answer checks. |
| `tests/test_rag_integration.py` | Eight fixed questions with real models/indexing and optional real Foundry answers. |
| `tests/test_validate_corpus.py` | Existing corpus validation coverage. |

Fast component checks use temporary files/databases, fixed vectors, and mocked
model/API responses. These establish component behavior, not pretrained-model
quality. Real-corpus integration uses BGE and the cross-encoder with a fresh
temporary index. Only `test_live_answers` makes LLM requests.

The reorganized full non-live suite passed **163 tests**, with one live test
deselected. Live answers were not regenerated during this refactor because the
agent's Python networking restriction remains; the user's prior successful
eight-answer run is recorded above. No new dependencies or workflow edits were
needed for this test split.

Pending work: integrate the harness into CI and record hosted run results.

CI will use `OPENAI_API_KEY`, `OPENAI_BASE_URL`, and `OPENAI_MODEL` from GitHub
Actions secrets for live evaluation. The pytest live test reads environment
settings or the local `.env`; when credentials are available it calls the real
LLM. Missing credentials explicitly skip the test. API errors with configured
credentials fail it rather than being silently skipped or scored as incorrect
policy answers. Both behaviors were verified with controlled checks.
Retrieval-only CI will not replace a successful hosted live run.

## Limitations

These are development regression cases drawn from the existing corpus, not a
held-out benchmark. Key-information checks accept useful paraphrases but cannot
prove complete semantic correctness or detect every contradictory extra claim.
Live outputs may vary between runs. Retrieval success and answer-check success
will be reported separately so failures can be traced to the appropriate stage.
All three modes tie on Hit@3 and Recall@4, so these results do not establish a
retrieval-quality improvement. Exact ranks can differ: E01's required table
was rank one in vector/hybrid results and rank two after reranking. Broader or
harder fixed cases would be needed to assess differences beyond this dataset.

## Completion criteria

The dataset, runner, pytest coverage, real retrieval results, and Python live
answer checks are complete and recorded. CI execution and hosted live evidence
remain outstanding for submission. Publish through a branch and PR after the
files have been reviewed individually.
