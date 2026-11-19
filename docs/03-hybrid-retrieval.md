# Part 3: Hybrid retrieval

Status: Implemented and locally verified on 2026-11-19. Keyword/RRF retrieval,
40 focused test cases, CLI mode selection, and hybrid CI checks are in place.
All 70 tests pass locally. Hosted verification of the Part 3 changes is pending
a feature-branch push and PR; Part 3 is not yet marked complete.

## Objective and scope

Add keyword matching alongside the existing BGE/Chroma vector search, combine
their results, and demonstrate a measured retrieval improvement. Reuse the
current 49 indexed chunks, IDs, text, and metadata. This stage needs no additional
search service, dependency, embedding model, or corpus revision.

Cross-encoder reranking belongs to Part 4. The fixed eight-question evaluation
harness belongs to Part 5. Both Remote Work versions remain available for the
later source-data diagnosis; hybrid retrieval does not select a current version.

## Implementation decisions

| Decision | Reason |
| --- | --- |
| Make hybrid search the `ask` default; retain `--retrieval vector` for comparison. | Both routes have been checked locally. Explicit `--retrieval hybrid` also selects the default route. |
| Keep the existing `rag/retrieval.py` as the vector-only baseline. | Enables a fair comparison and preserves the Part 2 behavior. |
| Run vector retrieval first, keyword search second, and RRF after both lists are available. | Follows the user's requested execution order. Keyword search independently covers the full corpus rather than only the vector candidates. |
| Search stored chunk text with case-insensitive whole-term matching. | Simple standard-library code is sufficient for this corpus; punctuation within identifiers must be retained. |
| Remove a small, documented set of common question words; count distinct matching terms rather than repeated occurrences. | Avoids rewarding repeated filler words or longer chunks solely for repeating a term. |
| Prefer the owning section/table when an exact query identifier matches `section_id` or `table_id`; break remaining ties by chunk ID. | An identifier's definition should rank ahead of procedural text that merely references it. Full identifiers must match, so prefixes such as `PTO-03.2` do not accidentally match `PTO-03.20`. |
| Return only positive keyword matches. | Empty lexical results allow the vector list to determine the result. |
| Retrieve 12 candidates from each method initially and return four final chunks by default. | Provides a broader candidate set than the final context. If `top_k` is larger, request at least that many candidates, capped at the corpus size. |
| Combine rankings with equal-weight Reciprocal Rank Fusion (RRF), using a rank constant of 60. | Avoids comparing incompatible keyword scores and cosine distances. These are initial parameters to validate, not proven optima. |
| Deduplicate by full chunk ID and retain source metadata. | The same hit can receive support from both methods while different document versions remain distinct. |
| Keep generation's interface unchanged. | It already accepts retrieved records containing IDs and text. |
| Add focused retrieval tests and extend the existing CI workflow. | Verifies the new behavior without Foundry credentials or live generation calls. |
| Implement and review one file at a time. | Follows the user's learning/review workflow; future publication uses a branch and PR. |

## Fusion and record structures

For each candidate, add `1 / (60 + rank)` for every result list containing it.
Ranks start at one. Sort by descending total, with chunk ID as a deterministic
tie-breaker, then take the final `top_k`. A candidate appearing in only one list
receives only that list's contribution. This is rank fusion, not the learned
cross-encoder scoring planned for Part 4.

Keyword results contain `id`, `text`, `metadata`, and `keyword_score`. Hybrid
results retain the source fields and expose `rrf_score`, `vector_rank`, and
`keyword_rank` for inspection. A missing rank is `None`; vector distance can be
retained when available, and must not be invented for keyword-only candidates.
These diagnostic fields are returned records, not new Chroma metadata.

Implemented functions:

- `keyword_search(question, top_k=12, path=DB_PATH)` reads stored documents and
  metadata, performs keyword matching, and returns its ranked candidates.
- `hybrid_retrieve(question, top_k=4, path=DB_PATH, candidate_k=12)` obtains both
  lists, fuses them, and returns the selected records.

Keyword search validates empty questions, positive integer result counts, and
an indexed database. Hybrid retrieval validates both count parameters and
delegates question/database validation to vector retrieval. Unsupported IDs do
not become prefix matches. Both methods read the existing corpus; indexing is
separate and should not run concurrently with a comparison.

Keyword tokenization retains letters, digits, and internal hyphens/dots, so
`PTO-03.2` remains one term. The small `STOP_WORDS` set is defined explicitly in
`rag/keyword_search.py`. Each distinct non-stopword contributes at most one
point per chunk, regardless of repetition in the question or source. Sorting
first prefers an exact identifier owner, then higher `keyword_score`, then
chunk ID. No stemming, synonym expansion, model loading, or embedding generation
occurs during keyword search.

The hybrid module requests `max(candidate_k, top_k)` candidates from each
retriever; each retriever naturally limits its output to available matches.
It constructs new result dictionaries and preserves input text/metadata. RRF
scores are higher-is-better ranking values, not confidence probabilities.
Keyword-only hits have `distance=None` and `vector_rank=None`; absent keyword
matches have `keyword_rank=None`.

Current call paths:

```text
pipeline ask (default or --retrieval hybrid)
  -> hybrid_retrieve -> retrieval.retrieve -> vector candidates
                     -> keyword_search -> independent keyword candidates
                     -> RRF -> selected records -> generation

pipeline ask --retrieval vector
  -> retrieval.retrieve -> vector results -> generation
```

`retrieval.py` remains the vector component in both routes. The `index` command
keeps its existing behavior; asking questions does not rebuild the index.

Run the following in the user's working terminal to compare one-chunk context
for the same query. These `ask` commands make live Foundry requests:

```sh
cd /Users/Abdul.2.Shaik/RAGsfs
export HF_HOME="$PWD/.venv/huggingface" HF_HUB_OFFLINE=1
.venv/bin/python -B -m rag.pipeline ask "PTO-T01" --top-k 1
.venv/bin/python -B -m rag.pipeline ask "PTO-T01" --top-k 1 --retrieval vector
```

The first command uses hybrid retrieval. Local routing checks used real
retrieval and replaced only generation with a test function: default and
explicit hybrid modes forwarded `pto-v1.0:PTO-03.2:0` first, while vector mode
forwarded `pto-v1.0:PTO-01:0`. Checks also confirmed selected evidence was passed
to generation, invalid modes exited with status 2, and index command routing
was preserved. No live hybrid-generated answer is claimed from those checks.

## File sequence

| File | Responsibility | Status |
| --- | --- | --- |
| `docs/03-hybrid-retrieval.md` | Decisions, actual results, limitations, and remaining work. | Updated through separate review turns |
| `rag/keyword_search.py` | Keyword-search component using the existing Chroma records. | Implemented; terminal checks pass |
| `rag/hybrid_retrieval.py` | Candidate gathering, deduplication, and RRF. | Implemented; terminal checks pass |
| `tests/test_hybrid_retrieval.py` | Focused tests for keyword boundaries, ranking, fusion, and distinct source versions. | Implemented; 40 cases pass locally |
| `rag/pipeline.py` | Select `--retrieval hybrid\|vector`; default to hybrid. | Implemented; local routing checks pass |
| `.github/workflows/ci.yml` | Run focused tests and compare real vector/hybrid retrieval after indexing. | Extended; local checks pass, hosted run pending |

No extra configuration file, benchmark script, or third-party keyword library is
planned. Read-only comparison snippets can inspect the component functions.

## Vector baseline and paired comparison

Initial read-only checks on 2026-11-19 measured vector ranks across the full
49-result list. A later paired run called both implemented retrievers with
`top_k=4` for the same eight queries, reproduced those vector ranks, and produced
the hybrid ranks below. Expected sources were identified from policy IDs before
the comparison. All targets appeared in both methods' top four.

| Query | Expected source | Vector rank | Hybrid rank |
| --- | --- | --- | --- |
| `SEC-T01` | Table `SEC-T01` | 1 | 1 |
| `Explain SEC-T01.` | Table `SEC-T01` | 1 | 1 |
| `EXP-T02` | Table `EXP-T02` | 1 | 1 |
| `What supporting evidence is listed in EXP-T02?` | Table `EXP-T02` | 1 | 1 |
| `PTO-T01` | Table `PTO-T01` | 4 | 1 |
| `PTO-03.2` | Section `PTO-03.2` | 3 | 1 |
| `EXP-05` | Section `EXP-05` | 1 | 1 |
| `SEC-03.2` | Section `SEC-03.2` | 2 | 1 |

For this small development set, **Hit@1 was 5/8 for vector-only and 8/8 for
hybrid**. Hit@4 was **8/8 for both**. No source-rank regressions occurred on
these eight queries; this is not a claim about unseen questions.

For `PTO-T01`, vector search's top four were Purpose and scope, Roles and
eligibility, Procedures, and finally the requested table. Before component
implementation, a small in-memory prototype matched the full identifier,
ranked the owning table before its
procedural reference, and fused the keyword list with 12 vector candidates.
It ranked `pto-v1.0:PTO-03.2:0` first.

That exploratory result improves the requested table's rank from **4 to 1**:
Hit@1 changes from **0 to 1**, and reciprocal rank from **0.25 to 1.0**. Hit@4
remains **1 for both**; this is not evidence of a vector-only miss at four.
The query was selected during development, not held out as an independent
benchmark. It is a development demonstration, not a claim that hybrid always
improves retrieval. No hybrid module or LLM call was used in that initial probe,
and the stored count remained 49.

## Actual module verification

Direct terminal checks on 2026-11-19 verified keyword search against the stored
corpus. Queries for `PTO-T01`, `EXP-T01`, `EXP-T02`, and `SEC-T01` ranked the
owning table first. Case changes and repeated query terms preserved results.
Unmatched terms and stopword-only queries returned no hits; `PTO-03.20` did not
match `PTO-03.2`. Controlled temporary records verified owner priority and stable
ties independently of repeated source terms. Invalid input and missing/empty
databases were rejected. Checks prohibited calls to model loading and embedding
functions, confirming neither was used by keyword search.

The actual `retrieve("PTO-T01", top_k=4)` and
`hybrid_retrieve("PTO-T01", top_k=4)` functions produced:

| Measure | Vector-only | Hybrid |
| --- | --- | --- |
| Rank of `pto-v1.0:PTO-03.2:0` | 4 | 1 |
| Hit@1 | 0 | 1 |
| Hit@4 | 1 | 1 |
| Reciprocal rank | 0.25 | 1.0 |

Hybrid used 12 initial candidates per retriever and selected four final hits.
This confirms the prototype's ranking improvement through the implemented
components. It demonstrates a same-cutoff success at one result, not an
improvement in Hit@4 or a general benchmark result.

Seven additional natural-language questions retrieved the expected source
section within the hybrid top four:

| Question | Expected section | Hybrid Hit@4 |
| --- | --- | --- |
| What is the daily meal reimbursement limit during business travel? | `EXP-03.1` | 1 |
| How many hours of PTO do I earn for each completed month? | `PTO-03.1` | 1 |
| How many unused PTO hours can I carry into next year? | `PTO-03.1` | 1 |
| What is the lodging reimbursement limit per room per night? | `EXP-03.1` | 1 |
| How soon must I submit an expense reimbursement claim? | `EXP-04` | 1 |
| What should I do if I lose my company laptop? | `SEC-04` | 1 |
| How many days can I work remotely per calendar week? | `RW-03.1` | 1 |

Both Remote Work versions remained in the allowance query's results. These
checks verified expected-source inclusion, not universal rank improvements or
generated-answer accuracy. The comparison is developmental, not held out.

Controlled fusion checks verified vector-before-keyword execution, candidates
appearing in both lists only once, consensus ranking, per-method rank fields,
keyword-only distances, stable ties, unchanged input records, and preservation
of vector ordering when keyword results are empty. Requests for more final
results increased the candidate budget appropriately; oversized final requests
returned the 49 available chunks. Invalid count parameters, questions, and
unindexed paths produced errors.

The initial component checks above were run directly in the terminal. The
focused test file has since been added, and the latest full suite reports
**70 passed**: 30 existing corpus cases plus **40 keyword/fusion cases**. The new
tests use temporary Chroma records with synthetic storage vectors and stubbed
ranked lists. Guards fail if model loading or embedding functions are called.
They require no model downloads, Foundry settings, or LLM calls and do not alter
the real policy files or stored corpus.

Coverage includes complete/case-insensitive identifiers, owning-source priority,
distinct terms, repetition and ties, no matches, invalid questions/counts,
unindexed databases, rank consensus, deduplication, distinct policy versions,
empty-keyword fallback, input preservation, per-source ranks, candidate budgets,
and final result limits. Chroma reported 20 legacy embedding-configuration
deprecation warnings in the local test run; all cases passed. The real corpus
remains at **49 chunks**.

## CI extension and local verification

The existing workflow still validates the corpus, runs pytest, checks the
minimal retrieval loop, and indexes the full corpus. Pytest automatically
discovers the 40 new focused cases.

After indexing, the extended retrieval step runs four questions through both
`retrieve` and `hybrid_retrieve`: meals, monthly PTO, lost laptops, and remote
work. It checks expected sections in the top four and both Remote Work versions
in the allowance results. All **eight source-inclusion checks passed locally**.

The same step compares `PTO-T01` at an identical four-result cutoff. It requires
the hybrid table rank to be one and better than vector-only, and prints ranks,
Hit@1, and Hit@4. The exact updated CI block ran locally with this output:

```text
PASS PTO-T01: vector rank 4; hybrid rank 1
Hit@1: vector=0, hybrid=1
Hit@4: vector=1, hybrid=1
```

Actionlint 1.7.12, YAML structure checks, and shell syntax checks passed. These
results used the local macOS environment and cached BGE model. The updated
workflow has not yet run on a hosted GitHub runner; passing Part 2 CI is not
substituted for that verification. No workflow secrets or Foundry calls are
needed. If a future model/corpus change removes this demonstration's measured
gain, the comparison assertion deliberately requests review of the example.

## Remaining verification

Publish the Part 3 changes on a feature branch after review and confirm the
updated hosted workflow passes before merging. Keep the measured comparison
and its limitations in the PR description. Live answer checks can run from the
user's terminal; this assistant runtime still has the previously diagnosed
Foundry connection limitation. Systematic answer accuracy and the fixed
evaluation harness remain Part 5 work.

## Completion criteria and handoff

Part 3 is complete when keyword and vector candidates are fused through separate
components, the pipeline exposes hybrid and vector modes, the actual code shows
at least one measured same-cutoff improvement, tests and CI pass, and this log
contains the results and limitations. Preserve the original policy conflict.

The implementation, measured improvement, focused tests, CLI integration, and
CI extension are complete locally. Part 3 awaits hosted CI and PR review before
being marked complete. Next after this log review: prepare the Part 3 feature
branch and PR when requested, then verify its hosted checks.

## References

- [Microsoft hybrid-search overview](https://learn.microsoft.com/azure/search/hybrid-search-overview)
- [Reciprocal Rank Fusion explanation](https://learn.microsoft.com/en-us/azure/search/hybrid-search-ranking)
- [Elastic hybrid-search guidance](https://www.elastic.co/docs/solutions/search/hybrid-search)

These references support the retrieval design; this lab continues using local
Chroma and does not introduce Azure AI Search or Elasticsearch.
