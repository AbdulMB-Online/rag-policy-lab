# Part 4: Cross-encoder reranking

Status: Implementation and real model inference verified locally and on hosted
CI on 2026-11-19. All 100 tests pass in both environments. Part 4 is available
for review in [PR #4](https://github.com/AbdulMB-Online/rag-policy-lab/pull/4).
Live generation with reranked context remains unverified.

## Objective

Score the initial retrieval candidates with a cross-encoder and select the best
few chunks for generation, as suggested in Part 4 of the assignment.

## Scope

Add a separate reranking component, connect it to `ask`, add focused tests, and
extend CI. Reuse the existing corpus, Chroma index, and dependencies. The fixed
eight-question evaluation harness belongs to Part 5.

## Decisions and rationale

| Decision | Rationale |
| --- | --- |
| Use `cross-encoder/ms-marco-MiniLM-L6-v2` through `sentence_transformers.CrossEncoder`. | A passage-ranking model supported by our existing library. The assignment suggests CrossEncoder but does not name a specific model slug. |
| Keep `BAAI/bge-small-en-v1.5` for embeddings. | Embedding retrieval and cross-encoder scoring have separate responsibilities; no reindexing is required. |
| Load the reranker on CPU and cache it once per Python process. | Supports local execution and avoids loading weights for every question. |
| Retrieve 12 candidates and select four context chunks by default. | Gives the reranker a broader selection than the final context. These initial settings are not tuned optima. |
| Enable reranking by default for both hybrid and vector retrieval. | Both routes can use the same reranking component. `--no-rerank` preserves a retrieval-only comparison. |
| Use raw model scores with an identity activation. | Higher means more relevant; scores can be negative and are not confidence probabilities. RRF scores and vector distances are preserved but not added to model scores. |
| Preserve retrieval order for equal scores. | Provides deterministic ties without introducing another scoring rule. |
| Check the full question–chunk pair against the 512-token limit. | The reranker has its own tokenizer. Reject oversized pairs instead of silently truncating policy evidence. |
| Preserve chunk IDs, text, metadata, and existing retrieval diagnostics. | Keeps sources and policy versions traceable. Reranking does not decide which Remote Work version applies. |
| Implement and review one file at a time. | Follows the agreed review workflow. Publication uses a branch and PR. |

Model usage reference: [official model card](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2).

## Files changed

| File | Responsibility |
| --- | --- |
| `rag/reranking.py` | Model loading, pair validation, scoring, and final selection. |
| `rag/pipeline.py` | Candidate count, reranking integration, and comparison options. |
| `tests/test_reranking.py` | 30 controlled test cases for the component and CLI integration. |
| `.github/workflows/ci.yml` | Real inference smoke checks without LLM calls. |
| `docs/04-reranking.md` | Decisions, usage, evidence, and remaining checks. |

## Interfaces and data structures

`rerank(question, chunks, top_k=4)` accepts retrieved records and returns new
record dictionaries with a floating-point `rerank_score`, sorted highest first.
It preserves the existing fields, including `id`, `text`, `metadata`, and any
retrieval scores. It does not write scores back to Chroma or mutate input
records. Empty candidates return an empty list without loading the model.

The model receives `(question, chunk_text)` pairs. BGE's embedding query
instruction is not added. Metadata remains available on returned records but
is not included in model input or newly forwarded to generation by this change.

```text
ask -> vector search -> keyword search -> RRF top 12 -> cross-encoder -> top 4 -> LLM
ask --retrieval vector -> vector top 12 -> cross-encoder -> top 4 -> LLM
ask --no-rerank -> hybrid top 4 -> LLM
```

For the default hybrid route, each retriever requests 12 candidates; RRF returns
up to 12 unique chunks for the cross-encoder. `--candidate-k` changes that count;
the pipeline requests at least `--top-k`. Available corpus size can limit the
actual count. With `--no-rerank`, the pipeline directly requests `--top-k`
results. Indexing remains independent of reranking.

### Usage

The first real reranker call needs network access to download its weights.
This command inspects retrieval and reranking without an LLM request:

```sh
cd /Users/Abdul.2.Shaik/RAGsfs
export HF_HOME="$PWD/.venv/huggingface"
unset HF_HUB_OFFLINE
.venv/bin/python -B - <<'PY'
from rag.hybrid_retrieval import hybrid_retrieve
from rag.reranking import rerank

question = "How many hours of PTO do I earn for each completed month?"
candidates = hybrid_retrieve(question, top_k=12)
for hit in rerank(question, candidates, top_k=4):
    print(hit["id"], hit["rerank_score"], hit["metadata"]["section_path"])
PY
```

After the required models are cached in this location, `HF_HUB_OFFLINE=1` can
be used for model loading. It does not disable Foundry API requests.

The following commands call the LLM and require the existing Foundry settings:

```sh
.venv/bin/python -B -m rag.pipeline ask "How many hours of PTO do I earn each month?"
.venv/bin/python -B -m rag.pipeline ask "How many hours of PTO do I earn each month?" --no-rerank
.venv/bin/python -B -m rag.pipeline ask "How many hours of PTO do I earn each month?" --retrieval vector --no-rerank
```

Use `--candidate-k 20 --top-k 4` to experiment with a larger reranking set.
To reproduce the Part 3 hybrid-versus-vector comparison, include `--no-rerank`
on both commands; selecting `--retrieval vector` alone still enables reranking.

## Validation and evidence

Actual local test command on 2026-11-19:

```sh
HF_HOME="$PWD/.venv/huggingface" HF_HUB_OFFLINE=1 \
  .venv/bin/python -B -m pytest -q -p no:cacheprovider
```

Result: **100 passed, 20 warnings in 3.68 seconds**. The warnings are existing
Chroma legacy embedding-configuration deprecations. The 30 new cases use mocked
model outputs and cover caching, raw scores, descending selection, stable ties,
preserved metadata and versions, unchanged inputs, invalid arguments, empty
candidates, exact-limit acceptance, oversized-pair rejection, mismatched score
counts, CLI routing, candidate counts, bypass behavior, and index independence.
These tests do not demonstrate pretrained-model ranking quality.

`actionlint` 1.7.12 passed for the updated workflow. Both embedded CI Python
blocks also passed AST syntax checks.

Hosted Linux CI passed for implementation commit
`5c8874ebfab92db0a0562e156235045c96c20a5a` in both the
[PR run](https://github.com/AbdulMB-Online/rag-policy-lab/actions/runs/44537471567)
and [push run](https://github.com/AbdulMB-Online/rag-policy-lab/actions/runs/44537418836).
The PR job recorded **100 passed, 20 warnings in 7.39 seconds**, a successful
two-question minimal loop, 49 indexed chunks, the existing vector/hybrid
checks, and the new real cross-encoder checks. The control passage moved from
rank 2 to 1, and all three expected corpus sections stayed at rank 1, matching
the local results below. No LLM API calls or credentials were needed.

The added CI step is configured to load the real model, promote a relevant PTO
passage over an unrelated passage, and rerank 12 hybrid candidates down to four
for meals, monthly PTO accrual, and lost-laptop questions. It checks finite,
descending scores and expected source sections, and prints before/after ranks.
That exact CI reranking block passed locally with the real cached models on
2026-11-19. The controlled PTO passage moved from rank 2 to rank 1. The corpus
comparison produced the following results:

| Question topic | Expected source | Hybrid rank | Reranked rank |
| --- | --- | --- | --- |
| Daily business-travel meal limit | `expense-reimbursement`, `EXP-03.1` | 1 | 1 |
| Monthly PTO accrual | `pto`, `PTO-03.1` | 1 | 1 |
| Lost company laptop | `information-security`, `SEC-04` | 1 | 1 |

All three runs returned four candidates with finite, descending scores, and
every returned ID belonged to the 12 retrieved candidates. Expected-source
Hit@1 and Hit@4 were both 3/3 before and after reranking. This confirms real
inference and no regressions on these three questions; it does not demonstrate
an improvement over hybrid retrieval. The two-passage control demonstrates
reordering, not an improvement on corpus retrieval or held-out evaluation.

A local attempt ran that exact CI reranking block on 2026-11-19 with
`HF_HUB_OFFLINE=0`. Loading the uncached cross-encoder failed to reach
`huggingface.co`: `[Errno 8] nodename nor servname provided, or not known`.
The download retries were interrupted before pair scoring. A second attempt
reproduced the Python downloader's DNS failure. Direct HTTPS downloads through
`curl` succeeded, resolving the local model-download blocker.

Six files were downloaded into `.venv/huggingface` from model revision
`233902d25c440f23af6f7d6e94d2946bac0bee0a`: configuration, safetensors weights,
tokenizer JSON/configuration, special-token mappings, and vocabulary. The five
configuration/tokenizer files matched their published Git blob hashes; the
90,870,598-byte weights matched the published SHA-256
`821d1aa69520101d6e0737f78a042ae25b19e5cb9160701909d10434f4aeb0ae`.
The successful inference checks then ran with `HF_HUB_OFFLINE=1`. Cached model
weights are runtime artifacts and must not be included in the PR.

For a fresh environment with a working Python network connection, the normal
download command is:

```sh
cd /Users/Abdul.2.Shaik/RAGsfs
HF_HOME="$PWD/.venv/huggingface" HF_HUB_OFFLINE=0 \
  .venv/bin/python -B -c 'from rag.reranking import load_model; load_model()'
```

This loads the reranker into the shared Hugging Face cache without calling
the LLM or changing the policy index. Once cached, the comparison can run with
`HF_HUB_OFFLINE=1`.

## Limitations

Real inference passed locally; improvement over hybrid retrieval is unproven.
The synthetic relevance check and three development questions are smoke checks,
not the Part 5 evaluation harness or evidence of general improvement. Reranking
cannot recover evidence absent from its candidate set. Long pairs fail with a
clear token-limit error. Source-version correction and generated citations
remain later-stage work.

## Completion criteria

Implementation, controlled tests, workflow syntax checks, real model inference
on the corpus, and hosted CI are complete for the implementation commit above.
Actual source ranks and hosted evidence are recorded. PR #4 remains for manual
review and merge. Live generation with reranked context has not been tested,
and no retrieval-quality improvement is claimed from the current three-question
comparison.
