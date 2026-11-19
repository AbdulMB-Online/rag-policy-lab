# Part 2: Build the pipeline

Status: Complete on 2026-11-18. The minimal retrieval prerequisite passed before
the full build; all seven components are implemented, with 49 indexed chunks.
Successful live Foundry answers were verified from terminal output supplied by
the user. The assistant execution environment still has a separate DNS/network
limitation; it did not independently reproduce those live answers.

## Objective and scope

Build a RAG pipeline that loads the generated policy corpus, chunks it, creates
embeddings, retrieves relevant chunks, and generates an answer through the
configured Microsoft Foundry deployment.

The required two-text retrieval check is a prerequisite within Part 2. It has
already passed before the full pipeline components were created. Hybrid
retrieval, reranking, evaluation, data quality diagnosis, source attribution, and
final CI/submission remain Parts 3–8 of the updated plan. The user requested a
baseline CI workflow before the Part 2 PR; it is recorded below without claiming
the later evaluation harness or final submission is complete.

## Approved decisions

| Decision | Reason |
| --- | --- |
| Use `BAAI/bge-small-en-v1.5` throughout the current embedding code and minimal demonstration. | Supports 512-token inputs and produces 384-dimensional vectors; gives longer sections room to remain together. |
| Generate document and query embeddings explicitly with Sentence Transformers. | Makes the embedding step visible and ensures both use the same model. |
| Prefix questions with `Represent this sentence for searching relevant passages: `; leave passages unprefixed. | Follows BGE's retrieval convention. The token-limit check includes this prefix for questions. |
| Use section-aware chunks capped at 440 tokens with up to 80 tokens of prose overlap. | Counts headings and special tokens within the cap, below BGE's 512-token input limit. Section boundaries remain meaningful; chunks need not fill the cap. |
| Expand all five source documents to 1,200–1,500 body words. | User-approved change provides longer procedures that exercise real splitting and overlap. Corpus decisions and counts are in `docs/01-corpus-creation.md`. |
| Use an in-memory ChromaDB collection for the preliminary two-text check. | Demonstrates embedding, storage, and retrieval independently of persistence or chunking. |
| Store the full corpus in the persistent `policy-chunks` collection under `.venv/chroma`. | Keeps derived database files with existing local runtime data; no new dependency or configuration file is needed. The path can be overridden through the storage functions. |
| Use cosine distance and disable Chroma's automatic embedding function. | Storage receives vectors explicitly from `rag/embeddings.py`. |
| Tag the collection with its embedding model and check the tag and distance metric on opening. | Rejects incompatible collections rather than silently mixing vectors or metrics. |
| Upsert by stable chunk ID and omit metadata values that are `None`. | Repeated indexing updates existing IDs without duplicates; source metadata otherwise remains intact, including both Remote Work versions. |
| Retrieve four chunks by default, with adjustable `top_k` capped at the collection count. | Gives a small initial context set and handles requests larger than the corpus without exceeding the available records. |
| Return chunk IDs, text, metadata, and cosine distances without filtering policy versions. | Makes retrieval inspectable and preserves the planted conflict for later diagnosis; no reranking or version resolution is implemented yet. |
| Use the OpenAI SDK's Chat Completions client with the existing Foundry v1 endpoint and deployment settings. | Keeps generation separate from retrieval while using the configured `OPENAI_API_KEY`, `OPENAI_BASE_URL`, and `OPENAI_MODEL`. |
| Ask the model to use only retrieved excerpts, acknowledge insufficient evidence, and describe conflicting statements. | Encourages grounded answers without implementing version filtering or claiming that prompt instructions guarantee correctness. |
| Use a 30-second API timeout and disable automatic retries. | Keeps failures visible and avoids repeated requests during initial validation. |
| Pin `openai==3.55.1` and `python-dotenv==1.2.4` as direct dependencies. | These installed versions passed compatibility and mocked generation checks. |
| Use successful user-terminal runs as live API evidence, identified separately from assistant-run checks. | The normal Mac terminal reaches Foundry; the assistant's Python runtime fails during DNS resolution. |
| Expose `index` and `ask` commands in `rag/pipeline.py`. | Coordinates the existing components while keeping indexing separate from repeated questions. |
| Remove stored IDs absent from the current corpus only after successful upsert during pipeline indexing. | Prevents stale chunks after source or chunking changes; both Remote Work versions remain because both are current corpus inputs. |
| Add baseline GitHub Actions CI before the Part 2 PR, on pushes and pull requests. | Checks the current corpus and real local retrieval path before review; later evaluation tests will extend it. |
| Run CI on Ubuntu with Python 3.12, CPU PyTorch, and public BGE downloads. | Exercises embedding and Chroma integration without GPU dependencies or a preexisting local model cache. |
| Exclude live Foundry calls and credentials from CI. | Keeps the initial checks independent of API secrets, paid generation, and the assistant runtime's connection limitation. |
| Use two fixed policy excerpts for that check. | Gives unambiguous expected rankings before the full corpus introduces conflicting versions. |
| Use a dedicated Python 3.12 virtual environment. | Python 3.15 could not resolve Chroma's ONNX Runtime dependency. Python 3.12.16 now runs the pinned packages successfully. |
| Record tested dependency versions in `requirements.txt`. | Makes the working environment reproducible without extra configuration files. |
| Check expected retrieval results inside the preliminary script. | A failed demonstration exits with a nonzero status; this check is separate from the Part 5 evaluation harness. |
| Keep `scripts/minimal_retrieval.py` as a standalone demonstration. | Makes the required pre-build check easy to rerun and explain, including exactly which code produces each PASS line. |
| Put each full-pipeline component in its own file. | Keeps ingestion, chunking, embedding, storage, retrieval, and generation independently understandable. |
| Create or edit one file at a time, pausing for approval. | Preserves the user's review and learning workflow. |
| Make coordinated edits for this corpus/model revision only. | Explicit user exception; the usual one-file review workflow resumes after this change. |
| Publish future commits on a separate branch through a pull request. | Do not push directly to `main` or merge without user approval. |

## Component files and responsibilities

Existing supporting files are `requirements.txt`, this decision log, and
`scripts/minimal_retrieval.py`. Full-pipeline component status is:

| File | Responsibility | Status |
| --- | --- | --- |
| `rag/ingestion.py` | Read the manifest and Markdown sources with their version metadata. | Implemented |
| `rag/chunking.py` | Split source content while preserving headings, section IDs, and table context. | Implemented |
| `rag/embeddings.py` | Load BGE and generate normalized document/query vectors. | Implemented |
| `rag/vector_store.py` | Open the persistent Chroma collection and upsert chunk records with explicit embeddings. | Implemented |
| `rag/retrieval.py` | Embed a natural-language question and return ranked chunks from persistent Chroma. | Implemented |
| `rag/generation.py` | Load local Foundry settings and generate an answer from retrieved context. | Implemented; live verified in user terminal |
| `rag/pipeline.py` | Connect the components and expose indexing and questioning commands. | Implemented; live verified in user terminal |

Review future component changes one file at a time. Keep shared helpers
inside their owning component; no extra framework or configuration file is
planned. `requirements.txt` now declares the direct OpenAI SDK and dotenv
dependencies through a separate reviewed update.

Retain the standalone minimal script independently of the full pipeline. Its
fixed inputs and checks make the original demonstration directly inspectable.
Preserve the pre-build evidence recorded below; later reruns demonstrate that the
check still works, while the recorded initial run establishes the required
sequence. A duplicate verification command in the full pipeline is unnecessary.

The virtual environment and downloaded model are local runtime dependencies,
not submission artifacts. Do not commit them. Confirm environment and download
access before claiming that installation or model loading works.

## Preliminary check: fixed inputs

Each record has a stable ID and text. The script uses these excerpts directly;
it does not read or modify the policy files or manifest.

| Record ID | Text | Origin |
| --- | --- | --- |
| `remote-work` | Eligible employees may work remotely up to 2 days per calendar week, subject to manager approval. | Remote Work v2.0, `RW-03.1` |
| `expense-meals` | Meals: $75 per employee per calendar day. Actual eligible food and nonalcoholic drinks during business travel, including tax and tips. | Expense Reimbursement v1.0, `EXP-03.1`, table `EXP-T01`; row rendered as plain text |

| Question | Expected first result |
| --- | --- |
| How many days can I work from home each week? | `remote-work` |
| What is the daily meal reimbursement limit during business travel? | `expense-meals` |

The outdated Remote Work policy remains in the original corpus. It is excluded
from this two-text demonstration so the initial retrieval check has an
unambiguous expected result. Diagnosis of the version conflict remains Part 6.

## Preliminary check: implementation

1. Load the embedding model once.
2. Embed and store the Remote Work record first; verify the collection has one
   record before adding the Expense Reimbursement record.
3. Embed and store the second record; verify the collection has two records.
4. Embed each question with the same model and BGE's query prefix, then query
   ChromaDB for both records. The original MiniLM run did not need this prefix.
5. Print the model name, embedding dimensions, stored record count, question,
   ranked IDs, retrieved text, and distances returned by ChromaDB.
6. Check that each expected ID is ranked first. Print a clear success result only
   after both questions pass; otherwise fail with a nonzero exit status.

The implemented metric is cosine distance. Smaller values indicate closer
vectors; they are not answer-confidence scores. Both document and query
embeddings are normalized.

The check is triggered by running `scripts/minimal_retrieval.py`: its
`if __name__ == "__main__"` block calls `main()`. The storage loop checks each
embedding shape against `(1, 384)` and the collection count against 1, then 2.
The question loop reads the two fixed entries in `QUESTIONS`, performs real vector
queries, and checks that each expected record is ranked first among two results.
A failed condition raises `RuntimeError` and stops execution. Per-question PASS
lines follow successful ranking checks; the final `2/2` PASS line is reached only
after both questions pass. These are script-level checks, not pytest cases or
assertions about every possible question.

## Preliminary check: actual evidence

The original MiniLM model and Chroma collection ran successfully on 2026-11-18, before any
full-pipeline component file was created. This satisfies the sequence in the
Suggested Approach; building the full pipeline can now follow.

Runtime versions: Python 3.12.16, Sentence Transformers 6.3.0, ChromaDB 1.5.9,
ONNX Runtime 1.32.0, PyTorch 2.15.0, Pydantic 2.14.3, and pytest 9.2.0.
`pip check` reported no broken requirements. The previous Python 3.15 environment
was retained as a temporary backup when `.venv` was rebuilt.

Command run from the project root:

```sh
HF_HOME="$PWD/.venv/huggingface" HF_HUB_OFFLINE=1 \
  .venv/bin/python -B scripts/minimal_retrieval.py
```

Python's sandboxed network access could not download the model directly. The
public MiniLM model was cached under `.venv/huggingface` at revision
`1110a243fdf4706b3f48f1d95db1a4f5529b4d41` and loaded offline. No substitute
embeddings or mocked database were used.

Selected terminal output from the original successful run:

```text
Stored remote-work: embedding shape (1, 384); collection count 1
Stored expense-meals: embedding shape (1, 384); collection count 2
PASS: expected first result is remote-work
PASS: expected first result is expense-meals
PASS: minimal embed-store-retrieve loop; 2/2 questions passed
```

| Question | First result / cosine distance | Second result / cosine distance |
| --- | --- | --- |
| How many days can I work from home each week? | `remote-work` / 0.3772 | `expense-meals` / 0.7743 |
| What is the daily meal reimbursement limit during business travel? | `expense-meals` / 0.2801 | `remote-work` / 0.7647 |

The script exited with status 0. Corpus validation passed and all 30 existing
pytest cases passed in the new Python 3.12 environment. These tests check the
corpus validator; they do not measure full-pipeline retrieval recall or answer
accuracy. A terminal screenshot for the final PDF is still required; this log is
a written record, not that screenshot. The MiniLM distances above are historical
evidence; the current script uses BGE, with rerun results below.

## Current BGE demonstration

The standalone script was updated to BGE, including the question prefix, so
current model references agree across `scripts/minimal_retrieval.py`,
`rag/chunking.py`, and `rag/embeddings.py`. The original proof above is retained
without relabeling its model or distances.

The BGE rerun on 2026-11-18 stored two real embeddings of shape `(1, 384)`,
confirmed collection counts of one and two, and passed both expected-first-result
checks. The cached BGE revision is
`5c38ec7c405ec4b44b94cc5a9bb96e735b38267a`; it loaded offline from
`.venv/huggingface` using the same environment settings shown above.

| Question | First result / cosine distance | Second result / cosine distance |
| --- | --- | --- |
| How many days can I work from home each week? | `remote-work` / 0.3533 | `expense-meals` / 0.3790 |
| What is the daily meal reimbursement limit during business travel? | `expense-meals` / 0.1904 | `remote-work` / 0.4373 |

Result: **2/2 questions passed**. This verifies the two-excerpt loop, not retrieval
quality over the expanded corpus. Distances across embedding models are not a
quality comparison.

## Full pipeline: implementation and remaining design

- Read all five document versions through the manifest, including the outdated
  Remote Work copy. Preserve version/status metadata without silently removing
  the planted conflict.
- Use section-aware chunks capped at 440 model tokens, including heading context
  and special tokens, with up to 80 tokens of overlap inside the same section.
  BGE's 512-token input limit motivates the cap; overlap preserves context near
  boundaries. These are initial parameters to evaluate, not proven optima.
- Preserve heading ancestry and section IDs. Keep each captioned Markdown table
  intact when it fits; raise an explicit error for oversized tables or tables
  without captions. Table-row splitting is deferred to later experiments.
  Never silently truncate a chunk to fit the embedding model.
- Represent chunks as an ID, text, and metadata. Use document-version IDs and
  section IDs so outdated/current chunks cannot overwrite each other.
- Use persistent local ChromaDB for the full corpus, with cosine distance and
  explicitly generated embeddings. The implemented default is `.venv/chroma`,
  collection `policy-chunks`. Keep generated database files out of Git.
- Retrieve the top four chunks initially. Make top-k adjustable from the command
  line and inspect retrieved text alongside generated answers while validating.
- Use `.env` values `OPENAI_API_KEY`, `OPENAI_BASE_URL`, and `OPENAI_MODEL` for
  Foundry's OpenAI-compatible v1 API. Answer from retrieved context and acknowledge
  missing evidence. Source-attribution presentation is developed in Part 7.
- Keep the question path separate from indexing so repeated questions can reuse
  the stored corpus. Retain `scripts/minimal_retrieval.py` for the independent
  two-text verification; it runs only when explicitly invoked.

Implemented command interface, run from the project root:

```sh
export HF_HOME="$PWD/.venv/huggingface" HF_HUB_OFFLINE=1
.venv/bin/python -B -m rag.pipeline index
.venv/bin/python -B -m rag.pipeline ask "What is the daily meal reimbursement limit?" --top-k 4
```

The environment flags load the cached embedding model; Foundry generation still
uses a remote API. Index after changing source documents or chunking settings;
repeated questions reuse the stored corpus.

The local credential file's three fields are present and its endpoint has the
expected v1 URL structure. After configuration troubleshooting, successful
user-terminal requests confirmed working authentication and deployment access.
Assistant-run requests still fail before an HTTP response during DNS resolution.
Do not log credentials or include `.env` in a PR.

## Implemented interfaces and validation

- `load_documents()` returns records with `id`, `text`, and manifest `metadata`.
  Both Remote Work versions remain present. It rejects duplicate IDs, missing or
  empty sources, unsupported manifests, and paths outside the policy directory.
- `chunk_documents(documents, tokenizer, max_tokens=440, overlap=80)` walks
  heading sections, preserves parent headings, and splits long prose at whole
  words. Overlap applies within a prose block, not across section or table
  boundaries. Each record contains `id`, `text`, and inherited metadata plus
  `section_id`, `section_path`, `chunk_index`, `token_count`, `content_type`, and
  `table_id` for tables. IDs include the document version, section, and index.
- `embed_texts(texts, is_query=False)` uses a cached BGE model on CPU and returns
  normalized vectors as lists. Pass `is_query=True` for questions. Empty input
  and inputs beyond the actual 512-token model limit raise errors.

Validation of the expanded corpus on 2026-11-18 produced:

| Document | Chunks |
| --- | --- |
| Remote Work v1.0 | 9 |
| Remote Work v2.0 | 9 |
| Expense Reimbursement v1.0 | 11 |
| PTO v1.0 | 10 |
| Information Security v1.0 | 10 |
| Total | 49 |

Chunks range from **29 to 440 tokens**, including headings and special tokens.
Each policy's procedures section splits into two prose chunks: five actual
split blocks across the five documents. All four tables remain whole, with
heading-inclusive sizes of 156, 136, 135, and 150 tokens for `EXP-T01`, `EXP-T02`,
`PTO-T01`, and `SEC-T01`, respectively. Shorter sections remain smaller than
440 tokens rather than being merged just to reach the cap.

Terminal checks verified every source body word is covered, overlapping tails
stay within 80 tokens, chunk IDs are unique and deterministic, and inherited
metadata is unchanged. Invalid settings and oversized tables raise errors.
Real BGE inference produced a **(49, 384)** matrix of finite, unit-length
embeddings. A question matched the explicitly prefixed BGE input, and inputs
beyond 512 tokens were rejected for both passages and questions.

The dependency-free corpus validator passes the revised 1,200–1,500-body-word
rule; all **30 pytest cases pass**, including the updated four length boundaries.
The additional chunking/embedding checks were run directly in the terminal,
not added to the corpus pytest suite. These results do not measure full RAG
retrieval recall or answer accuracy; those remain Part 5 work.

## Persistent storage: implementation and actual results

`rag/vector_store.py` provides two functions:

- `get_collection(path=DB_PATH)` opens or creates the local persistent
  `policy-chunks` collection. It checks that the collection's `embedding_model`
  matches the current BGE model and that its distance metric is cosine.
- `store_chunks(chunks, embeddings, path=DB_PATH)` upserts IDs, text, vectors,
  and metadata, then returns the collection's total record count. Empty input,
  mismatched chunk/vector counts, and duplicate input IDs raise `ValueError`.
  Metadata is copied with `None` values omitted; the source records are not
  modified. In particular, the current Remote Work version retains its
  `supersedes` reference.

Standalone storage command used during initial verification:

```sh
HF_HOME="$PWD/.venv/huggingface" HF_HUB_OFFLINE=1 \
  .venv/bin/python -B -m rag.vector_store
```

The module entry point loads the policies, chunks them, embeds their text, and
stores the records. The command completed on 2026-11-18 with:

```text
Collection: policy-chunks
Upserted 49 chunks; collection count: 49
```

Checks run afterward in a separate Python process confirmed:

- All 49 records survived the indexing process exiting. Stored IDs, text,
  sanitized metadata, and 384-dimensional normalized embeddings matched the
  current chunks. Both Remote Work versions remained present.
- Upserting the same corpus again left the count at 49.
- A direct Chroma query for the daily business-travel meal limit returned
  `expense-reimbursement-v1.0:EXP-03.1:0` first, using a BGE question embedding.
  This was a storage/search smoke check before the retrieval component was
  implemented; its results are separate from the checks below.
- Temporary database checks confirmed that upserting an existing ID replaces
  its text and metadata without increasing the count. Invalid input and
  incompatible model tags or distance metrics were rejected.

All **30 existing pytest cases passed** after implementation. Storage checks
were run directly in the terminal; no additional test or configuration files
were created.

Limitations: calling `store_chunks` or its standalone module command directly
does not delete stored IDs absent from later submissions. The pipeline `index`
command now performs that cleanup, as described below. Recreating `.venv` removes
this local database, which can be regenerated from the policy sources. No model
migration is implemented; incompatible collections are still rejected.

## Vector retrieval: implementation and actual results

`retrieve(question, top_k=4, path=DB_PATH)` in `rag/retrieval.py` validates the
question and positive integer `top_k`, opens the indexed collection, embeds the
stripped question with `is_query=True`, and queries Chroma for
`min(top_k, collection.count())` results. It returns a ranked list of records
containing `id`, `text`, `metadata`, and `distance`. Lower cosine distance means
closer vectors; it is not an answer-confidence score.

The terminal command prints rank, chunk ID, distance, source path, section path,
version status, and full text. Run it from the project root:

```sh
HF_HOME="$PWD/.venv/huggingface" HF_HUB_OFFLINE=1 \
  .venv/bin/python -B -m rag.retrieval \
  "What is the daily meal reimbursement limit during business travel?" --top-k 4
```

Actual checks on 2026-11-18 against the 49 stored chunks produced:

| Question | First chunk | Cosine distance |
| --- | --- | --- |
| What is the daily meal reimbursement limit during business travel? | `expense-reimbursement-v1.0:EXP-03.1:0` | 0.1166 |
| How many hours of PTO do I earn for each completed month? | `pto-v1.0:PTO-03.1:0` | 0.2277 |
| What should I do if I receive an unexpected authentication prompt? | `information-security-v1.0:SEC-04:0` | 0.3430 |
| How many days can I work remotely per calendar week? | `remote-work-v1.0:RW-03.1:0` | 0.1308 |

All four questions returned four distinct, distance-ordered records with text
and metadata matching storage. The Remote Work question retrieved both versions
and ranked the outdated version first. This preserves a visible source-data
conflict; it is not yet evidence of a flawed generated answer or a completed
Part 6 diagnosis.

Additional checks passed for `top_k=1` and a request larger than the corpus,
which returned all 49 records. Queries left the stored count unchanged. Empty
or non-string questions, invalid `top_k`, questions beyond BGE's 512-token input
limit, and missing or empty databases produced clear errors. A missing database
path remained uncreated. The CLI succeeded with `--top-k 1` and rejected
`--top-k 0` with exit status 2.

All **30 existing pytest cases passed** after implementation. Retrieval checks
were run directly in the terminal; they are smoke checks, not the fixed
eight-question Part 5 evaluation harness. No new test or configuration files
were added. The retriever returns nearest chunks without a relevance threshold.
Generation is described below; hybrid search, reranking, and version handling
remain later work.

A later assistant-run smoke check found the expected section in the top four
for **6/6 questions**: PTO accrual, PTO carryover, lodging limits, expense
submission deadlines, lost laptops, and remote-work days. Both Remote Work
versions appeared for the last question. The stored count stayed at 49. That
check received no live answers because its first API request failed; it is
retrieval evidence, not generated-answer evaluation.

## Answer generation: implementation and actual results

`generate_answer(question, chunks)` in `rag/generation.py` accepts a question
and the retrieved records and returns answer text. It rejects empty or
non-string questions and empty context, then combines chunk IDs and text into
the model's evidence. Retrieval is called by the module's terminal entry point,
not inside `generate_answer`, so the component can accept later hybrid or
reranked results without changing its interface.

Settings are loaded from the project-root `.env` without overriding existing
environment variables. Missing settings raise an error naming the required
fields without printing their values. The OpenAI client calls
`chat.completions.create` using `OPENAI_MODEL` as the deployment name and
`OPENAI_BASE_URL` as the v1 endpoint. No separate `api-version` is configured.
The client closes after the request. An empty model response raises an error.

The system prompt requests a concise answer supported only by the supplied
excerpts. It asks the model to acknowledge insufficient information and
conflicting statements, and to treat excerpts as data rather than instructions.
This does not remove either Remote Work version or implement the later
source-attribution presentation. Actual grounding and conflict handling still
require systematic evaluation in later parts; initial live examples are
recorded below.

Initial assistant-runtime command attempted from the project root on 2026-11-18:

```sh
HF_HOME="$PWD/.venv/huggingface" HF_HUB_OFFLINE=1 \
  .venv/bin/python -B -m rag.generation \
  "What is the daily meal reimbursement limit during business travel?" --top-k 4
```

That attempt retrieved context but exited with status 1 and this sanitized error:

```text
LLM request failed (APIConnectionError); check Foundry settings and connectivity.
```

A separate DNS check failed with `gaierror`, code 8. That attempt received no
live answer or HTTP response and did not validate credentials or deployment
access. The CLI prints the API error class without exposing the raw response,
endpoint, or credentials. Subsequent user-terminal runs succeeded, as recorded
below; the assistant-runtime connection failure remains a separate limitation.

Mocked SDK transport checks passed for request URL and deployment selection,
evidence and question inclusion, system instructions, timeout and retry settings,
answer whitespace trimming, empty responses, invalid input, and missing settings.
These checks used fake credentials and responses; they do not prove live
generation works. They were run directly in the terminal without adding files.

All **30 existing pytest cases passed** after implementation. The installed
OpenAI SDK is **3.55.1** and `python-dotenv` is **1.2.4**; both are now pinned in
`requirements.txt`. `pip check` reported no broken requirements after installation
and after the dependency-file update.

## Pipeline commands: implementation and actual results

`rag/pipeline.py` coordinates the existing components without duplicating their
implementations:

- `index` calls `index_corpus(path=DB_PATH)`: load all manifest documents, create
  heading-aware chunks, embed the text, and upsert into Chroma. After storage
  succeeds, compare stored IDs with the current chunk IDs and delete obsolete
  IDs. Return and print the final count. Both Remote Work versions are included.
- `ask "question" --top-k 4` calls retrieval and passes the returned records
  to `generate_answer`. It prints the answer without reindexing. Missing command
  arguments and invalid input exit with status 2; API failures exit with status 1
  and the same sanitized error format used by generation.

The real `index` command completed on 2026-11-18 and reported **49 indexed
chunks** in `.venv/chroma`. All **30 existing pytest cases passed**.
Additional terminal checks verified:

- Real indexing in a temporary database created 49 records. Adding an obsolete
  test record increased the count to 50; indexing again removed it and restored
  the exact original 49 IDs while retaining both Remote Work versions.
- A simulated storage failure prevented the cleanup step from running. Existing
  records, including the obsolete test record, remained present.
- The `ask` command used real retrieval to forward meal-policy evidence to a
  mocked SDK transport and printed its test answer. It did not call indexing
  or change the main collection's 49-record count.
- Missing commands, invalid `top_k`, and sanitized API failure handling behaved
  as expected, including keeping a fake secret out of the error output.

These particular checks added no test files and made no live LLM call. Index
upsert and stale-ID deletion are separate operations, not an atomic replacement.
The mocked answer is separate from the later successful live evidence below.

## Successful live runs: user-terminal evidence

On 2026-11-18, the user ran the full pipeline in the normal Mac terminal and
shared the resulting output in this conversation. This is user-provided live
evidence, not a mocked response or an assistant-run API result.

Command used:

```sh
cd /Users/Abdul.2.Shaik/RAGsfs
export HF_HOME="$PWD/.venv/huggingface" HF_HUB_OFFLINE=1
.venv/bin/python -B -m rag.pipeline ask \
  "What is the daily meal reimbursement limit during business travel?"
```

The actual generated answer supplied by the user was:

> The limit is **$75 per employee per calendar day** for eligible meals during
> approved business travel, including tax and tips. Meals paid directly by the
> company reduce the amount available for reimbursement.

This agrees with `EXP-03.1`: table `EXP-T01` defines the daily limit and included
costs, and the surrounding prose explains deductions for company-paid meals.
The run therefore demonstrated retrieval from the stored corpus followed by a
real Foundry API answer containing the expected policy information.

The user then tested the same retrieval/generation functions in an interactive
terminal session and shared these additional live results:

| Question | Observed answer | Assessment |
| --- | --- | --- |
| How many hours of PTO do I earn for each completed month? | 10 hours per completed month of employment, credited on the final calendar day. | Matches `PTO-03.1`. |
| How many days can I work remotely each week? | Identified v1.0's three-day allowance and v2.0's two-day allowance, both subject to eligibility and manager approval, but could not establish which version applied. | Both source versions surfaced; current-version selection remains Part 6 work. |

The Remote Work answer's uncertainty reflects an implementation limitation:
generation receives chunk IDs and text, not the manifest's status and effective
date metadata. The manifest marks v2.0 current and v1.0 superseded. Preserve this
observed behavior for later diagnosis rather than silently filtering the older
version now. These examples do not complete the Part 5 evaluation harness or
Part 6 diagnosis.

Earlier user-terminal attempts returned HTTP 401 before configuration
troubleshooting. The successful answers supersede that earlier authentication
failure as the latest user-terminal evidence; the precise credential change is
not recorded here. No keys were included in the log. `HF_HUB_OFFLINE=1` loaded
the cached BGE model successfully and did not disable remote Foundry requests.

Assistant-run API calls still fail with `APIConnectionError`, no HTTP status,
and an underlying `ConnectError [Errno 8]` during DNS lookup. This is distinct
from the successful user-terminal runs and does not invalidate their evidence.
The exact runtime network policy causing the difference was not identified.

## Baseline CI before the Part 2 PR

`.github/workflows/ci.yml` runs on pushes and pull requests with read-only
repository contents permission, an Ubuntu runner, Python 3.12, and a 20-minute
job timeout. It installs CPU PyTorch and `requirements.txt`, checks dependency
compatibility, then runs:

1. Corpus validation and the 30 existing pytest cases.
2. The real two-excerpt embed-store-retrieve check, expecting both questions to pass.
3. Full-corpus indexing through `rag.pipeline index`.
4. Four top-four retrieval checks across Expenses, PTO, Information Security,
   and Remote Work, including both Remote Work versions for the allowance query.

The workflow uses the public BGE model and no Foundry settings or secrets. It
tests real embeddings and Chroma retrieval; it does not run live generation or
claim to implement the Part 5 evaluation harness. The later harness must be
added to CI when built, as required for final submission.

Local verification on 2026-11-18 passed YAML structure and shell syntax checks,
corpus validation, **30 pytest cases**, **2/2 minimal retrieval questions**,
indexing of **49 chunks**, and **4/4 full-corpus retrieval checks**. Those commands
used the installed macOS environment and cached model. Linux CPU-wheel
installation and hosted GitHub Actions execution require a branch push and are
not claimed as locally verified. The PR's checks provide the hosted result.

## Completion criteria and handoff

Part 2 is complete when the pre-build retrieval proof is preserved, all policy
documents are ingested and chunked with the documented parameters, their
embeddings are indexed in ChromaDB, and a natural-language question produces an
LLM API answer grounded in retrieved chunks. Record actual chunk counts, retrieval
results, API execution, and limitations here. Existing validation/tests must
continue to pass, and all commits must use a separate branch and PR.

Part 2 is complete: the pre-build proof, documented chunking, Sentence
Transformers embeddings, persistent Chroma storage, top-k retrieval, and live
Foundry answer requirement are satisfied. All seven component files are
implemented, and existing tests plus the recorded smoke checks pass. Live
evidence comes from the user's working terminal, as explicitly attributed above.

Next, plan Part 3: add keyword matching alongside vector retrieval, combine
results, and measure at least one improvement over vector-only search. Record
its decisions when that part begins. Continue the usual one-file review
workflow; no Part 3 implementation is included in this update.

## References

- [MiniLM model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- [BGE model card](https://huggingface.co/BAAI/bge-small-en-v1.5)
- [Sentence Transformers installation](https://sbert.net/docs/installation.html)
- [Chroma Python client reference](https://docs.trychroma.com/reference/python)
- [Microsoft Foundry v1 API](https://learn.microsoft.com/en-us/azure/ai-foundry/openai/api-version-lifecycle)
