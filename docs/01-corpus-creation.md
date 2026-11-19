# Stage 1: Corpus creation

Status: Corpus revision complete. All five expanded policies pass validation;
all 30 validator tests pass. Latest results are recorded below.

## Objective and scope

Create a synthetic corpus for an internal company policy Q&A assistant. This
stage covers policy content, version metadata, corpus validation, and engineering
documentation. Embeddings, vector search, reranking, programmatic LLM calls, and
CI implementation belong to later stages.

## Approved decisions

| Decision | Rationale |
| --- | --- |
| Use Northstar Analytics, a fictional US software company. | Gives all documents a consistent organizational setting without using real company policies. |
| Create four policy types across five Markdown files. | Remote Work, Expense Reimbursement, PTO, and Information Security require four current documents; the outdated Remote Work version adds a fifth file. |
| Make every document 1,200–1,500 body words. | User-approved extension on 2026-11-18 replaces the assignment's original 500–800-word range to provide longer material for chunking experiments. |
| Count body words in paragraphs, lists, and table headers/cells, excluding headings, metadata, and Markdown syntax. | Keeps the revised word-count requirement reproducible when policies contain tables. |
| Expand explanations, examples, handoffs, and exception procedures while retaining existing policy facts. | Provides substantive content and longer sections without changing approvals, limits, or the planted conflict. |
| Author the synthetic policy text directly during implementation. | Uses AI-generated source material without adding programmatic LLM calls to Stage 1. |
| Use stable section identifiers, such as `RW-01` and `SEC-01`. | Supports later chunk attribution, exact-match retrieval examples, and evaluation references. |
| Preserve a consistent H1/H2/H3 hierarchy and distinct paragraph, list, and table blocks. | Provides explicit boundaries for later experiments with structured chunking. |
| Include two tables in Expense Reimbursement and one each in PTO and Information Security. | Provides varied table content for later chunking experiments without adding more version conflicts. |
| Keep structured metadata authoritative in `data/manifest.json`. | Avoids maintaining duplicate structured metadata in policy files. Markdown titles will identify the policy and version. |
| Retain both Remote Work versions in the manifest. | Preserves the deliberate source-data conflict for diagnosis in Part 6. |
| Keep the validator dependency-free and stop at the first error. | Makes a small lab utility easier to explain and maintain; failures identify the check and document where applicable. |
| Use pytest for focused validator tests. | Confirms valid input passes and deliberately broken input fails; these tests are separate from the Part 5 retrieval and answer evaluation harness. |
| Create and review one file at a time. | The user must approve each file and confirm sufficient clarity before work moves to another file. |
| Allow multiple existing files to change together for this corpus/model revision only. | Explicit user exception permits coordinated policy, validation, and model updates; the usual review workflow resumes afterward. |

## File inventory

All files below exist. The reusable documentation outline is included in this
file; no separate template file is needed.

| Path | Purpose |
| --- | --- |
| `docs/01-corpus-creation.md` | Stage decisions, validation evidence, and handoff. |
| `data/policies/remote-work-v1.0.md` | Superseded Remote Work policy permitting three remote days per week. |
| `data/policies/remote-work-v2.0.md` | Current Remote Work policy permitting two remote days per week. |
| `data/policies/expense-reimbursement-v1.0.md` | Current Expense Reimbursement policy. |
| `data/policies/pto-v1.0.md` | Current PTO policy. |
| `data/policies/information-security-v1.0.md` | Current Information Security policy. |
| `data/manifest.json` | Metadata and source paths for all five document versions. |
| `scripts/validate_corpus.py` | Standard-library-only corpus validation. |
| `tests/test_validate_corpus.py` | 30 pytest cases for valid and invalid corpus inputs. |

## Metadata contract

The manifest contains `schema_version` (integer `1`), `company` (string
`Northstar Analytics`), and `documents` (an array of document objects).

| Document field | Type and meaning |
| --- | --- |
| `document_id` | Unique string identifying a version, for example `remote-work-v2.0`. |
| `policy_id` | Stable string grouping versions, for example `remote-work`. |
| `title` | Human-readable policy name. |
| `version` | String in major.minor form, for example `2.0`. |
| `effective_date` | Valid ISO date in `YYYY-MM-DD` form. |
| `status` | Either `current` or `superseded`. |
| `source` | Repository-relative path to the source Markdown file. |
| `supersedes` | Previous `document_id`, or `null` when no predecessor is included. |

Remote Work v1.0 is effective `2025-01-01` and marked `superseded`.
Remote Work v2.0 is effective `2026-01-01`, marked `current`, and supersedes
`remote-work-v1.0`. The other three policies use version `1.0`, are effective
`2026-01-01`, have status `current`, and have no predecessor in the corpus.

`source` records the local document location. Generation provenance is recorded
here: these documents are synthetic policy text authored by the AI assistant
for this lab, rather than policies obtained from a real employer.

## Content constraints and planted defect

Each policy follows the heading and body contract below. Content covers
practical rules, procedures, approval roles, responsibilities, and exceptions as
appropriate to the policy.

### Heading and body contract

- Exactly one H1 title: `# <Policy name> v<version>`.
- Six H2 sections, in order: Purpose and scope; Roles and eligibility; Policy
  rules; Procedures; Exceptions and escalation; Ownership and review.
- Each H2 has a stable identifier and descriptive heading, for example
  `## RW-03: Policy rules`. Prefixes are `RW`, `EXP`, `PTO`, and `SEC`.
- Use H3 subsections where they group meaningful content, for example
  `### RW-03.1: Weekly allowance`. Include at least one H3 subsection per
  document so the corpus contains nested section boundaries.
- Do not skip heading levels. Keep each section or subsection nonempty, and
  preserve section identifiers between Remote Work versions.
- Separate paragraphs, lists, headings, captions, and tables with blank lines.
  Use short explanatory paragraphs, numbered lists for ordered procedures, and
  bullet lists for parallel rules. Include at least one procedural list per
  document.
- Keep tables in Markdown text, rather than images, HTML, or code fences. Give
  each table a descriptive caption with a stable ID, such as `EXP-T01`, directly
  above it. Include a header row, delimiter row, and consistently sized data rows.
- Keep an explanation of the table's scope and any qualifications in its owning
  section. State units in headers or cells and avoid merged cells, multiline
  cells, and ambiguous shorthand.

The implemented table inventory is:

| Policy | Tables | Intended content |
| --- | --- | --- |
| Remote Work, both versions | None | Paragraphs, nested headings, and a request procedure; the remote-day rule appears once in each version. |
| Expense Reimbursement | `EXP-T01`, `EXP-T02` | Expense categories and limits; required supporting evidence by expense type. |
| PTO | `PTO-T01` | Request notice and approval requirements by planned absence duration. |
| Information Security | `SEC-T01` | Data classifications and permitted handling methods. |

Tables contain substantive policy facts, with at least three data rows and
three columns each. Surrounding prose must agree with their values. The body-word
count includes table header and cell text and list item text; it excludes heading
lines, table delimiters, pipes, and list markers. Each policy contains
1,200–1,500 body words. Longer procedural sections exercise actual chunk
splitting; shorter sections remain focused rather than being padded to a cap.

This structure prepares later comparisons of fixed-size chunks, chunks based on
heading boundaries, parent/child sections, whole-table chunks, and table-row
chunks carrying their headers and section context. These are future experiment
options; Stage 1 implements only the source structure. A later chunking decision
must specify how it preserves heading ancestry, table headers, captions, and
qualifications when splitting content.

Company terminology and cross-policy rules must agree. Remote work must comply
with Information Security; expense rules must not introduce conflicting benefits;
PTO must remain separate from remote work eligibility. Documents must contain no
unresolved placeholders, missing referenced sections, or additional deliberate
contradictions.

The sole intentional policy-rule conflict is the weekly remote work allowance:

| Document | Allowance |
| --- | --- |
| Remote Work v1.0, superseded | 3 remote days per week |
| Remote Work v2.0, current | 2 remote days per week |

The Remote Work policy bodies are otherwise identical; their titles and
version metadata will also differ. Explanations of the planted defect stay in
engineering documentation, outside the policy prose. Both versions remain
available for later indexing; Stage 1 does not implement runtime version filters.

In Part 6, demonstrate the flawed answer, inspect the retrieved sources, and
trace the failure to the conflicting source versions. Actual retrieval behavior
and diagnosis evidence remain pending until that part is implemented.

## Policy content decisions

| Policy | Main rules chosen for the synthetic corpus |
| --- | --- |
| Remote Work | Standard weekly cap differs by version; recurring requests need five business days' notice and a manager response within three business days. Remote work remains working time and uses company-managed devices. |
| Expense Reimbursement | Actual eligible spending: meals up to $75 per employee per calendar day; lodging up to $200 per room per night; ground transport at actual standard fare. Submit within 30 calendar days; manager review within five business days; Finance payment within ten business days after receiving a complete approved claim. |
| PTO | Ten hours accrue per completed month, totaling 120 hours per full year; up to 40 hours carry over. Planned absence notice is two, five, or ten business days by duration; absences longer than five scheduled workdays also require People Operations approval. |
| Information Security | Public, Internal, and Confidential classifications; company-managed devices, multifactor authentication, and VPN access to internal network resources when remote. Automatic screen lock after five minutes; suspected incidents reported immediately through the incident portal or IT help desk. |

The full source documents define qualifications and exception procedures. These
are fictional company choices, not claims about a real employer or legal rules.

## Validation approach and evidence

The validator reads the manifest, checks each source, and then compares the two
Remote Work versions. Its four check functions keep the flow easy to explain:

| Function | Responsibility |
| --- | --- |
| `check_metadata` | Required fields and types, approved inventory, IDs, source paths, versions, dates, statuses, and supersession. |
| `check_structure` | Body length, versioned title, six ordered H2 sections, nested H3 sections, unique IDs, nonempty sections, and numbered procedures. |
| `check_tables` | Expected table IDs, captions, nonempty headers/cells, delimiter rows, consistent columns, and minimum dimensions. |
| `check_conflict` | Exactly one three-day allowance in v1.0 and one two-day allowance in v2.0; otherwise identical bodies after whitespace normalization. |

`count_words` includes paragraphs, list text, table captions, and table cells,
while excluding headings and Markdown markers. `validate_corpus` also checks
that the source inventory matches the manifest. Explicit checks raise
`ValueError`; the command prints the first failure and exits with status 1.
On success it prints per-document counts and the preserved-conflict result and
exits with status 0. The script never repairs or rewrites source files.

The script uses Python's standard library; pytest is needed only for tests. It
checks the Markdown forms used by this corpus, rather than implementing a general
Markdown parser. Checks are deliberately specific to the approved Stage 1
inventory; changing that inventory requires updating the validation rules.

Commands run from the repository root:

```sh
.venv/bin/python -B scripts/validate_corpus.py
.venv/bin/python -B -m pytest -q -p no:cacheprovider
```

Results recorded on 2026-11-18 with Python 3.12.16 and pytest 9.2.0:

| Document | Body words | Tables | Result |
| --- | --- | --- | --- |
| Remote Work v1.0 | 1,405 | 0 | PASS |
| Remote Work v2.0 | 1,405 | 0 | PASS |
| Expense Reimbursement v1.0 | 1,397 | 2 | PASS |
| PTO v1.0 | 1,397 | 1 | PASS |
| Information Security v1.0 | 1,418 | 1 | PASS |

Corpus validation exited with status 0. Metadata, heading structure, table
structure, and version relationships passed. The expected three-day versus
two-day conflict remains intact; no other Remote Work body differences were
found. All five policies meet the revised 1,200–1,500-body-word range. Their
existing IDs, versions, effective dates, statuses, source paths, and tables
remain unchanged, so the manifest required no edits.

Pytest exited with status 0: **30 passed**. Tests cover the real corpus, invalid
metadata, duplicate IDs, Markdown word counting, 1199/1200/1500/1501-word boundaries,
broken headings and procedures, malformed tables, altered Remote Work rules,
missing/unregistered sources, invalid JSON, and command execution from another
working directory. Negative tests modify in-memory strings or temporary copies;
the real policies and planted conflict remain unchanged by tests. The validator
and its four boundary-test cases now enforce the revised length range.

Manual review checked that remote work remains working time, PTO does not create
expense entitlements, expense and leave records use the approved portals, and
security rules apply across work locations. No additional policy contradiction
was identified in that review. Automated structural checks do not prove semantic
consistency. Retrieval recall and generated-answer accuracy remain unmeasured
until Part 5; these 30 tests are corpus validation tests. Full-corpus chunking
and BGE embedding checks for the expanded sources are recorded in
`docs/02-build-pipeline.md`.

## Documentation template for later stages

Use this outline inside each stage's documentation file:

1. Objective.
2. Scope.
3. Decisions and rationale.
4. Files changed.
5. Interfaces and data structures.
6. Validation and evidence.
7. Limitations.
8. Completion criteria.

Create each remaining document only when its part begins. Follow the approved
order: source documents; pipeline, including the minimal prerequisite; hybrid
retrieval; reranking; evaluation; data quality diagnosis; source attribution;
CI and submission. No additional template file is planned.

## Completion criteria and handoff

The revised corpus meets its content and word-count constraints, matches the
manifest, passes structural validation, and preserves the intended conflict.
The reusable documentation outline is recorded above. No optional files were
created for this revision.

Part 2's minimal embed-store-retrieve prerequisite already passed before the
full build. Ingestion, chunking, and embeddings are implemented; persistent
storage, full-corpus retrieval, and generation remain next. The present
revision does not implement those remaining components.
