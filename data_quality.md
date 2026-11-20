# Data quality diagnosis

**Question:** How many days per week can I work remotely?

## Evidence and observed answer

The [manifest](data/manifest.json) identifies v2.0 as current and v1.0 as superseded.

| Source: RW-03.1 | Status | Weekly allowance | Retrieved rank |
| --- | --- | --- | --- |
| [Remote Work v1.0](data/policies/remote-work-v1.0.md) | Superseded | 3 days | 1 |
| [Remote Work v2.0](data/policies/remote-work-v2.0.md) | Current | 2 days | 2 |

Hybrid retrieval selected 12 candidates; reranking retained four, including
`remote-work-v1.0:RW-03.1:0` and `remote-work-v2.0:RW-03.1:0`.

**Actual answer:**

> The excerpts disagree: **v1.0 allows up to 3 days per calendar week**, while **v2.0 allows up to 2**. Both require manager approval. The supplied information does not establish which version applies to you.

The expected current-policy answer is **2 days with manager approval**.
The system reported the disagreement but did not resolve it.

## Diagnosis

**Was it retrieval?** Both relevant sections reached the final context.
The current rule was not missed or truncated. Relevance ranking placed the
outdated version first; it does not establish policy authority.

**Was it source data?** Yes. The conflicting numbers exist in the source
documents. The two policies differ only in their title version and allowance.
Generation accurately reported the contradiction. Superseded records remain
eligible for retrieval, and generation omits their status/effective-date metadata.

## Current-only comparison

Removing the two superseded chunks from the same context, without changing
the question, model, or prompt, produced:

> Eligible employees may work remotely **up to 2 days per calendar week**, subject to manager approval. The week runs Monday through Sunday. [RW-03.1]

This supports a source-version diagnosis rather than a missing-source retrieval
failure or invented answer.

