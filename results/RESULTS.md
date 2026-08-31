# Results

Every number here was produced by a real run on the committed corpora and is reproducible from a
clean clone. The JSON beside this file is the machine-readable record; the logs are the raw
capture of the runs that produced the indexes.

| file | what it is | regenerate with |
|---|---|---|
| `ablation-ci.json` | retrieval ladder, CI corpus | `make ablation` (no model needed) |
| `ablation-demo.json` | retrieval ladder, demo corpus | `make ablation-demo` |
| `build-stats-ci.json` | CI index build: chunks, entities, edges, communities, wall-clock | `make graph` |
| `build-stats-demo.json` | demo index build, same fields | `make demo-index` |
| `demo-index-build.log` | raw capture of the 6.4-hour demo index build | `make demo-index` |
| `demo-summaries.log` | raw capture of the demo community-report run | `make demo-summaries` |

## The indexes

|  | CI corpus | demo corpus |
|---|---|---|
| documents | 15 | 68 |
| chunks | 212 | 948 |
| entities | 1,577 | 6,722 |
| edges | 12,166 | 55,146 |
| communities (Leiden, resolution 1.0) | 27 | 66 |
| extractions truncated | 0 | 0 |
| build wall-clock | 85 min | 6 h 23 min |

Both built by `llama3.1:8b` through the provider seam at temperature 0, one call per chunk, every
result cached and committed under `fixtures/extraction/`.

## Retrieval ablation — demo corpus

Two query manuscripts, 36 relevant documents between them, k = 10.

| rung | relevant found | recall@10 | ceiling | % of ceiling | NDCG@10 | median latency |
|---|---|---|---|---|---|---|
| BM25 | 18 / 36 | 0.366 | 0.679 | 54% | **0.770** | 26 ms |
| vector | 17 / 36 | 0.366 | 0.679 | 54% | **0.770** | 1 ms* |
| RRF hybrid | 17 / 36 | 0.366 | 0.679 | 54% | **0.770** | 24 ms |
| GraphRAG local | **18 / 36** | **0.393** | 0.679 | **58%** | 0.713 | 437 ms |
| GraphRAG global | 10 / 36 | 0.312 | 0.679 | 46% | 0.586 | 6 ms |

\* the vector rung's query embedding is served from the committed fixture, so its latency here is
lookup only, not encoding.

**Read this honestly, including the parts that do not flatter the graph:**

- **GraphRAG global is the worst rung on this corpus, on every measure.** It finds 10 of 36
  relevant documents where BM25 finds 18. Community-level routing is built for corpus-wide
  questions ("what themes exist here"), and these queries are specific-document lookups — the
  wrong tool, measured rather than quietly omitted.
- **GraphRAG local wins on recall and loses on ranking.** It surfaces the most relevant documents
  (18, and the highest share of the achievable ceiling at 58%) but orders them worse than BM25
  does (NDCG 0.713 vs 0.770) — entity-neighbourhood voting reaches documents lexical matching
  misses, then ranks them by graph proximity rather than by textual fit.
- **BM25 is the cost-effectiveness winner.** It matches or beats every other rung's NDCG at 26 ms
  and no index beyond a token count. A retrieval layer that cannot beat BM25 on a corpus like this
  has not earned its complexity, and on ranking, this one does not.
- **Recall@10 is reported against its ceiling for a reason.** With 28 relevant documents for one
  manuscript, no system can exceed 10/28 = 0.357 recall in a top-10 list. A bare "recall@10 =
  0.366" would read as poor when it is 54% of what is achievable. NDCG needs no such caveat — its
  denominator already accounts for the ceiling.
- **The corpus is citation-seeded** (see `DECISIONS.md` D6): it was built from these manuscripts'
  own reference lists, so it contains the answers by construction. These numbers compare rungs
  against each other. They are not an estimate of real-world retrieval difficulty.

## Retrieval ablation — CI corpus

`ablation-ci.json` runs the same five rungs over the 15-document CI corpus with no model at all
(3 s from a clean clone). It reports **no rates**: the CI corpus is topical rather than
citation-seeded, so it contains 0 ground-truth documents, and the N ≥ 20 gate withholds recall and
NDCG rather than printing a rate over nothing. Two of the three query manuscripts are skipped
there and say why — their published twins are not CI-corpus members, so those runs would have
nothing to exclude and are not valid runs.

That file exists to prove the harness runs offline and reports honestly when it has nothing to
measure, not to carry evidence.
