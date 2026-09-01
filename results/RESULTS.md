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
| `panel-review-met17.json` / `.log` | a real panel run on a real manuscript | `make review` |
| `panel-review-met17-run2.json` | an independent second run of the same panel | `make review` |

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

| rung | found @10 | found @30 | recall@10 | ceiling | % of ceiling | NDCG@10 | mean latency |
|---|---|---|---|---|---|---|---|
| BM25 | **13 / 36** | 18 / 36 | 0.366 | 0.679 | 54% | **0.770** | 28 ms |
| vector | **13 / 36** | 17 / 36 | 0.366 | 0.679 | 54% | **0.770** | 4 ms* |
| RRF hybrid | **13 / 36** | 17 / 36 | 0.366 | 0.679 | 54% | **0.770** | 27 ms |
| GraphRAG local | 12 / 36 | 18 / 36 | **0.393** | 0.679 | **58%** | 0.713 | 624 ms |
| GraphRAG global | 10 / 36 | 10 / 36 | 0.312 | 0.679 | 46% | 0.586 | 8 ms |

Two hit columns, both labelled, because they disagree and the disagreement is the point.
**found @10** counts relevant documents inside the top-10 list the reported rates are computed over.
**found @30** counts them inside a list three times as deep — where a rung with a long tail catches
up. An earlier version of this table published only the deeper count under a `k = 10` heading,
which flattered exactly the rung whose ranking is weakest. Both are here now.

\* the vector rung's query embedding is served from the committed fixture, so its latency here is
lookup only, not encoding. Latencies are means over the two cases, not medians.

**Read this honestly, including the parts that do not flatter the graph:**

- **GraphRAG global is the worst rung on this corpus, on every measure.** It finds 10 of 36
  relevant documents where BM25 finds 18. Community-level routing is built for corpus-wide
  questions ("what themes exist here"), and these queries are specific-document lookups — the
  wrong tool, measured rather than quietly omitted.
- **At the reported depth, BM25 beats GraphRAG local outright.** In the top 10, BM25 finds 13
  relevant documents to GraphRAG local's 12, and ranks them better (NDCG 0.770 vs 0.713). The graph
  only draws level three times deeper (18 each at @30), which is another way of saying its ranking
  is the weak part: it reaches documents lexical matching misses, then puts them too far down the
  list to help. GraphRAG local's one genuine win is recall@10 (0.393 vs 0.366) — it retrieves a
  larger *share* of what exists, while placing fewer documents in the top 10 than BM25 does.
- **BM25 is the cost-effectiveness winner, and it is not close.** Best NDCG, most documents in the
  top 10, 28 ms, and no index beyond a token count — against 624 ms for GraphRAG local, which is
  22× slower for a worse top-10. A retrieval layer that cannot beat BM25 on a corpus like this has
  not earned its complexity, and on this corpus, at this depth, it does not.
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


## The panel, on a real manuscript

`make review` on the MET17 preprint against the 68-document demo corpus, with that manuscript's
own published twin withheld from the index and from the graph. 142,852 tokens, 39 minutes, $0.

| | |
|---|---|
| reviewers | 2, blind and parallel, different retrieval scopes and model families |
| findings | 8 each |
| scores | methods 4/4/4 · novelty 4/4/**3** (they disagree on contribution) |
| claims verified | 24, each judged twice with the evidence order reversed |
| verdicts | 9 SUPPORTS · 2 REFUTES · 13 NOT_ENOUGH_INFO |
| conflicts detected | 1 (evidence: a reviewer's own claim refuted by retrieval) |
| deterministic lens | 0 findings — a published paper; the lens is proven on planted defects |

### The result worth reading: swap-consistency 0.42–0.50, and it is noisy

Two independent runs of the same panel, same manuscript, same withheld twin, same models:

| run | swap-consistency | verdicts forced to abstain | tokens | wall |
|---|---|---|---|---|
| 1 | 0.500 | 12 / 24 | 142,852 | 2,351 s |
| 2 | 0.417 | 14 / 24 | 137,561 | 2,964 s |

**Between two and five of every twelve claim verdicts flipped when the evidence order was
reversed**, and were forced to abstain. This is the most useful result the system has produced, and
the spread between the two runs is part of it: a single run's rate is not precise to three digits,
and an earlier version of this section published `0.50` as though it were. Anyone re-running will
get a third number in this neighbourhood, not this one.

Position bias in LLM judges is documented — first-shown options are picked ~64% of the time, and
the median model flips on ~41% of decisive swapped-order cases — and both runs land at or above the
high end of that on a small local model. Without the order-swap, four to five in every ten of these
verdicts would have been artifacts of which evidence chunk happened to come first, and they would
have looked exactly like judgements. The mitigation is not decoration; on this evidence it does
more work than any other component in the panel.

(The wall-clock figures are not comparable to each other: the two runs overlapped on one GPU. That
is also why sampling temperature, not timing, explains the token difference.)

### And one thing the run got wrong

One of the 16 findings — *"The authors use a commercial TAG assay kit to quantify triacylglycerol
content"* — describes a **different paper**. None of `triacylglycerol`, `TAG assay` or `lipid`
appears anywhere in the MET17 manuscript; the reviewer took retrieved literature context and
attributed it to the manuscript under review, despite the prompt labelling the two sections
separately.

That is 1 in 16 findings hallucinated by conflation, from an 8-billion-parameter model. It is
reported here rather than trimmed, because the number a reader needs in order to calibrate how much
to trust the other fifteen is exactly this one. It is also a concrete argument for the claim
verifier: a grounding check that demands a quote be a substring of a retrieved chunk is the kind of
mechanism that catches this class of error, and a larger model would reduce but not eliminate it.
