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
| `planted-eval-demo.json` | planted errors: panel vs single-agent baseline | `make eval` |

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
| BM25 | 13 / 36 | 25 / 36 | 0.366 | 0.679 | 69% | 0.770 | 20 ms |
| **vector** | **14 / 36** | **26 / 36** | **0.429** | 0.679 | **75%** | **0.806** | **0.4 ms** |
| RRF hybrid | 13 / 36 | 25 / 36 | 0.366 | 0.679 | 69% | 0.770 | 22 ms |
| GraphRAG local | 12 / 36 | 22 / 36 | 0.393 | 0.679 | 65% | 0.713 | 85 ms |
| GraphRAG global | 13 / 36 | 25 / 36 | 0.366 | 0.679 | 69% | 0.685 | 5 ms |

Every rung is scored over a document list of the **same length**. That sounds obvious and an
earlier version of this table did not do it: it retrieved a fixed 30 *chunks* per rung and then
scored the *document* ranking those chunks happened to produce — 4 documents for RRF against 13 for
GraphRAG local on the same query, both reported as "recall@10". Chunk depth is now grown until every
rung offers the same number of documents. Correcting it moved real conclusions, which is the point
of publishing the harness alongside the numbers.

"% of ceiling" is macro-averaged across cases like every other rate here; it was previously a
ratio-of-means, which silently reweighted toward the case with the larger denominator and inverted
which rung appeared to win.

Two hit columns, both labelled, because they disagree and the disagreement is the point.
**found @10** counts relevant documents inside the top-10 list the reported rates are computed over.
**found @30** counts them inside a list three times as deep — where a rung with a long tail catches
up. An earlier version of this table published only the deeper count under a `k = 10` heading,
which flattered exactly the rung whose ranking is weakest. Both are here now.

\* the vector rung's query embedding is served from the committed fixture, so its latency here is
lookup only, not encoding. Latencies are means over the two cases, not medians.

**Read this honestly, including the parts that do not flatter the graph:**

- **The plain vector rung wins on every aggregate measure.** Highest found@10 and found@30,
  highest recall, highest NDCG. A cosine lookup over committed embeddings beats the entire graph
  pipeline — the extraction, the Leiden partitioning, the community reports, all of it — on this
  corpus at this scale. Its 0.4 ms is *not* a fair speed claim: query embeddings are served from a
  committed fixture, so that figure excludes encoding. BM25's 20 ms is the honest cost baseline,
  and GraphRAG local runs 4x that.
- **GraphRAG local has the worst deep recall of any rung** (22 of 36 against vector's 26) and the
  second-worst NDCG, at 4x BM25's latency. Its one aggregate win is recall@10
  over BM25 (0.393 vs 0.366), and that is itself a mean over a reversal (below).
- **GraphRAG global is mid-pack, not worst.** An earlier version of this document called it "the
  worst rung on every measure" at 10 of 36. That was the depth defect: given a document list as
  long as the other rungs', it finds 25 of 36 and ties BM25 on recall. The correction is bigger
  than the original claim.
- **n = 2, and the ordering reverses between the two cases — so this establishes behaviour, not a
  ranking.** Every aggregate above is a mean over two manuscripts that disagree:

  | rung | MET17: recall / NDCG | biopolymer: recall / NDCG |
  |---|---|---|
  | BM25 · RRF | 0.375 / 0.539 | 0.357 / **1.000** |
  | vector | **0.500** / 0.612 | 0.357 / **1.000** |
  | GraphRAG local | **0.500** / 0.590 | 0.286 / 0.837 |
  | GraphRAG global | 0.375 / 0.370 | 0.357 / **1.000** |

  GraphRAG local ties vector for best on MET17 (0.500) and is the **worst rung** on the biopolymer
  paper (0.286). At this n no rung ordering survives per-case inspection, and any sentence of the
  form "rung X wins" is an artifact of averaging. This is the same discipline as the N ≥ 20 gate
  and the swap-consistency range: a mean over two opposing cases does not earn three significant
  digits.
- **The corpus is citation-seeded** (see `DECISIONS.md` D6): it was built from these manuscripts'
  own reference lists, so it contains the answers by construction. These numbers compare rungs
  against each other. They are not an estimate of real-world retrieval difficulty, and there is no
  random-selection floor here to say how much of any rung's score is the corpus rather than the
  method.


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

**Between six and seven of every twelve claim verdicts flipped when the evidence order was
reversed**, and were forced to abstain. This is the most useful result the system has produced, and
the spread between the two runs is part of it: a single run's rate is not precise to three digits,
and an earlier version of this section published `0.50` as though it were. Anyone re-running will
get a third number in this neighbourhood, not this one.

Position bias in LLM judges is documented — first-shown options are picked ~64% of the time, and
the median model flips on ~41% of decisive swapped-order cases — and both runs land at or above the
high end of that on a small local model. Without the order-swap, five to six in every ten of these
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


## The headline measurement: the panel loses to a single agent

Three errors planted into a held-out manuscript — a swapped gene symbol (`Dcr1`→`Dcr2`), a
reversed effect direction, and a fabricated citation — inside the 900-word window both systems
read. Neither system could retrieve the unperturbed original: the subject is held out of the corpus
entirely, so no answer key existed for either.

| system | detected | tokens | wall |
|---|---|---|---|
| **panel** (2 reviewers + verifier + deterministic lens + converger) | **1 / 3** | 165,334 | 2,751 s |
| **single agent**, chain-of-thought + self-consistency | **3 / 3** | 36,857 | 344 s |

**The single agent found every planted error. The panel found one, using 4.5× the tokens and 8× the
wall-clock.** It missed the swapped gene symbol and the reversed effect direction; it caught only
the fabricated citation.

This is the row the multi-agent literature says is always missing. Debate and panel architectures
"often fail to outperform simple single-agent baselines such as Chain-of-Thought and
Self-Consistency, even when consuming significantly more inference-time computation"
([arXiv:2502.08788](https://arxiv.org/abs/2502.08788)). That is reproduced here, against the system
this repository was built to demonstrate, and it is published for the same reason the losing
retrieval rung is: a result that only appears when it flatters the architecture is not a
measurement.

### Reading it honestly — including what would make it fairer

Three things about this comparison a careful reader should weigh, none of which reverses it:

- **The two systems were asked different questions, and that is the biggest confound.** The
  baseline's prompt is *"reviewing a manuscript excerpt for errors and weaknesses… quote the exact
  problematic text"*. The panel's reviewers are asked for a structured pre-submission review scored
  on soundness, presentation and contribution. One was pointed at the task being measured; the
  other was pointed at reviewing. A fairer comparison would give both the same instruction — and
  the fact that a general-purpose review panel misses a swapped gene symbol *is itself the finding*,
  because catching that is a reviewer's job.
- **The budget did not match, in the baseline's disfavour.** The baseline hit its sampling ceiling
  at 36,857 tokens — 22% of the panel's spend. It won with a fifth of the compute, so the gap is
  not explained by resources.
- **n = 3 errors, one manuscript.** Per D14, this establishes behaviour, not a rate. Two error
  kinds could not be planted inside the reviewed window at all and are recorded as skipped rather
  than quietly shrinking the denominator.

### What the panel is for, then

The panel's measured strengths are elsewhere in this document and they are real: order-swapped claim
verification that abstains on disagreement (five to six verdicts in ten flip on evidence order), a
deterministic lens no model can talk out of a finding, and evidence grounding enforced in code. What
this measurement says is that **assembling those parts into a panel did not, here, make it better at
finding planted errors than one well-prompted agent** — and that a system's architecture has to earn
its cost against the simplest thing that could work, every time, in public.
