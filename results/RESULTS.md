# Results

Every number here was produced by a real run on the committed corpora and is reproducible from a
clean clone. The JSON beside this file is the machine-readable record; the logs are the raw
capture of the runs that produced the indexes.

| file | what it is | regenerate with |
|---|---|---|
| `ablation-ci.json` | retrieval ladder, CI corpus | `make ablation-publish` (no model needed; only `latency_ms` moves) |
| `ablation-demo.json` | retrieval ladder, demo corpus | `make ablation-demo-publish` (needs the fetched corpus, no model) |
| `build-stats-ci.json` | CI index build: chunks, entities, edges, communities, wall-clock | `make graph` after emptying `fixtures/extraction/ci` — the committed record is the cold build; on the committed cache the same target replays with no model calls and reports `cached_before` = 234 |
| `build-stats-demo.json` | demo index build, same fields | `make demo-index` |
| `demo-index-build.log` | raw capture of the 6.4-hour demo index build | `make demo-index` |
| `demo-summaries.log` | raw capture of the demo community-report run | `make demo-summaries` |
| `panel-review-met17-auxotroph.json` / `.log` | a real panel run on a real manuscript | `make review` |
| `panel-review-met17-auxotroph-run2.json` / `.log` | the same panel run again — byte-identical apart from wall-clock, which is the evidence that it is deterministic | `make review RUN=2` |
| `planted-eval-caprin-heterochromatin.json` / `.log` | planted errors on the held-out manuscript: panel vs single-agent baseline | `make eval` |
| `planted-eval-met17-auxotroph.json` / `.log` | the same protocol on the second manuscript | `make eval SUBJECT=met17-auxotroph` |

Every row's command is bound by a test: `tests/test_regenerate_commands.py` expands each cited
`make` line, runs the underlying command with `--where`, and fails if the path it names is not the
file in its row. Model-run records (the panel, the planted evaluation, the index builds) reproduce
the protocol, not the bytes — small local models are not deterministic across runs, and this file
says so wherever a number depends on one run.

## The indexes

|  | CI corpus | demo corpus |
|---|---|---|
| documents | 15 | 68 |
| chunks | 234 | 1,077 |
| entities | 1,759 | 7,697 |
| edges | 13,516 | 63,130 |
| communities (Leiden, resolution 1.0) | 32 | 79 |
| extractions truncated | 0 | 0 |
| extraction calls | 234 | 1,077 |
| largest prompt / smallest margin | 5,238 / 2,747 tokens | 7,666 / 2,613 tokens |
| build wall-clock | 2 h 31 min | 11 h 53 min |

Both built by `llama3.1:8b` through the provider seam at temperature 0, one call per chunk, every
result cached and committed under `fixtures/extraction/`.

## Retrieval ablation — demo corpus

Two query manuscripts, 36 relevant documents between them, k = 10.

| rung | found @10 | found @30 | recall@10 | ceiling | % of ceiling | NDCG@10 | mean latency |
|---|---|---|---|---|---|---|---|
| BM25 | 13 / 36 | 24 / 36 | 0.366 | 0.679 | 69% | 0.770 | 30.8 ms |
| vector | 13 / 36 | 26 / 36 | 0.366 | 0.679 | 69% | 0.770 | 0.5 ms |
| RRF hybrid | 13 / 36 | 25 / 36 | 0.366 | 0.679 | 69% | 0.770 | 36.2 ms |
| GraphRAG local | 10 / 36 | 23 / 36 | 0.312 | 0.679 | 54% | 0.641 | 97.2 ms |
| **GraphRAG global** | **14 / 36** | 25 / 36 | **0.473** | 0.679 | **76%** | **0.830** | 6.7 ms |

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
own published twin withheld from the index and from the graph. 226,585 tokens, 31 minutes, $0.

| | |
|---|---|
| reviewers | 2, blind and parallel, different retrieval scopes and model families |
| findings | 8 each, and neither reviewer failed to parse |
| scores | methods 4/4/4 · novelty 4/4/**3** (they disagree on contribution) |
| claims verified | 20 verdicts, every one judged twice with the evidence order reversed |
| verdicts | 0 SUPPORTS · 0 REFUTES · **20 NOT_ENOUGH_INFO** |
| verdicts carrying an evidence span | 6 of 20 |
| conflicts detected | 0 |
| deterministic lens | 0 findings — a published paper; the lens is proven on planted defects |

### The result worth reading: the verifier decides nothing

Every verdict in both committed runs is an abstention. Not one claim was SUPPORTED or REFUTED.

That is not the swap rule forcing caution. Of the 20 claims, **14 were swap-consistent** — the
verifier returned `NOT_ENOUGH_INFO` in both evidence orderings, of its own accord — and the other 6
disagreed with themselves when the order was reversed and were forced to abstain. So on this
configuration the judge either cannot decide, or decides differently depending on what it read
first. No claim was ever decided the same way twice.

The clearest single case: the claim *"The MET17 gene catalyzes homocysteine synthesis by reacting
H2S with O-acetyl homoserine"* was verified against a retrieved span reading *"Met17p catalyzes the
fixation of inorganic sulfide with O-acetylhomoserine (OAHS) to form…"* — the evidence states the
claim, the span survived the substring check, and the verdict is still `NOT_ENOUGH_INFO`.

An earlier version of this page reported 9 SUPPORTS and 2 REFUTES from this same panel. Those runs
were measured while every reviewer and verifier prompt was being silently cut to a 4,096-token
window (DECISIONS.md D19), so the decisive-looking verdicts came from a judge that had read part of
its evidence and none of its instructions. Reading the prompt whole made it strictly more
conservative, and the honest reading of that is not an improvement: **a judge that cannot say
SUPPORTS is not a judge**, and this one is currently an expensive way to produce abstentions.

### Swap consistency: 0.70, over a population where it means little

| run | swap-consistent | forced to abstain | rate | tokens | wall |
|---|---|---|---|---|---|
| 1 | 14 / 20 | 6 / 20 | swap-consistency 0.70 | 226,585 | 1,859 s |
| 2 | 14 / 20 | 6 / 20 | swap-consistency 0.70 | 226,585 | 1,654 s |

The rate is now exact rather than a bound: each verdict records whether it was actually judged
twice (`swapped`), and all 20 were, so the denominator is real. An earlier version could only
publish `[0.00, 0.50]` because the records did not distinguish "agreed" from "never tested".

It is also nearly meaningless as a measure of position bias here, and saying so is the point:
consistency between two abstentions is not resistance to order effects, because there was no
decision to lose. What the number really reports is that **30% of claims changed the verifier's
answer when the evidence order was reversed** — which is the documented position-bias effect
(~41% flip rates on decisive cases in the literature), reproduced by a small local model, on a
population that otherwise refuses to commit.

### The two runs are byte-identical, and that is the finding

Run 2 reproduces run 1 exactly: same 20 verdicts, same 16 findings, same converger prose, same
226,585 tokens — every field identical apart from wall-clock. At temperature 0 this panel is
deterministic, so a second run is a **reproduction, not an independent sample**.

An earlier version of this page presented two runs as evidence of variability ("12 of 24, then 14
of 24"). Whatever produced that spread, it is not present in this configuration, and two identical
runs cannot support a claim about variance. What they do support is reproducibility: anyone with
the pinned models can re-run `make review` and get this record back.

### What the run got wrong: nothing it quoted

Across both runs, **0 of 32 reviewer findings quote text that is not in the manuscript.** Every
quote in every finding is real manuscript text.

An earlier version of this section reported three findings in sixteen describing a different paper
— a reviewer attributing retrieved literature to the manuscript under review. That reviewer was
reading a cut prompt: the excerpt's head was gone and the rubric with it, leaving retrieved
literature as the most recent thing in its context. Read whole, the behaviour disappears entirely
on this manuscript. One manuscript and 32 findings is not a licence to call the failure mode
solved, and `tests/test_artifact_conformance.py` recomputes this count from the records on every
run, so if it returns the number returns with it.

Reproduce with a substring check of each finding's `quote` against `manuscripts/met17-auxotroph.txt`
and the committed `results/panel-review-met17-auxotroph.json`.

### The mechanism that had never fired, and now has

A verdict's citation survives only if the chunk was actually retrieved **and** the quote is a
substring of that chunk's own text, so a fabricated citation cannot pass the type system. Until
this rebuild its yield across every committed run was exactly zero — 0 of 24 verdicts carried an
evidence span — and this page said so, calling it an unfired safety check rather than a
demonstrated capability.

It fires now: **12 of 40 verdicts across the two committed runs carry a surviving evidence span**,
6 in each. The spans are real retrieved text that passed the substring check, and one of them is
quoted above.

Two things keep this from being a success story. The verdicts carrying those spans are all
`NOT_ENOUGH_INFO`, so the grounding demonstrably works and the judge it feeds does not use it. And
the reviewers' own citations remain nearly absent: of 16 findings in a run, **1 cites a retrieved
chunk at all**. The mechanism is proven; what it is attached to is not.

The deterministic lens is the capability still without positive evidence: **0 findings** on every
committed run, exactly as before. It is proven only on planted defects, in the evaluation below.

## The headline measurement: the panel loses to a single agent

Three errors planted into a held-out manuscript — a swapped gene symbol (`Dcr1`→`Dcr2`), a
reversed effect direction, and a fabricated citation — inside the 900-word window both systems
read. Neither system could retrieve the unperturbed original: the subject is held out of the corpus
entirely, so no answer key existed for either.

| system | detected | tokens | wall |
|---|---|---|---|
| **panel** (2 reviewers + verifier + deterministic lens + converger) | **1 / 3** | 240,852 | 1,743 s |
| **single agent**, chain-of-thought + self-consistency | **2 / 3** | 36,062 | 339 s |

**The single agent found twice what the panel found, using 15% of the tokens.** The panel caught
the swapped gene symbol and nothing else; the baseline caught that and the reversed effect
direction. Neither caught the fabricated citation. The panel spent 6.7x the tokens and
5.1x the wall-clock to find less.

Both arms record how many of their calls produced nothing because the output would not parse:
**0 for the panel and 0 for the baseline**. That field exists because the run this page used
to report was measured with one of the panel's two reviewers silently dead — its JSON never parsed,
it contributed no findings, and nothing in the record said so. A detection count depressed by a
parse failure is not a measurement of an architecture.

This is the row the multi-agent literature says is always missing. Debate and panel architectures
"often fail to outperform simple single-agent baselines such as Chain-of-Thought and
Self-Consistency, even when consuming significantly more inference-time computation"
([arXiv:2502.08788](https://arxiv.org/abs/2502.08788)). That is reproduced here, against the system
this repository was built to demonstrate, and it is published for the same reason the losing
retrieval rung is: a result that only appears when it flatters the architecture is not a
measurement.

The rebuild did not rescue this. Every other headline on this page moved when the prompts were read
whole — the retrieval ranking inverted, the hallucinated attributions went to zero, the grounding
mechanism fired for the first time — and the panel still loses this comparison, now by a slightly
smaller margin against a baseline that also got worse (2 of 3 where it had scored 3 of 3 while
reading half its prompt). The loss is the most robust finding here.

### It replicates on a second manuscript

The same protocol on `met17-auxotroph`, whose published twin **is** in the corpus and is therefore
withheld from both arms' index (26 chunks dropped for each):

| manuscript | twin | panel | single agent | panel tokens | baseline tokens | baseline share |
|---|---|---|---|---|---|---|
| `caprin-heterochromatin` | absent from corpus | 1 / 3 | 2 / 3 | 240,852 | 36,062 | 15% |
| `met17-auxotroph` | in corpus, excluded | 1 / 3 | 2 / 3 | 244,460 | 34,902 | 14% |

Same outcome both times, on two different manuscripts with two different exclusion situations, and
in both the panel caught only the swapped gene symbol while the baseline caught that and the
reversed effect direction. Neither system, on either manuscript, caught the fabricated citation —
the error kind a literature-grounded panel ought to be best placed to catch.

This page previously reported one such record. Two agreeing records is not a rate either (n = 3
errors each, and two error kinds could not be planted inside the reviewed window at all), but it
removes the most obvious escape route from the earlier result: that the loss was one unlucky
manuscript.

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
  at 36,062 tokens — 15% of the panel's spend. It won on a sixth of the compute, so the gap is
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
