# Results

Every number on this page is the output of the code at this commit, read from the JSON record beside this file.

What "reproducible" means differs by layer, and this page states the difference rather than averaging it away.
The deterministic layers — the retrieval ladder, the graph rebuild from committed extractions, the quickstart — reproduce byte for byte from committed bytes, and `tests/test_artifact_conformance.py` regenerates one of them on every test run to prove it.
The model layers reproduce the protocol, not the bytes: small local models at temperature 0 are nearly but not exactly repeatable, and the measured size of that gap is published below instead of hedged about.
The demo-scale tables need one more thing before their commands can run at all — the 68 open-access papers they measure are fetched rather than committed, because they are third-party CC-BY works pinned by md5 in `corpus/demo.manifest.json`, so `make corpus` comes first.
The CI-scale artifacts need neither a fetch nor a model.

## Every committed record, and what writes it

| file | what it is | regenerate with |
|---|---|---|
| `ablation-ci.json` | retrieval ladder, CI corpus | `make ablation-publish` (no model; only `latency_ms`, `run_utc` and `load_average_1m` move) |
| `ablation-demo.json` | retrieval ladder, demo corpus | `make ablation-demo-publish` (needs the fetched corpus; no model) |
| `build-stats-ci.json` | CI index build: chunks, entities, edges, communities, run conditions, wall-clock | `make graph` after emptying `fixtures/extraction/ci` — the committed record is the cold build; on the committed cache the same target replays with no model calls and reports `cached_before` = 234 |
| `build-stats-demo.json` | demo index build, same fields | `make demo-index` after emptying `fixtures/extraction/demo` |
| `demo-index-build.log` | the demo index build's own console output: the embedding-fixture hash, the published record, the path it was written to | `make demo-index` |
| `summaries-stats-ci.json` | CI community reports: wire, model, calls, per-call margin, reports generated and served from cache | `make summaries` after emptying `fixtures/summaries/ci` — a warm run makes no calls and records none |
| `summaries-stats-demo.json` | demo community reports, same fields, at the one resolution retrieval reads | `make demo-summaries` after emptying `fixtures/summaries/demo` |
| `demo-summaries.log` | that run's console output: report count and the two paths written | `make demo-summaries` |
| `panel-review-met17-auxotroph.json` / `.log` | a real panel run on a real manuscript | `make review` |
| `panel-review-met17-auxotroph-run2.json` / `.log` | the same panel run again, so a reader can diff two runs of one configuration | `make review RUN=2` |
| `planted-eval-caprin-heterochromatin.json` / `.log` | planted errors on a manuscript with no published twin in this corpus: the panel and one single-agent arm per named local model, each arm's written strings and the residue actually scored | `make eval` |
| `planted-eval-met17-auxotroph.json` / `.log` | the same protocol on the second manuscript, whose twin IS a corpus member and is withheld per run | `make eval SUBJECT=met17-auxotroph` |
| `non-model-records.json` | which JSON under `results/` no model produced, and why | hand-maintained — proven by `tests/test_run_conditions.py` |
| `derived-fields.json` | which field on which record was written by derivation rather than measurement | hand-maintained — proven by `tests/test_run_conditions.py` |
| `evaluation-loop.json` | the adversarial review loop in numbers: rounds, findings per round, the pinned evaluator prompt's sha256 | hand-maintained — proven by `tests/test_run_conditions.py` |

Every `make` row is bound: `tests/test_regenerate_commands.py` expands the cited line, asks the underlying command where it would write (`--where`, no work done), and fails if that path is not the file in the row.
The three hand-maintained rows take a second shape on purpose — no command produces them, and inventing one would be a reproduction instruction that does not reproduce — so each names the test that reads it instead, and that test's existence is itself checked.
`tests/test_run_conditions.py` then sweeps every JSON under `results/`: each file either carries the run conditions of the model run that produced it, or is named in `results/non-model-records.json` with the reason no model produced it.

Run conditions mean, per record: which wire and model served it, how many calls it made, the largest prompt any of those calls sent, the smallest window any of them ran under, the smallest margin between the two, and the source the window figure was read from.
The two wires report their window differently and say so in the record itself — the native wire names the `options.num_ctx` it configured for the call, the OpenAI-compatible wire names the server's loaded window as read before each call — so a reader can tell from the record alone that no prompt was cut.

## What reproduces, and what does not

**The retrieval ladder reproduces exactly, except for the clock and the machine.**
`results/ablation-demo.json` has now been published from five separate runs in this repository's history, either side of a full regeneration of the community reports the graph rungs read.
Every quality figure is identical in all five: the same hits, the same ranks, the same recall, the same NDCG, for every rung on every case.
The three fields the schema declares run-varying — `latency_ms`, `run_utc` and `load_average_1m`, named in `RUN_VARYING_FIELDS` in `src/peerpanel/evals/ablation.py` — are the only ones that moved, and the conformance test strips exactly those and requires the timestamp to have changed.

**The latency column is the one measurement here that depends on the machine rather than the code, and the record now states the machine it was measured on.**
Every reading in the series below is one a reader can open — `git show 0fa86c1:results/ablation-demo.json`, then `c607ff4`, `b1f58dc`, `192e804`, and the record committed beside this page.
Across those five runs BM25's mean read 30.8 ms, then 63.3 ms, then 30.9 ms, then 27.0 ms, then 33.6 ms, and GraphRAG local's read 97.2 ms, then 118.6 ms, then 99.6 ms, then 94.3 ms, then 162.2 ms.
The ratio between the slowest graph rung and the lexical baseline has therefore read 3.16, then 1.87, then 3.22, then 3.49, then 4.83 on identical bytes — a spread of more than two and a half to one, on a column whose quality figures did not move by a digit across any of the five.
An earlier version of this paragraph published a reading that no commit of this file holds — it was measured in a working tree and overwritten before it was committed — and reported four runs in one sentence and five four lines below it; the reading is withdrawn rather than described, and the count is now the same number in both places.

**What that column is worth is a question about the machine, so the record now answers it.**
`AblationReport` carries `load_average_1m` beside `run_utc`, declared run-varying with `latency_ms` and read as the run starts, and both ladders committed here name theirs: the demo ladder ran at a one-minute load average of 18.49 and the CI ladder at 19.43.
Neither is a quiet machine, and the figure is published rather than the run repeated until it looked like one — so the 162.2 ms in the table below can be read for what it is, a busy laptop, which is more than any of the four readings before it allows.
What those four say about their own conditions is prose written afterwards rather than a field in the file: `DECISIONS.md` D23 records that the second was taken while a community-report regeneration was saturating the GPU, and answered that with a rule about when the ladders are republished — a promise about the machine, which this record replaces with a measurement of it.
This repository's bar for every other record is that a measurement whose conditions the record cannot state is not evidence for the number it carries, and the latency column was the one place that bar was kept as a rule about behaviour instead of as a field.
It is now kept as a field, on the repository's own terms: an idle laptop is not something a laptop can be held to, and a reader who can see the load can weigh the number instead of trusting the promise.
So this column supports the **ordering** of the rungs and nothing finer, every ratio drawn from it below is written as an approximation on purpose, and no conclusion on this page rests on a latency difference.

**The panel reproduces byte for byte, across two rebuilds.**
`results/panel-review-met17-auxotroph.json` regenerated on this commit is identical to the record published at `52238c8` — every verdict, every finding, every score, the converger's whole prose, the same 226,585 tokens — apart from `wall_s` and the four fields the schema has gained since: `models`, the window source inside `model_calls`, and the two citation-hygiene counters on each reviewer.
The two committed runs of that same configuration differ from each other in `wall_s` alone, at 1,297.9 s and 1,296.7 s.

**The single-agent arm does not, and its record says why.**
The baseline is chain-of-thought with self-consistency: 12 samples, the first at temperature 0 and the other 11 at 0.7 (`src/peerpanel/evals/baseline.py`), so it is a sampler and two runs of it are two samples rather than a reproduction.
Re-running the planted evaluation on this commit moved its output — the `qwen2:7b` arm wrote 40 finding strings on `caprin-heterochromatin` where the run published at `192e804` wrote 35, and the `llama3.1:8b` arm 75 where that run wrote 87, so the count moved in both directions between two runs of one configuration.
The panel's 16 strings per run did not move.
Every conclusion drawn from the baseline arms below is therefore drawn from one sample of a stochastic system, and the section on the detection rule says where that mattered.

**The summariser is nearly, but not perfectly, reproducible at temperature 0, and here is the number.**
Regenerating all 79 demo community reports cold, same prompt, same model, same wire, changed the wording of exactly **1 of 79**: one cluster's summary prose came back differently phrased, with its title, its member list and every other report unchanged.
That is the whole of the drift, and it is why the model layers are described as reproducing the protocol rather than the bytes.

**One field on two records is derived rather than measured, and is disclosed as such.**
`build-stats-ci.json` and `build-stats-demo.json` predate the window-source field.
They were not re-run: the field is a constant of the wire each record already names in `provider`, and a fresh extraction is non-deterministic, so re-running would have moved every downstream number in order to write a string the code path determines.
They gained it through the emitter's own derivation (`make derive-window-sources`), which is idempotent and proven equal to what a live build writes; `results/derived-fields.json` names the field, the record, the date and the command that would re-measure it.
This is also why `demo-index-build.log` shows a `model_calls` block without that field while the JSON beside it carries one — the log is the run's own output, captured and never edited.

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

Both indexes were built by `llama3.1:8b` on the native wire at temperature 0, one call per chunk, every extraction cached and committed under `fixtures/extraction/`.
Every row of this table is bound to its build record by row label and column, so a row that drifts fails rather than sits.
The demo figure is 11 h 53 min because that is what `extraction_wall_s` says in the record the log itself prints; an earlier version of this page labelled the same log a 6.4-hour build, a pre-rebuild figure `DECISIONS.md` D8 already records as superseded by D19.

## The community reports

The community reports are the layer `graphrag-global` ranks over, and they now carry a run record of their own.

|  | CI reports | demo reports |
|---|---|---|
| reports | 110 | 79 |
| generated by a model call | 75 | 79 |
| served from cache | 35 | 0 |
| model calls | 75 | 79 |
| reports truncated | 0 | 0 |
| Leiden resolutions | 0.3, 1.0, 3.0 | 1.0 |
| largest prompt / smallest margin | 903 / 3,193 tokens | 1,030 / 3,066 tokens |

Both runs were cold — the cache directory was emptied first — and both ran on the OpenAI-compatible wire against `llama3.1:8b`, which is the wire that cannot set the window and therefore reads the server's loaded window before every call.

**The CI gap between 110 reports and 75 calls is not a warm cache, and the record says which it is.**
The report cache is keyed by a community's *content* — the top 30 members by degree and the relations among them — and deliberately not by its resolution, so a community that is identical at two Leiden levels is generated once and read back from an entry the same run wrote minutes earlier.
110 reports over exactly 75 distinct member sets is what that produces.
The record separates `reports_generated` from `reports_from_cache` so `generated + from_cache == reports` and `calls == generated` are both checkable from the file alone, and a warm run publishes `calls: 0` rather than looking like a cold one.
The demo corpus runs at a single resolution, so it has no within-run hits at all: 79 reports, 79 calls.

**That cache sharing had a real defect, found by this regeneration and fixed before these records were published.**
Two of the 35 CI cache hits came back carrying a *different* community's `member_names` and `size` — 255 members served under a community of 250, and 111 under one of 102.
Sharing the prose is correct, because the prose was written from the shared head alone.
Sharing the identity is not: `graphrag-global` ranks over `member_names`, and `LIMITATIONS.md` counts residual contamination in it, so a lent membership put a wrong list under two published numbers.
`community_id`, `resolution`, `member_names` and `size` are now recomputed for the community a cached report is served *for*, and `tests/test_graph_summaries.py` builds the collision deliberately — two communities agreeing on their 30-member head and differing only in an isolated tail — so removing the fix turns the test red.
The demo corpus had 0 cache hits and was unaffected, so no published demo number moved.

## Retrieval ablation — demo corpus

Two query manuscripts, 36 relevant documents between them, k = 10.

| rung | found @10 | found @30 | recall@10 | ceiling | % of ceiling | NDCG@10 | mean latency |
|---|---|---|---|---|---|---|---|
| BM25 | 13 / 36 | 24 / 36 | 0.366 | 0.679 | 69% | 0.770 | 33.6 ms |
| vector | 13 / 36 | 26 / 36 | 0.366 | 0.679 | 69% | 0.770 | 0.6 ms |
| RRF hybrid | 13 / 36 | 25 / 36 | 0.366 | 0.679 | 69% | 0.770 | 36.4 ms |
| GraphRAG local | 10 / 36 | 23 / 36 | 0.312 | 0.679 | 54% | 0.641 | 162.2 ms |
| **GraphRAG global** | **14 / 36** | 25 / 36 | **0.473** | 0.679 | **76%** | **0.830** | 13.3 ms |

Every rung is scored over a document list of the **same length**.
That sounds obvious and an earlier version of this table did not do it: it retrieved a fixed 30 *chunks* per rung and then scored the *document* ranking those chunks happened to produce — 4 documents for RRF against 13 for GraphRAG local on the same query, both reported as "recall@10".
Chunk depth is now grown until every rung offers the same number of documents.
Correcting it moved real conclusions, which is the point of publishing the harness alongside the numbers.

"% of ceiling" is macro-averaged across cases like every other rate here; it was previously a ratio-of-means, which silently reweighted toward the case with the larger denominator and inverted which rung appeared to lead.

Two hit columns, both labelled, because they disagree and the disagreement is the point.
**found @10** counts relevant documents inside the top-10 list the reported rates are computed over.
**found @30** counts them inside a list three times as deep — where a rung with a long tail catches up.
An earlier version of this table published only the deeper count under a `k = 10` heading, which flattered exactly the rung whose ranking is weakest.

The vector rung's query embedding is served from the committed fixture, so its 0.6 ms is a lookup, not an encoding cost.
Latencies are means over the two cases, not medians.

**Read this honestly, including the parts that do not flatter the graph:**

- **Community-scoped graph search leads the aggregates; entity-scoped graph search is the worst rung on the page.** GraphRAG global has the highest found@10 (14 of 36), the highest recall@10 (0.473) and the highest NDCG (0.830). GraphRAG local has the lowest of all three (10 of 36, 0.312, 0.641) and the slowest mean latency (162.2 ms), and it has no aggregate win over any other rung.
- **The lexical, dense and hybrid rungs are indistinguishable on the reported rates.** BM25, vector and RRF hybrid return identical recall@10 and identical NDCG on both cases, because at k = 10 they surface the same relevant documents; they separate only at depth 30, where vector's longer tail reaches 26 of 36 against RRF's 25 and BM25's 24. Three rungs tying to four decimal places is a fact about a 36-document ground truth on a citation-seeded corpus, not evidence that the retrievers are equivalent.
- **The graph's lead comes from the layer with the most machinery behind it, and the least deterministic.** GraphRAG global ranks over community reports written by a local model, whose regeneration is only 78-of-79 reproducible and whose cache carried the identity defect described above until `c607ff4`. The rung leads; the input it leads on is the softest input in the system, and that is stated here rather than left for a reader to discover.
- **Speed: BM25 at 33.6 ms is the honest lexical baseline on this run, and GraphRAG local runs about 5x BM25's latency.** That multiplier is deliberately imprecise, and it is a reading of one machine under a load the record names (18.49) rather than a property of the two rungs. The same comparison on identical bytes has read 3.16, 1.87, 3.22, 3.49 and 4.83 across the five committed runs of this ladder, so the column carries the ordering and not a fine-grained ratio, and that spread is the reason the claim is written with a single significant figure. Vector's 0.6 ms is excluded from any cost claim: it is a fixture lookup with the encoder taken out.
- **No rung reaches the ceiling.** The best recall@10 on the page, 0.473, is 76% of the 0.679 that a perfect ranking could have achieved at this depth; the ceiling is below 1.0 because one case has more relevant documents than k.
- **n = 2, and the ordering reverses between the two cases — so this establishes behaviour, not a ranking.**

| rung | MET17: recall / NDCG | biopolymer: recall / NDCG |
|---|---|---|
| BM25 · vector · RRF | 0.375 / 0.539 | 0.357 / **1.000** |
| GraphRAG local | 0.375 / 0.486 | **0.250** / 0.797 |
| GraphRAG global | **0.625** / **0.746** | 0.321 / 0.915 |

GraphRAG global takes MET17 outright — 0.625 recall against 0.375 for every other rung — and then finishes below all three lexical-and-dense rungs on the biopolymer paper (0.321 against 0.357).
GraphRAG local is the weakest rung on the biopolymer case (0.250) and matches the pack's recall on MET17 at a lower NDCG (0.486 against 0.539).
At this n no rung ordering survives per-case inspection, and any sentence of the form "rung X is best" is an artifact of averaging two manuscripts that disagree.
This is the same discipline as the N ≥ 20 gate and the swap-consistency population below: a mean over two opposing cases does not earn three significant digits.

- **The corpus is citation-seeded** (see `DECISIONS.md` D6): it was built from these manuscripts' own reference lists, so it contains the answers by construction. These numbers compare rungs against each other. They are not an estimate of real-world retrieval difficulty, and there is no random-selection floor here to say how much of any rung's score is the corpus rather than the method.

## Retrieval ablation — CI corpus

`ablation-ci.json` runs the same five rungs over the 15-document CI corpus with no model at all, in seconds, from a clean clone.
It reports **no rates**: the CI corpus is topical rather than citation-seeded, so it contains 0 ground-truth documents for these queries, and the N ≥ 20 gate withholds recall and NDCG rather than printing a rate over nothing.
Two of the three query manuscripts are skipped there and the record says why — their published twins are not CI-corpus members, so those runs would have nothing to exclude and are not valid runs.
That file exists to prove the harness runs offline and reports honestly when it has nothing to measure, and it is the record the conformance test regenerates to prove this code produced it.
It carries no evidence about retrieval quality and none is drawn from it.

## The panel, on a real manuscript

`make review` on the `met17-auxotroph` preprint against the 68-document demo corpus, with that manuscript's own published twin withheld from the index and from the graph — 26 chunks dropped, recorded in the record.
226,585 tokens and 21 minutes over 46 model calls, with a largest prompt of 7,317 tokens, a smallest window of 4,096 tokens, and a smallest per-call margin of 3,695 tokens between a prompt and the window that was sized for it.

| | |
|---|---|
| reviewers | 2, blind and parallel, different retrieval scopes and model families |
| findings | 8 each — which is `MAX_FINDINGS`, the cap, not a number either reviewer chose |
| findings lost to unparseable output | 0: neither reviewer failed to parse |
| citations deleted by the hygiene pass | 0 on every reviewer of both runs |
| whole findings discarded as malformed | 0 on every reviewer of both runs |
| scores | methods 4/4/4 · novelty 4/4/**3** (they disagree on contribution) |
| claims verified | 20 verdicts, every one judged twice with the evidence order reversed |
| verdicts | 0 SUPPORTS · 0 REFUTES · **20 NOT_ENOUGH_INFO** |
| verdicts carrying an evidence span | 6 of 20 |
| conflicts detected | 0 |
| deterministic lens | 0 findings |

**Both reviewers stopped at exactly 8, and 8 is the cap.**
`MAX_FINDINGS = 8` in `src/peerpanel/agents/reviewer_base.py` is enforced twice — as `maxItems` in the schema the model is given, and as a hard break in the loop that reads the reply — so "16 findings in a run" is two reviewers at their ceiling, not two reviewers' natural output.
Nothing in the record distinguishes a reviewer that had eight things to say from one that was cut off at eight, and every per-finding rate below has that 16 as its denominator.
It is the same class of defect as the two hygiene counters above — a number a reader cannot tell apart from the filter that produced it — and here the answer is a sentence rather than a counter, because a cap that fires leaves nothing behind to count.

### The result worth reading: the verifier decides nothing

Every verdict in both committed runs is an abstention.
Not one claim was SUPPORTED or REFUTED.

That is not the swap rule forcing caution.
Of the 20 claims, **14 were swap-consistent** — the verifier returned `NOT_ENOUGH_INFO` in both evidence orderings, of its own accord — and the other 6 disagreed with themselves when the order was reversed and were forced to abstain.
So on this configuration the judge either cannot decide, or decides differently depending on what it read first.
No claim was ever decided the same way twice.

The clearest single case: the claim *"The MET17 gene catalyzes homocysteine synthesis by reacting H2S with O-acetyl homoserine"* was verified against a retrieved span from `PMC9757880` reading *"Met17p catalyzes the fixation of inorganic sulfide with O-acetylhomoserine (OAHS) to form homocysteine (Fig 1A)."* — the evidence states the claim, the span survived the substring check, and the verdict is still `NOT_ENOUGH_INFO`.

An earlier version of this page reported 9 SUPPORTS and 2 REFUTES from this same panel.
Those runs were measured while every reviewer and verifier prompt was being silently cut to a 4,096-token window (`DECISIONS.md` D19), so the decisive-looking verdicts came from a judge that had read part of its evidence and none of its instructions.
Reading the prompt whole made it strictly more conservative, and the honest reading of that is not an improvement: **a judge that cannot say SUPPORTS is not a judge**, and this one is currently an expensive way to produce abstentions.

### Swap consistency: 0.70, over a population where it means little

| run | swap-consistent | forced to abstain | rate | tokens | wall |
|---|---|---|---|---|---|
| 1 | 14 / 20 | 6 / 20 | swap-consistency 0.70 | 226,585 | 1,297.9 s |
| 2 | 14 / 20 | 6 / 20 | swap-consistency 0.70 | 226,585 | 1,296.7 s |

The rate is exact rather than a bound: each verdict records whether it was actually judged twice (`swapped`), and all 20 were, so the denominator is real.
An earlier version could only publish `[0.00, 0.50]` because the records did not distinguish "agreed" from "never tested".

It is also nearly meaningless as a measure of position bias here, and saying so is the point: consistency between two abstentions is not resistance to order effects, because there was no decision to lose.
What the number reports is that 6 of 20 claims — **three verdicts in ten** — changed the verifier's answer when the evidence order was reversed, on a population that otherwise refuses to commit.
An earlier version of this page glossed the same 6 of 20 at roughly double that rate, in its closing paragraph, in the sentence that told a reader what the panel was for.

### The two runs are identical, and that is the finding

Run 2 reproduces run 1 exactly: same 20 verdicts, same 16 findings, same converger prose, same 226,585 tokens — every field identical apart from wall-clock.
At temperature 0 this panel is deterministic, so a second run is a **reproduction, not an independent sample**.
The same 16 findings are also both reviewers at `MAX_FINDINGS`, so what reproduces is a capped list.

An earlier version of this page presented two runs as evidence of variability ("12 of 24, then 14 of 24").
Whatever produced that spread, it is not present in this configuration, and two identical runs cannot support a claim about variance.
What they do support is reproducibility, and that survived two rebuilds: the record regenerated on this commit matches the one published at `52238c8` in every field except wall-clock and the four fields the schema has gained since, and matches the one published at `edea64f` in every field except wall-clock and the two hygiene counters added at `192e804`.

### What the run got wrong: nothing it quoted

In each committed run, **0 of 16 reviewer findings quote text that is not in the manuscript.**
Every quote in every finding is real manuscript text.
The count is published per run because the second run reproduces the first: adding the two gives 32, which is one population of 16 counted twice rather than a sample twice the size.

An earlier version of this section reported three findings in sixteen describing a different paper — a reviewer attributing retrieved literature to the manuscript under review.
That reviewer was reading a cut prompt: the excerpt's head was gone and the rubric with it, leaving retrieved literature as the most recent thing in its context.
Read whole, the behaviour disappears entirely on this manuscript.
One manuscript and 16 findings in a run is not a licence to call the failure mode solved, and `tests/test_artifact_conformance.py` recomputes this count from the records on every run, so if it returns the number returns with it.

To reproduce: compare each finding's `quote` in `results/panel-review-met17-auxotroph.json` with `manuscripts/met17-auxotroph.txt` **after collapsing runs of whitespace in both**.
A literal substring check on that record reports five apparent misses (ten across both runs), all of them paragraph breaks where the chunker joined two lines with a space; the conformance test normalises, and this instruction now says so.

### The mechanism that had never fired, and the two that still have not

A verdict's citation survives only if the chunk was actually retrieved **and** the quote is a substring of that chunk's own text, so a fabricated citation cannot pass the type system.
Until the rebuild its yield across every committed run was exactly zero — 0 of 24 verdicts carried an evidence span — and this page said so, calling it an unfired safety check rather than a demonstrated capability.

It fires now: **6 of 20 verdicts carry a surviving evidence span in each of the two committed runs**.
The population is 20 and not 40: the second run reproduces the first, so adding them counts one set of verdicts twice.
The spans are real retrieved text that passed the substring check, and one of them is quoted above.

Two things keep that from being a success story.
The verdicts carrying those spans are all `NOT_ENOUGH_INFO`, so the grounding demonstrably works and the judge it feeds does not use it.
And the reviewers' own citations remain nearly absent: of 16 findings in a run, **1 cites a retrieved chunk at all**.
That number is now readable as reviewer behaviour rather than as the filter's, because the filter publishes its own yield.
A reviewer's citation to a chunk it was never given is deleted before the finding is recorded, and a finding with no rubric dimension or no text is discarded whole; both committed runs record **0 citations deleted and 0 findings discarded**, on every reviewer (`unretrieved_citations_dropped` and `malformed_findings_dropped` in each `reviewer_outputs` entry).
Until `192e804` neither drop was counted anywhere, so a reviewer that cited nothing could not be told apart from one whose citations this pass had removed — and the mitigation `LIMITATIONS.md` lists was the one safeguard on this page with no published yield, while lens findings, conflicts, unparsed calls and verdict grounding all carried theirs.
0 is the informative answer here: the citations are missing because the reviewers never wrote them.
The denominator is still the cap — 16 is `MAX_FINDINGS` twice over — so this is 1 in 16 capped findings rather than 1 in everything the reviewers had to say.
The mechanism is proven; what it is attached to is not.

Two mechanisms this architecture advertises have still never fired on any committed run, and neither is described anywhere on this page as a strength:

- **The deterministic lens: 0 findings**, on both panel runs and on the panel arm of both planted-error evaluations — it is a panel channel, and the single-agent arms have no equivalent. Its three checks are spreadsheet gene-symbol corruption, reference integrity and statistical impossibility. The two error kinds it could catch are the two the planting recorded as *not planted* — `impossible_pvalue` and `impossible_percentage`, both skipped for want of a candidate site inside the reviewed window — so the evaluation below cannot exercise it either, and this page no longer points at it as proof. Its only positive evidence anywhere is `tests/test_deterministic_lens.py`, against hand-written strings. A capability with no output on any committed run is not a measured strength, and it is listed here as a gap.
- **Conflict detection: 0 conflicts**, on both runs. Its evidence path needs a REFUTES verdict carrying a span, and there are 0 REFUTES. Its sentiment path needs a score gap of at least 2, and the one disagreement the reviewers had — novelty 4 against 3 — is a gap of 1. So the mechanism reported nothing on the one case a reader would expect it to catch, and the threshold that explains why is stated here rather than left in the code.

## The headline measurement: a null result

Three errors were planted into each of two manuscripts — a swapped gene symbol (`Dcr1`→`Dcr2`), a reversed effect direction, and a fabricated citation — inside the 900-word window every arm reads.
The two are not held out of the corpus in the same sense, and each record says which it is: `caprin-heterochromatin` has no published twin in this corpus at all (`twin_in_corpus: false`), while `met17-auxotroph` does, so its twin is withheld per run — `PMC10729969` and 26 chunks dropped on every arm.
Three systems reviewed each manuscript: the **panel** (a two-family mixture — methods and verifier on `llama3.1:8b`, novelty and converger on `qwen2:7b`), and one single-agent chain-of-thought-with-self-consistency arm per named local model, so each named model carries its own count rather than one number over a mixture.
Count, not rate: n = 3 planted errors on 2 manuscripts establishes behaviour, and this page says so wherever the number appears.

**Under the committed detection rule, no arm asserted any planted defect on either manuscript.**

| arm | caprin asserted | caprin named token | caprin tokens | caprin wall | met17 asserted | met17 named token | met17 tokens | met17 wall |
|---|---|---|---|---|---|---|---|---|
| **panel** (`llama3.1:8b` + `qwen2:7b`) | **0 / 3** | 1 / 3 | 240,852 | 1,449.8 s | **0 / 3** | 0 / 3 | 244,460 | 1,594.5 s |
| single agent · `llama3.1:8b` | **0 / 3** | 1 / 3 | 36,475 | 330.8 s | **0 / 3** | 0 / 3 | 31,914 | 338.1 s |
| single agent · `qwen2:7b` | **0 / 3** | 1 / 3 | 39,462 | 232.2 s | **0 / 3** | 0 / 3 | 29,758 | 135.0 s |

**asserted** is the headline, and it is published at 0.
**named token** is a labelled upper bound printed beside it, never in its place: it counts planted errors that some finding named while asserting nothing about them.
Since `192e804` that bound has been computed over the same stripped residue as the headline, and it can no longer be minted by quoting the perturbed sentence back — which, measured below, is what five of the eight credits these records would otherwise carry turned out to be.
Cost is tokens and wall-clock; no currency figure is published for this evaluation.
Every arm of every run recorded 0 unparsed calls, so no count here is depressed by output that failed to parse — that field exists because an earlier run of this comparison was measured with one of the panel's two reviewers silently dead, contributing no findings, with nothing in the record saying so.

On `caprin-heterochromatin` the panel spent 6.6x the tokens of the single-agent `llama3.1:8b` arm and 4.4x the wall-clock, to assert the same nothing.
Against `qwen2:7b` the same run is 6.1x the tokens and 6.2x the wall-clock.
On `met17-auxotroph` the gap against `llama3.1:8b` is 7.7x the tokens and 4.7x the wall-clock.
The same run against `qwen2:7b` is 8.2x the tokens and 11.8x the wall-clock.
The `llama3.1:8b` arm reached that result on 15% of the panel's tokens for the caprin run, and 13% of them on `met17-auxotroph`.
The `qwen2:7b` arm used 16% of the panel's tokens on `caprin-heterochromatin` and 12% on `met17-auxotroph`.
Neither single-agent arm was given an equal budget: both hit their 12-sample ceiling before spending what the panel spent, so this is not an equal-compute comparison, and the direction of that inequality favours the panel.
Each multiple is a ratio of two figures in the record beside this page, and `tests/test_planted_published_numbers.py` reads every one of them against the manuscript its own sentence names — a multiple true of one manuscript is false of the other, and until `192e804` the binding could not tell them apart.

### The rule, and what a 0 means under it

The published count is produced by `asserts` in `src/peerpanel/evals/planted.py`, rule id `assertion-v2`, written into every record.
A finding is credited only if one of its own sentences both names the planted detection token and carries an assertion cue — `incorrect`, `wrong`, `does not exist`, `should be`, `inconsistent`, `fabricated`, and about twenty more — that no negation in the cue's own clause stands in front of.

**Only the finding's own prose is scored — and since `192e804` that sentence is true of the records, where before it was only stated.**
The prose reaches the rule through two strippings.
First, the `quote` field is dropped: a quote is a verbatim slice of the manuscript and the planted token was planted *into* the manuscript, so scoring it credits an arm for reproducing the perturbed sentence.
Second, every sentence of what remains that is itself a verbatim slice of the perturbed manuscript is dropped as well — because nothing ever stopped a model putting manuscript text in the `text` field, and these models mostly do.
That second stripping was added at `192e804`, after round 4 filed twice what the first one on its own did not do: the code deleted the quote FIELD, not quotation, and the sentence a reader was asked to trust was not true of the records beside it.
The rule id moved from `assertion-v1` to `assertion-v2` for that reason and no other — the cue list, the negation scope and the same-sentence clause are byte-identical; what changed is what they are handed.
The residue each arm was actually scored on is committed as `scored_texts` beside the unstripped `finding_texts`, with `quoted_sentences_dropped` counting what the second stripping removed, so a reader can see both and measure the difference.
Stripping can only remove text before an existential rule sees it, so it can only lower a count and never raise one: a scoring change made after seeing the result that is arithmetically incapable of flattering the headline is the one shape of post-hoc change that cannot be fitted to it.

**The sentence test now tolerates the punctuation a quotation picks up, which is the repair this commit makes.**
A model that copies a mid-paragraph clause and ends it with a full stop writes a string that is not a substring of the manuscript, so until this commit that string survived the strip and was scored as prose the arm had written itself.
`own_prose` now takes a run of `.,;:!?"')]` off the end of a sentence, and a run of `"'([` off its front, before the substring test — off the sentence only, never off the manuscript and never anywhere inside the sentence (`_quotation_form` in `src/peerpanel/evals/planted.py`).
Nothing about it is fuzzy: a word added to a quotation still survives, because only those characters are removed.
Like the stripping it belongs to, it removes text before an existential rule sees it, so it can only lower a count.
The id stays `assertion-v2`, and the code says why where the constant is set: the repair corrects what v2 was already removing rather than changing what is scored or how it is judged.
Both evaluations were re-run on this commit so that every record agrees with the repaired rule, which means the numbers below carry two changes at once — a stricter strip and a fresh draw from a sampler — and no sentence here attributes a movement to one of them alone.

The rule errs in both directions and the record says which:

- it does not credit a finding that asserts the defect without naming the token, so the count is a floor on real detection;
- it does credit a finding that names the token and uses a cue about something else in the same sentence, so that floor is not a clean one;
- a negation earlier in the cue's clause suppresses the cue, so a real assertion phrased "Dcr2, not Dcr1, is incorrect" earns nothing — the rule under-credits rather than over-credits on purpose;
- it is same-sentence only, so an assertion split across two sentences earns nothing.

Cues that occur in these manuscripts as ordinary vocabulary were dropped before any control was run — `invert` (inverted microscope), `opposite`, bare `revers` (reverse transcribed), `no such` (no such enrichment was observed) — and `error` keeps an `error bars` exception.
The rule was frozen at that text before it scored anything, and it was never edited to make a control pass.
Two pre-run controls did defeat the mechanisms around the cue list, before any record existed: a clause-scoped negation replaced a fixed five-word lookback, and parenthesised spans are held whole so a citation containing `Nat. Metab.` is not cut into two sentences.
The cue list itself has never been touched, and the judgement the rule makes is the one it was frozen at; the id says `v2` because the input to that judgement changed, not the judgement.

### How much of what these arms wrote was quotation

The second stripping is a measurement as well as a fix, and this is what it measured on the six committed arms.

| arm | caprin: strings scored / written | caprin: own prose, in characters | met17: strings scored / written | met17: own prose, in characters |
|---|---|---|---|---|
| the two-family mixture | 10 / 16 | 70% | 8 / 16 | 62% |
| `llama3.1:8b` alone | 17 / 75 | 18% | 26 / 72 | 36% |
| `qwen2:7b` alone | 14 / 40 | 32% | 11 / 33 | 38% |

Of the 252 finding strings these six arms wrote, 166 leave no prose of their own once verbatim manuscript sentences are taken out — two strings in three — and 175 sentences were dropped in all, so **64% of the characters they wrote back were slices of the manuscript they were given**.
The two counts move independently because a finding string can hold more than one sentence: `quoted_sentences_dropped` counts sentences removed, and a string disappears from `scored_texts` only when every sentence in it was quotation.
The counts in that table are the lengths of `scored_texts` and `finding_texts` in the record beside this page, and the percentages are those same two arrays measured in characters, so every cell can be recounted without running anything.
On the two committed panel review runs the same behaviour is visible without any arithmetic at all: 8 of the 16 findings have a `text` field byte-identical to their own `quote`.

**This strengthens the null rather than softening it.**
The headline did not move: `asserted` is 0 under both the old input and the new one, on every arm of both manuscripts, so nothing was subtracted from detection.
What moved is the labelled bound beside it, including where a reader meets it first: the caprin column published 2 of 3 for both single-agent arms at `192e804` and reads 1 of 3 for every arm here.
Run the substring rule over the raw strings and it credits 8 of the 18 arm-and-error pairs; run it over the residue and it credits 3.
Three of the five credits it loses are on `met17-auxotroph` — the panel's single credit and both of the `llama3.1:8b` arm's — and two on `caprin-heterochromatin`, both of them the `qwen2:7b` arm's.
Every one of the five was the perturbed sentence itself, quoted back — including *"The MET18 gene, also known as MET15 or MET25 [13–15], catalyzes homocysteine synthesis by reacting H2S with O-acetyl homoserine (i.e. displaying OAH sulfhydrylase activity) [16–18]."*, which names the planted token only because the planting put it there.
The three that survive are all on `caprin-heterochromatin`, one per arm, and each one names its token in a sentence the arm wrote itself while asserting nothing about it — which is what the labelled bound is for.
So the honest summary of these systems on this task is not that they looked and missed: given a manuscript with three defects in it, they mostly handed the manuscript back.

### The 0 is not an instrument that cannot move

A detection rule that credits nothing is worthless if it *could* not credit anything, so the question is asked of the strings these runs actually produced.

| arm | caprin: strings / carrying an assertion cue | met17: strings / carrying an assertion cue |
|---|---|---|
| the two-family mixture | 16 / 0 | 16 / 0 |
| `llama3.1:8b` alone | 75 / 0 | 72 / 0 |
| `qwen2:7b` alone | 40 / 0 | 33 / 0 |

On these records the answer is stark: **not one of the 252 strings carries an assertion cue at all**, before or after the stripping.
These arms did not write "incorrect", "wrong", "should be", "inconsistent" or any of the twenty-odd others — about the planted defects or about anything else.
That is a fact about the arms, and on its own it cannot tell a silent system apart from a deaf instrument, so the instrument is evidenced separately.

Two things evidence it, and both can be run.
The rule's positive and boundary controls in `tests/test_planted_soundness.py` are built from real manuscript sentences — an assertion appended to a quotation scores 1, the quotation alone scores 0, a negated assertion scores 0 — so a rule that stopped firing would turn them red.
And the previous published run of this same protocol did produce cue-bearing prose: at commit `edea64f` the `qwen2:7b` arm wrote two real assertions about `caprin-heterochromatin` — one about "inconsistencies in the numbering of references", one about "inconsistent or incorrect formatting for citations" — neither about a planted defect, and so neither credited.
`git show edea64f:results/planted-eval-caprin-heterochromatin.json` still holds them.
That the two runs differ here is the sampler, not the rule: the single-agent arm draws 11 of its 12 samples at temperature 0.7, so its prose is redrawn every run while the panel's is not.
Reproduce either reading by running `ASSERTION_CUES` from `src/peerpanel/evals/planted.py` over the `finding_texts` and `scored_texts` arrays of the records.

### The 230 strings the old rule credited

The rule that produced the previous headline was a substring test over the finding's prose **plus the manuscript text it quoted**.
Three independent reviewers of commit `52238c8` opened the records and reached the same conclusion: every credited catch was a verbatim quotation of the perturbed sentence, and the number measured quotation rather than detection.

So the new rule ships with a negative control over that whole population.
All **230** strings the previous evaluation scored are committed at `tests/fixtures/round3_scored_strings.json`, extracted from the records at commit `52238c8`, each labelled with the errors the old rule credited it for.
Run against every error each was scored against — 690 decisions — `asserts` credits **0** of them, and `detect` credits exactly the **14** rows the fixture marks as credited, no more and no fewer.
That control is a floor rather than evidence, and `tests/test_planted_soundness.py` says so: none of the 230 contains a cue at all, so the same-sentence and negation clauses are exercised by separate boundary controls built from real manuscript sentences — a sentence containing a planted token beside `error bars represent one S.D.` must score 0, a real record string with a genuine assertion appended must score 1, and a negated one must score 0.

### The asymmetries, complete, with a direction for each

This comparison is not a controlled experiment, and an early version of this page listed three of its asymmetries while a tracked file claimed the list was complete.
The set below is the code's and the records', and it has been caught incomplete twice now: round 3 caught the claim, and round 4 caught row 16, the residue the exclusion leaves in the layer the panel's second reviewer ranks on.
A row was added rather than the claim softened, and row 13 now says at which levels the exclusion is symmetric instead of saying it is.

| # | asymmetry | panel | single-agent arms | favours |
|---|---|---|---|---|
| 1 | task instruction | a structured pre-submission review scored on soundness, presentation and contribution | "reviewing a manuscript excerpt for errors and weaknesses… quote the exact problematic text" | **the baselines** — one arm is pointed at the task being measured |
| 2 | token budget | 240,852 / 244,460 | 36,475 and 39,462 / 31,914 and 29,758 | **the panel** |
| 3 | wall-clock | 1,449.8 s / 1,594.5 s | 330.8 and 232.2 s / 338.1 and 135.0 s | **the panel** — and its reviewers run in parallel while the baseline loop is serial, so the published multiple understates the gap |
| 4 | model calls | 47 / 46 | 12 on every arm of both manuscripts — the sampler's ceiling, and no sample needed the single retry `call_json` allows | **the panel** |
| 5 | retrieval breadth | 2 reviewers × 3 queries × 3 chunks, plus per-claim retrieval on all 20 claims | one hybrid lookup on the title, 3 chunks | **the panel** |
| 6 | scored surface, as it is now scored: after quotation is stripped | 10 strings, 3,234 chars / 8 strings, 2,063 chars | 11–26 strings, 1,544–3,424 chars | **not established** — 0.69 to 1.66 times the panel's own prose, above it on one arm and below it on the other three; credit is still an OR over strings, which favours whichever arm writes more of them |
| 7 | output cap | `MAX_FINDINGS` 8 × 2 reviewers = 16 | 10 findings × 12 samples, deduped | **the baselines** |
| 8 | aggregation | one deterministic pass per reviewer at temperature 0 | 12 samples, the first at temperature 0 and the rest at 0.7, findings unioned | **the baselines** — union is the recall-maximising aggregation, and the panel was not given an equivalent |
| 9 | channels scored | three: reviewer findings, REFUTES verdicts, deterministic-lens findings | one: model-authored findings | **the panel** nominally, **the baselines** in practice — two of the panel's three channels produced nothing on any committed run |
| 10 | window into the manuscript | the deterministic lens reads the whole body; every other channel reads the 900-word excerpt | the 900-word excerpt | **the panel** nominally; nil in practice, since the lens returned 0 findings |
| 11 | retrieval rung | graph-local and graph-global | RRF hybrid | deliberate, and neither arm's rung produced a credited finding |
| 12 | detection rule | `asserts`, then `detect` as the labelled bound, both over the stripped residue | the same two functions, the same stripping, same thresholds | **symmetric in mechanism**; its false-positive rate scales with scored surface, so under the old rule — which scored quotation, and the baselines quoted more of it — it favoured the baselines in effect |
| 13 | exclusion, at the chunk and graph level | the same twin table, recorded per arm; `met17-auxotroph` drops `PMC10729969` and 26 chunks on every arm | the same | **symmetric at the chunk and graph level**, and enforced by a test |
| 14 | lost calls | 0 unparsed calls | 0 unparsed calls | **symmetric** |
| 15 | prompt and quote budgets | 1,024 review tokens, 240-character quote cap, one retry | the same | **symmetric** |
| 16 | exclusion residue in the ranked layer | the `prior-work-novelty` reviewer retrieves with `graphrag-global`, which ranks over community reports keyed to the FULL-corpus partition (`orchestration/panel.py` loads them through `load_artifacts` and never rebuilds them per run); on `met17-auxotroph` 101 entities that exist only because of the withheld twin appear in the `member_names` that rung tokenises, and `LIMITATIONS.md`'s own recipe prints `101 101 5`. The methods reviewer and the claim verifier read the rebuilt per-run graph and carry none of it | one RRF lookup over the chunk index the twin's 26 chunks were dropped from: no residue | **the panel** — on the one manuscript whose twin is a corpus member, one panel channel's ranking is influenced by the withheld document and no baseline channel is |

**Does the conclusion survive the set?**
The result to survive is now a null, not a ranking, and that changes what the asymmetries can do to it.

- **The null itself survives, and is the most robust thing on this page.** Every asymmetry that favours the baselines (1, 7, 8, and 9 in practice) is a reason a baseline might have asserted *more*, and every one that favours the panel (2, 3, 4, 5, and 16 on `met17-auxotroph`) is a reason the panel might have. All three arms asserted nothing, on both manuscripts. No asymmetry in this table turns a non-zero into a 0.
- **Row 16 pushes the panel's way and the panel still asserted nothing.** On the one manuscript where the exclusion leaves residue, the arm carrying that residue — reports built over a corpus that includes the document stating every planted fact correctly — asserted 0 of 3, and its labelled naming bound there is 0 as well. A contaminated arm that finds nothing is a stronger null than a clean one, not a weaker one.
- **The cost half survives.** The panel spent between 6.1 and 8.2 times the tokens of a single agent on identical inputs, metered through one ledger type on both arms, and asserted the same nothing. Asymmetries 2, 3, 4 and 5 all run the panel's way, so its spend is if anything understated relative to a fairer design.
- **The previous conclusion — that a single agent found twice what the panel found — does not survive, and is withdrawn.** It rested on a rule that credited quotation, over a scored surface several times larger for the arm that appeared to lead — most of that surface being, as row 6 now records, quotation itself — under a union aggregation the panel was not given. A one-error margin cannot carry two undisclosed confounds that both push the same way. The three-arm table above is what the records support instead.
- **What is not settled by this measurement.** Whether a panel would assert more than a single agent under matched instructions, matched aggregation and matched scored surface is untested here; asymmetry 1 alone would justify running it. n = 3 errors on 2 manuscripts establishes behaviour, not a rate, and two of the five error kinds could not be planted inside the reviewed window at all and are recorded as skipped rather than quietly shrinking the denominator.

### What the records say the panel is for

This repository was built to demonstrate a multi-agent review panel, and the measurement it ran on itself returns a null for the thing the panel exists to do.
The literature predicts it: architectures of this kind "often fail to outperform simple single-agent baselines such as Chain-of-Thought and Self-Consistency, even when consuming significantly more inference-time computation" ([arXiv:2502.08788](https://arxiv.org/abs/2502.08788)).

Read only against the committed records, what the panel demonstrably has is narrower than the architecture promises, and each item is a number on this page rather than a description:

- order-swapped claim verification whose disagreements are visible — 6 of 20 verdicts change with the evidence order, and the rate is published rather than smoothed;
- evidence grounding enforced in code, which now fires on 6 of 20 verdicts in each committed run where every run committed before the prompt-window fix carried none;
- reproducibility at temperature 0: two runs and two rebuilds that differ only in wall-clock and in fields the schema gained afterwards.

What it does not have, on this evidence, is a verdict it will commit to (0 SUPPORTS and 0 REFUTES in either run), a deterministic lens with any output at all (0 findings, everywhere), a conflict detector that has ever fired, or an advantage over one well-prompted local model at finding planted defects.
The honest summary is that assembling these parts into a panel did not, here, make it better at the task it was assembled for — and that a system's architecture has to earn its cost against the simplest thing that could work, every time, in public.
