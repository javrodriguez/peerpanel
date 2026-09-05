# PeerPanel

**A scientific-review system built so that its own measurements can be checked.**
Every number on this page is read out of a committed record under [results/](results/) by a test in this repository — including the numbers that look bad.

This repository is a **demonstration system**. It plants known defects into two manuscripts, runs a review panel and single agents over them under one
protocol, and publishes what each arm asserted, what it cost and under what run conditions — so a reader can check the headline, not take it.
Only one of the two is held out: `caprin-heterochromatin` is absent from the retrieval corpus, while `met17-auxotroph`'s published twin *is* a corpus
member, withheld from every arm at run time (26 chunks dropped on each) — a weaker guarantee, whose residue [LIMITATIONS.md](LIMITATIONS.md) measures.

## The headline: three defects planted per manuscript, and no arm asserted one

A finding is credited only when one sentence of its **own prose** names the planted token **and asserts that something is wrong** — rule `assertion-v2`,
in [src/peerpanel/evals/planted.py](src/peerpanel/evals/planted.py). Manuscript text an arm quotes back is stripped before scoring, out of the `text`
field as readily as the `quote` field: counting quotation as detection was the old flaw.

| arm | caprin asserted | caprin named token | caprin tokens | caprin wall | met17 asserted | met17 named token | met17 tokens | met17 wall |
|---|---|---|---|---|---|---|---|---|
| **panel** (llama3.1:8b + qwen2:7b) | **0 / 3** | 1 / 3 | 240,852 | 1,383.5 s | **0 / 3** | 0 / 3 | 244,460 | 1,438.4 s |
| **single agent** llama3.1:8b | **0 / 3** | 2 / 3 | 36,577 | 330.5 s | **0 / 3** | 0 / 3 | 32,395 | 358.6 s |
| **single agent** qwen2:7b | **0 / 3** | 2 / 3 | 41,747 | 234.9 s | **0 / 3** | 0 / 3 | 29,737 | 122.5 s |

**Asserted is the headline, and it reads 0 for every arm on both manuscripts: a null result.** The `named token` column beside it is the older
substring rule, kept as a labelled upper bound only — the gap between the two columns is restatement, not detection.

That upper bound used to be higher, and it fell because quotation stopped counting rather than because any arm got worse. Sentences an arm copies from
the manuscript are now dropped before either rule sees them: the panel's caprin arm was scored on 10 of the 16 strings it wrote, and every record
publishes the residue it was scored on (`scored_texts`) beside the count of quoted sentences removed (`quoted_sentences_dropped`). Stripping only ever
removes text before the rule reads it, so it can only lower a count — which is how a scoring change made after the fact can be told apart from a tune.

The panel spent 6.6x the tokens of the llama3.1:8b single agent on caprin and 7.5x the tokens of that arm on met17.
Against qwen2:7b it spent 5.8x the tokens on caprin and 8.2x the tokens on met17 — for the same asserted count as the cheapest arm: none.

Every committed model-run record names the model and wire that served it, how many calls it made, the margin between each prompt and the window that
call was given, and where that window figure was read from — so a reader can tell from the record alone that no prompt was silently truncated.

This repository is also judged from outside, by evaluators handed a clean clone of one commit and a pinned prompt and nothing else:
4 blind rounds under a pinned evaluator prompt (sha256 `6e1bcda6…`) have filed 89 findings against it; the reports are this project's private record.
None of the four has come back clean; the bar is all three reports of a round ending in the literal token CLEAN, and the loop's cap is 8 rounds.

Where this evaluation design does and does not line up with the seven-step credibility framework in FDA's draft guidance on AI in regulatory decision-making:
**[CREDIBILITY.md](CREDIBILITY.md)** — a mapping onto that vocabulary, never a claim about this system's regulatory standing.

## Try it in 60 seconds, with no model and no setup

```bash
git clone https://github.com/javrodriguez/peerpanel && cd peerpanel
uv sync --dev
make quickstart     # ~3s: rebuilds the graph from committed extractions, detects communities, runs retrieval
make ablation       # ~3s: the retrieval ladder, from committed bytes
make test           # ~1 min: the deterministic suite — every binding on this page is checked here
```

None of those touch a model, the network or an API key. They rebuild the knowledge graph from committed extraction records, run Leiden, and execute real
retrieval — because every expensive artifact this project produced is committed, not just described.

Full tables and the honest reading of every measurement, retrieval included: **[results/RESULTS.md](results/RESULTS.md)**.

## What runs, and in what order

```mermaid
flowchart TB
    M[Manuscript] --> ORCH[Orchestrator]
    C[(Open-access corpus<br/>68 papers, demo scale)] --> IDX[Index<br/>self-exclusion applied]
    IDX --> KG[Knowledge graph<br/>7,697 entities · 63,130 edges]
    KG --> LEI[Leiden communities<br/>3 resolutions]
    LEI --> REP[Community reports]

    ORCH -->|local: entity scope<br/>llama3.1| R1[Methods & statistics]
    ORCH -->|global: community scope<br/>qwen2| R2[Prior work & novelty]
    ORCH -->|no LLM| R3[Deterministic lens]
    IDX --> R1
    REP --> R2
    M --> R3

    R1 --> V[Claim verifier<br/>order-swapped, abstains on disagreement]
    R2 --> V
    M --> V
    V --> CONV[Converger<br/>no new claims]
    R3 --> CONV
    CONV --> OUT[Structured review + meta-review]
```

The second tier reproduces the model runs and needs [Ollama](https://ollama.com) with `llama3.1:8b`, `qwen2:7b` and `nomic-embed-text`:

```bash
make corpus         # fetch the 68-document demo corpus (license-gated, md5-verified)
make demo-index     # rebuild the index for real (~12h on an M-series laptop)
make review         # run the panel on a real manuscript
make eval           # planted errors: the panel and one single agent per model, same protocol
```

## How it is built

**GraphRAG, from scratch.** Chunk → LLM entity/relation extraction → merged knowledge graph →
multi-level Leiden communities → LLM community reports → local (entity-anchored) *and* global
(community-anchored) search. This is an independent implementation of the architecture class
Microsoft's GraphRAG describes; it is not affiliated with that project and does not use its code
(which cannot install on Python 3.14 — see [DECISIONS.md](DECISIONS.md) D2).

**Two scales, both stated as measured.** A 15-paper pinned corpus the deterministic suite exercises on every run, and the 68-paper open-access corpus
the captured runs use: 1,077 chunks, 7,697 entities, 63,130 edges, no extraction prompt truncated
([results/build-stats-demo.json](results/build-stats-demo.json)).

**Reviewers differ structurally, not by persona.** The methods reviewer searches entity
neighbourhoods; the novelty reviewer searches community summaries; they run on different model
families and never see each other's output. Prompt-level "you are a rigorous reviewer" personas
[measurably do not improve performance](https://arxiv.org/abs/2311.10054), so the differentiation
is in retrieval scope, rubric and model — things that can be inspected.

**Position bias is mitigated and measured.** Every claim is judged twice with the evidence order
reversed. Agreement stands; disagreement forces `NOT_ENOUGH_INFO`. The rate ships with its n, and is
withheld when n is too small to mean anything.

**A manuscript can never retrieve itself.** Several of these papers exist in the corpus in
published form. When one is reviewed, its twin's chunks leave the index and the knowledge graph is
*rebuilt* with that document's extractions withheld — removing not just its entities but the edge
weight it contributed between surviving ones. The mechanism is defended by tests that drive the
real orchestration and fail when it is disabled.

## What the panel measurably does, and does not do

**The verifier decides nothing.** Across both committed panel runs **every verdict is
`NOT_ENOUGH_INFO`** — 0 SUPPORTS, 0 REFUTES, 20 abstentions per run — including claims whose
retrieved evidence states them almost verbatim. The measured swap-consistency 0.70 says the rest:
three verdicts in ten change when the evidence order is reversed, which is position bias doing
exactly what the literature says it does. A judge that cannot say SUPPORTS is not a judge, and this
one is currently an expensive way to produce abstentions.

**Evidence grounding is enforced in code, and it now fires.** A verdict's citation survives only
if the chunk was actually retrieved *and* the quote is a substring of that chunk, so a fabricated
citation cannot pass the type system. Its yield was zero across every run this project had shipped
until 2 September 2026; on the rebuilt index it is **12 of 40 verdicts**, 6 in each run. What that
buys is smaller than it sounds: every verdict carrying a span is still an abstention, so the
grounding works and the judge it feeds ignores it, and only 1 reviewer finding in 16 cites a
retrieved chunk at all. That 16 is a ceiling, not an output: each reviewer returned exactly 8
findings, which is `MAX_FINDINGS`, the hard cap in `src/peerpanel/agents/reviewer_base.py`, and both
runs record `malformed_findings_dropped: 0` and `unretrieved_citations_dropped: 0`, so the cap is
what set the denominator and how much more either reviewer would have written is unmeasured. The
same ceiling sets the 32 below.

**The deterministic lens has no positive evidence at all.** It finds **0 findings on every committed
run**, and fires only on the constructed cases in `tests/test_deterministic_lens.py`. It is a
mechanism this repository ships and has never seen catch anything — not a demonstrated strength.

**One earlier self-report was itself wrong, and this is the correction.** A previous version of this
page reported that three of sixteen reviewer findings quoted text absent from the manuscript. Read
whole, no reviewer finding in either committed run quotes text that is not in the manuscript (0 of
32 — two runs, two reviewers, eight findings each, the cap again), and the swap rate is exact rather
than a bound because each verdict now records whether it was really judged twice.

The null result above is the same shape the literature reports for architectures that "often fail to
outperform simple single-agent baselines… even when consuming significantly more inference-time
computation" ([arXiv:2502.08788](https://arxiv.org/abs/2502.08788)) — reproduced here against the
system this repository exists to demonstrate, with the comparison's own asymmetries listed in
[results/RESULTS.md](results/RESULTS.md) rather than argued away. The one that matters most: the
baseline was asked to hunt errors and the panel was asked to review, and each single agent stopped
at its sampling ceiling well below the panel's spend, so this is not an equal-compute comparison.

## What is real, and what is recorded

Nothing here is mocked. The distinction that matters:

- **Runs for real, everywhere, every time:** chunking, sanitation, graph construction, Leiden,
  every retrieval rung, the eval harness, the deterministic lens.
- **Recorded from real runs and committed:** per-chunk extractions, community reports, embeddings.
  These are actual `llama3.1:8b` and `nomic-embed-text` outputs, cached by content hash, each with
  the command that regenerates it.
- **Proven by captured runs:** the index builds, both panel reviews and both three-arm evaluations,
  whose raw logs and structured records are committed under `results/` beside the numbers they produced.

CI has no GPU and calls no model, so it checks those recordings against the code rather than against
a fresh model. Concretely, it fails when:

- the knowledge graph rebuilt from the committed extractions no longer has the entity, edge and
  community counts in `results/build-stats-*.json`;
- the retrieval ladder no longer regenerates its record byte-for-byte apart from the fields the code
  itself declares run-varying (`latency_ms`, `run_utc`);
- a published detection count is not what the committed rule produces when re-run over the committed
  finding texts, or a record under `results/` cannot state its own run conditions;
- a number in this README, in `results/RESULTS.md` or in `LIMITATIONS.md` stops matching the record
  it is read from.

What no replay can catch: whether the same model would write the same text again. A cached
extraction proves what the model said the day it was called, not what it would say tomorrow — the
model layers reproduce the protocol, not the bytes, and `results/RESULTS.md` puts a measured figure
on the difference.

The one adapter that has never run — Anthropic's — says so in its own docstring and is tested for
request shape only. No API key exists on the development machine and none was requested.

## Read this next

- **[results/RESULTS.md](results/RESULTS.md)** — every measurement, with its honest reading.
- **[CREDIBILITY.md](CREDIBILITY.md)** — the evaluation design mapped onto FDA's seven-step
  credibility framework, including the steps this repository does not evidence.
- **[LIMITATIONS.md](LIMITATIONS.md)** — what this does not do, what the numbers do not prove, and
  eight known failure modes of LLM review with what is done about each, including the ones where
  the honest answer is "partly" or "not addressed".
- **[DECISIONS.md](DECISIONS.md)** — the design calls and the measurements behind them.

## Scope

A pre-submission self-check on already-public manuscripts, by their authors. **Not a substitute
for peer review**, and not for manuscripts under confidential review — publishers and funders
prohibit uploading those to language models.

## License

MIT for the code. Corpus and manuscript documents keep their own Creative Commons licenses, held
by their authors and recorded per-document in [CITATIONS.md](CITATIONS.md).
