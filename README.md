# PeerPanel

**A multi-agent scientific-review system with GraphRAG retrieval, measured against its own
baselines — including where it loses.**

Independent reviewer agents with structurally different lenses read a manuscript against a pinned
open-access corpus, an adversarial verifier checks every claim against retrieved evidence, a
deterministic non-LLM lens catches what language models miss, and a converger writes the
meta-review over the structured record. Everything runs locally on small open models at zero cost.

```mermaid
flowchart TB
    M[Manuscript] --> ORCH[Orchestrator]
    C[(Pinned corpus<br/>68 open-access papers)] --> IDX[Index<br/>self-exclusion applied]
    IDX --> KG[Knowledge graph<br/>6,722 entities · 55,146 edges]
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

## Try it in 60 seconds, with no model and no setup

```bash
git clone https://github.com/javrodriguez/peerpanel && cd peerpanel
uv sync --dev
make quickstart     # ~4s: rebuilds the graph, detects communities, runs retrieval
make test           # ~20s: the full deterministic suite
make ablation       # ~3s: the retrieval ladder, from committed bytes
```

None of those touch a model or the network. They rebuild the knowledge graph from committed
extraction records, run Leiden, and execute real retrieval — because every expensive artifact this
project produced is committed, not just described.

The second tier needs [Ollama](https://ollama.com) with `llama3.1:8b`, `qwen2:7b` and
`nomic-embed-text`:

```bash
make corpus         # fetch the 68-document demo corpus (license-gated, md5-verified)
make demo-index     # rebuild the index for real (~6.4h on an M-series laptop)
make review         # run the panel on a real manuscript
make eval           # planted errors: panel vs equal-compute single agent
```

## What the measurements say

Full tables and the honest reading: **[results/RESULTS.md](results/RESULTS.md)**. The headline,
over 68 documents and two query manuscripts with 36 relevant documents between them:

| rung | relevant found | recall@10 (of 0.679 ceiling) | NDCG@10 | latency |
|---|---|---|---|---|
| BM25 | 18 / 36 | 54% | **0.770** | 26 ms |
| vector | 17 / 36 | 54% | **0.770** | — |
| RRF hybrid | 17 / 36 | 54% | **0.770** | 24 ms |
| GraphRAG local | **18 / 36** | **58%** | 0.713 | 437 ms |
| GraphRAG global | 10 / 36 | 46% | 0.586 | 6 ms |

**GraphRAG global is the worst rung here on every measure, and BM25 wins on ranking at a fraction
of the cost.** GraphRAG local reaches documents lexical matching misses — the best recall of any
rung — then orders them worse. Those are the results; they are published rather than omitted
because a retrieval layer that cannot beat BM25 on a corpus like this has not earned its
complexity, and knowing *which* part earned it is the useful finding.

Recall is always shown against its achievable ceiling: with 28 relevant documents, no top-10 list
can exceed 0.357 recall, so a bare number would misrepresent a good result as a poor one.

## How it is built

**GraphRAG, from scratch.** Chunk → LLM entity/relation extraction → merged knowledge graph →
multi-level Leiden communities → LLM community reports → local (entity-anchored) *and* global
(community-anchored) search. This is an independent implementation of the architecture class
Microsoft's GraphRAG describes; it is not affiliated with that project and does not use its code
(which cannot install on Python 3.14 — see [DECISIONS.md](DECISIONS.md) D2).

**Reviewers differ structurally, not by persona.** The methods reviewer searches entity
neighbourhoods; the novelty reviewer searches community summaries; they run on different model
families and never see each other's output. Prompt-level "you are a rigorous reviewer" personas
[measurably do not improve performance](https://arxiv.org/abs/2311.10054), so the differentiation
is in retrieval scope, rubric and model — things that can be inspected.

**Position bias is mitigated and measured.** Every claim is judged twice with the evidence order
reversed. Agreement stands; disagreement forces `NOT_ENOUGH_INFO`. The swap-consistency rate ships
with its n, and is withheld when n is too small to mean anything.

**Evidence grounding is enforced in code.** A verdict's evidence survives only if the chunk was
actually retrieved *and* the quote is a substring of that chunk. Fabricated citations cannot pass
the type system, let alone the reviewer.

**A manuscript can never retrieve itself.** Several of these papers exist in the corpus in
published form. When one is reviewed, its twin's chunks leave the index and the knowledge graph is
*rebuilt* with that document's extractions withheld — removing not just its entities but the edge
weight it contributed between surviving ones. The mechanism is defended by tests that drive the
real orchestration and fail when it is disabled.

## What is real, and what is recorded

Nothing here is mocked. The distinction that matters:

- **Runs for real, everywhere, every time:** chunking, sanitation, graph construction, Leiden,
  every retrieval rung, the eval harness, the deterministic lens.
- **Recorded from real runs and committed:** per-chunk extractions, community reports, embeddings.
  These are actual `llama3.1:8b` and `nomic-embed-text` outputs, cached by content hash, each with
  the command that regenerates it. CI replays them — it has no GPU — and any drift fails loudly.
- **Proven by captured runs:** the panel and the planted-error evaluation, whose raw logs are
  committed under `results/`.

The one adapter that has never run — Anthropic's — says so in its own docstring and is tested for
request shape only. No API key exists on the development machine and none was requested.

## Read this next

- **[results/RESULTS.md](results/RESULTS.md)** — every measurement, with its honest reading.
- **[LIMITATIONS.md](LIMITATIONS.md)** — what this does not do, what the numbers do not prove, and
  nine known failure modes of LLM review with what is done about each.
- **[DECISIONS.md](DECISIONS.md)** — the design calls and the measurements behind them.

## Scope

A pre-submission self-check on already-public manuscripts, by their authors. **Not a substitute
for peer review**, and not for manuscripts under confidential review — publishers and funders
prohibit uploading those to language models.

## License

MIT for the code. Corpus and manuscript documents keep their own Creative Commons licenses, held
by their authors and recorded per-document in [CITATIONS.md](CITATIONS.md).
