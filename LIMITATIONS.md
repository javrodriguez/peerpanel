# Limitations

PeerPanel is a demonstration system. This page is the part of the repo most worth reading
carefully: it says what the design does not do, what the numbers do not prove, and which known
failure modes of multi-agent LLM review it mitigates, measures, or simply carries.

## Scope

**What this is for:** a pre-submission self-check on manuscripts that are already public, run
against a pinned open-access corpus, by their own authors.

**What it is not for:** it is not a substitute for peer review, and it must not be used on
manuscripts under confidential review — publishers and funders prohibit uploading such
manuscripts to language models, and this system would do exactly that.

## Structural limitations of this implementation

- **The manuscript is reviewed as an excerpt.** Reviewer prompts are budgeted to a 4096-token
  serving slot, so each reviewer sees roughly the first 900 words plus retrieved literature
  context — not the full paper, and never the figures or tables. Reviews of a paper's later
  sections are therefore out of reach by construction. (The reviewer that reads figures does not
  exist here; the best-known automated reviewer cannot read them either, and its evaluators
  called that out.)
- **Small local models.** Every model call runs against `llama3.1:8b` and `qwen2:7b` on one
  laptop. That keeps the whole system reproducible at zero cost, and it caps quality well below
  what a frontier model would produce. Numbers here are numbers for *this* configuration.
- **Two reviewers, not a large panel.** Panel diversity is largely illusory in the literature —
  nine LLM judges have been measured to behave like roughly two independent votes under
  correlated error ([arXiv:2605.29800](https://arxiv.org/pdf/2605.29800)) — so this system runs
  two structurally different reviewers rather than performing breadth it cannot deliver.
- **The converger shares a model family with one reviewer.** Both the novelty reviewer and the
  converger run on `qwen2`. That is a self-preference risk (see below) and it is a deliberate
  trade for keeping everything on two locally-available families; a third family would be the
  clean fix.
- **Propositional conflict detection is not implemented.** Conflicts are detected
  deterministically for score divergence (*sentiment*) and for reviewer claims the evidence
  refutes (*evidence*). Detecting that two reviewers assert logically incompatible propositions
  needs entailment between free-text findings, which this version does not do.
- **The corpus is citation-seeded.** The demo corpus is built from the query manuscripts' own
  cited references (plus one-hop neighbours and a topical top-up), which is what makes
  ground truth exist at all — and it means retrieval numbers are measured on a corpus that was
  constructed to contain the answers. Read them as a comparison *between rungs*, never as an
  estimate of real-world retrieval difficulty.
- **A stale corpus, deliberately.** The corpus is a pinned snapshot with per-document md5s, not a
  live search. The demo therefore re-runs identically next year and cannot see anything published
  after the snapshot. That trade is the point: reproducible verdicts over current ones.

## Known failure modes of LLM review, and what this system does about each

| Failure mode | Evidence | What PeerPanel does |
|---|---|---|
| **Position bias** | Across 36 models and 193 pairs, the first-shown option is picked 64.3% of the time, and the median model flips on 41.3% of decisive swapped-order cases ([benchmark](https://github.com/lechmazur/position_bias)) | **Mitigated and measured.** Every claim is judged twice with the evidence order reversed; disagreement forces `NOT_ENOUGH_INFO`. The swap-consistency rate is reported with its n, or withheld when n < 10. |
| **Fabricated citations** | 19.9% of GPT-4o citations across six simulated literature reviews were untraceable ([JMIR Mental Health](https://mental.jmir.org/2025/1/e80371)); retrieval reduces but does not eliminate this ([arXiv:2409.13740](https://arxiv.org/html/2409.13740v1)) | **Mitigated in code, not by instruction.** A verdict's evidence spans survive only if the chunk id was actually retrieved and the quote is a substring of that chunk's own text. Reviewer findings citing unprovided chunks have those citations dropped. |
| **Prompt injection** | Hidden instructions in a manuscript reach up to 100% acceptance across 1,000 generated reviews ([arXiv:2509.10248](https://arxiv.org/abs/2509.10248)) | **Mitigated.** Every corpus document and manuscript passes a deterministic screen before chunking: invisible and control characters are stripped, instruction-shaped lines are neutralised (prefixed and quoted as data, never silently deleted), and every action is reported. |
| **Multi-agent may not beat a single agent** | Debate "often fails to outperform simple single-agent baselines… even when consuming significantly more inference-time computation" ([arXiv:2502.08788](https://arxiv.org/abs/2502.08788)) | **Measured, not assumed.** The planted-error evaluation reports the panel beside an equal-compute single-agent baseline driven through the same token ledger. Whatever the delta is, it ships. |
| **Self-preference** | Judges favour their own generations, and the bias tracks self-recognition causally ([Panickssery et al.](https://arxiv.org/abs/2404.13076)) | **Partly mitigated, partly carried.** The claim verifier runs on a different family from the novelty reviewer; the converger does not. Recorded above rather than hidden. |
| **Sycophancy** | 58.19% sycophantic responses with 78.5% persistence, worst under citation-based rebuttals ([SycEval](https://arxiv.org/abs/2502.08177)) | **Reduced by construction.** Judges never see an author, a rebuttal, or their own earlier answer — there is no multi-turn pressure channel. Not separately measured. |
| **Aggregation anchoring** | LLM meta-reviewers anchor harder than humans (0.255 vs 0.193) and suppress minority views ([arXiv:2503.13879](https://arxiv.org/html/2503.13879)) | **Constrained, not solved.** The converger may only restate the structured findings it is given and is instructed to present both sides of a disagreement, but the structured record — not its prose — is what a reader should trust. Anchoring is not measured here. |
| **Verbosity bias** | Length control raises correlation with human preference from 0.94 to 0.98 ([arXiv:2404.04475](https://arxiv.org/abs/2404.04475)) | **Not addressed.** Findings are capped in number, not scored for length, and no length control is applied. |

## What the numbers do not prove

- Retrieval metrics are reported only when at least 20 relevant documents exist across the
  evaluated manuscripts; below that the per-item hit table is shown and rates are withheld
  entirely. A rate over a handful of items is decorative.
- Inter-reviewer correlation is reported only at n ≥ 30 observations, for the same reason.
- The graph, communities and retrieval run for real in CI; the model-dependent layers are proven
  by captured real runs whose artifacts are committed and labelled with the command that
  regenerates them. Which is which is stated in the README, and no result anywhere is produced by
  a mocked pipeline.
