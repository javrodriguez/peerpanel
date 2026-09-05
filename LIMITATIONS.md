# Limitations

PeerPanel is a demonstration system. This page is the part of the repo most worth reading
carefully: it says what the design does not do, what the numbers do not prove, and which known
failure modes of multi-agent LLM review it mitigates, measures, or simply carries.

Every figure this page measures about this system is recomputed from committed bytes by `tests/test_limitations_numbers.py` or `tests/test_run_graph.py`, so a number that drifts from the measurement fails the suite rather than sitting here unread.
The demo-corpus figures are recomputed offline, by merging the per-chunk extractions in `fixtures/extraction/demo`, with no network, no model and no fetched corpus text.
The percentages in the failure-mode table further down are other people's published measurements, cited to their source.

## Scope

**What this is for:** a pre-submission self-check on manuscripts that are already public, run
against a pinned open-access corpus, by their own authors.

**What it is not for:** it is not a substitute for peer review, and it must not be used on
manuscripts under confidential review — publishers and funders prohibit uploading such
manuscripts to language models, and this system would do exactly that.

## Structural limitations of this implementation

- **The manuscript is reviewed as an excerpt.** Each reviewer sees the first 900 words plus a
  few retrieved literature chunks — not the full paper, and never the figures or tables — so
  reviews of a paper's later sections are out of reach by construction. The excerpt is a cost
  decision, one chunk-sized prompt per reviewer; it is no longer a serving-slot limit, because
  the window is now sized to whatever the prompt needs (DECISIONS.md D19). Until 2 September
  2026 it was both, and the slot silently cut what the budget did not. (The reviewer that reads figures does not
  exist here; the best-known automated reviewer cannot read them either, and its evaluators
  called that out.)
- **Small local models.** Every model call runs against `llama3.1:8b` and `qwen2:7b` on one
  laptop. That keeps the whole system reproducible at zero cost, and it caps quality well below
  what a frontier model would produce. Numbers here are numbers for *this* configuration.
- **Two reviewers, not a large panel.** Panel diversity is largely illusory in the literature —
  nine LLM judges have been measured to behave like roughly two independent votes under
  correlated error ([arXiv:2605.29800](https://arxiv.org/pdf/2605.29800)) — so this system runs
  two structurally different reviewers rather than performing breadth it cannot deliver.
- **Four roles, two model families.** The converger runs on the novelty reviewer's model
  (`qwen2:7b`) and the claims verifier runs on the methods reviewer's model (`llama3.1:8b`), so the
  judge of each reviewer's claims shares a family with one of them and the writer of the meta-review
  shares a family with the other. Both are self-preference risks (see below), taken deliberately to
  keep everything on two locally-available families; a third family would be the clean fix. The
  assignment lives in one place, `PanelProviders.local_default()`.
- **Propositional conflict detection is not implemented.** Conflicts are detected
  deterministically for score divergence (*sentiment*) and for reviewer claims the evidence
  refutes (*evidence*). Detecting that two reviewers assert logically incompatible propositions
  needs entailment between free-text findings, which this version does not do.
- **The corpus is citation-seeded.** The demo corpus is built from the query manuscripts' own
  cited references (plus one-hop neighbours and a topical top-up), which is what makes
  ground truth exist at all — and it means retrieval numbers are measured on a corpus that was
  constructed to contain the answers. Read them as a comparison *between rungs*, never as an
  estimate of real-world retrieval difficulty.
- **Exclusion is exact for the graph, approximate for summary wording.** When a manuscript is
  reviewed, its published twin's chunks leave the index and the knowledge graph is *rebuilt* for
  that run with the twin's extractions withheld (a pure merge over cached extractions plus seeded
  Leiden — measured at ~0.2 s, so the exact thing is affordable). On the CI corpus that removes
  142 entities and additionally strips 27.25 units of twin-contributed weight from 48 edges
  joining entities that both survive — the part node-filtering cannot reach, and the tests
  demonstrate it by doing the filtering instead and showing those 48 edges keep their weight. Chunk-level retrieval and
  `graphrag-local` therefore carry no residue at all. Community summary *text*, however, is a
  model artifact generated once over the whole corpus, so `graphrag-global` — which ranks over
  that text — can still be influenced by a document it may not retrieve.
  **Measured on the committed bytes:** on the MET17 run, 101 graph entities exist only because of the excluded twin.
  All 101 of them appear in the community reports' `member_names` at the resolution retrieval reads, which is the honest measure of this residue: `graphrag-global` tokenises `title`, `summary` and `member_names` together, so membership in that list is what carries a twin-only entity into the scores the rung ranks by.
  At most 5 of the 101 are named in a report's own title or summary prose — a substring test, and therefore an over-count: two of the five (`age`, `cre`) match only inside ordinary words such as *management* and *creation*, while `academic press`, `arganda-carreras i` and `longair m` are named outright.
  The figure published until this commit, 96, was 101 minus those five, stated with its sense inverted: in the text that ranking actually reads the residue is complete rather than partial, so the page understated the weakness it exists to disclose.
  The 101 are mostly the twin's own apparatus and acknowledged colleagues — "Costar 3370 96-well plate", "Teleshake magnetic shaking device", "Nikon AZ100 upright microscope", "Maitreya Dunham" — so its fingerprint is in the summary text that ranking scores, even though not one of its chunks can be retrieved.
  Reproduce it from a clean clone, with no network and no model, by merging the committed per-chunk extractions rather than the fetched corpus text:

  ```python
  import json, pathlib
  from peerpanel.graph.build import build_graph, _norm
  from peerpanel.graph.models import ChunkExtraction

  root = pathlib.Path('.')
  records = [ChunkExtraction.model_validate_json(f.read_text())
             for f in sorted((root / 'fixtures/extraction/demo').glob('*.json'))]
  full = build_graph(records)
  without_twin = build_graph([r for r in records if not r.chunk_id.startswith('PMC10729969:')])
  twin_only = set(full) - set(without_twin)
  members, summary_text = set(), []
  for path in sorted((root / 'fixtures/summaries/demo').glob('*.json')):
      report = json.loads(path.read_text())
      members |= {_norm(name) for name in report['member_names']}
      summary_text.append((report['title'] + ' ' + report['summary']).lower())
  blob = ' '.join(summary_text)
  print(len(twin_only), len(twin_only & members), sum(1 for e in twin_only if e in blob))
  ```

  It prints `101 101 5`, and the same merge reproduces the published index exactly (7,697 entities, 63,130 edges — `results/build-stats-demo.json`).
  The recipe this page gave until this commit ran `build_run_graph` over the demo corpus, which raises `DemoCorpusMissing` in a clean clone because that corpus is fetched, so the figure it claimed to reproduce could not be checked by anyone who had not already run the fetch.
  Regenerating summaries per run would cost roughly an hour of local model time per manuscript and is not done; the LLM face of global search (`answer()`, which would put that summary text in front of a reviewer) instead refuses to run when anything is excluded.
- **The graph is mostly adjacency, not stated relations.** An extracted relation weighs four times a
  bare co-mention per edge, but 93.1% of the demo graph's edges (58,797 of 63,130) are co-mention
  only, and just 4,333 edges carry a relation the model actually asserted.
  When `graphrag-local` reaches a document lexical matching misses, it is usually because two
  entities appeared in the same chunk — not because the model asserted a link between them.
  All three figures come from the same offline merge as the residue above, and `tests/test_run_graph.py` holds `graph/build.py`'s own docstring to the same measurement.
- **No entity resolution.** Nodes are case-normalised strings. `MET17`, `Met17` and `met17` merge;
  `S. cerevisiae` and `Saccharomyces cerevisiae` do not, and neither do a protein and the gene that
  encodes it. The entity count is therefore an upper bound on distinct concepts.
- **`graphrag-global` breaks ties alphabetically.** Chunks inside one community share that
  community's score, so ties are resolved by chunk-id string order — which begins with the PMCID.
  Some of that rung's ranking, and therefore some of its NDCG, is alphabetical rather than
  relevance-driven. It is deterministic and reproducible; it is not meaningful.
- **The ablation and the panel use `graphrag-local` differently.** The ablation blends graph votes
  with cosine similarity; the panel's reviewer runs it with the embedding blend off. The retrieval
  numbers above therefore describe a configuration the panel does not use, and the two should not
  be read as measuring the same component.
- **Attribution travels with the repository; the corpus text does not.** `CITATIONS.md` carries per-document TASL attribution for the 15 documents committed to this repository, and `corpus/DEMO_CITATIONS.md` — tracked at this commit — carries the same for all 68 pinned documents of the demo corpus, with title, authors, source, licence, changes and md5 for each.
  A reader of the repository alone therefore sees attribution for 83 of 83 documents the project touches.
  What that reader does not get offline is the corpus TEXT: the demo documents are fetched by `make corpus` against the pinned md5s, and only their attribution is committed.
  Until this commit this paragraph said 15 of 83: it was written before `corpus/DEMO_CITATIONS.md` was tracked, and it left the page understating — in the paragraph about licence compliance — work the repository had already done.
- **No random-selection floor.** 52.9% of the demo corpus is relevant to one query or another by construction — the two query manuscripts' own cited sets, intersected with the pinned manifest and unioned, cover 36 of 68 documents — so a rung that returned documents at random would not score zero, and nothing here measures what it *would* score.
  Every reported rate should be read against that missing baseline rather than against zero.
  Recompute the share from `corpus/ground_truth.json` and `corpus/demo.manifest.json`, both committed.
  The share published until this commit, 42%, was one manuscript's own cited share against a single run's 67-document index rather than the union this sentence describes, and it understated the missing baseline by eleven points.
- **A stale corpus, deliberately.** The corpus is a pinned snapshot with per-document md5s, not a
  live search. The demo therefore re-runs identically next year and cannot see anything published
  after the snapshot. That trade is the point: reproducible verdicts over current ones.

## Known failure modes of LLM review, and what this system does about each

| Failure mode | Evidence | What PeerPanel does |
|---|---|---|
| **Position bias** | Across 36 models and 193 pairs, the first-shown option is picked 64.3% of the time, and the median model flips on 41.3% of decisive swapped-order cases ([benchmark](https://github.com/lechmazur/position_bias)) | **Mitigated and measured.** Every claim is judged twice with the evidence order reversed; disagreement forces `NOT_ENOUGH_INFO`. The swap-consistency rate is reported with its n, or withheld when n < 10. |
| **Fabricated citations** | 19.9% of GPT-4o citations across six simulated literature reviews were untraceable ([JMIR Mental Health](https://mental.jmir.org/2025/1/e80371)); retrieval reduces but does not eliminate this ([arXiv:2409.13740](https://arxiv.org/html/2409.13740v1)) | **Mitigated in code, not by instruction.** A verdict's evidence spans survive only if the chunk id was actually retrieved and the quote is a substring of that chunk's own text. Reviewer findings citing unprovided chunks have those citations dropped. |
| **Prompt injection** | Hidden instructions in a manuscript reach up to 100% acceptance across 1,000 generated reviews ([arXiv:2509.10248](https://arxiv.org/abs/2509.10248)) | **Mitigated.** Every corpus document and manuscript passes a deterministic screen before chunking: invisible and control characters are stripped, instruction-shaped lines are neutralised (prefixed and quoted as data, never silently deleted), and every action is reported. |
| **Multi-agent may not beat a single agent** | Debate "often fails to outperform simple single-agent baselines… even when consuming significantly more inference-time computation" ([arXiv:2502.08788](https://arxiv.org/abs/2502.08788)) | **Measured, not assumed.** The planted-error evaluation reports the panel beside a single-agent baseline given the panel's token spend as a *ceiling* through the same ledger — a ceiling it has never reached on any committed run, so the baseline always had less. Whatever the delta is, it ships, with each arm's spend beside it. |
| **Self-preference** | Judges favour their own generations, and the bias tracks self-recognition causally ([Panickssery et al.](https://arxiv.org/abs/2404.13076)) | **Partly mitigated, partly carried.** The claim verifier (`llama3.1`) judges the novelty reviewer's claims across a family boundary but the methods reviewer's within one; the converger (`qwen2`) writes over the novelty reviewer's own family. Recorded above rather than hidden. |
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
