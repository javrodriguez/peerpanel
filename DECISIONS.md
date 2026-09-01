# Decisions

Design decisions, with the why. Newest last.

## D1 — The name: PeerPanel
`peerpanel` on PyPI is unclaimed and GitHub has zero repos by that name (checked 2026-08-30).
The name says what the system is: a panel of independent reviewers doing peer review — "panel" is also the literature's own term for judge ensembles (Panel of LLM evaluators, PoLL).
Rejected for collisions: `colloquy` (established macOS IRC client + PyPI taken), `parley` (linebender's 715★ Rust text library), `conclave` (R3's confidential-computing product), `synod`/`scrutineer` (PyPI taken, starred repos), `paperjury` (a 1,035★ project owns it), `collegium` (shadowed by the CollegiumAI edtech platform).
No affiliation is implied by the name, and no paper by this name exists in the automated-review literature (searched 2026-08-30).

## D2 — From scratch, not an import
`microsoft/graphrag` cannot be installed on Python 3.14 (every published version pins `<3.14`), and its Leiden dependency chain (graspologic) fails to build.
More importantly, the graph, the Leiden partitioning and the dual search modes are the object of study here — importing them would leave nothing to demonstrate.
When you would NOT build this yourself: with Python ≤3.13 and no need to instrument the internals, LlamaIndex's PropertyGraphIndex or Microsoft's GraphRAG give you the pipeline off the shelf, and LazyGraphRAG's deferred-summarisation design is the better cost profile for large corpora.

## D3 — Two wire protocols, honestly labeled
The provider seam is exercised over two genuinely different wire protocols against the same local models (the OpenAI client pointed at Ollama, and the native Ollama client).
The Anthropic adapter is implemented and wire-shape-tested but has never made a call from this repo — no API key exists on the development machine and none was requested; the code and tests say exactly that.

## D4 — Workspace folder keeps its working name
The development folder is `workspaces/graphrag-review` in a private workspace registry; only the public repo and the package carry the PeerPanel name.
Renaming the folder would touch registry state for zero public benefit — minimal blast radius wins.

## D5 — CI corpus composition
15 committed CC BY full texts (~1.2 MB): 14 topical yeast-sulfur papers from the PMC OA CC-BY subset (most recent first at authoring time) plus one force-included pin, PMC10729969 — the published form of a manuscript the demo reviews.
That pin exists as a labeled twin-retrieval showcase only; the self-exclusion rule bars it from every metric involving its own manuscript.
Every document is pinned by (PMCID, version, md5), byte-verified by `python -m peerpanel corpus verify` and by a committed-data test, and attributed in CITATIONS.md.
PMC10729969 fails the topical [TIAB] query (its title says "budding yeast", not "Saccharomyces cerevisiae") — which is why it is a pinned addition rather than a topical pick.

## D6 — Demo candidate construction, and why twins are members
`corpus/demo.candidates.json` is ordered: citation seeds first (the three query manuscripts' cited papers, via their published versions' reference lists — 140 candidates, forced), then the three published twins themselves (forced), then one-hop citation neighbours, then topical top-up (447 total).
Forced candidates are never dropped by size truncation — they are the retrieval eval's ground truth.
The twins are corpus MEMBERS by design: each manuscript's review run drops its OWN twin from the index via the recorded twin table, so the exclusion mechanism demonstrably removes a real in-index document on every run; at other manuscripts' runs a foreign twin is an ordinary corpus document.
Keeping twins out of the corpus entirely would make each run's exclusion set empty — an untested mechanism wearing a passing test.

## D7 — Extraction cost: the measured arithmetic behind chunking, model and parallelism
Graph extraction pays one local-LLM call per chunk, so every parameter here was measured, not assumed (all timings on this machine, llama3.1:8b Q4, Apple Silicon, 2026-08-31):
- **Extractor model**: llama3.1:8b. qwen2:7b measured faster (23.0s vs 31.7s per 300-word chunk) but returned ZERO relations on the same input — edges are what make the graph a graph, so speed lost to quality.
- **Output caps enforced in the schema**: asking the model for "at most 12 entities" in prose was ignored (40+ entities, >1000 output tokens); `maxItems` in the JSON schema enforces it through constrained decoding (exactly 12 came back, ~half the output tokens). Prompt version bumped so the cache re-keys.
- **One chunking everywhere, 900 words (~1200 tokens)**: extraction, embeddings and retrieval share one chunk id space (entity→chunk→vector joins by id), and 900 words cuts the CI corpus from 785 to 212 chunks — 3.7× fewer LLM calls for the same coverage. The embedding fixture was regenerated for real (212×768 f16, nomic-embed-text, sha256 in its manifest).
- **Server parallelism**: the local Ollama daemon shipped serving one request at a time (measured 1.0× speedup on 4 concurrent calls). Restarted with OLLAMA_NUM_PARALLEL=4 · OLLAMA_CONTEXT_LENGTH=4096 (per-slot; the first attempt at 16384 per-slot ballooned the KV cache past GPU memory — 65536 total context, partial CPU offload, generation crawled — and was corrected). Post-restart toy-probe speedup 1.9×; the CI bake's own wall-clock is the number that governs demo-corpus sizing (the plan's 8-hour trigger reads the measured rate).
- A truncated extraction is retried once at doubled output budget, then recorded honestly as `truncated=true` — counted in build stats, never a crash, never an invented result.

## D8 — Demo scope: two query manuscripts, measured against two hard gates
Two constraints bind the demo corpus in opposite directions: the wall-clock trigger (a full-index build over 8 measured hours forces the minimum cut) and the ground-truth gate (recall/NDCG are only reported at >=20 aggregate relevant documents; below it, per-item hit tables only).
The measurements (2026-08-31): effective extraction throughput 25.6 s/chunk (parallel factor 1.36x over the 34.7 s single stream); CC-gate pass rates on the three manuscripts' citation seeds — MET17 8/21, caprin 28/67, biopolymer 28/52.
- Three manuscripts ≈ 160+ docs ≈ 16 h — over the trigger.
- One manuscript (MET17) = 8 relevant — under the ground-truth gate: no rates, no correlation, real evidence lost.
- **Two manuscripts (MET17 + biopolymer) = 36 aggregate relevant, ~68 docs ≈ 960 chunks ≈ 6.8 h — clears both gates with margin.**
So the demo runs MET17 + the oleaginous-biopolymer manuscript; caprin is deferred to the three-manuscript expansion (its ingest, twin row and ground truth are all committed and waiting).
Biopolymer was kept over caprin deliberately: nitrogen-metabolism text is topically adjacent to the sulfur-heavy corpus, which makes retrieval harder and the reported numbers more honest than caprin's easily-separable heterochromatin domain would.
The caprin twin (PMC11918387) stays out of the demo corpus entirely — the non-empty-exclusion assertion binds per RUN, and caprin has no run; it re-enters WITH its manuscript if the three-manuscript expansion lands (its ingest, twin row and ground truth are already committed).
Why this scope reduction is legal under both laws: the wall-clock trigger sanctions reducing MANUSCRIPT scope (the plan's own minimum cut drops to one manuscript, so whole-manuscript reduction is the designed escape hatch), while the ground-truth tie-break bars cutting ground-truth DOCUMENTS within whatever scope is kept — and every kept manuscript's forced includes are untouched here.
**Outcome:** the demo index built in 6.4 h against the 6.8 h projection and the 8 h ceiling, 948 chunks, zero truncated — the measure-then-rule ordering was worth it, and a scope guessed rather than measured would have been wrong in one direction or the other.

## D9 — Community reports are generated at the resolution retrieval reads
Leiden runs at three resolutions on both corpora and the full hierarchy is persisted (`artifacts/<corpus>/communities.json`) — that is the cheap, deterministic half.
Each community REPORT costs one model call, and only the resolution-1.0 level is read by global search and the novelty reviewer, so the demo corpus is summarised at 1.0 (65 eligible communities) rather than all three levels (~213).
The CI corpus, being small, carries reports at every level.
`graph summaries --resolution <r>` makes the choice explicit rather than implicit, and the README states which levels carry reports on which corpus.
This is a cost decision, not a claim about what GraphRAG requires: summarising every level is what Microsoft's implementation does, and doing so here would add roughly 150 further model calls for levels nothing currently retrieves through.

## D10 — Exclusion binds the graph, not only the chunks
Chunk-level filtering alone left a real leak, found by testing rather than reasoning: on the CI graph 131 entities were evidenced ONLY by the excluded twin, and 110 of them still matched queries and voted for their neighbours — the twin's own author surnames and MET17 among them.
No twin text could be retrieved (measured 0 twin chunks in 30 hits across both graph modes), but the twin was still steering which surviving chunks ranked.
The first fix filtered the built graph (`Index.live_nodes()` drops entities whose every evidencing chunk is excluded), and it was justified with a wrong cost: "a per-run rebuild would take 6.4 h".
That conflated rebuilding the INDEX with rebuilding the GRAPH. Extraction is per-chunk, cached, and already paid; `build_graph()` is a pure merge over those cached extractions, so withholding a document costs a merge, not an extraction.
Measured on the CI corpus: full rebuild from cache **0.03 s** (1577 nodes / 12166 edges, matching the committed graph), rebuild with the twin withheld **0.03 s** (1446 / 10919 — 131 nodes and 1247 edges removed), plus **0.12 s** for seeded Leiden.
So the exact thing is affordable and the approximation was never needed: `build_run_graph()` REBUILDS the graph per run with the excluded document's extractions withheld. That removes the 131 entities *and* strips twin-contributed weight from **38 edges joining entities that both legitimately survive** (21.25 weight units on the CI graph) — the part node filtering structurally cannot reach. A concrete example: the link between *Saccharomyces cerevisiae* and *yeast* carried weight 1.25, of which 0.25 came from the twin's text alone; after the rebuild it carries 1.0.
`live_nodes()` is kept as a second line of defence and as a cross-check: a test asserts the two mechanisms agree exactly on which entities survive.
**The residual is now narrow and accurately priced:** community summary TEXT is an LLM artifact generated once over the full corpus, so `graph-global`, which ranks over that text, is the one mode where an excluded document can still influence wording. Everything chunk-level and `graph-local` carry no residual at all.
Note also what the non-empty-exclusion assertion does and does not prove: it shows the exclusion SET is non-empty, not that anything was removed. The assertion is now `dropped_chunk_count > 0` — by this repo's own D6 argument, an assertion that passes when the mechanism is a no-op is an untested mechanism wearing a passing test.

## D11 — The same failure shape, three times: assertions that pass on a no-op
Three separate defects in this build shared one shape — a check that could not fail when the thing it guarded stopped working.
(1) The self-exclusion law was asserted as "the exclusion SET is non-empty", which passes when exclusion removes nothing; now `dropped_chunk_count > 0`.
(2) That strengthened assertion immediately earned its keep by catching a real defect of its own: the planted-error subject is held out by construction, so nothing drops, and the check had to learn the difference between *nothing to exclude* and *exclusion failed*.
(3) Worst of the three, found by a peer session's reviewer: no test imported `run_ablation`, `run_panel` or `run_planted_eval` at all, so every exclusion guard in the orchestration layer could be deleted with a green suite — demonstrated by mutation (five guards disabled, 252 passed → 252 passed, identical).
The standing rule this leaves: **an invariant is only defended where a test drives the real entry point.** Toy-fixture sweeps and data-membership pins are worth having, but they do not defend orchestration, and a suite that stays green through a mutation of the mechanism is measuring something other than the mechanism.

## D12 — Both evaluation arms must withhold the same documents
The planted-error evaluation built the panel's index with exclusions and the baseline's without.
Latent rather than active — the shipped subject's twin is not a corpus member — but the published NOTE claimed "no unperturbed original is retrievable", which was false for both in-scope manuscripts, and D8 schedules exactly the corpus change that would have made the default case contaminated.
Measured on the committed demo corpus before the fix: the baseline's top three chunks for MET17 were the manuscript's own published twin (scores 64.7 / 46.1 / 45.0), which states every planted fact correctly — the answer key, handed to one arm only.
Both arms now build from the same twin lookup, the run refuses if the subject's twin is a corpus member and the baseline index dropped nothing, and each arm's exclusion state is recorded in the report rather than described in prose.

## D13 — A mutation proof needs a clean clone AND a fresh environment
The fourth near-miss of the D11 shape was in the proof, not the code. Mutating a `cp -R` copy of the
workspace and running its tests reported **5 passed on a mutant with every exclusion guard removed** —
which looked exactly like "the tests do not defend".
The copy carried `.venv`, whose editable install still resolved `peerpanel` to the ORIGINAL source
tree, so under pytest the mutations were never imported. A probe test printing `module.__file__`
showed it pointing back at the real workspace.
Redone as `git clone` + `uv sync --dev` — no venv, no `__pycache__`, tracked files only — the same
three mutations turn **4 of 5 tests red**, including the one asserting no twin chunk reaches a
reviewer's prompt.
The rule: a mutation proof is only evidence if it first proves it is running the mutant. Clone, never
copy; sync fresh; and when a result says "the guard does not fire", suspect the harness before
believing it.

## D14 — A small-n aggregate publishes its per-unit numbers, or publishes nothing
One error class was caught four times in this build, each time in something already written down as
though it were settled:
1. A hits column computed at k=30 published under a `k = 10` heading — which happened to flatter the
   graph rung, and reversed the true ordering at the reported depth (BM25 13, GraphRAG local 12).
2. Swap-consistency published as `0.50` from a single run; an independent second run of the same
   panel returned 0.417.
3. `GraphRAG local wins on recall, 0.393` — the mean of a decisive win on one manuscript (0.500) and
   a clear loss on the other (0.286), with the rung ordering reversing between the only two cases.
4. Latency labelled "median" while the code computed a mean.
None was a lie and none was caught by a test; each was a number carrying more confidence than its
evidence supported, in a repo whose entire credibility rests on the opposite.
**The rule:** an aggregate over fewer than roughly ten units ships the per-unit numbers beside it, or
does not ship. Where a mean can invert the ordering it implies, the per-unit numbers are the result
and the mean is a convenience. This is enforced in the tool, not just in prose — `render_table`
always emits per-case rows — because a prose-only fix decays the moment another case lands.
The repo already applied this discipline twice before noticing it was a rule (the N ≥ 20 gate that
withholds rates over thin ground truth; swap-consistency published as a range). The inconsistency was
ours.
