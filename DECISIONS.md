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
