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
