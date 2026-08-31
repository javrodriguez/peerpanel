# PeerPanel — every target is real work on real data; nothing is mocked.
# Tier 1 (no LLM): quickstart. Tier 2 (needs Ollama + pinned local models): demo, ablation, eval.

.PHONY: quickstart demo demo-replay corpus embeddings graph summaries ablation eval test lint

## Tier 1 — deterministic, committed CI corpus + fixture embeddings, no LLM, <60s post-install
quickstart:
	uv run python -m peerpanel quickstart

## Tier 2 — the captured real demo run on the demo-scale corpus (needs Ollama)
demo:
	uv run python -m peerpanel demo

## Replay the committed machine-captured demo (labeled recording; see demo/README)
demo-replay:
	uv run python -m peerpanel demo --replay

## Fetch + md5-verify the demo-scale corpus per corpus/demo.manifest.json
corpus:
	uv run python -m peerpanel corpus fetch

## Regenerate embedding fixtures and diff their sha256 manifest
embeddings:
	uv run python -m peerpanel embeddings --check

## Regenerate the extraction + graph artifacts for the CI corpus
graph:
	uv run python -m peerpanel graph build

## Regenerate community summaries (needs Ollama)
summaries:
	uv run python -m peerpanel graph summaries

## The retrieval ablation ladder (BM25 → vector → RRF → GraphRAG-local → GraphRAG-global)
ablation:
	uv run python -m peerpanel ablation

## Planted-error evaluation: panel vs equal-compute single-agent baseline
eval:
	uv run python -m peerpanel eval

test:
	uv run pytest -m "not network and not ollama"

lint:
	uv run ruff check src tests && uv run mypy
