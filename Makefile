# PeerPanel — every target is real work on real data; nothing is mocked.
#
# Tier 1 (no model, no network): quickstart, test, lint — run from committed bytes.
# Tier 2 (needs Ollama + the pinned local models): everything else.
#
# Full reproduction of the demo scale, in order:
#   make corpus && make demo-index && make demo-summaries && make ablation && make demo

.PHONY: quickstart demo demo-replay demo-index demo-summaries corpus embeddings \
        graph summaries ablation eval review test lint

## --- Tier 1: deterministic, from committed bytes ---

## The 60-second first look: graph + communities + retrieval, rebuilt from
## committed caches with a provider that refuses to be called.
quickstart:
	uv run python -m peerpanel quickstart

test:
	uv run pytest -m "not network and not ollama"

lint:
	uv run ruff check src tests && uv run mypy

## --- Tier 2: real model runs (needs Ollama) ---

## Fetch + md5-verify the demo-scale corpus per corpus/demo.manifest.json.
corpus:
	uv run python -m peerpanel corpus fetch

## Verify the committed CI corpus byte-for-byte against its pins.
corpus-verify:
	uv run python -m peerpanel corpus verify

## Regenerate the CI embedding fixture and diff its sha256 manifest.
embeddings:
	uv run python -m peerpanel embeddings --check

## Regenerate the CI extraction + graph artifacts.
graph:
	uv run python -m peerpanel graph build

## Regenerate CI community reports (all Leiden levels).
summaries:
	uv run python -m peerpanel graph summaries

## Build the demo-scale index: embeddings + extraction + graph + Leiden (~6.4h).
demo-index:
	uv run python -m peerpanel graph build --corpus demo

## Demo community reports at the resolution retrieval reads (see DECISIONS D9).
demo-summaries:
	uv run python -m peerpanel graph summaries --corpus demo --resolution 1.0

## The retrieval ablation ladder over the demo corpus.
ablation:
	uv run python -m peerpanel ablation --corpus demo

## Review one manuscript with the full panel (MANUSCRIPT=<stem>).
MANUSCRIPT ?= met17-auxotroph
review:
	uv run python -m peerpanel review $(MANUSCRIPT)

## The headline measurement: planted errors, panel vs equal-compute baseline.
eval:
	uv run python -m peerpanel eval

## The captured demo run (writes demo/raw-*.log beside its curated transcript).
demo:
	uv run python -m peerpanel demo

## Replay the committed machine-captured demo (a labelled recording).
demo-replay:
	uv run python -m peerpanel demo --replay
