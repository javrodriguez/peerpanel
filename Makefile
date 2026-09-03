# PeerPanel — every target is real work on real data; nothing is mocked.
#
# bash + `set -o pipefail` on every piped recipe, on purpose: several targets pipe a real
# run through `tee` so the raw log is committed beside the record. Without it a crashed
# run's exit code is tee's, which is 0 — the target reports success, the log is left
# empty or half-written, and the stale record beside it stands as if it had been
# regenerated. That is a mocked artifact produced by nothing but a broken pipe.
# The guard is written into each recipe rather than into .SHELLFLAGS because GNU Make
# 3.81 (what macOS ships) ignores .SHELLFLAGS silently — a guard that cannot fire on
# the machine these runs happen on is the defect it was meant to close (D11).
# test_regenerate_commands proves no piped recipe is missing it.
SHELL := /bin/bash
#
# Tier 1 (no model, no network): quickstart, test, lint — run from committed bytes.
# Tier 2 (needs Ollama + the pinned local models): everything else.
#
# Full reproduction of the demo scale, in order:
#   make corpus && make demo-index && make demo-summaries && make ablation && make demo

.PHONY: quickstart demo demo-replay demo-index demo-summaries corpus corpus-verify \
        embeddings query-embeddings graph summaries ablation ablation-demo publish-results eval review test lint

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

## Regenerate the CI extraction + graph artifacts, and publish the build record
## RESULTS.md cites (results/build-stats-ci.json). On the committed extraction cache
## this replays in seconds, makes no model calls and reports cached_before = 234; the
## committed record is the COLD build (cached_before = 0, model_calls = 234, the real
## extraction wall-clock), reproduced by emptying fixtures/extraction/ci first. Both
## forms are honest and the record says which it is — a warm record claiming a cold
## build's wall-clock would not be.
graph:
	uv run python -m peerpanel graph build --publish

## Regenerate CI community reports (all Leiden levels).
summaries:
	uv run python -m peerpanel graph summaries

## Build the demo-scale index: embeddings + extraction + graph + Leiden (~6.4h).
## Writes the log RESULTS.md cites as this build's raw capture — a documented
## regenerate command that produces no artifact is not a regenerate command.
demo-index:
	@mkdir -p results
	set -o pipefail; uv run python -m peerpanel graph build --corpus demo --publish 2>&1 | tee results/demo-index-build.log

## Demo community reports at the resolution retrieval reads (see DECISIONS D9).
demo-summaries:
	@mkdir -p results
	set -o pipefail; uv run python -m peerpanel graph summaries --corpus demo --resolution 1.0 2>&1 | tee results/demo-summaries.log

## The retrieval ablation ladder — CI corpus, from committed bytes, no model needed.
ablation:
	uv run python -m peerpanel ablation --corpus ci

## The same ladder at demo scale (needs the fetched demo corpus).
ablation-demo:
	uv run python -m peerpanel ablation --corpus demo

## The same ladders written into the tracked results/ records RESULTS.md cites.
## `make ablation` leaves the tracked file alone on purpose; these are the doors
## that regenerate it, and a test proves each cited command names its file.
ablation-publish:
	uv run python -m peerpanel ablation --corpus ci --publish

ablation-demo-publish:
	uv run python -m peerpanel ablation --corpus demo --publish

publish-results: ablation-publish ablation-demo-publish

## Regenerate the committed query-embedding fixture against the live embedder.
query-embeddings:
	uv run python -m peerpanel ablation --corpus ci --live
	uv run python -m peerpanel ablation --corpus demo --live

## Review one manuscript with the full panel (MANUSCRIPT=<stem>, RUN=<n>).
## Publishes the tracked record + raw log RESULTS.md cites:
##   results/panel-review-<stem>[-run<n>].json and .log (RUN=1 carries no suffix).
MANUSCRIPT ?= met17-auxotroph
RUN ?= 1
RUN_SUFFIX = $(if $(filter 1,$(RUN)),,-run$(RUN))
review:
	@mkdir -p results
	set -o pipefail; uv run python -m peerpanel review $(MANUSCRIPT) --publish --run $(RUN) 2>&1 | tee results/panel-review-$(MANUSCRIPT)$(RUN_SUFFIX).log

## The headline measurement: planted errors, panel vs single-agent baseline
## (SUBJECT=<stem>). Publishes results/planted-eval-<stem>.json and .log.
SUBJECT ?= caprin-heterochromatin
eval:
	@mkdir -p results
	set -o pipefail; uv run python -m peerpanel eval $(SUBJECT) --publish --run $(RUN) 2>&1 | tee results/planted-eval-$(SUBJECT)$(RUN_SUFFIX).log

## The captured demo run — tee'd verbatim to demo/raw-<utc>.log.
## The captured demo walk. Writes the raw capture to the path --replay reads,
## so the committed evidence and the replay can never drift apart.
demo:
	@mkdir -p results
	set -o pipefail; uv run python -m peerpanel demo 2>&1 | tee results/demo-capture.log

## Print the committed capture instead of re-running it (no model needed).
demo-replay:
	uv run python -m peerpanel demo --replay

