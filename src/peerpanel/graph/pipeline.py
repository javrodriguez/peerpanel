"""Corpus graph pipeline, parameterized over the two scales (ci | demo).

Both scales ride the SAME road: the corpus's committed manifest + fetched
texts -> sanitize -> one shared chunking -> cached LLM extraction -> graph ->
seeded Leiden -> persisted artifacts. The CI scale's chunks come from
peerpanel.embeddings.pipeline.ci_chunks (the embedding fixture shares those
ids); the demo scale reads corpus/demo.manifest.json + corpus/demo/ and its
embedding fixture is written here (fixtures/demo_embeddings.npz).

Extraction caches are committed: real recorded model outputs, regenerated for
real by `make graph` / `graph build --corpus demo` (see the README beside
each cache). CI rebuilds graphs from the caches with no LLM.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from peerpanel.corpus.models import CorpusManifest
from peerpanel.embeddings import store
from peerpanel.embeddings.pipeline import ci_chunks
from peerpanel.providers.base import ChatProvider, EmbedProvider
from peerpanel.text.chunks import Chunk, chunk_document
from peerpanel.text.sanitize import Finding, sanitize

from .build import build_graph, save_graph, stats
from .communities import detect
from .extract import extract_many

CACHE_README = """\
# {corpus} extraction cache — real recorded fixtures

Every file here is the REAL output of the extraction LLM (llama3.1:8b via the
provider seam, temperature 0, prompt v{version}) over one {corpus}-corpus
chunk, produced by an actual call and cached by (chunk id, provider, prompt
version). Nothing is hand-written. Regenerate for real:
`{regenerate}` with Ollama running (a changed prompt version re-extracts
everything).
"""

_REGENERATE = {"ci": "make graph", "demo": "python -m peerpanel graph build --corpus demo"}


def corpus_chunks(
    root: Path, manifest_rel: Path, texts_dir: Path
) -> tuple[list[Chunk], list[tuple[str, Finding]]]:
    """Sanitize + chunk every corpus doc, in pmcid order. Deterministic."""
    manifest = CorpusManifest.load(root / manifest_rel)
    chunks: list[Chunk] = []
    findings: list[tuple[str, Finding]] = []
    for doc in sorted(manifest.docs, key=lambda d: d.pmcid):
        raw = (root / texts_dir / f"{doc.pmcid}.txt").read_text(encoding="utf-8")
        clean, found = sanitize(raw)
        findings.extend((doc.pmcid, f) for f in found)
        chunks.extend(chunk_document(doc.pmcid, clean))
    return chunks, findings


def demo_chunks(root: Path) -> tuple[list[Chunk], list[tuple[str, Finding]]]:
    return corpus_chunks(root, Path("corpus") / "demo.manifest.json", Path("corpus") / "demo")


def chunks_for(root: Path, corpus: str) -> list[Chunk]:
    if corpus == "ci":
        return ci_chunks(root)[0]
    if corpus == "demo":
        return demo_chunks(root)[0]
    raise ValueError(f"unknown corpus {corpus!r} (ci | demo)")


def embedding_fixture_path(root: Path, corpus: str) -> Path:
    return root / "fixtures" / ("ci_embeddings.npz" if corpus == "ci" else "demo_embeddings.npz")


def write_demo_embedding_fixture(root: Path, embedder: EmbedProvider) -> str:
    """Real recompute of the demo-scale embedding fixture; returns its sha256."""
    chunks = chunks_for(root, "demo")
    ids = [c.chunk_id for c in chunks]
    vectors = store.build(ids, [c.text for c in chunks], embedder)
    return store.save(embedding_fixture_path(root, "demo"), ids, vectors, embedder.name)


def graph_build(
    root: Path, provider: ChatProvider, *, corpus: str = "ci", concurrency: int = 4
) -> dict[str, object]:
    """Build one corpus's graph end to end; returns (and persists) honest stats."""
    from .extract import PROMPT_VERSION

    chunks = chunks_for(root, corpus)
    cache_dir = root / "fixtures" / "extraction" / corpus
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "README.md").write_text(
        CACHE_README.format(corpus=corpus, version=PROMPT_VERSION, regenerate=_REGENERATE[corpus])
    )
    cached_before = sum(1 for _ in cache_dir.glob("*.json"))
    t0 = time.monotonic()
    extractions = extract_many(chunks, provider, cache_dir=cache_dir, concurrency=concurrency)
    wall_s = time.monotonic() - t0
    graph = build_graph(extractions)
    communities = detect(graph)
    out_dir = root / "artifacts" / corpus
    save_graph(graph, out_dir / "graph.json")
    (out_dir / "communities.json").write_text(
        json.dumps({str(r): m for r, m in communities.items()}, indent=0, sort_keys=True) + "\n"
    )
    result: dict[str, object] = {
        "corpus": corpus,
        "chunks": len(chunks),
        "cached_before": cached_before,
        "extraction_wall_s": round(wall_s, 1),
        "truncated_chunks": sum(1 for e in extractions if e.truncated),
        **stats(graph),
        "communities_per_resolution": {
            str(r): len(set(m.values())) for r, m in communities.items()
        },
        "provider": provider.name,
    }
    (out_dir / "build_stats.json").write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    return result


def ci_graph_build(
    root: Path, provider: ChatProvider, concurrency: int = 4
) -> dict[str, object]:
    return graph_build(root, provider, corpus="ci", concurrency=concurrency)


def cache_complete(root: Path, corpus: str = "ci") -> bool:
    """Pure-local: does every current chunk have a committed extraction?
    (What lets CI rebuild the graph with no LLM — any drift is loud.)"""
    from .extract import _cache_key

    chunks = chunks_for(root, corpus)
    cache_dir = root / "fixtures" / "extraction" / corpus
    provider_name = "ollama-openai:llama3.1:8b"
    return all(
        (cache_dir / f"{_cache_key(c.chunk_id, provider_name)}.json").exists() for c in chunks
    )
