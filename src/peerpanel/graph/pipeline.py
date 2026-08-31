"""CI-corpus graph pipeline: the SAME chunks the embedding fixture uses
(peerpanel.embeddings.pipeline.ci_chunks — shared ids are what let local
search join entities to vectors), extracted through the cached LLM road,
assembled and partitioned, persisted under artifacts/.

The extraction cache under fixtures/extraction/ci is committed: real recorded
llama3.1:8b outputs, regenerated for real by `make graph` (see the README
beside the cache). CI rebuilds graph + communities from the cache with no LLM.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from peerpanel.embeddings.pipeline import ci_chunks
from peerpanel.providers.base import ChatProvider

from .build import build_graph, save_graph, stats
from .communities import detect
from .extract import extract_many

EXTRACTION_CACHE = Path("fixtures") / "extraction" / "ci"
ARTIFACTS = Path("artifacts") / "ci"

CACHE_README = """\
# CI extraction cache — real recorded fixtures

Every file here is the REAL output of the extraction LLM (llama3.1:8b via the
provider seam, temperature 0, prompt v{version}) over one committed CI-corpus
chunk, produced by an actual call and cached by (chunk id, provider, prompt
version). Nothing is hand-written. Regenerate for real: `make graph` with
Ollama running (a changed prompt version re-extracts everything).
"""


def ci_graph_build(
    root: Path, provider: ChatProvider, concurrency: int = 4
) -> dict[str, object]:
    """Build the CI graph end to end; returns (and persists) honest stats."""
    chunks, _ = ci_chunks(root)
    cache_dir = root / EXTRACTION_CACHE
    cache_dir.mkdir(parents=True, exist_ok=True)
    from .extract import PROMPT_VERSION

    (cache_dir / "README.md").write_text(CACHE_README.format(version=PROMPT_VERSION))
    cached_before = sum(1 for _ in cache_dir.glob("*.json"))
    t0 = time.monotonic()
    extractions = extract_many(chunks, provider, cache_dir=cache_dir, concurrency=concurrency)
    wall_s = time.monotonic() - t0
    graph = build_graph(extractions)
    communities = detect(graph)
    out_dir = root / ARTIFACTS
    save_graph(graph, out_dir / "graph.json")
    (out_dir / "communities.json").write_text(
        json.dumps({str(r): m for r, m in communities.items()}, indent=0, sort_keys=True) + "\n"
    )
    result: dict[str, object] = {
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


def cache_complete(root: Path) -> bool:
    """Pure-local: does every current CI chunk have a committed extraction?
    (What lets CI rebuild the graph with no LLM — any drift is loud.)"""
    from .extract import _cache_key

    chunks, _ = ci_chunks(root)
    cache_dir = root / EXTRACTION_CACHE
    provider_name = "ollama-openai:llama3.1:8b"
    return all(
        (cache_dir / f"{_cache_key(c.chunk_id, provider_name)}.json").exists() for c in chunks
    )
