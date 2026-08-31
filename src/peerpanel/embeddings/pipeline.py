"""CI-corpus embedding fixture pipeline: manifest -> sanitize -> chunk -> embed -> store.

The committed fixture is real output of the pinned embedder (nomic-embed-text
via Ollama) over the committed CI corpus. CI has no Ollama, so CI verifies the
committed bytes two ways without a model: the vector payload matches its
sha256 manifest, and the stored chunk ids match a fresh sanitize+chunk pass
over the committed corpus (any drift in corpus text, sanitizer or chunker is
loud). The live road regenerates for real: `make embeddings`.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from peerpanel.corpus.models import CorpusManifest
from peerpanel.embeddings import store
from peerpanel.providers.base import EmbedProvider
from peerpanel.text.chunks import Chunk, chunk_document
from peerpanel.text.sanitize import Finding, sanitize

FIXTURE = Path("fixtures") / "ci_embeddings.npz"
CI_MANIFEST = Path("corpus") / "ci.manifest.json"
CI_TEXTS = Path("corpus") / "ci"


def ci_chunks(root: Path) -> tuple[list[Chunk], list[tuple[str, Finding]]]:
    """Sanitize + chunk every CI-corpus doc, in pmcid order. Deterministic."""
    manifest = CorpusManifest.load(root / CI_MANIFEST)
    chunks: list[Chunk] = []
    findings: list[tuple[str, Finding]] = []
    for doc in sorted(manifest.docs, key=lambda d: d.pmcid):
        raw = (root / CI_TEXTS / f"{doc.pmcid}.txt").read_text(encoding="utf-8")
        clean, found = sanitize(raw)
        findings.extend((doc.pmcid, f) for f in found)
        chunks.extend(chunk_document(doc.pmcid, clean))
    return chunks, findings


def compute(root: Path, provider: EmbedProvider) -> tuple[list[str], NDArray[np.float16]]:
    """Embed the current CI chunks (live model call — needs the provider up)."""
    chunks, _ = ci_chunks(root)
    ids = [c.chunk_id for c in chunks]
    return ids, store.build(ids, [c.text for c in chunks], provider)


def write(root: Path, provider: EmbedProvider) -> str:
    """Regenerate and persist the fixture; returns the payload sha256."""
    ids, vectors = compute(root, provider)
    return store.save(root / FIXTURE, ids, vectors, provider.name)


def check_live(root: Path, provider: EmbedProvider) -> tuple[bool, str, str]:
    """Recompute live and diff against the committed manifest hash.

    Returns (match, recomputed_sha256, committed_sha256) — nothing is written.
    """
    _, vectors = compute(root, provider)
    manifest = json.loads((root / FIXTURE).with_suffix(".manifest.json").read_text())
    recomputed = hashlib.sha256(vectors.tobytes()).hexdigest()
    committed: str = manifest["sha256"]
    return recomputed == committed, recomputed, committed


def check_stored(root: Path) -> bool:
    """Pure-local fixture truth (no model): bytes match the manifest hash AND
    the stored chunk ids match a fresh pass over the committed corpus."""
    if not store.check(root / FIXTURE):
        return False
    stored_ids, _ = store.load(root / FIXTURE)
    current_ids = [c.chunk_id for c in ci_chunks(root)[0]]
    return stored_ids == current_ids
