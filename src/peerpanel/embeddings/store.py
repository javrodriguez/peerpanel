"""Embedding fixture store: committed f16 vectors + a sha256 manifest.

CI has no Ollama, so the pinned CI corpus's embeddings are precomputed here,
committed, and hash-manifested; `make embeddings` regenerates and diffs the
hash (a drifted fixture fails loudly). f16 storage measured 1.000 top-10
ranking overlap vs f32 in the design research — the README says all of this
plainly, which is what keeps a committed fixture honest.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from peerpanel.providers.base import EmbedProvider


def build(
    chunk_ids: list[str], texts: list[str], provider: EmbedProvider, batch_size: int = 32
) -> NDArray[np.float16]:
    if len(chunk_ids) != len(texts):
        raise ValueError("chunk_ids and texts must align")
    parts: list[NDArray[np.float32]] = []
    for start in range(0, len(texts), batch_size):
        parts.append(provider.embed(texts[start : start + batch_size]))
    return np.concatenate(parts).astype(np.float16)


def save(path: Path, chunk_ids: list[str], vectors: NDArray[np.float16], provider_name: str) -> str:
    """Write the .npz + manifest; returns the sha256 of the vector payload."""
    if vectors.dtype != np.float16:
        raise ValueError("fixtures are stored f16")
    if len(chunk_ids) != vectors.shape[0]:
        raise ValueError("one vector per chunk id")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, chunk_ids=np.array(chunk_ids), vectors=vectors)
    digest = hashlib.sha256(vectors.tobytes()).hexdigest()
    manifest = {
        "provider": provider_name,
        "n_vectors": int(vectors.shape[0]),
        "dim": int(vectors.shape[1]),
        "dtype": "float16",
        "sha256": digest,
        "regenerate": "make embeddings",
    }
    path.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    return digest


def load(path: Path) -> tuple[list[str], NDArray[np.float16]]:
    data = np.load(path, allow_pickle=False)
    return [str(c) for c in data["chunk_ids"]], data["vectors"].astype(np.float16)


def check(path: Path) -> bool:
    """Does the stored payload still match its manifest hash?"""
    manifest = json.loads(path.with_suffix(".manifest.json").read_text())
    _, vectors = load(path)
    return bool(hashlib.sha256(vectors.tobytes()).hexdigest() == manifest["sha256"])
