"""Committed query-embedding fixtures.

The evaluation's queries are deterministic — a manuscript's title plus a fixed
prefix of its body — so their embeddings can be recorded once and committed,
exactly as the chunk-embedding fixture is. That removes the last live-model
dependency from the ablation, so a stranger can reproduce every retrieval
number from a clean clone with no GPU.

These are REAL recorded outputs of the pinned embedder, keyed by the sha256 of
the query text: change the query and the key misses, which fails loudly rather
than silently serving the wrong vector.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from peerpanel.providers.base import EmbedProvider

FIXTURE = Path("fixtures") / "query_embeddings.npz"
MANIFEST = Path("fixtures") / "query_embeddings.manifest.json"


def key(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:32]


class CachedQueryEmbedder:
    """An EmbedProvider serving committed query vectors; misses are loud.

    `live` is used only when regenerating the fixture — during a normal run it
    is None and an unknown query raises rather than quietly reaching for a model
    that a clean clone does not have.
    """

    dim = 768

    def __init__(self, root: Path, live: EmbedProvider | None = None) -> None:
        self._root = root
        self._live = live
        path = root / FIXTURE
        if path.exists():
            data = np.load(path, allow_pickle=False)
            self._vectors = {
                str(k): v for k, v in zip(data["keys"], data["vectors"], strict=True)
            }
        else:
            self._vectors = {}

    @property
    def name(self) -> str:
        return "query-fixture:nomic-embed-text"

    def embed(self, texts: list[str]) -> NDArray[np.float32]:
        out = []
        for text in texts:
            cached = self._vectors.get(key(text))
            if cached is None:
                if self._live is None:
                    raise KeyError(
                        f"no committed embedding for this query (key {key(text)}). "
                        "The eval queries are fixed, so this means the query text "
                        "changed: regenerate with `make query-embeddings`."
                    )
                cached = self._live.embed([text])[0]
                self._vectors[key(text)] = cached.astype(np.float16)
            out.append(np.asarray(cached, dtype=np.float32))
        return np.asarray(out, dtype=np.float32)

    def save(self, provider_name: str) -> str:
        keys = sorted(self._vectors)
        vectors = np.asarray([self._vectors[k] for k in keys], dtype=np.float16)
        path = self._root / FIXTURE
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, keys=np.array(keys), vectors=vectors)
        digest = hashlib.sha256(vectors.tobytes()).hexdigest()
        (self._root / MANIFEST).write_text(
            json.dumps(
                {
                    "provider": provider_name,
                    "n_queries": len(keys),
                    "dim": int(vectors.shape[1]) if len(keys) else 0,
                    "dtype": "float16",
                    "sha256": digest,
                    "regenerate": "make query-embeddings",
                },
                indent=1,
            )
            + "\n"
        )
        return digest
