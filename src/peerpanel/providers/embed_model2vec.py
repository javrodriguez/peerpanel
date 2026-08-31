"""Pure-CPU static embeddings via model2vec — the second embedding backend.

No torch, no GPU, deterministic. The model downloads once from HuggingFace at
first use (network; no key), so tests touching it carry the network marker.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

DEFAULT_MODEL = "minishlab/potion-base-8M"


class Model2VecEmbed:
    def __init__(self, model_name: str = DEFAULT_MODEL, dim: int = 256) -> None:
        from model2vec import StaticModel

        self._model = StaticModel.from_pretrained(model_name)
        self._model_name = model_name
        self._dim = dim

    @property
    def name(self) -> str:
        return f"model2vec:{self._model_name}"

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts: list[str]) -> NDArray[np.float32]:
        out = np.asarray(self._model.encode(texts), dtype=np.float32)
        if out.shape != (len(texts), self._dim):
            raise ValueError(f"embedding shape {out.shape} != ({len(texts)}, {self._dim})")
        return out
