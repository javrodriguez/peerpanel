"""Deterministic chunking: paragraph-packed windows with stable content-hashed ids."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

_PARA_SPLIT = re.compile(r"\n\s*\n")


@dataclass(frozen=True)
class Chunk:
    chunk_id: str  # "<doc_id>:<index>:<sha8 of text>"
    doc_id: str
    index: int
    text: str

    @property
    def n_words(self) -> int:
        return len(self.text.split())


def _sha8(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:8]


def chunk_document(
    doc_id: str, text: str, *, target_words: int = 900, overlap_paras: int = 1
) -> list[Chunk]:
    """Pack paragraphs into ~target_words windows, overlapping by whole paragraphs.

    Deterministic for identical input; chunk ids carry a content hash so any
    text drift is visible in every downstream artifact.

    The 900-word default (~1200 tokens) is a measured cost decision, not taste:
    graph extraction pays one LLM call per chunk, and one chunking is shared by
    extraction, embeddings and retrieval so entity->chunk->vector joins stay on
    one id space (DECISIONS.md D7 has the arithmetic).
    """
    paras = [p.strip() for p in _PARA_SPLIT.split(text) if p.strip()]
    chunks: list[Chunk] = []
    window: list[str] = []
    count = 0

    def flush() -> None:
        nonlocal window, count
        if not window:
            return
        body = "\n\n".join(window)
        chunks.append(Chunk(f"{doc_id}:{len(chunks)}:{_sha8(body)}", doc_id, len(chunks), body))
        window = window[-overlap_paras:] if overlap_paras else []
        count = sum(len(p.split()) for p in window)

    for para in paras:
        words = len(para.split())
        if count + words > target_words and window:
            flush()
        window.append(para)
        count += words
    flush()
    return chunks
