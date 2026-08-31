"""Hybrid extraction: deterministic spaCy noun-chunk candidates + LLM typing.

The candidate pass is pure and CI-testable (the pinned en_core_web_sm wheel is
a regular dependency). The LLM pass types candidates and proposes relations at
temperature 0 with a JSON schema; results cache on disk keyed by
(chunk id, provider, prompt version), so re-runs are free and CI replays the
committed cache as labeled recorded fixtures.

Cost design (measured live 2026-08-31: ~80 s warm per uncapped chunk on
llama3.1:8b): the prompt caps entities/relations, output budget starts at
3072 tokens with ONE doubled retry on truncated JSON, and a chunk that still
fails is recorded honestly as truncated=True (cached, counted, never a crash).
extract_many runs chunks on a thread pool — Ollama serves parallel requests.
"""

from __future__ import annotations

import hashlib
import json
import re
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path
from typing import Any

from peerpanel.providers.base import ChatProvider
from peerpanel.text.chunks import Chunk

from .models import ENTITY_TYPES, ChunkExtraction, Entity, Relation

PROMPT_VERSION = 2  # v2: entity/relation caps in the prompt

MAX_ENTITIES = 12
MAX_RELATIONS = 12

_WORD = re.compile(r"[A-Za-z]")

SYSTEM_PROMPT = (
    "You extract a knowledge graph from scientific text about molecular biology. "
    "Given a passage and candidate terms, return entities (typed) and relations "
    "(subject-predicate-object, short predicates) that the passage explicitly supports. "
    "Only include entities grounded in the passage. Use candidate terms where correct, "
    "but fix their boundaries and add clearly-present entities the candidates missed. "
    f"Return AT MOST the {MAX_ENTITIES} most important entities and "
    f"{MAX_RELATIONS} most important relations."
)

EXTRACTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "type": {"type": "string", "enum": list(ENTITY_TYPES)},
                },
                "required": ["name", "type"],
            },
        },
        "relations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "source": {"type": "string"},
                    "predicate": {"type": "string"},
                    "target": {"type": "string"},
                },
                "required": ["source", "predicate", "target"],
            },
        },
    },
    "required": ["entities", "relations"],
}


@lru_cache(maxsize=1)
def _nlp() -> Any:
    import spacy

    return spacy.load("en_core_web_sm", exclude=["ner", "lemmatizer"])


def candidate_terms(text: str) -> list[str]:
    """Deterministic noun-chunk candidates, deduplicated in document order.

    Determiners are stripped, casing is preserved (gene symbols are
    case-bearing), and chunks without alphabetic content are dropped.
    """
    seen: set[str] = set()
    out: list[str] = []
    for chunk in _nlp()(text).noun_chunks:
        tokens = [t for t in chunk if t.pos_ not in ("DET", "PRON")]
        term = " ".join(t.text for t in tokens).strip()
        if len(term) < 3 or not _WORD.search(term):
            continue
        key = term.lower()
        if key not in seen:
            seen.add(key)
            out.append(term)
    return out


def _cache_key(chunk_id: str, provider_name: str) -> str:
    raw = f"{chunk_id}|{provider_name}|v{PROMPT_VERSION}"
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


def _parse(text: str) -> dict[str, Any] | None:
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _hygiene(chunk_id: str, data: dict[str, Any]) -> ChunkExtraction:
    """Deterministic post-pass: schema-typed entities only, relations must join
    known entities (dangling endpoints are dropped, not invented)."""
    entities = [
        Entity(name=e["name"].strip(), type=e["type"])
        for e in data.get("entities", [])
        if isinstance(e, dict) and e.get("name", "").strip() and e.get("type") in ENTITY_TYPES
    ]
    names = {e.name.lower() for e in entities}
    relations = [
        Relation(
            source=r["source"].strip(),
            predicate=r["predicate"].strip(),
            target=r["target"].strip(),
        )
        for r in data.get("relations", [])
        if isinstance(r, dict)
        and r.get("source", "").strip().lower() in names
        and r.get("target", "").strip().lower() in names
        and r.get("predicate", "").strip()
    ]
    return ChunkExtraction(chunk_id=chunk_id, entities=entities, relations=relations)


def extract_chunk(
    chunk: Chunk,
    provider: ChatProvider,
    cache_dir: Path | None = None,
    max_tokens: int = 3072,
) -> ChunkExtraction:
    """Extract one chunk's graph fragment; disk-cached when cache_dir is given.

    Truncated/invalid JSON gets ONE retry at double the output budget; a chunk
    that still fails is returned (and cached) as truncated=True with no
    entities — an honest, countable gap rather than a crash or an invention.
    """
    cache_path = None
    if cache_dir is not None:
        cache_path = cache_dir / f"{_cache_key(chunk.chunk_id, provider.name)}.json"
        if cache_path.exists():
            return ChunkExtraction.model_validate_json(cache_path.read_text())
    candidates = candidate_terms(chunk.text)
    user = (
        f"PASSAGE:\n{chunk.text}\n\n"
        f"CANDIDATE TERMS (from a deterministic parser; correct and extend):\n"
        f"{json.dumps(candidates)}"
    )

    def _call(budget: int) -> dict[str, Any] | None:
        response = provider.chat(
            system=SYSTEM_PROMPT,
            user=user,
            json_schema=EXTRACTION_SCHEMA,
            temperature=0.0,
            max_tokens=budget,
        )
        return _parse(response.text)

    data = _call(max_tokens)
    if data is None:
        data = _call(max_tokens * 2)
    if data is None:
        extraction = ChunkExtraction(
            chunk_id=chunk.chunk_id, entities=[], relations=[], truncated=True
        )
    else:
        extraction = _hygiene(chunk.chunk_id, data)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(extraction.model_dump_json(indent=1))
    return extraction


def extract_many(
    chunks: list[Chunk],
    provider: ChatProvider,
    cache_dir: Path | None = None,
    concurrency: int = 4,
) -> list[ChunkExtraction]:
    """Order-preserving parallel extraction (thread pool; clients are thread-safe)."""
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        return list(pool.map(lambda c: extract_chunk(c, provider, cache_dir), chunks))
