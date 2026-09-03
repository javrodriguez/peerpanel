"""Hybrid extraction: deterministic spaCy noun-chunk candidates + LLM typing.

The candidate pass is pure and CI-testable (the pinned en_core_web_sm wheel is
a regular dependency). The LLM pass types candidates and proposes relations at
temperature 0 with a JSON schema; results cache on disk keyed by
(chunk id, provider, prompt version), so re-runs are free and CI replays the
committed cache as labeled recorded fixtures.

Cost design (measured live 2026-08-31: ~80 s warm per uncapped chunk on
llama3.1:8b): entity/relation caps are enforced in the schema, the output
budget starts at 1024 tokens (measured capped output is ~500-600; prompt +
output must stay inside the serve slot's context) with ONE doubled retry on
truncated JSON, and a chunk that still fails is recorded honestly as
truncated=True (cached, counted, never a crash).
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

from peerpanel.providers.base import ChatProvider, TokenLedger
from peerpanel.text.chunks import Chunk

from .models import ENTITY_TYPES, ChunkExtraction, Entity, Relation

PROMPT_VERSION = 3  # v3: caps enforced in the SCHEMA (maxItems), not just asked
EXTRACTION_MODEL = "llama3.1:8b"
# The ONE cache identity every committed extraction is filed under: the model on the
# native wire, which sizes its window per chunk (DECISIONS.md D19). The replay stubs
# and the completeness check read it from here, so it cannot drift in three places.
EXTRACTION_PROVIDER_NAME = f"ollama-native:{EXTRACTION_MODEL}"

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
            "maxItems": MAX_ENTITIES,  # enforced by constrained decoding, not just asked
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
            "maxItems": MAX_RELATIONS,
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
    max_tokens: int = 1024,
    ledger: TokenLedger | None = None,
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
        if ledger is not None:
            ledger.record(response)
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
    ledger: TokenLedger | None = None,
) -> list[ChunkExtraction]:
    """Order-preserving parallel extraction (thread pool; clients are thread-safe).

    Dispatched longest chunk first, which is NOT an optimisation of this code but of the
    server underneath it: the window a call needs follows its prompt size, and Ollama
    reloads the runner whenever the window changes between calls. In corpus order the
    three window sizes interleave and the runner reloads hundreds of times across a build,
    each reload waiting for the in-flight calls to drain first. Longest-first walks the
    window sizes downward, so the reloads happen once per size.

    Order in, order out: results are returned in the caller's chunk order regardless of
    the order they were computed in, because the graph builder's relation pass depends on
    which entities exist by the time it reads each extraction.
    """
    # Results are keyed by chunk id below, which is what preserves the caller's order —
    # and is also how a repeated id would disappear: the dict would keep one extraction
    # and the returned list would hand it out twice, at exactly the right LENGTH, so
    # nothing downstream could notice. Ids carry a content hash, so a repeat means two
    # chunks with identical text AND index. Refuse rather than merge.
    if len({c.chunk_id for c in chunks}) != len(chunks):
        raise ValueError(
            f"{len(chunks)} chunks carry {len({c.chunk_id for c in chunks})} distinct ids — "
            "a chunk id repeats, and one extraction would stand in for another"
        )
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        by_size = sorted(chunks, key=lambda c: len(c.text), reverse=True)
        done = {
            e.chunk_id: e
            for e in pool.map(
                lambda c: extract_chunk(c, provider, cache_dir, ledger=ledger), by_size
            )
        }
    return [done[c.chunk_id] for c in chunks]
