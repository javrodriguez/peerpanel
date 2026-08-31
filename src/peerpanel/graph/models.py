"""Graph extraction models: typed entities and relations per chunk."""

from __future__ import annotations

from pydantic import BaseModel

ENTITY_TYPES = (
    "gene_or_protein",
    "chemical",
    "organism",
    "method_or_tool",
    "process_or_phenotype",
    "other",
)


class Entity(BaseModel):
    name: str
    type: str


class Relation(BaseModel):
    source: str
    predicate: str
    target: str


class ChunkExtraction(BaseModel):
    chunk_id: str
    entities: list[Entity]
    relations: list[Relation]
    truncated: bool = False
