"""Evaluation harnesses: retrieval ablation and (later) planted-error detection."""

from .retrieval_eval import (
    N_GATE,
    QueryCase,
    build_cases,
    doc_ranking,
    ndcg_at_k,
    per_item_hits,
    rates_allowed,
    recall_at_k,
)

__all__ = [
    "N_GATE",
    "QueryCase",
    "build_cases",
    "doc_ranking",
    "ndcg_at_k",
    "per_item_hits",
    "rates_allowed",
    "recall_at_k",
]
