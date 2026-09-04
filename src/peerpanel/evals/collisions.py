"""Where a planted token could be QUOTED from instead of found — measured, not assumed.

A detection token that also occurs in the retrieval corpus can be quoted back by
either arm out of retrieved literature, minting a catch nobody earned ("decreased"
alone occurs in 37 of the 68 demo documents, which is why a direction token is a
phrase). The guard that pins this used to skip whenever the fetched corpus was
absent — which is every clean clone and every CI run, so the check the evaluation
leans on never actually ran (round-3 finding F12).

So the collision measurement is made once, against the fetched corpus, and COMMITTED
as `corpus/token_collisions.json`: a model-free derived artifact that
`tests/test_planted_soundness.py` reads unconditionally, and re-derives whenever the
corpus is present. `documents_scanned` is part of the record on purpose — an empty
corpus would otherwise produce an all-clear that means nothing.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from peerpanel.agents.reviewer_base import EXCERPT_WORDS
from peerpanel.evals.planted import plant_errors
from peerpanel.manuscripts.store import read_manuscript

ARTIFACT = "token_collisions.json"
GENERATED_FROM = "make corpus + plant_errors(SEED)"


def token_collisions(root: Path, corpus: str, subjects: list[str]) -> dict[str, Any]:
    """For every planted token of every subject, the corpus documents containing it.

    `subjects` are manuscript stems (a `.txt` name is accepted too). Each subject is
    planted exactly as the evaluation plants it — `plant_errors(body, name,
    window_words=EXCERPT_WORDS)` after `read_manuscript`, same seed, same window — so
    the tokens listed here are the tokens that get scored, not a re-derivation that
    could drift from them.

    An empty corpus directory raises rather than returning an empty all-clear: a
    collision report nobody could have failed is the defect this artifact exists to
    close.
    """
    docs = sorted((root / "corpus" / corpus).glob("*.txt"))
    if not docs:
        raise FileNotFoundError(
            f"no documents in {root / 'corpus' / corpus} — fetch the corpus first "
            "(`make corpus`); an empty scan would report a clean collision check "
            "that never looked at anything"
        )
    texts = {path.name: path.read_text(errors="ignore").lower() for path in docs}
    found: dict[str, dict[str, list[str]]] = {}
    for subject in subjects:
        name = subject if subject.endswith(".txt") else f"{subject}.txt"
        path = root / "manuscripts" / name
        _header, body = read_manuscript(path)
        planted = plant_errors(body, path.name, window_words=EXCERPT_WORDS)
        found[path.stem] = {
            error.detection_token: [
                doc for doc, text in sorted(texts.items()) if error.detection_token.lower() in text
            ]
            for error in planted.errors
        }
    return {
        "corpus": corpus,
        "documents_scanned": len(docs),
        "generated_from": GENERATED_FROM,
        "subjects": found,
    }


def write_token_collisions(root: Path, corpus: str, subjects: list[str]) -> Path:
    """Write `corpus/token_collisions.json` and return its path."""
    path = root / "corpus" / ARTIFACT
    path.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(token_collisions(root, corpus, subjects), indent=1, sort_keys=True)
    path.write_text(body + "\n")
    return path
