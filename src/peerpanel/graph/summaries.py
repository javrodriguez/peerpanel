"""Community reports: the LLM half of GraphRAG's preprocessing.

Each Leiden community (>= MIN_MEMBERS) gets a grounded report — title +
summary written only from the community's own entities and intra-community
relations. Reports are content-addressed on disk (members + edges + prompt
version + provider), so an unchanged community never re-summarises and CI
replays the committed cache as labeled recorded fixtures. Global search
map-reduces over these reports.

Hand `summarise_communities` a `TokenLedger` and this layer becomes visible from
its own record: `graph summaries --publish` writes
`results/summaries-stats-<corpus>.json` out of that ledger — the wire and model
that served the reports, the calls THIS run made, the per-call margin between each
prompt and the window it ran in, and the source that window was read from. Until
round 3 of the review found it, the one model layer feeding graphrag-global's
ranking, on the one wire that cannot set its own window, shipped no record of its
run conditions at all (requirement 8).

The ledger counts a CALL, never a report: a report served from the cache above
makes no call and is counted nowhere. That is why the committed records are cold
runs — empty `fixtures/summaries/<corpus>` first, as the Makefile target says. A
warm run publishes `calls: 0`, which is honest and is evidence of nothing.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import networkx as nx
from pydantic import BaseModel

from peerpanel.providers.base import ChatProvider, TokenLedger

from .build import display_name, node_type
from .communities import members

PROMPT_VERSION = 1
# The community reports ride the OpenAI-compatible wire on purpose (DECISIONS.md D3):
# every report prompt fits the daemon's default window by construction, and it keeps
# a second wire exercised on the same local model. This is the cache identity the
# committed reports are filed under; the replay stubs present it.
SUMMARY_MODEL = "llama3.1:8b"
SUMMARY_PROVIDER_NAME = f"ollama-openai:{SUMMARY_MODEL}"
MIN_MEMBERS = 3
MAX_MEMBERS_IN_PROMPT = 30

SYSTEM_PROMPT = (
    "You write a community report for a cluster of entities from a scientific "
    "knowledge graph. Using ONLY the entities and relations given, produce a short "
    "title (<= 12 words) and a summary paragraph (<= 150 words) describing what this "
    "cluster is about and how its members relate. Never introduce facts not present "
    "in the input."
)

REPORT_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "summary": {"type": "string"},
    },
    "required": ["title", "summary"],
}

SUMMARY_CACHE_README = """\
# {corpus} community-report cache — real recorded fixtures

Every file here is the REAL output of the summarisation LLM (llama3.1:8b via
the provider seam, temperature 0, prompt v{version}) for one Leiden community,
produced by an actual call. Nothing is hand-written.

Cached by the CONTENT of the community (its member entities and the relations
between them) plus the provider and prompt version — so a community whose
membership changes re-summarises, and one that does not is served from here.
That is also why the file count can exceed the report count of any single run:
a community that changed shape leaves its earlier report behind, keyed to the
membership that produced it.

Regenerate for real: `{regenerate}` with Ollama running.

These reports are the residual contamination surface LIMITATIONS.md names: they
are generated once over the whole corpus, so a run that excludes a document can
still be influenced by a report describing it. The LLM face of global search
refuses to run when anything is excluded, for exactly that reason.
"""


def write_cache_readme(cache_dir: Path, corpus: str, regenerate: str) -> None:
    """Every recorded-fixture directory says what it is and how to remake it."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "README.md").write_text(
        SUMMARY_CACHE_README.format(corpus=corpus, version=PROMPT_VERSION, regenerate=regenerate)
    )


class CommunityReport(BaseModel):
    community_id: int
    resolution: float
    size: int
    member_names: list[str]
    title: str
    summary: str
    truncated: bool = False


def _community_payload(g: nx.Graph[str], nodes: list[str]) -> tuple[list[str], list[str]]:
    """Entity lines + relation lines for the prompt, deterministic order."""
    ranked = sorted(nodes, key=lambda n: (-g.degree(n, weight="weight"), n))
    kept = ranked[:MAX_MEMBERS_IN_PROMPT]
    entity_lines = [f"{display_name(g, n)} ({node_type(g, n)})" for n in kept]
    kept_set = set(kept)
    relation_lines: list[str] = []
    from .build import _top

    for a, b, attrs in g.edges(data=True):
        if a in kept_set and b in kept_set and attrs.get("predicates"):
            predicate = _top(attrs["predicates"])
            relation_lines.append(f"{display_name(g, a)} —{predicate}→ {display_name(g, b)}")
    return entity_lines, sorted(relation_lines)


def _cache_key(entity_lines: list[str], relation_lines: list[str], provider_name: str) -> str:
    raw = json.dumps([entity_lines, relation_lines, provider_name, PROMPT_VERSION])
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


def summarise_communities(
    g: nx.Graph[str],
    assignment: dict[str, int],
    resolution: float,
    provider: ChatProvider,
    cache_dir: Path | None = None,
    ledger: TokenLedger | None = None,
) -> list[CommunityReport]:
    reports: list[CommunityReport] = []
    for cid, nodes in sorted(members(assignment).items()):
        if len(nodes) < MIN_MEMBERS:
            continue
        entity_lines, relation_lines = _community_payload(g, nodes)
        cache_path = None
        if cache_dir is not None:
            key = _cache_key(entity_lines, relation_lines, provider.name)
            cache_path = cache_dir / f"{key}.json"
            if cache_path.exists():
                cached = CommunityReport.model_validate_json(cache_path.read_text())
                reports.append(
                    cached.model_copy(update={"community_id": cid, "resolution": resolution})
                )
                continue
        user = (
            "ENTITIES:\n" + "\n".join(entity_lines) + "\n\n"
            "RELATIONS:\n" + ("\n".join(relation_lines) or "(none recorded)")
        )
        response = provider.chat(
            system=SYSTEM_PROMPT,
            user=user,
            json_schema=REPORT_SCHEMA,
            temperature=0.0,
            max_tokens=1024,
        )
        # The one model layer that shipped no run record until round 3 found it: these
        # calls ride the OpenAI wire, which cannot set its window, and they are the only
        # input to graphrag-global's ranking. A cached report makes no call and records
        # nothing, which is correct — the record describes what THIS run sent.
        if ledger is not None:
            ledger.record(response)
        try:
            data = json.loads(response.text)
            report = CommunityReport(
                community_id=cid,
                resolution=resolution,
                size=len(nodes),
                member_names=[display_name(g, n) for n in nodes],
                title=str(data.get("title", "")).strip(),
                summary=str(data.get("summary", "")).strip(),
            )
        except json.JSONDecodeError:
            report = CommunityReport(
                community_id=cid,
                resolution=resolution,
                size=len(nodes),
                member_names=[display_name(g, n) for n in nodes],
                title="",
                summary="",
                truncated=True,
            )
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(report.model_dump_json(indent=1))
        reports.append(report)
    return reports
