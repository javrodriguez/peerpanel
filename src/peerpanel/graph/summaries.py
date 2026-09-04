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

The ledger counts a CALL, never a report, so the record carries `SummaryRunStats`
beside it: `reports_generated` and `reports_from_cache`, which always sum to
`reports`. Both are needed. The cache is keyed by a community's CONTENT and not by
its resolution, and the same community recurs across Leiden levels, so even a run
that starts with an empty cache reads back entries it wrote itself minutes earlier
— the committed CI record is 110 reports from 75 distinct communities and 75 calls.
A reader comparing 75 against 110 with no other field would call that a warm run.
The committed records are still cold runs — empty `fixtures/summaries/<corpus>`
first, as the Makefile target says — and a genuinely warm one publishes `calls: 0`
with `reports_generated: 0`, which is honest and is evidence of nothing.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
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


@dataclass
class SummaryRunStats:
    """How this run produced its reports: generated by a call, or read from cache.

    Both numbers are needed because `calls` alone cannot say whether a run was cold.
    The report cache is keyed by a community's CONTENT (`_cache_key` takes the entity
    and relation lines and the provider, and deliberately not the resolution), and the
    same community recurs across Leiden resolutions — so even a run that starts with an
    empty cache serves some reports from entries it wrote itself, minutes earlier. The
    committed CI record is the example: 110 reports, 75 distinct communities, 75 calls.
    Without these two fields a reader comparing 75 against 110 would read a cold run as
    a warm one, which is the class of question requirement 8 exists to make answerable
    from the record alone.
    """

    generated: int = 0
    from_cache: int = 0


def summarise_communities(
    g: nx.Graph[str],
    assignment: dict[str, int],
    resolution: float,
    provider: ChatProvider,
    cache_dir: Path | None = None,
    ledger: TokenLedger | None = None,
    stats: SummaryRunStats | None = None,
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
                # The cache key is the PROMPT — the top MAX_MEMBERS_IN_PROMPT members by
                # degree and the relations among them — so two communities that differ
                # only in their tail share an entry. That is right for the prose, which
                # was written from exactly those top members and says nothing about the
                # tail. It is wrong for anything describing WHICH community this is, so
                # every such field is taken from the community being served and never
                # from the entry: `graphrag-global` ranks over `member_names`, and
                # LIMITATIONS counts residual contamination in it, so a report carrying
                # a neighbouring community's membership would put a wrong list under
                # both. Measured on the CI corpus before this was fixed: 2 of 35 cache
                # hits carried a member list belonging to a different community — 255
                # members served under a community of 250, and 111 under one of 102.
                reports.append(
                    cached.model_copy(
                        update={
                            "community_id": cid,
                            "resolution": resolution,
                            "member_names": [display_name(g, n) for n in nodes],
                            "size": len(nodes),
                        }
                    )
                )
                if stats is not None:
                    stats.from_cache += 1
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
        if stats is not None:
            stats.generated += 1
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
