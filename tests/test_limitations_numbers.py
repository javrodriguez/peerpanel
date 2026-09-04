"""Every self-critical figure in LIMITATIONS.md, bound to committed bytes.

`LIMITATIONS.md` is the file the evaluator brief singles out and the one the README
calls the part of the repo most worth reading carefully. Round 3 found four numbers
wrong in it, all in the same direction: the random-selection floor understated by 11
points, the attribution coverage understating work the repository had actually done,
the residual-contamination figure stating a complement with the sense inverted, and one
of its two named examples not a member of the set it illustrates (report-1 F6, F8a;
report-2 F5). The brief is explicit that an understatement in a self-weakness section
is worse than the original error — so these figures need binding more than the
flattering ones do, not less.

Every value here is RECOMPUTED from committed bytes, offline, with no model and no
fetched corpus: the graph from `fixtures/extraction/demo`, the community reports from
`fixtures/summaries/demo`, the relevant set from `corpus/ground_truth.json` against
`corpus/demo.manifest.json`, the attribution counts from the two citation files. The
route through `build_run_graph` that LIMITATIONS.md used to name cannot run from a
clean clone — `corpus/demo/` is gitignored and fetched — which is why the figures were
unreproducible in the first place.

The sentence shapes this file reads (the prose keeps its freedom everywhere else):

* `... 52.9% of the demo corpus is relevant ...`, optionally with `(36 of 68)`
* `... attribution for 83 of 83 documents ...`
* `... 101 graph entities exist only because of the excluded twin ...`
* `... 101 of them appear in the community reports ...`
* `... at most 5 ... prose ...`
* `... 93.1% of the demo graph's edges (58,797 of 63,130) are co-mention only ...`
* every entity named as an example, in double quotes, is a member of the twin-only set

Each binding asserts it matched a real string in the real file: a guard over zero
strings is round 3's defect class, and this file exists because of it. Every failure
message carries the full recomputed set, so the prose can be written from the test's
own output rather than from a number somebody remembers.
"""

from __future__ import annotations

import json
import re
from functools import cache
from pathlib import Path

import pytest

from peerpanel.corpus.models import CorpusManifest

# The graph's own key normaliser: an example entity is compared to the node keys the
# same way the graph built them, never by eyeballing the surface form.
from peerpanel.graph.build import _norm as norm_entity
from peerpanel.graph.build import build_graph
from peerpanel.graph.models import ChunkExtraction

ROOT = Path(__file__).resolve().parents[1]
LIMITATIONS_PATH = ROOT / "LIMITATIONS.md"
# Bold and italic markers are formatting: "**42%**" and "42%" are the same claim, and a
# guard that reads one and not the other is a guard the next edit switches off. The
# underscore is NOT stripped — it is part of `member_names` and `build_run_graph`, and
# removing it would silently un-bind every rule that names a field or a function.
PROSE = LIMITATIONS_PATH.read_text().replace("*", "")
TWIN = "PMC10729969"


@cache
def _demo_graphs() -> tuple[frozenset[str], int, int, int]:
    """(twin-only entities, total edges, co-mention-only edges, relation-carrying edges).

    Both graphs are merges over the committed per-chunk extractions — a pure
    aggregation, no model, no corpus text — so this runs in a clean clone.
    """
    records = [
        ChunkExtraction.model_validate_json(path.read_text())
        for path in sorted((ROOT / "fixtures" / "extraction" / "demo").glob("*.json"))
    ]
    assert records, "no committed demo extractions — every figure here would be vacuous"
    full = build_graph(records)
    without_twin = build_graph([r for r in records if not r.chunk_id.startswith(f"{TWIN}:")])
    total = full.number_of_edges()
    co_only = sum(1 for _a, _b, at in full.edges(data=True) if not at["predicates"])
    return frozenset(set(full) - set(without_twin)), total, co_only, total - co_only


@cache
def _report_text() -> tuple[frozenset[str], str]:
    """(every entity named in a community report's members, the reports' own prose).

    `graphrag-global` ranks over `title + summary + member_names`, so membership in the
    member list is what puts a twin-only entity into the text retrieval scores — the
    honest measure of the residue, and the one report-2 F5 established.
    """
    members: set[str] = set()
    prose: list[str] = []
    for path in sorted((ROOT / "fixtures" / "summaries" / "demo").glob("*.json")):
        report = json.loads(path.read_text())
        members |= {norm_entity(name) for name in report["member_names"]}
        prose.append(f"{report['title']} {report['summary']}".lower())
    assert members, "no committed demo community reports — the residue measure is vacuous"
    return frozenset(members), " ".join(prose)


@cache
def _figures() -> dict[str, float]:
    """Every figure LIMITATIONS.md publishes about itself, recomputed."""
    twin_only, edges, co_only, with_relation = _demo_graphs()
    members, prose = _report_text()
    ground_truth = json.loads((ROOT / "corpus" / "ground_truth.json").read_text())["manuscripts"]
    demo_pmcids = CorpusManifest.load(ROOT / "corpus" / "demo.manifest.json").pmcids()
    relevant: set[str] = set()
    for entry in ground_truth.values():
        relevant |= set(entry["cited_pmcids"]) & demo_pmcids
    demo_pinned = (ROOT / "corpus" / "DEMO_CITATIONS.md").read_text().count("Pinned:")
    ci_pinned = (ROOT / "CITATIONS.md").read_text().count("Pinned:")
    return {
        "relevant_docs": len(relevant),
        "corpus_docs": len(demo_pmcids),
        "relevant_share": 100 * len(relevant) / len(demo_pmcids),
        "attributed": demo_pinned + ci_pinned,
        "documents_touched": len(demo_pmcids) + len(CorpusManifest.load(
            ROOT / "corpus" / "ci.manifest.json"
        ).docs),
        "twin_only": len(twin_only),
        "twin_only_in_members": len(twin_only & members),
        "twin_only_in_prose": sum(1 for entity in twin_only if entity in prose),
        "edges": edges,
        "co_mention_only": co_only,
        "co_mention_share": 100 * co_only / edges,
        "with_relation": with_relation,
    }


def _summary() -> str:
    """The recomputed figures, in the shape the prose needs them. Every assertion
    message carries this, so LIMITATIONS.md is written from the measurement."""
    f = _figures()
    return (
        "\n\nRECOMPUTED FROM COMMITTED BYTES:\n"
        f"  relevant share      {f['relevant_docs']:.0f} of {f['corpus_docs']:.0f} "
        f"= {f['relevant_share']:.1f}%\n"
        f"  attribution         {f['attributed']:.0f} of {f['documents_touched']:.0f} "
        "documents\n"
        f"  twin-only entities  {f['twin_only']:.0f}\n"
        f"    in member_names   {f['twin_only_in_members']:.0f}\n"
        f"    in report prose   {f['twin_only_in_prose']:.0f} (substring test, an over-count)\n"
        f"  co-mention only     {f['co_mention_only']:,.0f} of {f['edges']:,.0f} "
        f"= {f['co_mention_share']:.1f}%\n"
        f"  carrying a relation {f['with_relation']:,.0f}\n"
    )


def _slack(shown: str) -> float:
    """Half of the last digit the claim published — the honest rounding window."""
    decimals = len(shown.split(".")[1]) if "." in shown else 0
    return 0.5 * 10.0**-decimals + 1e-9


def _matches(pattern: re.Pattern[str], what: str, shape: str) -> list[re.Match[str]]:
    """Every match in the real file, and never zero of them."""
    found = list(pattern.finditer(PROSE))
    assert found, (
        f"LIMITATIONS.md states no {what}, so this binding covers nothing — round 3's "
        f"defect class (report-2 F6). The sentence shape read here is: {shape}"
        + _summary()
    )
    return found


def _bullet_around(position: int) -> str:
    """The list item a match sits in — LIMITATIONS.md is a document of bullets."""
    start = PROSE.rfind("\n- ", 0, position)
    end = PROSE.find("\n- ", position)
    return PROSE[start if start != -1 else 0 : end if end != -1 else len(PROSE)]


RELEVANT_SHARE = re.compile(r"(\d+(?:\.\d+)?)\s*%[^.]{0,60}of the demo corpus is relevant", re.I)
COUNTED_PAIR = re.compile(r"(\d[\d,]*) of (?:the )?(\d[\d,]*)")
ATTRIBUTION = re.compile(r"attribution for (\d[\d,]*) of (?:the )?(\d[\d,]*)", re.I)
TWIN_ONLY = re.compile(
    r"(\d+)\s+(?:graph\s+)?entities[^.]{0,40}only because of the (?:excluded )?twin", re.I
)
IN_REPORTS = re.compile(
    r"(\d+)\s+of (?:them|those|the \d+)[^.]{0,60}appear[^.]{0,80}"
    r"(?:community report|member_names|member names|report)",
    re.I,
)
IN_PROSE = re.compile(r"(?:at most\s+)?(\d+)[^.]{0,80}\bprose\b", re.I)
CO_SHARE = re.compile(r"(\d+(?:\.\d+)?)\s*%\s+of the demo graph'?s edges", re.I)
CO_PAIR = re.compile(r"\((\d[\d,]*) of (\d[\d,]*)\)[^.]{0,40}co-mention", re.I)
EXAMPLE = re.compile(r'"([^"]{2,60})"')


class TestTheMissingBaseline:
    """Round-1 F6a: the floor a rung scoring at random would clear. The paragraph's
    whole point is that the missing baseline is high; it was published 11 points low."""

    def test_the_relevant_share_matches_the_ground_truth(self) -> None:
        figures = _figures()
        for match in _matches(
            RELEVANT_SHARE,
            "share of the demo corpus that is relevant",
            "'52.9% of the demo corpus is relevant'",
        ):
            shown = match.group(1)
            assert float(shown) == pytest.approx(
                figures["relevant_share"], abs=_slack(shown)
            ), (
                f"LIMITATIONS.md says {shown}% of the demo corpus is relevant; the union "
                f"of the cited sets inside the manifest is "
                f"{figures['relevant_docs']:.0f} of {figures['corpus_docs']:.0f} = "
                f"{figures['relevant_share']:.1f}%" + _summary()
            )
            bullet = _bullet_around(match.start())
            for pair in COUNTED_PAIR.finditer(bullet):
                counts = [int(g.replace(",", "")) for g in pair.groups()]
                assert counts == [
                    int(figures["relevant_docs"]),
                    int(figures["corpus_docs"]),
                ], (
                    f"the same bullet says {pair.group(0)!r}; the measure is "
                    f"{figures['relevant_docs']:.0f} of {figures['corpus_docs']:.0f}"
                    + _summary()
                )


class TestAttributionCoverage:
    """Round-1 F6b: `corpus/DEMO_CITATIONS.md` is tracked and carries all 68 pinned demo
    documents, so a reader of the repository alone sees attribution for all of them —
    the paragraph understated work the repository had already done, in the paragraph
    about licence compliance."""

    def test_the_attribution_count_matches_the_citation_files(self) -> None:
        figures = _figures()
        for match in _matches(
            ATTRIBUTION, "attribution coverage count", "'attribution for 83 of 83 documents'"
        ):
            counts = [int(g.replace(",", "")) for g in match.groups()]
            assert counts == [int(figures["attributed"]), int(figures["documents_touched"])], (
                f"LIMITATIONS.md says {match.group(0)!r}; CITATIONS.md and "
                f"corpus/DEMO_CITATIONS.md pin {figures['attributed']:.0f} documents "
                f"between them, of {figures['documents_touched']:.0f} the project touches"
                + _summary()
            )

    def test_the_demo_citations_file_is_tracked(self) -> None:
        """The claim only holds because that file is committed — the reason the old
        sentence was wrong is that it was written before it was."""
        assert (ROOT / "corpus" / "DEMO_CITATIONS.md").is_file()


class TestResidualContamination:
    """Report-2 F5: the one channel exclusion cannot close. Every figure here is the
    complete residue, and the page must publish it as such — 101 of 101 in the ranked
    text, not 96, which was the complement with its sense inverted."""

    def test_the_twin_only_entity_count_matches_the_graph(self) -> None:
        figures = _figures()
        for match in _matches(
            TWIN_ONLY,
            "count of entities that exist only because of the excluded twin",
            "'101 graph entities exist only because of the excluded twin'",
        ):
            assert int(match.group(1)) == int(figures["twin_only"]), (
                f"LIMITATIONS.md says {match.group(0)!r}; rebuilding the demo graph "
                f"without the twin's chunks removes {figures['twin_only']:.0f} entities"
                + _summary()
            )

    def test_the_share_that_reaches_the_ranked_text_matches_the_reports(self) -> None:
        figures = _figures()
        for match in _matches(
            IN_REPORTS,
            "count of twin-only entities appearing in the community reports",
            "'101 of them appear in the community reports'",
        ):
            assert int(match.group(1)) == int(figures["twin_only_in_members"]), (
                f"LIMITATIONS.md says {match.group(0)!r}; "
                f"{figures['twin_only_in_members']:.0f} of the "
                f"{figures['twin_only']:.0f} twin-only entities are in the reports' "
                "member_names, which is what `graphrag-global` tokenises and ranks"
                + _summary()
            )

    def test_the_prose_count_matches_the_reports(self) -> None:
        figures = _figures()
        for match in _matches(
            IN_PROSE,
            "count of twin-only entities named in the reports' own prose",
            "'at most 5 are named in the reports' title or summary prose'",
        ):
            assert int(match.group(1)) == int(figures["twin_only_in_prose"]), (
                f"LIMITATIONS.md says {match.group(0)!r}; {figures['twin_only_in_prose']:.0f} "
                "of the twin-only entities appear as a substring of a report's title or "
                "summary (a substring test, so an over-count)" + _summary()
            )

    def test_every_named_example_is_in_the_twin_only_set(self) -> None:
        """Report-2 F5's falsified illustration: `Biotek Synergy MX plate reader` is in
        the graph and SURVIVES exclusion — another document contributes it — so it
        illustrates the opposite of the sentence it sits in. Code spans are removed
        before the examples are read, because a path or a call is not an entity."""
        twin_only, _edges, _co, _rel = _demo_graphs()
        figures = _figures()
        match = _matches(
            TWIN_ONLY,
            "count of entities that exist only because of the excluded twin",
            "'101 graph entities exist only because of the excluded twin'",
        )[0]
        bullet = re.sub(r"`[^`]*`", " ", _bullet_around(match.start()))
        examples = [e.strip() for e in EXAMPLE.findall(bullet)]
        assert examples, (
            "the residual-contamination bullet names no example entity in double quotes. "
            "It illustrates what leaks with entities a reader can look up, and each one "
            "is checked for membership in the twin-only set — an example that is not in "
            "the set falsifies the sentence it illustrates." + _summary()
        )
        missing = [e for e in examples if norm_entity(e) not in twin_only]
        assert not missing, (
            f"named as twin-only but not in the twin-only set: {missing}. The set holds "
            f"{figures['twin_only']:.0f} entities; pick examples from it — a few of them "
            f"are {sorted(twin_only)[:6]}" + _summary()
        )


class TestTheGraphIsMostlyAdjacency:
    """Report-1 F8a: published with no committed artifact behind it and no test, in the
    same paragraph as a CI figure that WAS bound — so the file mixed bound and
    unbindable numbers with nothing to tell them apart."""

    def test_the_co_mention_share_matches_the_graph(self) -> None:
        figures = _figures()
        for match in _matches(
            CO_SHARE, "co-mention share of the demo graph", "'93.1% of the demo graph's edges'"
        ):
            shown = match.group(1)
            assert float(shown) == pytest.approx(
                figures["co_mention_share"], abs=_slack(shown)
            ), (
                f"LIMITATIONS.md says {shown}% of the demo graph's edges are co-mention "
                f"only; the merge over the committed extractions gives "
                f"{figures['co_mention_share']:.1f}%" + _summary()
            )

    def test_the_co_mention_counts_match_the_graph(self) -> None:
        figures = _figures()
        for match in _matches(
            CO_PAIR, "co-mention edge counts", "'(58,797 of 63,130) are co-mention only'"
        ):
            counts = [int(g.replace(",", "")) for g in match.groups()]
            assert counts == [int(figures["co_mention_only"]), int(figures["edges"])], (
                f"LIMITATIONS.md says {match.group(0)!r}; the graph has "
                f"{figures['co_mention_only']:,.0f} co-mention-only edges of "
                f"{figures['edges']:,.0f}" + _summary()
            )


class TestTheFiguresAreReproducibleOffline:
    """The recipe the page gives a reader must be the one that works. The old one
    (`build_run_graph(root, "demo", …)`) raises `DemoCorpusMissing` from a clean clone,
    because `corpus/demo/` is fetched — so the figure it claimed to reproduce could not
    be checked by anyone who had not already run the fetch."""

    def test_every_input_this_file_reads_is_committed(self) -> None:
        for relative in (
            "fixtures/extraction/demo",
            "fixtures/summaries/demo",
            "corpus/ground_truth.json",
            "corpus/demo.manifest.json",
            "corpus/ci.manifest.json",
            "corpus/DEMO_CITATIONS.md",
            "CITATIONS.md",
        ):
            assert (ROOT / relative).exists(), (
                f"{relative} is missing, so the figures in LIMITATIONS.md cannot be "
                "recomputed from a clean clone — which is how they drifted"
            )

    def test_the_reproduce_recipe_does_not_send_a_reader_to_the_fetched_corpus(self) -> None:
        """`corpus/demo/` is gitignored; a recipe that needs it is not a recipe."""
        offenders = [
            line.strip()
            for line in PROSE.splitlines()
            if "build_run_graph(" in line and '"demo"' in line
        ]
        assert not offenders, (
            f"LIMITATIONS.md tells a reader to reproduce a demo figure with {offenders}, "
            "which raises DemoCorpusMissing from a clean clone (corpus/demo/ is fetched). "
            "The offline road is a merge over fixtures/extraction/demo — the one this "
            "test takes." + _summary()
        )
