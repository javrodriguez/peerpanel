"""Manuscript-lane tests: deterministic JATS extraction, the attributed on-disk
format round-tripping on the committed real files, the twin table's exclusion
predicate, and the stubbed ingest road — the live refetch carries the network
marker."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest

from peerpanel.corpus.s3 import LicenseRefused
from peerpanel.manuscripts import (
    ManuscriptHeader,
    Twin,
    TwinTable,
    extract,
    load_twins,
    read_manuscript,
    render_file,
    write_manuscript,
)
from peerpanel.manuscripts import ingest as ingest_mod
from peerpanel.manuscripts.store import HEADER_BEGIN, HEADER_END
from peerpanel.text import chunk_document

MANUSCRIPTS_DIR = Path(__file__).resolve().parent.parent / "manuscripts"

SYNTHETIC_JATS = b"""<?xml version="1.0" encoding="UTF-8"?>
<article xmlns:xlink="http://www.w3.org/1999/xlink">
 <front>
  <article-meta>
   <title-group>
    <article-title>A <italic>tiny</italic> study</article-title>
   </title-group>
   <contrib-group>
    <contrib contrib-type="author">
     <name><surname>Doe</surname><given-names>Jane</given-names></name>
    </contrib>
    <contrib contrib-type="author">
     <name><surname>Roe</surname><given-names>Rick</given-names></name>
    </contrib>
   </contrib-group>
  </article-meta>
 </front>
 <body>
  <sec id="s1">
   <title>Introduction</title>
   <p>First paragraph with an <italic>inline</italic>
      emphasis<xref ref-type="bibr" rid="r1">1</xref>.</p>
   <fig id="f1"><caption><p>A figure caption that must vanish.</p></caption></fig>
   <table-wrap id="t1"><table><tr><td>cell-noise</td></tr></table></table-wrap>
   <p>Equation<disp-formula>E = mc^2-noise</disp-formula> stripped,
      text after the formula survives.</p>
  </sec>
  <sec id="s2">
   <title>Methods</title>
   <sec id="s2a"><title>Strains</title><p>Nested section text.</p></sec>
   <list list-type="bullet">
    <list-item><p>item one</p></list-item>
    <list-item><p>item two</p></list-item>
   </list>
  </sec>
 </body>
 <back>
  <ref-list>
   <title>Reference-list-title</title>
   <ref id="r1"><mixed-citation>Reference noise never appears</mixed-citation></ref>
  </ref-list>
 </back>
</article>
"""


class TestJatsExtraction:
    def test_extract_twice_gives_identical_bytes(self) -> None:
        a = extract(SYNTHETIC_JATS)
        b = extract(SYNTHETIC_JATS)
        assert a == b
        assert a.body.encode() == b.body.encode()

    def test_title_and_authors_from_jats(self) -> None:
        e = extract(SYNTHETIC_JATS)
        assert e.title == "A tiny study"
        assert e.authors == ["Jane Doe", "Rick Roe"]

    def test_headings_are_their_own_paragraphs_in_order(self) -> None:
        paras = extract(SYNTHETIC_JATS).body.split("\n\n")
        assert paras[0] == "Introduction"
        assert "Methods" in paras
        assert "Strains" in paras
        assert paras.index("Methods") < paras.index("Strains")

    def test_figures_tables_display_math_and_refs_stripped(self) -> None:
        body = extract(SYNTHETIC_JATS).body
        for gone in ("vanish", "cell-noise", "mc^2", "Reference noise", "Reference-list-title"):
            assert gone not in body
        assert "Equation stripped, text after the formula survives." in body

    def test_inline_markup_flattened_with_collapsed_whitespace(self) -> None:
        body = extract(SYNTHETIC_JATS).body
        assert "First paragraph with an inline emphasis1." in body

    def test_list_items_become_paragraphs(self) -> None:
        paras = extract(SYNTHETIC_JATS).body.split("\n\n")
        assert "item one" in paras
        assert "item two" in paras

    def test_missing_body_raises(self) -> None:
        bodyless = (
            b"<article><front><article-meta>"
            b"<title-group><article-title>t</article-title></title-group>"
            b'<contrib-group><contrib contrib-type="author">'
            b"<name><surname>S</surname></name></contrib></contrib-group>"
            b"</article-meta></front></article>"
        )
        with pytest.raises(ValueError, match="body"):
            extract(bodyless)


def _header(**overrides: object) -> ManuscriptHeader:
    base: dict[str, object] = {
        "title": "A tiny study",
        "authors": ["Jane Doe", "Rick Roe"],
        "source": "https://www.biorxiv.org/content/10.1101/00.00.000000v1",
        "license": "CC BY 4.0 — https://creativecommons.org/licenses/by/4.0/",
        "changes": "converted from JATS XML to plain text",
        "preprint_doi": "10.1101/00.00.000000",
        "preprint_version": 1,
        "source_jats_md5": "0" * 32,
        "regenerate": "uv run python -m peerpanel.manuscripts.ingest 10.1101/00.00.000000",
    }
    base.update(overrides)
    return ManuscriptHeader(**base)  # type: ignore[arg-type]


class TestStoreRoundTrip:
    def test_write_read_render_is_byte_faithful(self, tmp_path: Path) -> None:
        body = "Introduction\n\nOne paragraph.\n\nAnother paragraph."
        path = tmp_path / "m.txt"
        write_manuscript(path, _header(), body)
        header, got_body = read_manuscript(path)
        assert header == _header()
        assert got_body == body
        assert render_file(header, got_body) == path.read_text()

    def test_missing_sentinel_refused(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.txt"
        path.write_text("no attribution block\n\nbody")
        with pytest.raises(ValueError, match="does not open"):
            read_manuscript(path)

    def test_missing_field_refused(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.txt"
        path.write_text(f"{HEADER_BEGIN}\n# Title: t\n{HEADER_END}\n\nbody\n")
        with pytest.raises(ValueError, match="missing"):
            read_manuscript(path)


class TestCommittedManuscripts:
    """The three real committed files — parseable, attributed, chunker-ready."""

    def _paths(self) -> list[Path]:
        return sorted(MANUSCRIPTS_DIR.glob("*.txt"))

    def test_exactly_three_committed(self) -> None:
        assert len(self._paths()) == 3

    def test_headers_complete_and_bodies_substantial(self) -> None:
        for path in self._paths():
            header, body = read_manuscript(path)
            assert header.title
            assert header.authors
            assert header.license.startswith("CC BY 4.0")
            assert re.fullmatch(r"[0-9a-f]{32}", header.source_jats_md5)
            assert header.preprint_doi.startswith("10.1101/")
            assert header.preprint_doi in header.regenerate
            assert "peerpanel.manuscripts.ingest" in header.regenerate
            assert len(body.split("\n\n")) >= 10

    def test_round_trip_reproduces_committed_bytes(self) -> None:
        for path in self._paths():
            header, body = read_manuscript(path)
            assert render_file(header, body) == path.read_text()
            assert HEADER_BEGIN not in body

    def test_bodies_chunk_cleanly(self) -> None:
        for path in self._paths():
            _, body = read_manuscript(path)
            chunks = chunk_document(path.stem, body)
            assert len(chunks) > 1
            assert HEADER_BEGIN not in "".join(c.text for c in chunks)

    def test_citations_md_covers_every_committed_file(self) -> None:
        citations = (MANUSCRIPTS_DIR / "CITATIONS.md").read_text()
        for path in self._paths():
            header, _ = read_manuscript(path)
            assert header.title in citations
            assert header.source_jats_md5 in citations
        assert "MIT license covers code only" in citations


class TestTwinTable:
    def test_committed_table_covers_every_committed_manuscript(self) -> None:
        table = load_twins(MANUSCRIPTS_DIR / "twins.json")
        assert len(table.twins) == 3
        for path in sorted(MANUSCRIPTS_DIR.glob("*.txt")):
            header, _ = read_manuscript(path)
            excluded = table.excluded_pmcids(header.preprint_doi)
            assert excluded, f"{path.name}: empty exclusion set"
            assert all(p.startswith("PMC") for p in excluded)

    def test_known_doi_maps_to_its_twin(self) -> None:
        table = load_twins(MANUSCRIPTS_DIR / "twins.json")
        assert table.excluded_pmcids("10.1101/2023.05.18.541364") == {"PMC10729969"}

    def test_unknown_doi_raises_rather_than_empty(self) -> None:
        table = load_twins(MANUSCRIPTS_DIR / "twins.json")
        with pytest.raises(KeyError, match="not in the twin table"):
            table.excluded_pmcids("10.1101/1900.01.01.000001")

    def test_upsert_replaces_not_duplicates(self, tmp_path: Path) -> None:
        twin = Twin(
            preprint_doi="10.1101/00.00.000000",
            preprint_version=1,
            published_doi="10.9/pub",
            pmcid="PMC7",
            title="t",
        )
        table = TwinTable(twins=[])
        table.upsert(twin)
        table.upsert(twin.model_copy(update={"pmcid": "PMC8"}))
        assert len(table.twins) == 1
        path = tmp_path / "twins.json"
        table.save(path)
        assert load_twins(path).excluded_pmcids("10.1101/00.00.000000") == {"PMC8"}


class TestIngestStubbed:
    """The full ingest writer through stubbed fetch seams (unit stubs of our
    own network layer — the live road is network-marked below)."""

    def _wire(self, monkeypatch: pytest.MonkeyPatch, license_code: str = "cc_by") -> None:
        meta = ingest_mod.BiorxivVersion(
            doi="10.1101/00.00.000000",
            version=1,
            license=license_code,
            jatsxml_url="https://example.invalid/x.source.xml",
        )
        monkeypatch.setattr(ingest_mod, "fetch_version_meta", lambda d, v, c=None: meta)
        monkeypatch.setattr(ingest_mod, "fetch_jats", lambda m, c=None: SYNTHETIC_JATS)

    def _run(self, out_dir: Path) -> Path:
        return ingest_mod.ingest("10.1101/00.00.000000", 1, "tiny", "10.9/pub", "PMC7", out_dir)

    def test_writes_txt_twins_and_citations_deterministically(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._wire(monkeypatch)
        path = self._run(tmp_path)
        first = {p.name: p.read_bytes() for p in sorted(tmp_path.iterdir())}
        self._run(tmp_path)
        second = {p.name: p.read_bytes() for p in sorted(tmp_path.iterdir())}
        assert first == second
        assert set(first) == {"tiny.txt", "twins.json", "CITATIONS.md"}

        header, body = read_manuscript(path)
        assert header.title == "A tiny study"
        assert header.source_jats_md5 == hashlib.md5(SYNTHETIC_JATS).hexdigest()
        assert "Introduction" in body.split("\n\n")

        table = load_twins(tmp_path / "twins.json")
        assert table.excluded_pmcids("10.1101/00.00.000000") == {"PMC7"}
        raw = json.loads((tmp_path / "twins.json").read_text())
        assert raw[0]["published_doi"] == "10.9/pub"

        citations = (tmp_path / "CITATIONS.md").read_text()
        for needle in (
            "**Title:** A tiny study",
            "**Authors:** Jane Doe; Rick Roe",
            "creativecommons.org/licenses/by/4.0",
            "**Published twin:** DOI 10.9/pub · PMC7",
            "JATS md5 " + hashlib.md5(SYNTHETIC_JATS).hexdigest(),
        ):
            assert needle in citations

    def test_non_redistributable_license_refused_before_any_write(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._wire(monkeypatch, license_code="cc_by_nc")
        with pytest.raises(LicenseRefused, match="not redistributable"):
            self._run(tmp_path)
        assert list(tmp_path.iterdir()) == []


@pytest.mark.network
class TestLiveRefetch:
    """Refetch one committed manuscript's JATS live and prove the committed
    artifact regenerates byte-for-byte (md5 pin + deterministic extraction)."""

    DOI = "10.1101/2023.05.18.541364"
    VERSION = 1

    def test_live_jats_matches_pin_and_reextracts_identically(self) -> None:
        committed = next(
            p
            for p in sorted(MANUSCRIPTS_DIR.glob("*.txt"))
            if read_manuscript(p)[0].preprint_doi == self.DOI
        )
        header, _ = read_manuscript(committed)

        meta = ingest_mod.fetch_version_meta(self.DOI, self.VERSION)
        assert meta.license == "cc_by"
        jats_bytes = ingest_mod.fetch_jats(meta)
        assert hashlib.md5(jats_bytes).hexdigest() == header.source_jats_md5

        extracted = ingest_mod.extract(jats_bytes)
        assert extracted.title == header.title
        assert extracted.authors == header.authors
        assert render_file(header, extracted.body) == committed.read_text()
