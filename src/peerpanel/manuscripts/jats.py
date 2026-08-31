"""JATS XML -> deterministic plain text for the query manuscripts.

The extraction is pure and byte-stable: the same JATS bytes always produce the
same text. Body sections become blank-line-separated paragraphs (the shape the
chunker expects), section headings become their own paragraphs, and figures,
tables, display math and the reference list are stripped entirely. Inline
markup (italic, xref, sup/sub) is flattened to its text with whitespace
collapsed.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass

# Subtrees that never enter the plain text: figures, tables, display math,
# graphics/media, supplementary material, and the reference list.
_STRIP = frozenset(
    {
        "fig",
        "fig-group",
        "table-wrap",
        "table-wrap-group",
        "table-wrap-foot",
        "disp-formula",
        "disp-formula-group",
        "graphic",
        "media",
        "supplementary-material",
        "inline-supplementary-material",
        "ref-list",
    }
)


@dataclass(frozen=True)
class JatsExtract:
    """What one JATS document yields: attribution metadata plus the body text."""

    title: str
    authors: list[str]
    body: str


def _local(tag: object) -> str:
    """Namespace-free local tag name; '' for comments/PIs (their tag is not a str)."""
    if not isinstance(tag, str):
        return ""
    return tag.rsplit("}", 1)[-1]


def _collect_text(el: ET.Element) -> str:
    """All character data under `el`, skipping stripped subtrees, whitespace collapsed."""
    parts: list[str] = []

    def walk(node: ET.Element) -> None:
        if node.text:
            parts.append(node.text)
        for child in node:
            tag = _local(child.tag)
            if tag and tag not in _STRIP:
                walk(child)
            if child.tail:
                parts.append(child.tail)

    walk(el)
    return " ".join("".join(parts).split())


def _block_paragraphs(node: ET.Element, out: list[str]) -> None:
    """Walk block-level children in document order, appending paragraphs."""
    for child in node:
        tag = _local(child.tag)
        if not tag or tag in _STRIP:
            continue
        if tag == "sec":
            _block_paragraphs(child, out)
        elif tag == "list":
            for item in child:
                if _local(item.tag) == "list-item":
                    text = _collect_text(item)
                    if text:
                        out.append(text)
        else:
            # <title>, <p>, and any other block element: one paragraph of text.
            text = _collect_text(child)
            if text:
                out.append(text)


def _first(root: ET.Element, local_name: str) -> ET.Element | None:
    return next((el for el in root.iter() if _local(el.tag) == local_name), None)


def extract_body(root: ET.Element) -> str:
    body = _first(root, "body")
    if body is None:
        raise ValueError("JATS document has no <body> element")
    paras: list[str] = []
    _block_paragraphs(body, paras)
    return "\n\n".join(paras)


def extract_title(root: ET.Element) -> str:
    title_el = _first(root, "article-title")
    if title_el is None:
        raise ValueError("JATS document has no <article-title> element")
    return _collect_text(title_el)


def extract_authors(root: ET.Element) -> list[str]:
    """Author names from <article-meta>, in document order, 'Given Surname'."""
    meta = _first(root, "article-meta")
    if meta is None:
        raise ValueError("JATS document has no <article-meta> element")
    authors: list[str] = []
    for contrib in meta.iter():
        if _local(contrib.tag) != "contrib" or contrib.get("contrib-type") != "author":
            continue
        name_el = _first(contrib, "name")
        if name_el is not None:
            given = _first(name_el, "given-names")
            surname = _first(name_el, "surname")
            pieces = [_collect_text(el) for el in (given, surname) if el is not None]
            full = " ".join(p for p in pieces if p)
            if full:
                authors.append(full)
            continue
        collab = _first(contrib, "collab")
        if collab is not None:
            text = _collect_text(collab)
            if text:
                authors.append(text)
    if not authors:
        raise ValueError("JATS document names no authors")
    return authors


def extract(jats_bytes: bytes) -> JatsExtract:
    """Parse once; return (title, authors, body) — deterministic for identical bytes."""
    root = ET.fromstring(jats_bytes)
    return JatsExtract(
        title=extract_title(root),
        authors=extract_authors(root),
        body=extract_body(root),
    )
