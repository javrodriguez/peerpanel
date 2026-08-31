"""The on-disk manuscript format: an attribution header + the plain-text body.

Every committed manuscript file opens with a comment-style attribution block
(CC BY requires attribution and a changes-made note) delimited by two fixed
sentinel lines, then one blank line, then the body as blank-line-separated
paragraphs. `read_manuscript` splits the two deterministically, so downstream
chunking and sanitation only ever see the body.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

HEADER_BEGIN = "# --- peerpanel manuscript attribution ---"
HEADER_END = "# --- end attribution ---"


@dataclass(frozen=True)
class ManuscriptHeader:
    """The attribution block: TASL + changes note + pin + regenerate command."""

    title: str
    authors: list[str]
    source: str
    license: str
    changes: str
    preprint_doi: str
    preprint_version: int
    source_jats_md5: str
    regenerate: str


_FIELD_MAP: tuple[tuple[str, str], ...] = (
    ("Title", "title"),
    ("Authors", "authors"),
    ("Source", "source"),
    ("License", "license"),
    ("Changes", "changes"),
    ("Preprint-DOI", "preprint_doi"),
    ("Preprint-Version", "preprint_version"),
    ("Source-JATS-md5", "source_jats_md5"),
    ("Regenerate", "regenerate"),
)


def render_header(header: ManuscriptHeader) -> str:
    lines = [HEADER_BEGIN]
    for key, attr in _FIELD_MAP:
        value = getattr(header, attr)
        text = "; ".join(value) if isinstance(value, list) else str(value)
        lines.append(f"# {key}: {text}")
    lines.append(HEADER_END)
    return "\n".join(lines) + "\n"


def render_file(header: ManuscriptHeader, body: str) -> str:
    """The full file content — the format `write_manuscript` commits to disk."""
    return render_header(header) + "\n" + body.strip("\n") + "\n"


def write_manuscript(path: Path, header: ManuscriptHeader, body: str) -> None:
    path.write_text(render_file(header, body), encoding="utf-8", newline="\n")


def read_manuscript(path: Path) -> tuple[ManuscriptHeader, str]:
    """Parse a committed manuscript into (header, body).

    The body comes back without the attribution block or surrounding blank
    lines; `render_file(header, body)` reproduces the file bytes exactly.
    """
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or lines[0] != HEADER_BEGIN:
        raise ValueError(f"{path.name}: does not open with {HEADER_BEGIN!r}")
    try:
        end = lines.index(HEADER_END)
    except ValueError:
        raise ValueError(f"{path.name}: attribution block never closed") from None

    fields: dict[str, str] = {}
    for raw in lines[1:end]:
        if not raw.startswith("# ") or ": " not in raw:
            raise ValueError(f"{path.name}: malformed attribution line {raw!r}")
        key, _, value = raw.removeprefix("# ").partition(": ")
        fields[key] = value
    missing = [key for key, _ in _FIELD_MAP if key not in fields]
    if missing:
        raise ValueError(f"{path.name}: attribution block missing {missing}")

    header = ManuscriptHeader(
        title=fields["Title"],
        authors=fields["Authors"].split("; "),
        source=fields["Source"],
        license=fields["License"],
        changes=fields["Changes"],
        preprint_doi=fields["Preprint-DOI"],
        preprint_version=int(fields["Preprint-Version"]),
        source_jats_md5=fields["Source-JATS-md5"],
        regenerate=fields["Regenerate"],
    )
    body = "\n".join(lines[end + 1 :]).strip("\n")
    return header, body
