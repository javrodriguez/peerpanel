"""The panel's one deterministic lens — no model, no prompt, no network.

Every finding here is produced by a regular expression and a rule, so the same
text always yields the same findings in the same order. That is the point: a
whole class of real defects in the genomics literature is mechanical rather than
judgmental. Abeysooriya et al. (doi:10.1371/journal.pcbi.1008984) found
spreadsheet-autoformatted gene-symbol corruption in 30.9% of 11,117 genomics
papers carrying supplementary Excel gene lists (2014-2020, with no decline after
the 2016 HGNC renaming). A checker catches that class exactly; a language model
catches it only probably, and cannot say why twice the same way.

Each check is deliberately conservative — a missed case costs less than a false
positive, because noise is what makes a review tool unusable. The limits are
written into each check's own docstring; they are limits, not defects.

Public surface: `check_gene_symbol_corruption`, `check_reference_integrity`,
`check_stat_sanity`, and `run_deterministic_lens` over all three. All are pure
functions of the manuscript body text (the attribution header is the caller's to
strip — see `peerpanel.manuscripts.store.read_manuscript`).
"""

from __future__ import annotations

import re

from .schemas import DeterministicFinding

SEVERITY_MAJOR = "major"
SEVERITY_MINOR = "minor"

CHECK_GENE_SYMBOL = "gene_symbol_corruption"
CHECK_REFERENCE_INTEGRITY = "reference_integrity"
CHECK_STAT_SANITY = "stat_sanity"

# Only the month names that collide with real HGNC symbol families are treated
# as corruption candidates. MARCH1-11/MARC1-2, SEPT1-15, DEC1, OCT1-11 and NOV
# have documented Excel date forms; "1-Feb" or "3-Aug" correspond to no gene, so
# flagging them would be pure noise.
_CORRUPTING_MONTHS: dict[str, str] = {
    "mar": "MARCH1-11 / MARC1-2 (now MARCHF / MTARC)",
    "march": "MARCH1-11 / MARC1-2 (now MARCHF / MTARC)",
    "sep": "SEPT1-15 (now SEPTIN)",
    "sept": "SEPT1-15 (now SEPTIN)",
    "september": "SEPT1-15 (now SEPTIN)",
    "dec": "DEC1 (BHLHE40)",
    "december": "DEC1 (BHLHE40)",
    "oct": "OCT1-11 (POU2F / SLC22A)",
    "october": "OCT1-11 (POU2F / SLC22A)",
    "nov": "NOV (CCN3)",
    "november": "NOV (CCN3)",
}

# The largest family member is SEPT15, so a number above this is a calendar day,
# never a gene. Cheap, and it removes most of the false-positive surface.
_MAX_FAMILY_NUMBER = 15

_MONTH_ALT = r"mar(?:ch)?|sep(?:t|tember)?|dec(?:ember)?|oct(?:ober)?|nov(?:ember)?"

_GENE_DATE_RE = re.compile(
    rf"(?<![\w-])(?:(?P<num_first>\d{{1,2}})-(?P<mon_last>{_MONTH_ALT})"
    rf"|(?P<mon_first>{_MONTH_ALT})-(?P<num_last>\d{{1,2}}))(?![\w-])",
    re.IGNORECASE,
)

# A line that is doing calendar/bibliographic work. "Received: 1 Mar 2024",
# "Accepted 2-Sep-2021", a citation with a year, anything with a DOI or a URL.
_DATE_CONTEXT_RE = re.compile(
    r"\b(?:received|accepted|revised|resubmitted|submitted|published|reprinted"
    r"|copyright|epub|retrieved|accessed|in\s+press|version\s+of\s+record"
    r"|first\s+(?:online|published)|available\s+(?:online|at|from)"
    r"|deadline|embargo|date[sd]?\s*[:=]|last\s+modified|et\s+al\.)"
    r"|\(\s*(?:19|20)\d{2}[a-z]?\s*\)"
    r"|\bdoi\s*[:=]|https?://",
    re.IGNORECASE,
)

# "1-Mar-2024" / "1-Mar 2024" / "1-Mar, 2024" is a date with its year attached.
_TRAILING_YEAR_RE = re.compile(r"^\s*[-,]?\s*(?:19|20)\d{2}\b")

# Excel turns an identifier like "2310009E13" into "2.31E+13". A negative
# exponent is ordinary scientific notation (a p-value, a rate) and is never
# flagged; only the "+" form is a corruption signature.
_SCI_NOTATION_RE = re.compile(r"(?<![\w.])\d(?:\.\d+)?[eE]\+\d{1,3}(?![\w.])")

_ID_CONTEXT_RE = re.compile(
    r"\b(?:gene|genes|symbol|symbols|accession|accessions|probe|probes|probeset"
    r"|clone|clones|transcript|transcripts|identifier|identifiers|locus|loci"
    r"|genbank|ensembl|refseq|affymetrix|uniprot|entrez|riken)\b",
    re.IGNORECASE,
)

# A measured quantity carries a unit; an identifier does not.
_SCI_UNIT_RE = re.compile(
    r"^\s*(?:cells?|CFU|copies|counts?|molecules?|reads?|particles?|events?"
    r"|bp|kb|Mb|Da|kDa|mol|M|mM|nM|/mL|/ml|/L|per\s+m[Ll])\b"
)

_REFS_HEADING_RE = re.compile(
    r"^\s*(?:#{1,6}\s*)?(?:\d{1,2}[.)]?\s*)?"
    r"(?:references|reference\s+list|bibliography|literature\s+cited|works\s+cited)"
    r"\s*:?\s*$",
    re.IGNORECASE,
)

_REF_ENTRY_RE = re.compile(r"^\s*\[?(\d{1,3})[\].)]\s+\S")

# Citation ranges are typeset with any of three dashes. Named escapes keep the
# source ASCII (and ruff's ambiguous-character rule quiet).
_DASHES = "\N{EN DASH}\N{EM DASH}-"

_CITATION_MARKER_RE = re.compile(rf"\[(\d{{1,3}}(?:\s*[,;{_DASHES}]\s*\d{{1,3}})*)\]")

_RANGE_RE = re.compile(rf"(\d{{1,3}})\s*[{_DASHES}]\s*(\d{{1,3}})")

# Trailing punctuation a DOI picks up in running prose, closing quote included.
_DOI_TRAILING_PUNCT = ".,;:)]}\N{RIGHT SINGLE QUOTATION MARK}\"'"

_DOI_CANDIDATE_RE = re.compile(
    r"(?:\bdoi\s*[:=]\s*|https?://(?:dx\.)?doi\.org/)(\S+)", re.IGNORECASE
)

_DOI_VALID_RE = re.compile(r"^10\.\d{4,9}/\S+$")

_P_VALUE_RE = re.compile(
    r"(?<![\w])[Pp](?:[-\s]?values?)?\s*"
    r"(?P<op><=|>=|=|<|>|≤|≥)\s*"
    r"(?P<val>\d*\.\d+|\d+)(?P<exp>[eE][-+]?\d+)?"
)

# "P = 100 mM" is phosphorus, not a probability; a unit after the number means
# the token was never a p-value.
_P_UNIT_RE = re.compile(
    r"^\s*(?:mM|uM|µM|nM|pM|M|mg|µg|ug|ng|g/L|nm|mm|cm|°C|K"
    r"|h|hr|hrs|min|s|sec|mL|ml|L|ppm|kb|bp|Da|kDa|%|fold|x)\b"
)

_PERCENT_RE = re.compile(r"(\d{1,3}(?:\.\d+)?)\s*%")

# "88.83 ± 4.87%": the share is the number in front of the marker, the number
# carrying the percent sign is its error bar.
_UNCERTAINTY_RE = re.compile(r"(\d{1,3}(?:\.\d+)?)\s*(?:\N{PLUS-MINUS SIGN}|\+/-|\+-)\s*$")

# Only a clause that actually claims a partition is checked for a sum. A list
# of reagent purities ("peptone >95% ... glucose >99.5%") claims no partition.
_PARTITION_RE = re.compile(
    r"\b(?:consisted\s+of|consisting\s+of|composed\s+of|comprised(?:\s+of)?"
    r"|comprising|made\s+up\s+of|account(?:s|ed)?\s+for"
    r"|divided\s+into|partitioned\s+into|split\s+into|broke\s+down\s+into"
    r"|distributed\s+as"
    r"|of\s+(?:the\s+|all\s+)?(?:samples|patients|participants|subjects|cells"
    r"|reads|genes|isolates|respondents|cases|individuals|strains|colonies"
    r"|total|population|cohort))\b",
    re.IGNORECASE,
)

_PERCENT_PARTITION_TOLERANCE = 105.0
_MIN_PARTITION_SHARES = 3

# Clause, not sentence: a semicolon joins two independent statements, and two
# partitions joined that way ("EPS A consisted of mannose (92.8%) and glucose
# (7.2%); EPS B consisted mainly of mannose (83%)") must never be summed as one.
_CLAUSE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\s*;\s*")


def _location(line_no: int) -> str:
    """Every finding points at a 1-based line of the text it was given."""
    return f"line {line_no}"


def _reference_section_start(lines: list[str]) -> int | None:
    """Index of the line AFTER a standalone references heading, if there is one.

    The last such heading wins — a table of contents can name "References" long
    before the section itself.
    """
    for index in range(len(lines) - 1, -1, -1):
        if _REFS_HEADING_RE.match(lines[index]):
            return index + 1
    return None


def check_gene_symbol_corruption(text: str) -> list[DeterministicFinding]:
    """Spreadsheet-mangled gene symbols: date forms and exponent forms.

    Two signatures, both from Abeysooriya et al. (doi:10.1371/journal.pcbi.1008984):

    1. A date form of a symbol whose prefix is a month abbreviation — "1-Mar" or
       "Mar-1" for MARCH1/MARC1, "2-Sep" for SEPT2, "1-Dec" for DEC1, and the
       zero-padded and full-month spellings of each.
    2. Scientific-notation corruption of an identifier — "2.31E+13", which Excel
       produces from a clone/accession token such as "2310009E13".

    Limits, deliberately:

    - Only MAR, SEP, DEC, OCT and NOV are treated as corrupting months, and only
      numbers 1-15 (SEPT15 is the largest real family member). "3-Aug" and
      "22-Mar" name no gene and are never flagged.
    - Lines doing calendar or bibliographic work are skipped entirely
      (received/accepted/published/copyright/accessed lines, anything with a
      DOI, a URL, "et al." or a parenthesised year), as is everything at or
      below a standalone references heading. A date form with a year attached
      ("1-Mar-2024") is a date, not a symbol.
    - The exponent form is only flagged when the same line carries a gene or
      accession context word AND the number is not followed by a unit, so
      "1.5E+06 cells/mL" in a growth-rate line stays quiet. A negative exponent
      is ordinary notation and is never flagged.

    The cost of those guards is recall on a corrupted symbol that appears with
    no surrounding context at all; that trade is the intended one.
    """
    findings: list[DeterministicFinding] = []
    lines = text.splitlines()
    refs_start = _reference_section_start(lines)

    for index, line in enumerate(lines):
        if refs_start is not None and index >= refs_start:
            continue
        if _DATE_CONTEXT_RE.search(line):
            continue
        line_no = index + 1

        for match in _GENE_DATE_RE.finditer(line):
            if _TRAILING_YEAR_RE.match(line[match.end() :]):
                continue
            raw_number = match.group("num_first") or match.group("num_last")
            raw_month = match.group("mon_first") or match.group("mon_last")
            number = int(raw_number)
            if not 1 <= number <= _MAX_FAMILY_NUMBER:
                continue
            family = _CORRUPTING_MONTHS[raw_month.lower()]
            findings.append(
                DeterministicFinding(
                    check=CHECK_GENE_SYMBOL,
                    severity=SEVERITY_MAJOR,
                    detail=(
                        f'"{match.group(0)}" is the spreadsheet date form of a '
                        f"{family} gene symbol, not a date"
                    ),
                    location=_location(line_no),
                )
            )

        if not _ID_CONTEXT_RE.search(line):
            continue
        for match in _SCI_NOTATION_RE.finditer(line):
            if _SCI_UNIT_RE.match(line[match.end() :]):
                continue
            findings.append(
                DeterministicFinding(
                    check=CHECK_GENE_SYMBOL,
                    severity=SEVERITY_MAJOR,
                    detail=(
                        f'"{match.group(0)}" is scientific notation where a gene or '
                        "accession identifier is expected — the spreadsheet "
                        "exponent corruption"
                    ),
                    location=_location(line_no),
                )
            )

    return findings


def _expand_citation_group(group: str) -> list[int]:
    """ "3,4" -> [3, 4]; "3-5" and its en-dash spelling -> [3, 4, 5]."""
    numbers: list[int] = []
    for part in re.split(r"[,;]", group):
        part = part.strip()
        if not part:
            continue
        span = re.fullmatch(_RANGE_RE, part)
        if span:
            low, high = int(span.group(1)), int(span.group(2))
            if low <= high and high - low <= 200:
                numbers.extend(range(low, high + 1))
            continue
        if part.isdigit():
            numbers.append(int(part))
    return numbers


def _normalise_doi(raw: str) -> str:
    return raw.strip().rstrip(_DOI_TRAILING_PUNCT).lower()


def check_reference_integrity(text: str) -> list[DeterministicFinding]:
    """Three mechanical reference checks, each with its own honest limit.

    (a) A numeric citation marker in the body that points past the end of the
        reference list — "[42]" against a 20-entry list. Severity major: the
        reader cannot reach the source at all. Runs ONLY when a standalone
        references heading is present AND its entries are numbered, because
        without both there is nothing to count against; a manuscript whose
        reference list has been stripped (as the committed ones have) is
        therefore silent here rather than guessing.
    (b) A DOI-shaped string failing DOI syntax (10.NNNN/suffix). Only strings
        introduced by a DOI marker are checked — "doi:", "DOI =",
        "https://doi.org/" — so an ordinary decimal such as "10.5 mM" is never
        mistaken for a broken DOI. Severity minor.
    (c) The same normalised DOI appearing in two different reference entries.
        Confined to the reference section, so a DOI legitimately cited twice in
        the body is not a duplicate entry. Severity minor.

    A duplicate entry that carries no DOI, and a citation marker in author-year
    style, are both out of reach of a deterministic check and are not attempted.
    """
    findings: list[DeterministicFinding] = []
    lines = text.splitlines()
    refs_start = _reference_section_start(lines)

    entry_numbers: list[int] = []
    if refs_start is not None:
        for line in lines[refs_start:]:
            match = _REF_ENTRY_RE.match(line)
            if match:
                entry_numbers.append(int(match.group(1)))

    if entry_numbers and refs_start is not None:
        n_refs = len(set(entry_numbers))
        body_lines = lines[: refs_start - 1]
        reported: set[int] = set()
        for index, line in enumerate(body_lines):
            for match in _CITATION_MARKER_RE.finditer(line):
                for number in _expand_citation_group(match.group(1)):
                    if number <= n_refs or number in reported:
                        continue
                    reported.add(number)
                    findings.append(
                        DeterministicFinding(
                            check=CHECK_REFERENCE_INTEGRITY,
                            severity=SEVERITY_MAJOR,
                            detail=(
                                f"citation [{number}] points past the reference list, "
                                f"which has {n_refs} entries"
                            ),
                            location=_location(index + 1),
                        )
                    )

    for index, line in enumerate(lines):
        for match in _DOI_CANDIDATE_RE.finditer(line):
            candidate = _normalise_doi(match.group(1))
            if _DOI_VALID_RE.match(candidate):
                continue
            findings.append(
                DeterministicFinding(
                    check=CHECK_REFERENCE_INTEGRITY,
                    severity=SEVERITY_MINOR,
                    detail=(
                        f'"{candidate}" is offered as a DOI but is not DOI syntax (10.NNNN/suffix)'
                    ),
                    location=_location(index + 1),
                )
            )

    if refs_start is not None:
        seen: dict[str, int] = {}
        for index, line in enumerate(lines[refs_start:], start=refs_start):
            for match in _DOI_CANDIDATE_RE.finditer(line):
                candidate = _normalise_doi(match.group(1))
                if not _DOI_VALID_RE.match(candidate):
                    continue
                first = seen.get(candidate)
                if first is None:
                    seen[candidate] = index + 1
                    continue
                findings.append(
                    DeterministicFinding(
                        check=CHECK_REFERENCE_INTEGRITY,
                        severity=SEVERITY_MINOR,
                        detail=(
                            f"duplicate reference entry: {candidate} already appears "
                            f"at line {first}"
                        ),
                        location=_location(index + 1),
                    )
                )

    return findings


def _percent_shares(clause: str) -> list[float]:
    """The shares a clause claims, reading "88.83 ± 4.87%" as 88.83, not 4.87.

    The percent sign in a measurement with an error bar attaches to the error,
    not to the share, so a naive reader would sum the uncertainties. Where the
    number is preceded by an uncertainty marker the value in front of it is the
    share; everywhere else the matched number is.
    """
    shares: list[float] = []
    for match in _PERCENT_RE.finditer(clause):
        uncertainty = _UNCERTAINTY_RE.search(clause[: match.start()])
        shares.append(float(uncertainty.group(1) if uncertainty else match.group(1)))
    return shares


def check_stat_sanity(text: str) -> list[DeterministicFinding]:
    """Statistical impossibilities that need no judgment to see.

    Three flags, all minor — each is a "this cannot be what you meant", not a
    verdict on the analysis:

    1. A p-value stated as exactly zero ("p = 0", "p=0.000"). A p-value is a
       probability under a continuous statistic; software reports "< 2.2e-16",
       never 0. Only the "=" form is flagged, and "p = 0.0001" is fine.
    2. A p-value above 1, which no probability can be, whatever the operator.
    3. Percentages in a single partition-claiming clause summing above 105.

    Limits, deliberately:

    - The p-value patterns require the letter p immediately before the operator,
      so "pH = 6.9" never matches, and a number carrying a unit ("P = 100 mM",
      phosphorus) is skipped. A p-value held in a variable named something else
      is out of reach.
    - The percentage check needs an explicit partition claim ("consisted of",
      "accounted for", "of the samples", ...) AND at least three shares within
      one clause, and it reads "88.83 ± 4.87%" as the share 88.83 rather than
      summing error bars. A list of reagent purities in one sentence is
      therefore silent, and so are two partitions of different things joined by
      a semicolon. The 105 tolerance absorbs honest rounding; a real partition
      claim that overshoots by less than five points will be missed, which is
      the intended direction.
    """
    findings: list[DeterministicFinding] = []

    for index, line in enumerate(text.splitlines()):
        line_no = index + 1
        for match in _P_VALUE_RE.finditer(line):
            if _P_UNIT_RE.match(line[match.end() :]):
                continue
            raw = match.group("val") + (match.group("exp") or "")
            try:
                value = float(raw)
            except ValueError:  # pragma: no cover - the pattern only yields floats
                continue
            operator = match.group("op")
            if value == 0.0 and operator == "=":
                findings.append(
                    DeterministicFinding(
                        check=CHECK_STAT_SANITY,
                        severity=SEVERITY_MINOR,
                        detail=(
                            f'"{match.group(0).strip()}" states a p-value of exactly '
                            "zero; report the software's bound instead (e.g. p < 2.2e-16)"
                        ),
                        location=_location(line_no),
                    )
                )
            elif value > 1.0:
                findings.append(
                    DeterministicFinding(
                        check=CHECK_STAT_SANITY,
                        severity=SEVERITY_MINOR,
                        detail=(
                            f'"{match.group(0).strip()}" states a p-value above 1, '
                            "which no probability can be"
                        ),
                        location=_location(line_no),
                    )
                )

        for clause in _CLAUSE_SPLIT_RE.split(line):
            if not _PARTITION_RE.search(clause):
                continue
            shares = _percent_shares(clause)
            if len(shares) < _MIN_PARTITION_SHARES:
                continue
            total = sum(shares)
            if total <= _PERCENT_PARTITION_TOLERANCE:
                continue
            rendered = ", ".join(f"{share:g}%" for share in shares)
            findings.append(
                DeterministicFinding(
                    check=CHECK_STAT_SANITY,
                    severity=SEVERITY_MINOR,
                    detail=(
                        f"a clause claiming a partition lists shares summing to "
                        f"{total:g}% ({rendered})"
                    ),
                    location=_location(line_no),
                )
            )

    return findings


def _line_number(location: str) -> int:
    match = re.fullmatch(r"line (\d+)", location)
    return int(match.group(1)) if match else 0


def run_deterministic_lens(text: str) -> list[DeterministicFinding]:
    """Every check, in one pass, ordered so the same text gives the same list.

    Findings are sorted by line, then check name, then detail — a total order on
    the finding content itself, so the result never depends on which check ran
    first or on any dict iteration.
    """
    findings = [
        *check_gene_symbol_corruption(text),
        *check_reference_integrity(text),
        *check_stat_sanity(text),
    ]
    return sorted(
        findings,
        key=lambda finding: (_line_number(finding.location), finding.check, finding.detail),
    )
