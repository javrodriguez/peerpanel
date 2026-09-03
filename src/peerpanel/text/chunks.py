"""Deterministic chunking: paragraph-packed windows with stable content-hashed ids."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

_PARA_SPLIT = re.compile(r"\n\s*\n")
# A sentence ends at . ! or ? followed by whitespace and an uppercase letter, a digit or
# an opening bracket/quote — the last of those because a bibliography paragraph's
# sentences begin "[12]", "2019" and "(2019)" as often as they begin with a capital.
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\[\(\"'])")
# The period in "et al. (2019)", "Fig. 3" and the initial in "J. Smith" is not a sentence
# end, and the pattern alone cannot tell: its lookahead admits a digit and "(" on purpose.
# So every candidate boundary is checked against the word before it (`_sentences`) — an
# abbreviation or a single character keeps the sentence whole. A wrong split here loses no
# text, it just cuts a piece in an odd place, which is exactly why nothing would notice it.
_ABBREVIATIONS = frozenset(
    {
        "al",
        "et",
        "fig",
        "figs",
        "eq",
        "eqs",
        "ref",
        "refs",
        "no",
        "nos",
        "vs",
        "cf",
        "approx",
        "e.g",
        "i.e",
        "sec",
        "pp",
        "vol",
        "ed",
        "eds",
        "dr",
        "prof",
        "mr",
        "ms",
        "st",
        "jr",
        "sr",
        "inc",
        "ltd",
        "co",
        "dept",
        "univ",
    }
)
# A window carrying less new text than this joins its neighbour instead of becoming a chunk
# of its own: a numbered heading or a stray "Data availability." line is not worth an
# extraction call or an embedding, and alone in the index it is retrievable noise.
#
# The cost is that a chunk can run over the target. Not by this much, and not once per
# document: a window is carried overlap plus fresh text, and holding a short window open
# lets its fresh half keep growing, so the honest bound is about twice the target plus this
# minimum. Measured on the corpora this repository ships: 1,042 words in CI and 1,183 in
# demo against a 900-word target, and randomised documents reach 3,059.
MIN_TAIL_WORDS = 60


@dataclass(frozen=True)
class Chunk:
    chunk_id: str  # "<doc_id>:<index>:<sha8 of text>"
    doc_id: str
    index: int
    text: str

    @property
    def n_words(self) -> int:
        return len(self.text.split())


def _sha8(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:8]


def _sentences(para: str) -> list[str]:
    """Sentences, with abbreviations and initials kept whole.

    `_SENTENCE_END` finds the candidate boundaries; a candidate whose preceding
    word is an abbreviation ("et al.", "Fig.") or a single character (an author
    initial, a list number) is not a sentence end, and the sentence runs on.

    The single-character rule is a trade, and it is wrong 216 times across the two
    corpora here: a sentence genuinely ending in one character — "conducted in R.",
    "with pestle A.", "the strains in panel B." — is joined to the one after it.
    Telling those from "J. Smith" needs to know whether a surname follows, which is
    a bigger machine than this earns. What it costs is bounded and small: the rule
    only moves where an OVERSIZE paragraph is cut, never what any chunk contains,
    and disabling it entirely moves one chunk in 1,077 across the demo corpus.
    """
    out: list[str] = []
    start = 0
    for match in _SENTENCE_END.finditer(para):
        head = para[start : match.start()]
        word = head.rsplit(None, 1)[-1].rstrip(".").lower() if head.split() else ""
        if word in _ABBREVIATIONS or len(word) <= 1:
            continue
        out.append(head)
        start = match.end()
    out.append(para[start:])
    return out


def split_oversize(para: str, target_words: int) -> list[str]:
    """A paragraph longer than the window, cut at sentence ends into pieces that fit.

    Before this, one paragraph over the target became one chunk of that size — a
    reference list or a results block of 12,000 words went to the extractor as a
    single prompt, which the serving window then silently cut to its tail
    (DECISIONS.md D19). Pieces pack like paragraphs but are never carried as
    overlap (a piece is most of a window; re-reading it would double the cost of
    every long paragraph). A lone sentence longer than the target (no boundary to
    cut at) stays whole: the provider sizes a window to fit it and refuses the
    call only if even the 32,768 ceiling cannot hold it, so nothing here can hand
    a model a prompt it will silently trim.
    """
    if len(para.split()) <= target_words:
        return [para]
    pieces: list[str] = []
    current: list[str] = []
    count = 0
    for sentence in _sentences(para):
        words = len(sentence.split())
        if count + words > target_words and current:
            pieces.append(" ".join(current))
            current, count = [], 0
        current.append(sentence)
        count += words
    if current:
        pieces.append(" ".join(current))
    return pieces


def chunk_document(
    doc_id: str,
    text: str,
    *,
    target_words: int = 900,
    overlap_paras: int = 1,
    min_tail_words: int = MIN_TAIL_WORDS,
) -> list[Chunk]:
    """Pack paragraphs into ~target_words windows, overlapping by whole paragraphs.

    Deterministic for identical input; chunk ids carry a content hash so any
    text drift is visible in every downstream artifact.

    The 900-word default (~1200 tokens) is a measured cost decision, not taste:
    graph extraction pays one LLM call per chunk, and one chunking is shared by
    extraction, embeddings and retrieval so entity->chunk->vector joins stay on
    one id space (DECISIONS.md D7 has the arithmetic). A paragraph longer than
    the target is split at sentence ends first (`split_oversize`), so no chunk
    exceeds the target plus one overlap paragraph unless a single sentence does,
    or unless a window held too little new text to stand as a chunk: below
    `min_tail_words` it takes the next paragraph in with it, and at the end of a
    document it joins the chunk before it instead — which can carry a chunk to
    roughly twice the target, the bound `MIN_TAIL_WORDS` records.
    """
    # (text, whole): a whole paragraph may be carried as overlap; a split piece may not.
    # `split_oversize` returns a single piece in two different situations — the paragraph
    # fitted, or it was oversize with no sentence boundary to cut at — and only the first
    # is safe to carry. Reading "one piece" as "whole" made the largest possible paragraph
    # the one most likely to be re-sent as overlap, which is the opposite of the intent.
    paras: list[tuple[str, bool]] = []
    for p in _PARA_SPLIT.split(text):
        if p.strip():
            body = p.strip()
            fitted = len(body.split()) <= target_words
            pieces = split_oversize(body, target_words)
            paras.extend((piece, fitted and len(pieces) == 1) for piece in pieces)
    chunks: list[Chunk] = []
    window: list[tuple[str, bool]] = []
    count = 0
    carried = 0  # leading entries of `window` re-read from the previous chunk

    def flush(final: bool = False) -> None:
        nonlocal window, count, carried
        if not window:
            return
        fresh = window[carried:]  # what this window adds; the rest is already in a chunk
        if final and chunks and sum(len(p.split()) for p, _ in fresh) < min_tail_words:
            # Too little new text to be worth its own extraction call, so it joins the
            # chunk before it rather than becoming retrievable noise. Only the fresh
            # paragraphs move — the carried ones are already in that chunk's text — and
            # a document with no previous chunk keeps its one short chunk.
            previous = chunks[-1]
            body = "\n\n".join([previous.text] + [p for p, _ in fresh])
            chunks[-1] = Chunk(
                f"{doc_id}:{previous.index}:{_sha8(body)}", doc_id, previous.index, body
            )
        else:
            body = "\n\n".join(p for p, _ in window)
            chunks.append(Chunk(f"{doc_id}:{len(chunks)}:{_sha8(body)}", doc_id, len(chunks), body))
        carry = window[-overlap_paras:] if overlap_paras else []
        window = [p for p in carry if p[1]]
        carried = len(window)
        count = sum(len(p.split()) for p, _ in window)

    def fresh_words() -> int:
        return sum(len(p.split()) for p, _ in window[carried:])

    for para, whole in paras:
        words = len(para.split())
        # A window is closed when the next paragraph would overrun the target — unless it
        # has too little new text to stand as a chunk, in which case it takes the next
        # paragraph with it rather than becoming a 17-word chunk of its own.
        if count + words > target_words and window and fresh_words() >= min_tail_words:
            flush()
        window.append((para, whole))
        count += words
    flush(final=True)
    return chunks
