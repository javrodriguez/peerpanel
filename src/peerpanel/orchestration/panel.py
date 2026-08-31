"""The panel orchestrator: blind parallel reviewers → adversarial claim
verification → deterministic lens → structural convergence.

Orchestrator-worker topology. Reviewers run blind (neither sees the other's
output), in parallel threads, on structurally different lenses and model
families. The verifier order-swaps every claim judgement and abstains on
disagreement. Conflicts are computed DETERMINISTICALLY (score divergence =
sentiment conflict; a reviewer finding whose claim the evidence REFUTES =
evidence conflict); propositional conflict classification is deliberately
absent in this version (a limitation, recorded — not an invisible gap). The
converger writes prose ONLY over the structured findings it is given
(no-new-claims rule; its family overlap with one reviewer is a recorded
limitation).
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from peerpanel.agents import methods_reviewer, novelty_reviewer
from peerpanel.agents.claims_verifier import (
    decompose_claims,
    swap_consistency_rate,
    verify_claims,
)
from peerpanel.agents.deterministic_lens import run_deterministic_lens
from peerpanel.agents.reviewer_base import excerpt
from peerpanel.agents.schemas import (
    VERDICT_REFUTES,
    AtomicClaim,
    ClaimVerdict,
    Conflict,
    PanelReview,
    ReviewerOutput,
)
from peerpanel.embeddings import store
from peerpanel.evals.ablation import load_artifacts
from peerpanel.graph.pipeline import chunks_for, embedding_fixture_path
from peerpanel.graph.run_graph import build_run_graph
from peerpanel.manuscripts.store import read_manuscript
from peerpanel.manuscripts.twins import load_twins
from peerpanel.providers.base import ChatProvider, TokenLedger
from peerpanel.retrieval import GraphGlobalRetriever, GraphLocalRetriever, Index

SENTIMENT_GAP = 2
CLAIMS_FROM_MANUSCRIPT = 8


class PanelProviders:
    """The panel's model assignment — heterogeneity is structural."""

    def __init__(
        self,
        methods: ChatProvider,
        novelty: ChatProvider,
        verifier: ChatProvider,
        converger: ChatProvider,
    ) -> None:
        self.methods = methods
        self.novelty = novelty
        self.verifier = verifier
        self.converger = converger


def detect_conflicts(
    outputs: list[ReviewerOutput], verdicts: list[ClaimVerdict]
) -> list[Conflict]:
    """Deterministic conflict detection (no LLM)."""
    conflicts: list[Conflict] = []
    for dim in outputs[0].scores if outputs else {}:
        scores = {o.reviewer: o.scores.get(dim) for o in outputs if dim in o.scores}
        values = [v for v in scores.values() if v is not None]
        if len(values) >= 2 and max(values) - min(values) >= SENTIMENT_GAP:
            conflicts.append(
                Conflict(
                    type="sentiment",
                    between=sorted(scores),
                    detail=f"{dim}: scores diverge {scores}",
                )
            )
    refuted = {v.claim_id: v for v in verdicts if v.verdict == VERDICT_REFUTES}
    for verdict in refuted.values():
        source = verdict.claim_id.split(":", 1)[0]
        if source != "manuscript":
            conflicts.append(
                Conflict(
                    type="evidence",
                    between=[source, "claims-verifier"],
                    detail=f"reviewer claim refuted by evidence: {verdict.claim_text[:120]}",
                )
            )
    return conflicts


def converge_summary(
    outputs: list[ReviewerOutput],
    verdicts: list[ClaimVerdict],
    conflicts: list[Conflict],
    provider: ChatProvider,
    ledger: TokenLedger,
) -> str:
    """Prose over the structured record ONLY (no-new-claims rule in the prompt;
    the structured record beside it is what a reader should trust)."""
    findings_block = "\n".join(
        f"- [{o.reviewer}] ({f.dimension}/{f.severity}) {f.text}"
        for o in outputs
        for f in o.findings
    )
    verdict_block = "\n".join(
        f"- {v.verdict}{'' if v.swap_consistent else ' (order-inconsistent -> abstained)'}: "
        f"{v.claim_text[:100]}"
        for v in verdicts
    )
    conflict_block = "\n".join(f"- {c.type}: {c.detail}" for c in conflicts) or "(none)"
    response = provider.chat(
        system=(
            "Write a short meta-review (<= 200 words) synthesising ONLY the findings, "
            "verdicts and conflicts given. Never add facts, findings or citations of "
            "your own; where reviewers disagree, present both sides."
        ),
        user=(
            f"REVIEWER FINDINGS:\n{findings_block or '(none)'}\n\n"
            f"CLAIM VERDICTS:\n{verdict_block or '(none)'}\n\n"
            f"CONFLICTS:\n{conflict_block}"
        ),
        temperature=0.0,
        max_tokens=512,
    )
    ledger.record(response)
    return response.text.strip()


def run_panel(
    root: Path,
    manuscript_path: Path,
    providers: PanelProviders,
    corpus: str = "demo",
) -> PanelReview:
    t0 = time.monotonic()
    ledger = TokenLedger()
    header, body = read_manuscript(manuscript_path)
    excluded = load_twins(root / "manuscripts" / "twins.json").excluded_pmcids(
        header.preprint_doi
    )
    chunks = chunks_for(root, corpus)
    chunk_ids, vectors = store.load(embedding_fixture_path(root, corpus))
    if [c.chunk_id for c in chunks] != chunk_ids:
        raise RuntimeError("embedding fixture stale relative to the corpus chunking")
    index = Index.build(chunks, vectors.astype("float32"), exclude_docs=excluded)
    if not index.dropped_chunk_count:
        # A non-empty exclusion SET proves nothing; this proves the mechanism removed
        # something. An assertion that passes when exclusion is a no-op is an untested
        # mechanism wearing a passing test.
        raise RuntimeError(
            f"exclusion removed no chunks for {header.preprint_doi} — no valid run (N3/D8)"
        )
    # The graph is REBUILT with the excluded document's extractions withheld (exact,
    # ~0.1s) rather than filtered after the fact — that removes the edge weight the
    # excluded text contributed, which filtering cannot reach (DECISIONS D10).
    graph, _run_assignment, _withheld = build_run_graph(root, corpus, excluded)
    # Community REPORTS are keyed to the full-corpus partition, so global search
    # keeps that assignment for report lookup while expanding membership through
    # the per-run graph — a node the rebuild removed cannot be expanded into.
    _full_graph, full_assignment, reports = load_artifacts(root, corpus)
    chunk_texts = {c.chunk_id: c.text for c in chunks}
    local = GraphLocalRetriever(index, graph, embedder=None)
    global_ = GraphGlobalRetriever(index, graph, reports, full_assignment)

    with ThreadPoolExecutor(max_workers=2) as pool:  # blind, parallel
        methods_future = pool.submit(
            methods_reviewer.review,
            manuscript_text=body,
            title=header.title,
            retriever=local,
            provider=providers.methods,
            chunk_texts=chunk_texts,
            ledger=ledger,
        )
        novelty_future = pool.submit(
            novelty_reviewer.review,
            manuscript_text=body,
            title=header.title,
            retriever=global_,
            provider=providers.novelty,
            chunk_texts=chunk_texts,
            ledger=ledger,
        )
        outputs = [methods_future.result(), novelty_future.result()]

    claims: list[AtomicClaim] = decompose_claims(
        excerpt(body),
        providers.verifier,
        source="manuscript",
        max_claims=CLAIMS_FROM_MANUSCRIPT,
        ledger=ledger,
    )
    for output in outputs:
        for i, finding in enumerate(output.findings):
            claims.append(
                AtomicClaim(
                    claim_id=f"{output.reviewer}:{i}:panel000",
                    text=finding.text,
                    source=output.reviewer,
                )
            )

    def retrieve(query: str) -> list[tuple[str, str]]:
        return [
            (h.chunk_id, chunk_texts.get(h.chunk_id, "")) for h in local.search(query, k=4)
        ]

    verdicts = verify_claims(claims, retrieve, providers.verifier, ledger)
    deterministic = run_deterministic_lens(body)
    conflicts = detect_conflicts(outputs, verdicts)
    summary = converge_summary(outputs, verdicts, conflicts, providers.converger, ledger)
    rate, n_checked = swap_consistency_rate(verdicts)
    return PanelReview(
        manuscript_doi=header.preprint_doi,
        corpus_manifest=f"corpus/{'ci' if corpus == 'ci' else 'demo'}.manifest.json",
        excluded_docs=sorted(index.excluded_docs),
        reviewer_outputs=outputs,
        verdicts=verdicts,
        deterministic_findings=deterministic,
        conflicts=conflicts,
        summary=summary,
        swap_consistency_rate=rate,
        n_swap_checked=n_checked,
        total_tokens=ledger.total_tokens,
        wall_s=round(time.monotonic() - t0, 1),
    )
