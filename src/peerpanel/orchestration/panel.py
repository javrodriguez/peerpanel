"""The panel orchestrator: blind parallel reviewers → adversarial claim
verification → deterministic lens → structural convergence.

Orchestrator-worker topology. Reviewers run blind (neither sees the other's
output), in parallel threads, on structurally different lenses and model
families. The verifier order-swaps every claim judgement and abstains on
disagreement. Conflicts are computed DETERMINISTICALLY (score divergence =
sentiment conflict; a reviewer-derived claim the evidence REFUTES, with the
refuting span present = evidence conflict); propositional conflict
classification is deliberately absent in this version (a limitation, recorded
— not an invisible gap). The converger writes prose ONLY over the structured
findings it is given (no-new-claims rule; its family overlap with one reviewer
is a recorded limitation).

What the verifier judges: the manuscript's own claims (decomposed from the
reviewed excerpt) and the scientific propositions inside each reviewer's
soundness / contribution findings — decomposed through the same claim splitter
under a rule that discards remarks about writing and presentation. A reviewer's
opinion is not a claim; only the checkable propositions inside it are.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from peerpanel.agents import methods_reviewer, novelty_reviewer
from peerpanel.agents.claims_verifier import (
    SOURCE_MANUSCRIPT,
    claim_source,
    decompose_claims,
    swap_consistency_by_source,
    swap_consistency_rate,
    verify_claims,
)
from peerpanel.agents.deterministic_lens import run_deterministic_lens
from peerpanel.agents.reviewer_base import excerpt
from peerpanel.agents.schemas import (
    VERDICT_REFUTES,
    AtomicClaim,
    CallStats,
    ClaimVerdict,
    Conflict,
    PanelReview,
    ReviewerOutput,
    SwapStat,
)
from peerpanel.corpus.models import CorpusManifest
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
CLAIMS_FROM_REVIEWER = 6
# Presentation findings are about the writing; nothing in a corpus can refute them.
CLAIM_DIMENSIONS = ("soundness", "contribution")


def _manifest_rel(corpus: str) -> Path:
    return Path("corpus") / ("ci.manifest.json" if corpus == "ci" else "demo.manifest.json")


class PanelProviders:
    """The panel's model assignment — heterogeneity is structural.

    Two model families serve four roles, so overlap is unavoidable and recorded
    (LIMITATIONS): the verifier shares its model with the methods reviewer, the
    converger shares its model with the novelty reviewer.
    """

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

    @classmethod
    def local_default(cls) -> PanelProviders:
        """The demo assignment: the ONE place the panel's models are named.

        Swapping a provider (an Anthropic adapter, a different local model) is
        an edit here, not at every command that runs the panel.
        """
        # Every panel role rides the native wire: it is the one that can size the
        # window per call, and the reviewer and verifier prompts exceed Ollama's
        # default 4,096 (providers/context.py, DECISIONS.md D19). The OpenAI wire
        # runs the community reports, whose prompts fit the default by construction.
        from peerpanel.providers.ollama_native import OllamaNativeChat

        return cls(
            methods=OllamaNativeChat("llama3.1:8b"),
            novelty=OllamaNativeChat("qwen2:7b"),
            verifier=OllamaNativeChat("llama3.1:8b"),
            converger=OllamaNativeChat("qwen2:7b"),
        )


def reviewer_claims(
    outputs: list[ReviewerOutput],
    provider: ChatProvider,
    ledger: TokenLedger | None = None,
) -> list[AtomicClaim]:
    """The checkable propositions inside each reviewer's findings.

    Only soundness / contribution findings are candidates (CLAIM_DIMENSIONS),
    and they pass through `decompose_claims` under the reviewer rule, so what
    reaches the verifier is a proposition about the science — never "the
    introduction is well organised" dressed as a claim. A reviewer whose
    candidate findings are empty contributes nothing and costs no call.
    """
    claims: list[AtomicClaim] = []
    for output in outputs:
        texts = [f.text for f in output.findings if f.dimension in CLAIM_DIMENSIONS]
        if not texts:
            continue
        claims.extend(
            decompose_claims(
                "\n".join(texts),
                provider,
                source=output.reviewer,
                max_claims=CLAIMS_FROM_REVIEWER,
                ledger=ledger,
            )
        )
    return claims


def detect_conflicts(outputs: list[ReviewerOutput], verdicts: list[ClaimVerdict]) -> list[Conflict]:
    """Deterministic conflict detection (no LLM)."""
    conflicts: list[Conflict] = []
    # Dimensions come from the UNION of what reviewers actually scored. Reading
    # them from outputs[0] alone meant one truncated response — which scores
    # nothing — silently disabled conflict detection for the whole panel.
    dimensions = {dim for o in outputs for dim in o.scores}
    for dim in sorted(dimensions):
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
    # An evidence conflict needs evidence: a REFUTES that carries no grounded span
    # is the judge's word alone, and a conflict minted from it would present an
    # ungrounded verdict as a documented contradiction.
    refuted = {v.claim_id: v for v in verdicts if v.verdict == VERDICT_REFUTES and v.evidence}
    for verdict in refuted.values():
        source = claim_source(verdict.claim_id)
        if source != SOURCE_MANUSCRIPT:
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
    excluded = load_twins(root / "manuscripts" / "twins.json").excluded_pmcids(header.preprint_doi)
    chunks = chunks_for(root, corpus)
    chunk_ids, vectors = store.load(embedding_fixture_path(root, corpus))
    if [c.chunk_id for c in chunks] != chunk_ids:
        raise RuntimeError("embedding fixture stale relative to the corpus chunking")
    index = Index.build(chunks, vectors.astype("float32"), exclude_docs=excluded)
    # Two safe states, one unsafe one. Safe: exclusion actually removed chunks, or
    # the manuscript is held out by construction (nothing of it is in the corpus at
    # all — how the planted-error subject is chosen). UNSAFE, and the reason this
    # check exists: its twin IS a corpus member and yet nothing was dropped, which
    # is exclusion silently behaving as a no-op.
    present = excluded & CorpusManifest.load(root / _manifest_rel(corpus)).pmcids()
    if present and not index.dropped_chunk_count:
        raise RuntimeError(
            f"{header.preprint_doi}: twin {sorted(present)} is in the corpus but exclusion "
            "removed no chunks — the mechanism is a no-op (N3/D8)"
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
        source=SOURCE_MANUSCRIPT,
        max_claims=CLAIMS_FROM_MANUSCRIPT,
        ledger=ledger,
    )
    claims.extend(reviewer_claims(outputs, providers.verifier, ledger))

    def retrieve(query: str) -> list[tuple[str, str]]:
        return [(h.chunk_id, chunk_texts.get(h.chunk_id, "")) for h in local.search(query, k=4)]

    verdicts = verify_claims(claims, retrieve, providers.verifier, ledger)
    deterministic = run_deterministic_lens(body)
    conflicts = detect_conflicts(outputs, verdicts)
    summary = converge_summary(outputs, verdicts, conflicts, providers.converger, ledger)
    rate, n_checked = swap_consistency_rate(verdicts)
    by_source = {
        source: SwapStat(rate=r, n=n)
        for source, (r, n) in swap_consistency_by_source(verdicts).items()
    }
    return PanelReview(
        manuscript_doi=header.preprint_doi,
        models={
            "methods-statistics": providers.methods.name,
            "prior-work-novelty": providers.novelty.name,
            "verifier": providers.verifier.name,
            "converger": providers.converger.name,
        },
        corpus_manifest=f"corpus/{'ci' if corpus == 'ci' else 'demo'}.manifest.json",
        excluded_docs=sorted(index.excluded_docs),
        dropped_chunks=index.dropped_chunk_count,
        reviewer_outputs=outputs,
        verdicts=verdicts,
        deterministic_findings=deterministic,
        conflicts=conflicts,
        summary=summary,
        swap_consistency_rate=rate,
        n_swap_checked=n_checked,
        swap_by_source=by_source,
        total_tokens=ledger.total_tokens,
        wall_s=round(time.monotonic() - t0, 1),
        model_calls=CallStats.from_ledger(ledger),
    )
