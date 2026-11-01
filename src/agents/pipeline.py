"""Classification pipeline: context -> retrieve -> classify -> critique/revise -> score -> decide.

Bare bones: the loop runs end to end and every step is recorded. Scoring
and decision use simple placeholders marked TODO.
"""

import logging
import time

from sqlalchemy.ext.asyncio import AsyncSession

from .agents import ClassifierAgent, CriticAgent
from src.services.llm import LLMClient, LLMError
from src.services.embedder import OllamaEmbedder   # adjust import path
from src.services.retrieval import related_category_ids, retrieve_candidates
from src.models.products import (
    Candidate, ClassificationResult, ClassifierOutput, CriticOutput,
    Decision, ProductContext, ProductIn, Step,
)

logger = logging.getLogger(__name__)

MAX_REVISIONS = 2
AUTO_LINK_THRESHOLD = 0.85
REVIEW_THRESHOLD = 0.60


class _Trace:
    def __init__(self) -> None:
        self.steps: list[Step] = []

    def add(self, role, output, started, valid=True, error=None) -> None:
        self.steps.append(Step(
            step_no=len(self.steps) + 1, role=role, output=output,
            valid_choice=valid, error=error,
            latency_ms=int((time.perf_counter() - started) * 1000),
        ))


async def build_context(session: AsyncSession, product: ProductIn) -> ProductContext:
    ctx = ProductContext(**product.model_dump())
    ctx.related_category_ids = await related_category_ids(session, ctx)
    # TODO: image captioning with a vision model, cached by image hash
    return ctx


async def _classify_valid(
        classifier: ClassifierAgent, ctx: ProductContext, candidates: list[Candidate],
        trace: _Trace, feedback: CriticOutput | None = None,
) -> ClassifierOutput | None:
    """Run the classifier; retry once if it errors or picks an id outside the candidates."""
    ids = {c.category_id for c in candidates}
    for _attempt in range(2):
        started = time.perf_counter()
        try:
            out = await classifier.run(ctx, candidates, feedback)
        except LLMError as e:
            trace.add("classifier", None, started, valid=False, error=str(e))
            continue
        valid = out.category_id in ids
        trace.add("classifier", out, started, valid=valid,
                  error=None if valid else f"id {out.category_id!r} not in candidates")
        if valid:
            return out
    return None


def score(
        final: ClassifierOutput,
        critique: CriticOutput | None,
        candidates: list[Candidate],
        related_ids: list[str],
) -> tuple[float, dict[str, float]]:
    """TODO: the real blend. Placeholder weights; calibrate on the gold set."""
    by_id = {c.category_id: c for c in candidates}
    chosen = by_id[final.category_id]
    signals = {
        "self": final.confidence,
        "critic_agrees": 1.0 if critique and critique.own_pick == final.category_id else 0.0,
        "related_agrees": 1.0 if final.category_id in related_ids else 0.0,
        "retrieval": max(0.0, chosen.similarity),
    }
    weights = {"self": 0.2, "critic_agrees": 0.4, "related_agrees": 0.2, "retrieval": 0.2}
    if not related_ids:   # no related products: spread that weight over the others
        weights = {"self": 0.25, "critic_agrees": 0.5, "related_agrees": 0.0, "retrieval": 0.25}
    return round(sum(signals[k] * w for k, w in weights.items()), 3), signals


def decide(confidence: float) -> Decision:
    if confidence >= AUTO_LINK_THRESHOLD:
        return "auto_linked"
    if confidence >= REVIEW_THRESHOLD:
        return "review"
    return "unclassified"


async def classify_product(
        session: AsyncSession,
        embedder: OllamaEmbedder,
        llm: LLMClient,
        product: ProductIn,
) -> ClassificationResult:
    trace = _Trace()
    classifier, critic = ClassifierAgent(llm), CriticAgent(llm)

    ctx = await build_context(session, product)
    candidates = await retrieve_candidates(session, embedder, ctx)

    def result(final, conf, breakdown, decision) -> ClassificationResult:
        path = next((c.full_path for c in candidates if final and c.category_id == final), None)
        return ClassificationResult(
            product_id=ctx.product_id, final_category_id=final, final_category_path=path,
            confidence=conf, confidence_breakdown=breakdown, decision=decision,
            candidates=candidates, steps=trace.steps,
        )

    if not candidates:
        return result(None, 0.0, {}, "failed")

    proposal = await _classify_valid(classifier, ctx, candidates, trace)
    if proposal is None:
        return result(None, 0.0, {}, "failed")

    ids = {c.category_id for c in candidates}
    critique: CriticOutput | None = None
    for _round in range(MAX_REVISIONS):
        started = time.perf_counter()
        try:
            critique = await critic.run(ctx, candidates, proposal)
        except LLMError as e:
            trace.add("critic", None, started, valid=False, error=str(e))
            critique = None
            break   # TODO: retry, or proceed without a critique (scores lower)?
        valid = critique.own_pick in ids and (
                critique.suggested_category_id is None or critique.suggested_category_id in ids
        )
        trace.add("critic", critique, started, valid=valid)
        if critique.verdict == "accept" or not valid:
            break
        revised = await _classify_valid(classifier, ctx, candidates, trace, feedback=critique)
        if revised is None:
            break   # keep the last valid proposal
        proposal = revised
        # TODO: stop early if the revision didn't change the category?

    confidence, breakdown = score(proposal, critique, candidates, ctx.related_category_ids)
    # TODO: persist to classification_runs / classification_steps
    return result(proposal.category_id, confidence, breakdown, decide(confidence))