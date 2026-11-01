"""Classifier and critic agents.

Bare bones: the plumbing (rendering, calling the LLM, returning typed output)
is done. The prompts are deliberately minimal v0 placeholders so the pipeline
runs end to end. Designing them is the real work; see the TODOs.
"""

import json

from src.services.llm import LLMClient
from src.models.products import Candidate, ClassifierOutput, CriticOutput, ProductContext


# ---------------------------------------------------------------------------
# Rendering shared by both agents
# ---------------------------------------------------------------------------

def render_product(ctx: ProductContext) -> str:
    data = {
        "title": ctx.title,
        "description": ctx.description,
        "attributes": ctx.attributes or None,
        "price": f"{ctx.price.amount} {ctx.price.currency}" if ctx.price else None,
        "image_caption": ctx.image_caption,
    }
    return json.dumps({k: v for k, v in data.items() if v is not None},
                      ensure_ascii=False, indent=2)


def render_candidates(candidates: list[Candidate], related_ids: list[str]) -> str:
    lines = []
    for c in candidates:
        tags = []
        if c.category_id in related_ids:
            tags.append("used by related products")
        line = f"- id={c.category_id} | {c.full_path}"
        if tags:
            line += f"  [{', '.join(tags)}]"
        if c.description:
            line += f"\n    {c.description}"
        lines.append(line)
    return "\n".join(lines)
    # TODO: should the model see similarity scores? It anchors on them.
    # Try with and without, and compare on the gold set.


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

# TODO(prompt): v0 placeholder. Things to decide and test:
#   - leaf preference: deepest correct category, never a parent
#   - how to weigh "used by related products" (strong prior, not a rule)
#   - what confidence should mean, with examples of 0.9 vs 0.5
#   - what to do when no candidate fits (pick closest + low confidence?)
CLASSIFIER_SYSTEM_V0 = """\
You classify marketplace products into a category tree.
Choose exactly one category id from the candidate list. Never invent an id.
Return JSON matching the schema."""


class ClassifierAgent:
    prompt_version = "classifier_v0"

    def __init__(self, llm: LLMClient):
        self.llm = llm

    async def run(
            self,
            ctx: ProductContext,
            candidates: list[Candidate],
            feedback: CriticOutput | None = None,
    ) -> ClassifierOutput:
        user = (
            f"PRODUCT\n{render_product(ctx)}\n\n"
            f"CANDIDATES\n{render_candidates(candidates, ctx.related_category_ids)}"
        )
        if feedback is not None:
            # TODO(prompt): how should a revision use the critique? Show the
            # previous answer too? Tell it the critic can be wrong?
            user += f"\n\nREVIEWER FEEDBACK\n{feedback.model_dump_json(indent=2)}"
        return await self.llm.complete_json(CLASSIFIER_SYSTEM_V0, user, ClassifierOutput)


# ---------------------------------------------------------------------------
# Critic
# ---------------------------------------------------------------------------

# TODO(prompt): v0 placeholder. The critic's value depends on independence:
#   - make it pick its own category BEFORE reading the classifier's answer
#     (own_pick comes first in the schema for this reason)
#   - give it the checklist: not_leaf, too_generic, better_sibling,
#     contradicts_related, unsupported_evidence
#   - consider a different model or temperature from the classifier
CRITIC_SYSTEM_V0 = """\
You review a product classification.
First choose your own category id from the candidates, then judge the proposed one.
Return "accept" or "revise"; if revising, give suggested_category_id from the candidates.
Return JSON matching the schema."""


class CriticAgent:
    prompt_version = "critic_v0"

    def __init__(self, llm: LLMClient):
        self.llm = llm

    async def run(
            self,
            ctx: ProductContext,
            candidates: list[Candidate],
            proposal: ClassifierOutput,
    ) -> CriticOutput:
        user = (
            f"PRODUCT\n{render_product(ctx)}\n\n"
            f"CANDIDATES\n{render_candidates(candidates, ctx.related_category_ids)}\n\n"
            f"PROPOSED CLASSIFICATION\n{proposal.model_dump_json(indent=2)}"
        )
        return await self.llm.complete_json(CRITIC_SYSTEM_V0, user, CriticOutput)