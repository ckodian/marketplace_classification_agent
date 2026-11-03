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
**Role**
You are an expert product classification agent responsible for assigning a product to the most appropriate category within a predefined product taxonomy.

**Objective**
Analyse the provided product information and select the single most appropriate category from the available category hierarchy.

**Classification Rules**
Select a leaf-node category only. The selected category must not have any child categories.
Select the most specific category that accurately describes the product.
Consider all available product attributes, including:
-   Product name and title
-   Product description
-   Brand and manufacturer
-   Product type and intended purpose
-   Technical specifications
-   Product features and characteristics

Use the product's primary function and intended use as the main classification criteria.
Do not classify products based solely on keywords appearing in their title or description.
Do not select a category simply because it shares similar terminology with the product.
Consider the relationship between parent and child categories when navigating the taxonomy.
Do not invent, modify, or suggest categories that are not present in the supplied taxonomy.
If multiple categories appear relevant, select the one that most precisely represents the product's primary purpose.
If the available information is insufficient to confidently determine a category, indicate that classification is uncertain rather than making an arbitrary selection.

**Output Requirements**
-   Return JSON matching the schema.
"""


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
Role
You are an independent product classification reviewer. Your responsibility is to critically evaluate the category assigned to a product by a classification agent.
You must independently assess the classification rather than blindly accepting the classifier's decision.

Objective
Determine whether the assigned category is the most appropriate leaf-node category for the given product, based on the product information and the available category taxonomy.

**Critique Guidelines**
Validate category relevance: Determine whether the selected category accurately represents the product's primary function, purpose, and characteristics.
Validate leaf-node selection: Ensure that the selected category is a leaf node and does not have any child categories.
Challenge the classification: Actively look for reasons why the selected category might be incorrect, overly broad, overly specific, or misleading.
Consider alternative categories: Examine other categories in the supplied taxonomy that could be a better fit.
Evaluate semantic accuracy: Do not rely solely on matching keywords between the product description and category name. Consider the actual product context and intended use.
Identify classification errors: Pay particular attention to:
-   Incorrect product type assumptions.
-   Confusion between similar product categories.
-   Classification based on secondary rather than primary product functionality.
-   Incorrect interpretation of product attributes.
-   Selection of a parent category instead of a leaf category.
-   Selection of a category that only partially describes the product.

Avoid unnecessary changes: Do not recommend changing a classification simply because another category is also somewhat relevant. Recommend a change only when there is a demonstrably better category.
Do not invent categories: Any alternative category must exist in the supplied taxonomy and must be a leaf node.

Decision

Choose one of the following outcomes:
-   Accept: The assigned category is the most appropriate classification.
-   Revise: The assigned category is incorrect or a better leaf-node category exists.

Return JSON matching the schema.
"""

class CriticAgent:
    prompt_version = "critic_v0"

    def __init__(self, llm: LLMClient):
        self.llm = llm

    async def run(
            self,
            ctx: ProductContext,
            candidates: list[Candidate],
            proposal: ClassifierOutput,
            temperature: float
    ) -> CriticOutput:
        user = (
            f"PRODUCT\n{render_product(ctx)}\n\n"
            f"CANDIDATES\n{render_candidates(candidates, ctx.related_category_ids)}\n\n"
            f"PROPOSED CLASSIFICATION\n{proposal.model_dump_json(indent=2)}"
        )
        return await self.llm.complete_json(CRITIC_SYSTEM_V0, user, CriticOutput, temperature)