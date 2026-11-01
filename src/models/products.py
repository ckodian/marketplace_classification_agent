"""Data contracts for the classification pipeline.

Three groups:
  API input      ProductIn               what a client or event sends
  Pipeline state ProductContext,         what the agents see
                 Candidate
  LLM output     ClassifierOutput,       structured output; their JSON schema
                 CriticOutput            is sent to the model as the format

Keep the LLM-output models small and flat. Every field is something the
model must produce, so every field costs tokens and is a chance to fail
validation.
"""

from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator

AttributeValue = str | int | float | bool | list[str]


# ---------------------------------------------------------------------------
# API input
# ---------------------------------------------------------------------------

class Price(BaseModel):
    amount: Decimal = Field(ge=0)
    currency: str = Field(min_length=3, max_length=3, examples=["AUD"])


class ProductIn(BaseModel):
    """What the caller sends. Only id and title are required."""
    model_config = ConfigDict(str_strip_whitespace=True)

    product_id: str = Field(min_length=1)
    title: str = Field(min_length=1, max_length=500)
    description: str | None = Field(None, max_length=10_000)
    attributes: dict[str, AttributeValue] = Field(default_factory=dict)
    price: Price | None = None
    image_urls: list[HttpUrl] = Field(default_factory=list, max_length=10)
    base_product_id: str | None = None
    sibling_ids: list[str] = Field(default_factory=list, max_length=50)


# ---------------------------------------------------------------------------
# Pipeline state
# ---------------------------------------------------------------------------

class ProductContext(ProductIn):
    """ProductIn plus everything the pipeline derives before the LLM runs."""
    image_caption: str | None = None
    # Categories already linked to the base product or siblings
    related_category_ids: list[str] = Field(default_factory=list)

    def to_query_text(self) -> str:
        """Text embedded for candidate retrieval. Deterministic, so runs are reproducible."""
        parts = [self.title]
        if self.description:
            parts.append(self.description[:2000])
        if self.attributes:
            parts.append("; ".join(f"{k}: {v}" for k, v in sorted(self.attributes.items())))
        if self.image_caption:
            parts.append(self.image_caption)
        return "\n".join(parts)


class Candidate(BaseModel):
    category_id: str
    name: str
    full_path: str
    description: str | None = None
    similarity: float = Field(ge=-1, le=1)   # cosine similarity
    rank: int | None = None                  # None for categories added from siblings
    source: Literal["vector", "sibling"]


# ---------------------------------------------------------------------------
# LLM outputs
# ---------------------------------------------------------------------------

class Alternate(BaseModel):
    category_id: str
    reason: str = Field(max_length=300)


class ClassifierOutput(BaseModel):
    category_id: str = Field(description="Must be one of the candidate category ids.")
    confidence: float = Field(ge=0, le=1)
    rationale: str = Field(max_length=800)
    evidence: list[str] = Field(
        default_factory=list, max_length=6,
        description="Facts from the product data that support the choice.",
    )
    alternates: list[Alternate] = Field(default_factory=list, max_length=3)


IssueType = Literal[
    "not_leaf",              # chose a parent when a child fits
    "too_generic",
    "better_sibling",
    "contradicts_related",   # disagrees with base product / sibling categories
    "unsupported_evidence",  # rationale cites facts not in the product data
    "other",
]


class CriticIssue(BaseModel):
    type: IssueType
    detail: str = Field(max_length=400)


class CriticOutput(BaseModel):
    # The critic commits to its own pick BEFORE judging the classifier's.
    # Small models otherwise just agree with whatever they're shown.
    own_pick: str = Field(description="The critic's own choice from the candidates.")
    verdict: Literal["accept", "revise"]
    issues: list[CriticIssue] = Field(default_factory=list, max_length=5)
    suggested_category_id: str | None = None
    confidence: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def _revise_needs_suggestion(self) -> "CriticOutput":
        if self.verdict == "revise" and not self.suggested_category_id:
            raise ValueError("verdict 'revise' requires suggested_category_id")
        return self


# ---------------------------------------------------------------------------
# Pipeline result
# ---------------------------------------------------------------------------

Decision = Literal["auto_linked", "review", "unclassified", "failed"]


class Step(BaseModel):
    step_no: int
    role: Literal["classifier", "critic"]
    output: ClassifierOutput | CriticOutput | None = None
    valid_choice: bool = True
    error: str | None = None
    latency_ms: int


class ClassificationResult(BaseModel):
    product_id: str
    final_category_id: str | None
    final_category_path: str | None
    confidence: float
    confidence_breakdown: dict[str, float]
    decision: Decision
    candidates: list[Candidate]
    steps: list[Step]