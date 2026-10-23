"""Untrusted agent outputs cross this strict, bounded validation boundary."""
from __future__ import annotations

import re
import unicodedata
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def normalized_text(value: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", value)).strip()


class Payload(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, allow_inf_nan=False)


class Evidence(Payload):
    page_number: int = Field(ge=1, strict=True)
    quote: str = Field(min_length=1)
    evidence_kind: Literal["text", "visual"] = "text"
    locator: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def visual_locator(self):
        if self.evidence_kind == "visual" and not self.locator:
            raise ValueError("Visual evidence requires a structured locator, e.g. figure/table/equation and bounding box")
        return self


IdeaKind = Literal["definition", "method", "assumption", "equation", "theory", "experiment", "baseline", "limitation", "open_question"]


class Idea(Payload):
    local_id: str = Field(min_length=1, max_length=160, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$")
    kind: IdeaKind
    title: str = Field(min_length=1)
    statement: str = Field(min_length=1)
    conditions: str = ""
    attribution: Literal["original", "background", "synthesis"]
    confidence: float = Field(ge=0, le=1)
    embedding_text: str = Field(min_length=1)
    validity_flags: list[str] = Field(default_factory=list)
    evidence: list[Evidence] = Field(min_length=1)


class Coverage(Payload):
    section_id: UUID
    status: Literal["covered", "unreadable"]
    notes: str = Field(min_length=1)


class Reference(Payload):
    citation_key: str = Field(min_length=1, max_length=160)
    title: str = Field(min_length=1)
    arxiv_id: str | None = None
    doi: str | None = None
    url: str | None = None


class Predecessor(Payload):
    idea_local_id: str = Field(min_length=1)
    citation_key: str = Field(min_length=1)
    evidence_page: int = Field(ge=1, strict=True)
    evidence_quote: str = Field(min_length=1)


class ExtractionPayload(Payload):
    ideas: list[Idea]
    coverage: list[Coverage] = Field(min_length=1)
    references: list[Reference] = Field(default_factory=list)
    predecessors: list[Predecessor] = Field(default_factory=list)

    @model_validator(mode="after")
    def unique_ids_and_links(self):
        for label, values in (
            ("idea local_id", [x.local_id for x in self.ideas]),
            ("section_id", [x.section_id for x in self.coverage]),
            ("citation_key", [x.citation_key for x in self.references]),
            ("predecessor pair", [(x.idea_local_id,x.citation_key) for x in self.predecessors]),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"Duplicate {label}")
        ideas = {x.local_id for x in self.ideas}
        refs = {x.citation_key for x in self.references}
        for nomination in self.predecessors:
            if nomination.idea_local_id not in ideas or nomination.citation_key not in refs:
                raise ValueError("Predecessor must name an idea and reference in this submission")
        return self


class Relevance(Payload):
    score: int = Field(ge=0, le=100, strict=True)
    reason: str = Field(min_length=1)


class IdeaRelevance(Relevance):
    local_id: str = Field(min_length=1)


class NominationDecision(Payload):
    idea_local_id: str = Field(min_length=1)
    citation_key: str = Field(min_length=1)
    approved: bool = Field(strict=True)
    reason: str = Field(min_length=1)


class ReviewPayload(Payload):
    accepted: bool = Field(strict=True)
    notes: str = Field(min_length=1)
    paper_relevance: Relevance | None = None
    idea_relevance: list[IdeaRelevance] = Field(default_factory=list)
    issues: list[str] = Field(default_factory=list)
    nomination_decisions: list[NominationDecision] = Field(default_factory=list)
    visual_evidence_confirmed: bool = Field(default=False, strict=True)

    @model_validator(mode="after")
    def coherent_review(self):
        scores = [item.local_id for item in self.idea_relevance]
        pairs = [(item.idea_local_id,item.citation_key) for item in self.nomination_decisions]
        if len(scores) != len(set(scores)) or len(pairs) != len(set(pairs)):
            raise ValueError("Duplicate idea scores or nomination decisions")
        if self.accepted and (self.paper_relevance is None or self.issues):
            raise ValueError("Acceptance requires paper relevance and no unresolved issues")
        if not self.accepted and not self.issues:
            raise ValueError("Rejected review requires at least one actionable issue")
        return self


RELEVANCE_RUBRIC = {
    "unrelated": {"min": 0, "max": 19, "description": "No substantiated connection to prompt/context compression"},
    "peripheral": {"min": 20, "max": 49, "description": "Adjacent technique or setting, without an established dependency"},
    "foundational": {"min": 50, "max": 69, "description": "Concept supporting a stated prompt/context compression mechanism"},
    "directly_applicable": {"min": 70, "max": 89, "description": "Direct method, comparison, or result applicable to prompt/context compression"},
    "central": {"min": 90, "max": 100, "description": "Primary subject is prompt/context compression"},
}
