"""Stable data models shared by the verification pipeline stages."""

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field

from builderlab_verify.schemas import CategorySchema


class ExtractionBranch(StrEnum):
    OCR = "ocr"
    VISION = "vision"


class RunStatus(StrEnum):
    VERIFIED = "verified"
    HUMAN_REVIEW = "human_review"


class AlignmentStatus(StrEnum):
    MATCH = "match"
    MISMATCH = "mismatch"
    UNCERTAIN = "uncertain"


class FieldProvenance(BaseModel):
    """An extracted value and the source details available for that value."""

    value: str | int | float | bool | None
    unit: str | None = None
    page: int | None = Field(default=None, ge=1)


class ExtractionResult(BaseModel):
    """Structured output from one extraction branch."""

    run_id: str = Field(min_length=1)
    branch: ExtractionBranch
    category: CategorySchema
    fields: dict[str, FieldProvenance]


class FieldComparison(BaseModel):
    """The alignment result and both extracted values being compared."""

    status: AlignmentStatus
    ocr: FieldProvenance
    vision: FieldProvenance


class TokenUsage(BaseModel):
    """Non-sensitive token counts returned by the checker API."""

    prompt: int | None = None
    candidates: int | None = None
    thoughts: int | None = None
    total: int | None = None


class VerificationResult(BaseModel):
    """Field-level OCR/vision alignment and the resulting run status."""

    run_id: str = Field(min_length=1)
    status: RunStatus
    fields: dict[str, FieldComparison]
    notes: dict[str, str] = Field(default_factory=dict)


class HumanResolution(BaseModel):
    """A human decision used to produce the final run result."""

    reviewer: str = Field(min_length=1)
    fields: dict[str, FieldProvenance]
    note: str | None = None
    resolved_at: datetime


class FinalResult(BaseModel):
    """Persisted verified or human-resolved output for a run."""

    run_id: str = Field(min_length=1)
    status: RunStatus
    fields: dict[str, FieldProvenance]
    resolution: HumanResolution | None = None
