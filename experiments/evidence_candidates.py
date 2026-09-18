"""Evidence-first part-number candidate harvesting and ranking experiment.

This module is intentionally isolated from the production extraction and
verification pipeline. It ranks values only from document evidence; filenames
are never accepted as candidate or ranking input.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from enum import StrEnum
from typing import Any, Iterable

from pydantic import BaseModel, ConfigDict, Field


class CandidateSource(StrEnum):
    NATIVE_TEXT = "native_text"
    TESSERACT_TSV = "tesseract_tsv"
    OCR_STRUCTURED = "ocr_structured"
    NATIVE_STRUCTURED = "native_structured"
    VISION = "vision"


class CandidateRole(StrEnum):
    ORDERABLE_PART = "orderable_part"
    FAMILY = "family"
    DOCUMENT_TITLE = "document_title"
    DRAWING_NUMBER = "drawing_number"
    VARIANT = "variant"
    UNKNOWN = "unknown"


class Candidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    raw_text: str = Field(min_length=1)
    normalized_value: str = Field(min_length=1)
    source: CandidateSource
    sources: list[CandidateSource] = Field(default_factory=list)
    page: int | None = Field(default=None, ge=1)
    bbox: list[float] | None = None
    region: str | None = None
    context: str = ""
    evidence_text: str = ""
    block_no: int | None = None
    line_no: int | None = None
    table_membership: bool = False
    occurrence_count: int = Field(default=1, ge=1)
    ocr_confidence: float | None = Field(default=None, ge=0, le=100)
    role: CandidateRole = CandidateRole.UNKNOWN
    score: float = 0.0
    accepted: bool = False
    abstention_reason: str | None = None


_PART_LABELS = (
    "manufacturer part number",
    "part number",
    "part no",
    "part #",
    "mpn",
    "order number",
    "order no",
    "catalog number",
    "catalog no",
    "product number",
    "product no",
    "item number",
)
_FAMILY_LABELS = ("product family", "family", "series", "product line")
_DRAWING_LABELS = ("drawing number", "drawing no", "dwg no", "dwg.")
_VARIANT_LABELS = ("variant", "option", "configuration", "revision", "rev")
_TITLE_WORDS = ("datasheet", "data sheet", "specification", "technical data", "brochure")


def normalize_candidate(value: str | None) -> str:
    return re.sub(r"[^a-z0-9]", "", (value or "").lower())


def _role_for_context(context: str, raw_text: str) -> CandidateRole:
    lowered = " ".join(f"{context} {raw_text}".lower().split())
    if any(label in lowered for label in _PART_LABELS):
        return CandidateRole.ORDERABLE_PART
    if any(label in lowered for label in _DRAWING_LABELS):
        return CandidateRole.DRAWING_NUMBER
    if any(label in lowered for label in _FAMILY_LABELS):
        return CandidateRole.FAMILY
    if any(label in lowered for label in _VARIANT_LABELS):
        return CandidateRole.VARIANT
    if any(word in lowered for word in _TITLE_WORDS):
        return CandidateRole.DOCUMENT_TITLE
    return CandidateRole.UNKNOWN


def _has_explicit_part_label(context: str) -> bool:
    lowered = " ".join(context.lower().split())
    return any(label in lowered for label in _PART_LABELS)


def _identifier_like(value: str) -> bool:
    normalized = normalize_candidate(value)
    if len(normalized) < 3 or not any(char.isdigit() for char in normalized):
        return False
    if len(normalized) > 80:
        return False
    return True


def _measurement_like(value: str) -> bool:
    """Reject short scalar measurements that commonly sit beside part labels."""

    compact = value.strip().lower()
    return bool(re.fullmatch(r"\d+(?:\.\d+)?(?:mm|cm|hz|w|v|a|f|c)?", compact)) and len(
        normalize_candidate(value)
    ) < 6


def build_candidate(
    *,
    raw_text: str,
    source: CandidateSource,
    page: int | None,
    bbox: list[float] | None,
    context: str,
    block_no: int | None = None,
    line_no: int | None = None,
    table_membership: bool = False,
    ocr_confidence: float | None = None,
    filename: str | None = None,
) -> Candidate:
    """Build one candidate; ``filename`` is accepted only to assert exclusion."""

    del filename
    raw_text = " ".join(str(raw_text).split()).strip()
    normalized = normalize_candidate(raw_text)
    if not normalized:
        raise ValueError("candidate text must contain an alphanumeric value")
    evidence_text = " ".join(part for part in (context.strip(), raw_text) if part)
    return Candidate(
        raw_text=raw_text,
        normalized_value=normalized,
        source=source,
        sources=[source],
        page=page,
        bbox=bbox,
        region="document_page" if page is not None else None,
        context=context.strip(),
        evidence_text=evidence_text,
        block_no=block_no,
        line_no=line_no,
        table_membership=table_membership,
        ocr_confidence=ocr_confidence,
        role=CandidateRole.UNKNOWN if _measurement_like(raw_text) else _role_for_context(context, raw_text),
    )


_ROLE_PRIORITY = {
    CandidateRole.ORDERABLE_PART: 6,
    CandidateRole.VARIANT: 4,
    CandidateRole.DRAWING_NUMBER: 3,
    CandidateRole.FAMILY: 2,
    CandidateRole.DOCUMENT_TITLE: 1,
    CandidateRole.UNKNOWN: 0,
}


def _merge_candidates(group: list[Candidate]) -> Candidate:
    representative = max(
        group,
        key=lambda candidate: (
            _ROLE_PRIORITY[candidate.role],
            _has_explicit_part_label(candidate.context),
            candidate.ocr_confidence or 0,
            len(candidate.raw_text),
        ),
    )
    roles = sorted(group, key=lambda candidate: _ROLE_PRIORITY[candidate.role], reverse=True)
    sources = list(dict.fromkeys(source for candidate in group for source in [candidate.source, *candidate.sources]))
    contexts = list(dict.fromkeys(candidate.context for candidate in group if candidate.context))
    confidence_values = [candidate.ocr_confidence for candidate in group if candidate.ocr_confidence is not None]
    return representative.model_copy(update={
        "sources": sources,
        "context": " | ".join(contexts[:4]),
        "evidence_text": " | ".join(candidate.evidence_text for candidate in group[:4]),
        "occurrence_count": len(group),
        "role": roles[0].role,
        "ocr_confidence": max(confidence_values) if confidence_values else None,
        "table_membership": any(candidate.table_membership for candidate in group),
    })


def _score(candidate: Candidate) -> float:
    score = {
        CandidateRole.ORDERABLE_PART: 6.0,
        CandidateRole.VARIANT: 1.0,
        CandidateRole.DRAWING_NUMBER: -1.0,
        CandidateRole.FAMILY: -2.0,
        CandidateRole.DOCUMENT_TITLE: -4.0,
        CandidateRole.UNKNOWN: 0.0,
    }[candidate.role]
    if _has_explicit_part_label(candidate.context):
        score += 4.0
    if candidate.table_membership:
        score += 1.0
    score += min(3.0, 1.5 * max(0, len(candidate.sources) - 1))
    score += min(2.0, math.log2(candidate.occurrence_count + 1))
    if candidate.ocr_confidence is not None:
        score += max(0.0, min(2.0, candidate.ocr_confidence / 50.0))
    return round(score, 4)


def rank_candidates(candidates: Iterable[Candidate]) -> list[Candidate]:
    """Aggregate identical values and rank them with conservative abstention."""

    groups: dict[str, list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        if candidate.normalized_value:
            groups[candidate.normalized_value].append(candidate)
    ranked = []
    for group in groups.values():
        merged = _merge_candidates(group)
        ranked.append(merged.model_copy(update={"score": _score(merged)}))
    ranked.sort(key=lambda candidate: (-candidate.score, candidate.normalized_value))

    for index, candidate in enumerate(ranked):
        next_score = ranked[index + 1].score if index + 1 < len(ranked) else None
        margin = candidate.score - next_score if next_score is not None else candidate.score
        explicit = _has_explicit_part_label(candidate.context)
        cross_branch = len(candidate.sources) >= 2
        accepted = (
            candidate.role is CandidateRole.ORDERABLE_PART
            and candidate.score >= 8.0
            and (explicit or cross_branch)
            and margin >= 1.5
        )
        reason = None if accepted else (
            "insufficient orderable-part evidence"
            if candidate.role is not CandidateRole.ORDERABLE_PART
            else "requires explicit label, cross-branch agreement, or stronger score margin"
        )
        ranked[index] = candidate.model_copy(update={
            "accepted": accepted,
            "abstention_reason": reason,
        })
    return ranked


def evaluate_ranked_cases(cases: Iterable[dict[str, Any]]) -> dict[str, Any]:
    cases = list(cases)
    total = len(cases)
    recall_1 = 0
    recall_3 = 0
    present = 0
    absent = 0
    misranked = 0
    rescues = 0
    false_positives = 0
    for case in cases:
        reference = normalize_candidate(str(case.get("reference") or ""))
        ranked = case.get("ranked", [])
        values = [candidate.normalized_value for candidate in ranked]
        if values[:1] == [reference]:
            recall_1 += 1
        if reference in values[:3]:
            recall_3 += 1
        if reference in values:
            present += 1
            if values[:1] != [reference]:
                misranked += 1
        else:
            absent += 1
        if ranked and ranked[0].accepted:
            if values[0] == reference:
                rescues += 1
            else:
                false_positives += 1
    return {
        "cases": total,
        "recall_at_1": recall_1 / total if total else 0.0,
        "recall_at_3": recall_3 / total if total else 0.0,
        "correct_value_present": present,
        "absent": absent,
        "misranked": misranked,
        "safe_rescues": rescues,
        "false_positives": false_positives,
    }
