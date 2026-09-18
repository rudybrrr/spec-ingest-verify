"""Isolated Docling layout/table candidate experiment.

This module is deliberately separate from production verification. It accepts
only Docling document objects and emits auditable candidate evidence.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Iterable

from pydantic import BaseModel, ConfigDict, Field

from experiments.evidence_candidates import CandidateRole, normalize_candidate


_PART_LABELS = ("manufacturer part number", "part number", "part no", "part #", "mpn", "order number", "order no", "catalog number", "catalog no", "product number", "product no", "item number")
_FAMILY_LABELS = ("product family", "family", "series", "product line")
_DRAWING_LABELS = ("drawing number", "drawing no", "dwg no", "dwg.")
_VARIANT_LABELS = ("variant", "option", "configuration", "revision", "rev")


class DoclingCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    raw_text: str = Field(min_length=1)
    normalized_value: str = Field(min_length=1)
    source: str = "docling"
    page: int | None = None
    bbox: list[float] | None = None
    region: str | None = None
    context: str = ""
    element_label: str | None = None
    source_ref: str | None = None
    table_index: int | None = None
    row_index: int | None = None
    column_index: int | None = None
    row_text: str = ""
    column_header: str | None = None
    occurrence_count: int = 1
    role: str = "unknown"
    score: float = 0.0
    accepted: bool = False
    abstention_reason: str | None = None


def _role(context: str, raw: str) -> str:
    text = f"{context} {raw}".lower()
    if any(x in text for x in _PART_LABELS):
        return CandidateRole.ORDERABLE_PART.value
    if any(x in text for x in _DRAWING_LABELS):
        return CandidateRole.DRAWING_NUMBER.value
    if any(x in text for x in _FAMILY_LABELS):
        return CandidateRole.FAMILY.value
    if any(x in text for x in _VARIANT_LABELS):
        return CandidateRole.VARIANT.value
    return CandidateRole.UNKNOWN.value


def _identifier_like(value: str) -> bool:
    compact = normalize_candidate(value)
    return 3 <= len(compact) <= 80 and any(ch.isdigit() for ch in compact)


def candidate_from_evidence(*, raw_text: str, page: int | None, context: str, bbox: list[float] | None = None, table_index: int | None = None, row_index: int | None = None, column_index: int | None = None, row_text: str = "", source_ref: str | None = None, element_label: str | None = None, column_header: str | None = None) -> DoclingCandidate:
    raw = " ".join(str(raw_text).split()).strip()
    normalized = normalize_candidate(raw)
    if not normalized:
        raise ValueError("candidate text must contain an alphanumeric value")
    return DoclingCandidate(
        raw_text=raw,
        normalized_value=normalized,
        page=page,
        bbox=bbox,
        region="document_page" if page is not None else None,
        context=" ".join(context.split()),
        element_label=element_label,
        source_ref=source_ref,
        table_index=table_index,
        row_index=row_index,
        column_index=column_index,
        row_text=row_text,
        column_header=column_header,
        role=_role(context, raw),
    )


def _provenance(item: Any) -> tuple[int | None, list[float] | None]:
    prov = getattr(item, "prov", None) or []
    if not prov:
        return None, None
    p = prov[0]
    page = getattr(p, "page_no", None)
    bbox = getattr(p, "bbox", None)
    if bbox is not None:
        bbox = [float(getattr(bbox, x)) for x in ("l", "t", "r", "b") if hasattr(bbox, x)] or None
    return int(page) if page is not None else None, bbox


def _text(item: Any) -> str:
    value = item.get("text", "") if isinstance(item, dict) else getattr(item, "text", "")
    return " ".join(str(value or "").split())


def _table_cells(table: Any) -> list[Any]:
    data = getattr(table, "data", None)
    return list(getattr(data, "table_cells", None) or [])


def harvest_docling_candidates(document: Any) -> tuple[list[DoclingCandidate], dict[str, Any]]:
    candidates: list[DoclingCandidate] = []
    element_count = 0
    group_meta: dict[str, tuple[int, int, int]] = {}
    group_text: dict[str, str] = {}
    for group in list(getattr(document, "groups", []) or []):
        self_ref = getattr(group, "self_ref", "")
        parent = getattr(getattr(group, "parent", None), "cref", "")
        match = re.search(r"#/tables/(\d+)", str(parent))
        name = str(getattr(group, "name", ""))
        numbers = [int(x) for x in re.findall(r"\d+", name)]
        if match and len(numbers) >= 3:
            group_meta[self_ref] = (int(match.group(1)), numbers[-3], numbers[-2])
            refs = [getattr(child, "cref", "") for child in getattr(group, "children", []) or []]
            group_text[self_ref] = " | ".join(_text(document.texts[int(ref.rsplit('/', 1)[-1])]) for ref in refs if ref.startswith("#/texts/"))
    row_texts: dict[tuple[int, int], list[str]] = defaultdict(list)
    for ref, (table_index, row, _col) in group_meta.items():
        if group_text.get(ref):
            row_texts[(table_index, row)].append(group_text[ref])
    for index, item in enumerate(list(getattr(document, "texts", []) or [])):
        raw = _text(item)
        element_count += 1
        if not _identifier_like(raw):
            continue
        page, bbox = _provenance(item)
        label = str(getattr(getattr(item, "label", None), "value", getattr(item, "label", "")) or "")
        parent_ref = getattr(getattr(item, "parent", None), "cref", "")
        meta = group_meta.get(parent_ref)
        row_text = " | ".join(row_texts.get((meta[0], meta[1]), [])) if meta else ""
        candidates.append(candidate_from_evidence(raw_text=raw, page=page, bbox=bbox, context=row_text or raw, element_label=label, source_ref=f"text-{index}", table_index=meta[0] if meta else None, row_index=meta[1] if meta else None, column_index=meta[2] if meta else None, row_text=row_text))
    for table_index, table in enumerate(list(getattr(document, "tables", []) or [])):
        cells = _table_cells(table)
        element_count += 1
        by_row: dict[int, list[Any]] = defaultdict(list)
        for cell in cells:
            row = int(getattr(cell, "start_row_offset_idx", getattr(cell, "row_index", 0)) or 0)
            by_row[row].append(cell)
        for row, row_cells in by_row.items():
            row_cells.sort(key=lambda c: int(getattr(c, "start_col_offset_idx", getattr(c, "col_index", 0)) or 0))
            row_text = " | ".join(_text(c) for c in row_cells if _text(c))
            for cell in row_cells:
                raw = _text(cell)
                if not _identifier_like(raw):
                    continue
                page, bbox = _provenance(cell)
                col = int(getattr(cell, "start_col_offset_idx", getattr(cell, "col_index", 0)) or 0)
                candidates.append(candidate_from_evidence(raw_text=raw, page=page, bbox=bbox, context=row_text, table_index=table_index, row_index=row, column_index=col, row_text=row_text, source_ref=f"table-{table_index}/row-{row}/cell-{col}"))
    return candidates, {"text_elements": len(getattr(document, "texts", []) or []), "tables": len(getattr(document, "tables", []) or []), "elements_seen": element_count}


def rank_docling_candidates(candidates: Iterable[DoclingCandidate]) -> list[DoclingCandidate]:
    groups: dict[str, list[DoclingCandidate]] = defaultdict(list)
    for candidate in candidates:
        groups[candidate.normalized_value].append(candidate)
    role_score = {"orderable_part": 8.0, "variant": 1.0, "drawing_number": -1.0, "family": -2.0, "document_title": -3.0, "unknown": 0.0}
    ranked: list[DoclingCandidate] = []
    for values in groups.values():
        best = max(values, key=lambda c: (role_score.get(c.role, 0), "part number" in c.context.lower(), c.table_index is not None, len(c.raw_text)))
        explicit = any(label in best.context.lower() for label in _PART_LABELS)
        score = role_score.get(best.role, 0.0) + (3.0 if explicit else 0.0) + (1.0 if best.table_index is not None else 0.0) + min(2.0, 0.5 * (len(values) - 1))
        ranked.append(best.model_copy(update={"occurrence_count": len(values), "score": round(score, 3)}))
    ranked.sort(key=lambda c: (-c.score, c.normalized_value))
    for i, candidate in enumerate(ranked):
        margin = candidate.score - (ranked[i + 1].score if i + 1 < len(ranked) else 0)
        explicit = any(label in candidate.context.lower() for label in _PART_LABELS)
        accepted = candidate.role == "orderable_part" and explicit and candidate.page is not None and margin >= 1.5
        ranked[i] = candidate.model_copy(update={"accepted": accepted, "abstention_reason": None if accepted else "insufficient explicit labeled evidence"})
    return ranked


def evaluate_docling_cases(cases: Iterable[dict[str, Any]], previous_missing: set[str] | None = None) -> dict[str, Any]:
    cases = list(cases)
    previous_missing = previous_missing or set()
    metrics = {"cases": len(cases), "recall_at_1": 0, "recall_at_3": 0, "correct_value_present": 0, "absent": 0, "misranked": 0, "previous_missing_recovered": 0, "safe_rescues": 0, "false_positives": 0}
    for case in cases:
        ref = normalize_candidate(str(case.get("reference") or ""))
        ranked = case.get("ranked", [])
        values = [x.normalized_value for x in ranked]
        present = ref in values
        if values[:1] == [ref]: metrics["recall_at_1"] += 1
        if ref in values[:3]: metrics["recall_at_3"] += 1
        if present: metrics["correct_value_present"] += 1
        else: metrics["absent"] += 1
        if present and values[:1] != [ref]: metrics["misranked"] += 1
        if case.get("run_id") in previous_missing and present: metrics["previous_missing_recovered"] += 1
        if ranked and ranked[0].accepted:
            if values[0] == ref: metrics["safe_rescues"] += 1
            else: metrics["false_positives"] += 1
    total = len(cases) or 1
    metrics["recall_at_1"] /= total
    metrics["recall_at_3"] /= total
    return metrics
