"""Small, UI-agnostic helpers for the Streamlit review flow."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from pydantic import ValidationError

from builderlab_verify.models import (
    FieldProvenance,
    FinalResult,
    HumanResolution,
    RunStatus,
    VerificationResult,
)
from builderlab_verify.schemas import CategorySchema


def parse_category_json(raw: str) -> CategorySchema:
    """Parse the manual category-schema input used by the dashboard."""

    try:
        return CategorySchema.model_validate_json(raw)
    except (ValueError, ValidationError) as error:
        raise ValueError(f"Invalid category schema JSON: {error}") from error


def parse_schema_mapping_json(raw: str) -> dict[str, CategorySchema]:
    """Parse an explicit filename-to-schema mapping; no category inference is used."""

    try:
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            raise ValueError("Schema mapping must be a JSON object keyed by PDF filename.")
        return {str(filename): CategorySchema.model_validate(schema) for filename, schema in payload.items()}
    except (ValueError, TypeError, ValidationError) as error:
        raise ValueError(f"Invalid filename-to-schema mapping JSON: {error}") from error


def parse_json_scalar(raw: str) -> str | int | float | bool | None:
    """Parse a human-entered JSON scalar while keeping ordinary text convenient."""

    value = raw.strip()
    if not value:
        return None
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return value
    if parsed is None or isinstance(parsed, (str, int, float, bool)):
        return parsed
    raise ValueError("A resolved value must be a JSON scalar or plain text.")


def build_final_result(
    verification: VerificationResult,
    fields: dict[str, FieldProvenance],
    *,
    reviewer: str | None = None,
    note: str | None = None,
    resolved_at: datetime | None = None,
) -> FinalResult:
    """Build the persisted final result from verified or human-resolved fields."""

    if verification.status is RunStatus.VERIFIED:
        return FinalResult(run_id=verification.run_id, status=RunStatus.VERIFIED, fields=fields)

    if not reviewer or not reviewer.strip():
        raise ValueError("A reviewer is required for human-resolved runs.")
    resolution = HumanResolution(
        reviewer=reviewer.strip(),
        fields=fields,
        note=note.strip() or None if note else None,
        resolved_at=resolved_at or datetime.now(timezone.utc),
    )
    return FinalResult(
        run_id=verification.run_id,
        status=RunStatus.VERIFIED,
        fields=fields,
        resolution=resolution,
    )


def write_final_result(path: Path, result: FinalResult) -> Path:
    """Persist a validated final result using the run's fixed artifact path."""

    path.write_text(result.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return path
