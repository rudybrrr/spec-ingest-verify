from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from builderlab_verify.models import (
    AlignmentStatus,
    ExtractionBranch,
    ExtractionResult,
    FieldProvenance,
    FinalResult,
    HumanResolution,
    RunStatus,
    VerificationResult,
)
from builderlab_verify.schemas import CategoryField, CategorySchema


def test_extraction_result_preserves_category_and_field_provenance() -> None:
    category = CategorySchema(
        name="laptop",
        fields=[CategoryField(name="screen_size")],
    )

    result = ExtractionResult(
        run_id="run-001",
        branch=ExtractionBranch.OCR,
        category=category,
        fields={
            "screen_size": FieldProvenance(value=15.6, unit="in", page=2),
            "color": FieldProvenance(value="silver"),
        },
    )

    assert result.model_dump() == {
        "run_id": "run-001",
        "branch": "ocr",
        "category": {
            "name": "laptop",
            "fields": [{"name": "screen_size", "description": None, "required": True}],
        },
        "fields": {
            "screen_size": {"value": 15.6, "unit": "in", "page": 2},
            "color": {"value": "silver", "unit": None, "page": None},
        },
    }


def test_extraction_result_rejects_non_positive_source_page() -> None:
    with pytest.raises(ValidationError):
        FieldProvenance(value="x", page=0)


def test_verification_result_records_alignment_and_human_review_status() -> None:
    result = VerificationResult(
        run_id="run-001",
        status=RunStatus.HUMAN_REVIEW,
        fields={
            "screen_size": {
                "status": AlignmentStatus.MATCH,
                "ocr": FieldProvenance(value=15.6, unit="in", page=2),
                "vision": FieldProvenance(value=15.6, unit="in", page=1),
            },
            "color": {
                "status": AlignmentStatus.MISMATCH,
                "ocr": FieldProvenance(value="silver", page=1),
                "vision": FieldProvenance(value="gray", page=1),
            },
        },
        notes={"color": "OCR says silver; vision says gray"},
    )

    assert result.model_dump() == {
        "run_id": "run-001",
        "status": "human_review",
        "fields": {
            "screen_size": {
                "status": "match",
                "ocr": {"value": 15.6, "unit": "in", "page": 2},
                "vision": {"value": 15.6, "unit": "in", "page": 1},
            },
            "color": {
                "status": "mismatch",
                "ocr": {"value": "silver", "unit": None, "page": 1},
                "vision": {"value": "gray", "unit": None, "page": 1},
            },
        },
        "notes": {"color": "OCR says silver; vision says gray"},
    }


def test_human_resolution_and_final_result_capture_resolved_fields() -> None:
    resolved_at = datetime(2026, 9, 17, 12, 0, tzinfo=timezone.utc)
    resolution = HumanResolution(
        reviewer="analyst@example.com",
        fields={"color": FieldProvenance(value="silver", page=1)},
        note="Confirmed against the source PDF.",
        resolved_at=resolved_at,
    )
    final = FinalResult(
        run_id="run-001",
        status=RunStatus.VERIFIED,
        fields=resolution.fields,
        resolution=resolution,
    )

    assert final.status is RunStatus.VERIFIED
    assert final.resolution.reviewer == "analyst@example.com"
    assert final.fields["color"].value == "silver"
