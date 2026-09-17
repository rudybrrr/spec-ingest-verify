"""OCR and Vision verification stage."""

import json
import logging
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types
from pydantic import BaseModel, ConfigDict, ValidationError

from builderlab_verify.models import (
    AlignmentStatus,
    ExtractionBranch,
    ExtractionResult,
    FieldComparison,
    FieldProvenance,
    RunStatus,
    TokenUsage,
    VerificationResult,
)
from builderlab_verify.ocr import _api_key_from_settings, _safe_error, _usage_value
from builderlab_verify.schemas import CategorySchema
from builderlab_verify.storage import RunDirectory

LOGGER = logging.getLogger(__name__)
MODEL_ID = "gemini-3.5-flash-lite"


class GeminiVerificationError(RuntimeError):
    """Raised when verification cannot produce a valid structured result."""


class _CheckerDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: AlignmentStatus
    note: str | None = None


class _CheckerResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    fields: dict[str, _CheckerDecision]


def build_checker_response_schema(
    category: CategorySchema,
    field_names: set[str] | None = None,
) -> dict[str, Any]:
    """Build a response schema that can express status and explanation only."""

    allowed_fields = field_names or {field.name for field in category.fields}
    return {
        "type": "OBJECT",
        "properties": {
            "run_id": {"type": "STRING"},
            "fields": {
                "type": "OBJECT",
                "properties": {
                    field.name: {
                        "type": "OBJECT",
                        "properties": {
                            "status": {
                                "type": "STRING",
                                "enum": [status.value for status in AlignmentStatus],
                            },
                            "note": {"type": "STRING", "nullable": True},
                        },
                        "required": ["status", "note"],
                    }
                    for field in category.fields
                    if field.name in allowed_fields
                },
                "required": [],
            },
        },
        "required": ["run_id", "fields"],
    }


def _read_extraction(path: Path, branch: ExtractionBranch, category: CategorySchema, run_id: str) -> ExtractionResult:
    if not path.exists():
        raise GeminiVerificationError(f"Verification requires extraction file: {path}")
    try:
        result = ExtractionResult.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, ValidationError) as error:
        raise GeminiVerificationError(f"Invalid {branch.value} extraction JSON: {error}") from error
    if result.run_id != run_id or result.branch is not branch:
        raise GeminiVerificationError(f"{branch.value} extraction has an unexpected run_id or branch.")
    if result.category != category:
        raise GeminiVerificationError(f"{branch.value} extraction category differs from supplied schema.")
    return result


def _snapshot(result: ExtractionResult, name: str) -> FieldProvenance:
    return result.fields.get(name, FieldProvenance(value=None))


def _usage(response: object) -> TokenUsage:
    metadata = getattr(response, "usage_metadata", None)
    return TokenUsage(
        prompt=_usage_value(metadata, "prompt_token_count"),
        candidates=_usage_value(metadata, "candidates_token_count"),
        thoughts=_usage_value(metadata, "thoughts_token_count"),
        total=_usage_value(metadata, "total_token_count"),
    )


def verify_ocr_vision(
    run: RunDirectory,
    category: CategorySchema,
    *,
    client: Any | None = None,
    api_key: str | None = None,
) -> Path:
    resolved_key = _api_key_from_settings(api_key)
    if not resolved_key:
        raise GeminiVerificationError(
            "Gemini verification requires GEMINI_API_KEY in the environment or .env."
        )

    ocr = _read_extraction(run.ocr_json, ExtractionBranch.OCR, category, run.root.name)
    vision = _read_extraction(run.vision_json, ExtractionBranch.VISION, category, run.root.name)
    comparisons: dict[str, FieldComparison] = {}
    unresolved: dict[str, dict[str, FieldProvenance]] = {}

    for field in category.fields:
        ocr_field = _snapshot(ocr, field.name)
        vision_field = _snapshot(vision, field.name)
        ocr_null = ocr_field.value is None
        vision_null = vision_field.value is None
        if ocr_null != vision_null:
            status = AlignmentStatus.MISMATCH
        elif ocr_null:
            status = AlignmentStatus.UNCERTAIN if field.required else AlignmentStatus.MATCH
        else:
            unresolved[field.name] = {"ocr": ocr_field, "vision": vision_field}
            continue
        comparisons[field.name] = FieldComparison(status=status, ocr=ocr_field, vision=vision_field)

    token_usage: TokenUsage | None = None
    notes: dict[str, str] = {}
    if unresolved:
        payload = {
            name: {branch: value.model_dump() for branch, value in snapshots.items()}
            for name, snapshots in unresolved.items()
        }
        prompt = (
            f"Compare semantic equivalence for corresponding non-null OCR and Vision values for run_id {run.root.name!r}. "
            "Return only the supplied field names and status match or mismatch. Do not correct, "
            "select, normalize, or invent values. Return the exact run_id supplied above. You may provide "
            "a brief note for a mismatch.\n\n"
            f"CATEGORY SCHEMA:\n{category.model_dump_json()}\n\nVALUES:\n{json.dumps(payload)}"
        )
        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=build_checker_response_schema(category, set(unresolved)),
            thinking_config=types.ThinkingConfig(thinking_level=types.ThinkingLevel.MINIMAL),
        )
        try:
            gemini_client = client or genai.Client(api_key=resolved_key)
            response = gemini_client.models.generate_content(
                model=MODEL_ID,
                contents=prompt,
                config=config,
            )
        except Exception as error:
            raise GeminiVerificationError(
                f"Gemini verification request failed: {_safe_error(error, resolved_key)}"
            ) from error
        token_usage = _usage(response)
        parsed = getattr(response, "parsed", None)
        if parsed is None:
            raise GeminiVerificationError("Gemini did not return structured verification JSON.")
        try:
            checked = _CheckerResponse.model_validate(parsed)
        except ValidationError as error:
            message = str(error)
            if "corrected" in message.lower():
                message = "Gemini returned a corrected value; corrected values are forbidden."
            raise GeminiVerificationError(f"Gemini returned invalid structured verification: {message}") from error
        if checked.run_id != run.root.name:
            raise GeminiVerificationError("Gemini returned an unexpected verification run_id.")
        expected = set(unresolved)
        actual = set(checked.fields)
        if actual != expected:
            raise GeminiVerificationError(
                f"Gemini returned checker fields {sorted(actual)}; expected {sorted(expected)}."
            )
        for name, decision in checked.fields.items():
            if decision.status is AlignmentStatus.UNCERTAIN:
                raise GeminiVerificationError("Gemini may not return uncertain for non-null values.")
            compared = unresolved[name]
            comparisons[name] = FieldComparison(
                status=decision.status,
                ocr=compared["ocr"],
                vision=compared["vision"],
            )
            if decision.note:
                notes[name] = decision.note

    status = (
        RunStatus.VERIFIED
        if all(comparison.status is AlignmentStatus.MATCH for comparison in comparisons.values())
        else RunStatus.HUMAN_REVIEW
    )
    result = VerificationResult(
        run_id=run.root.name,
        status=status,
        fields=comparisons,
        notes=notes,
    )
    artifact = result.model_dump()
    if token_usage is not None:
        artifact["token_usage"] = token_usage.model_dump()
    run.verification_json.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    return run.verification_json
