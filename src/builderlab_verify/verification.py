"""OCR and Vision verification stage."""

import json
import logging
import re
from time import perf_counter
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
from builderlab_verify.metrics import record_stage
from builderlab_verify.ocr import _api_key_from_settings, _safe_error, _usage_value
from builderlab_verify.retry import RetryFailure, RetryInfo, call_with_retry
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


def _part_number_equivalent(ocr: str, vision: str) -> bool:
    """Accept punctuation-only variants without choosing a canonical value."""
    normalize = lambda value: re.sub(r"[^a-z0-9]", "", value.lower())
    return bool(normalize(ocr)) and normalize(ocr) == normalize(vision)


def _description_equivalent(ocr: str, vision: str) -> bool:
    """Accept a compatible trailing elaboration, but not an apparent negation."""
    normalize = lambda value: " ".join(value.lower().split())
    shorter, longer = sorted((normalize(ocr), normalize(vision)), key=len)
    if not shorter or not longer.startswith(shorter):
        return False
    added = longer[len(shorter):].lower()
    return not re.search(r"\b(?:not|without|except|instead)\b", added)


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
    started = perf_counter()
    resolved_key = _api_key_from_settings(api_key)
    if not resolved_key:
        raise GeminiVerificationError(
            "Gemini verification requires GEMINI_API_KEY in the environment or .env."
        )

    ocr = _read_extraction(run.ocr_json, ExtractionBranch.OCR, category, run.root.name)
    vision = _read_extraction(run.vision_json, ExtractionBranch.VISION, category, run.root.name)
    comparisons: dict[str, FieldComparison] = {}
    unresolved: dict[str, dict[str, FieldProvenance]] = {}
    notes: dict[str, str] = {}

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
            if field.name == "part_number" and _part_number_equivalent(ocr_field.value, vision_field.value):
                comparisons[field.name] = FieldComparison(
                    status=AlignmentStatus.MATCH, ocr=ocr_field, vision=vision_field
                )
                continue
            if field.name == "description" and _description_equivalent(ocr_field.value, vision_field.value):
                comparisons[field.name] = FieldComparison(
                    status=AlignmentStatus.MATCH, ocr=ocr_field, vision=vision_field
                )
                continue
            unresolved[field.name] = {"ocr": ocr_field, "vision": vision_field}
            continue
        comparisons[field.name] = FieldComparison(status=status, ocr=ocr_field, vision=vision_field)

    token_usage: TokenUsage | None = None
    retry_info = RetryInfo()
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
            response, retry_info = call_with_retry(
                lambda: gemini_client.models.generate_content(
                    model=MODEL_ID,
                    contents=prompt,
                    config=config,
                )
            )
        except RetryFailure as failure:
            error = failure.error
            retry_info = failure.info
            record_stage(run.metrics_json, "checker", model=MODEL_ID, usage=None,
                         latency_ms=(perf_counter() - started) * 1000,
                         error_category="api", error=_safe_error(error, resolved_key),
                         **retry_info.as_dict())
            raise GeminiVerificationError(
                f"Gemini verification request failed: {_safe_error(error, resolved_key)}"
            ) from error
        token_usage = _usage(response)
        parsed = getattr(response, "parsed", None)
        if parsed is None:
            message = "Gemini did not return structured verification JSON."
            record_stage(run.metrics_json, "checker", model=MODEL_ID, usage=token_usage,
                         latency_ms=(perf_counter() - started) * 1000,
                         error_category="schema", error=message, **retry_info.as_dict())
            raise GeminiVerificationError(message)
        try:
            checked = _CheckerResponse.model_validate(parsed)
        except ValidationError as error:
            message = str(error)
            if "corrected" in message.lower():
                message = "Gemini returned a corrected value; corrected values are forbidden."
            message = f"Gemini returned invalid structured verification: {message}"
            record_stage(run.metrics_json, "checker", model=MODEL_ID, usage=token_usage,
                         latency_ms=(perf_counter() - started) * 1000,
                         error_category="schema", error=message, **retry_info.as_dict())
            raise GeminiVerificationError(message) from error
        if checked.run_id != run.root.name:
            message = "Gemini returned an unexpected verification run_id."
            record_stage(run.metrics_json, "checker", model=MODEL_ID, usage=token_usage,
                         latency_ms=(perf_counter() - started) * 1000,
                         error_category="schema", error=message, **retry_info.as_dict())
            raise GeminiVerificationError(message)
        expected = set(unresolved)
        actual = set(checked.fields)
        if actual != expected:
            message = f"Gemini returned checker fields {sorted(actual)}; expected {sorted(expected)}."
            record_stage(run.metrics_json, "checker", model=MODEL_ID, usage=token_usage,
                         latency_ms=(perf_counter() - started) * 1000,
                         error_category="schema", error=message, **retry_info.as_dict())
            raise GeminiVerificationError(message)
        for name, decision in checked.fields.items():
            if decision.status is AlignmentStatus.UNCERTAIN:
                message = "Gemini may not return uncertain for non-null values."
                record_stage(run.metrics_json, "checker", model=MODEL_ID, usage=token_usage,
                             latency_ms=(perf_counter() - started) * 1000,
                             error_category="schema", error=message, **retry_info.as_dict())
                raise GeminiVerificationError(message)
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
    record_stage(run.metrics_json, "checker", model=MODEL_ID, usage=token_usage,
                 latency_ms=(perf_counter() - started) * 1000,
                 **retry_info.as_dict())
    return run.verification_json


def native_part_number_fallback_needed(run: RunDirectory) -> bool:
    """Return whether the completed first pass needs a part-number rescue."""

    if not run.verification_json.exists():
        return False
    try:
        verification = json.loads(run.verification_json.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    part_number = verification.get("fields", {}).get("part_number", {})
    return (
        verification.get("status") == RunStatus.HUMAN_REVIEW.value
        and part_number.get("status") != AlignmentStatus.MATCH.value
    )


def apply_native_part_number_fallback(run: RunDirectory, category: CategorySchema) -> bool:
    """Rescue only a reviewed part number when native text agrees with Vision.

    The existing first-pass artifact remains byte-for-byte untouched unless native
    text proves an equivalent ``part_number`` under the established semantics.
    """

    if not native_part_number_fallback_needed(run) or not run.native_json.exists():
        return False
    try:
        native = _read_extraction(
            run.native_json, ExtractionBranch.NATIVE_TEXT, category, run.root.name
        )
        vision = _read_extraction(run.vision_json, ExtractionBranch.VISION, category, run.root.name)
        verification = json.loads(run.verification_json.read_text(encoding="utf-8"))
    except (GeminiVerificationError, OSError, ValueError):
        return False

    native_part = _snapshot(native, "part_number").value
    vision_part = _snapshot(vision, "part_number").value
    if (
        native_part is None
        or vision_part is None
        or not _part_number_equivalent(native_part, vision_part)
    ):
        return False

    verification["fields"]["part_number"]["status"] = AlignmentStatus.MATCH.value
    verification.setdefault("notes", {})["part_number"] = (
        "Native PDF text aligned with Vision while Tesseract was uncertain or mismatched."
    )
    verification["status"] = (
        RunStatus.VERIFIED.value
        if all(
            field.get("status") == AlignmentStatus.MATCH.value
            for field in verification["fields"].values()
        )
        else RunStatus.HUMAN_REVIEW.value
    )
    run.verification_json.write_text(json.dumps(verification, indent=2) + "\n", encoding="utf-8")
    return True
