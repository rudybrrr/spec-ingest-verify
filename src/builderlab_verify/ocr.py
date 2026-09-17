"""Gemini structured extraction from raw OCR text."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types
from pydantic import ValidationError

from builderlab_verify.config import Settings
from builderlab_verify.models import ExtractionBranch, ExtractionResult
from builderlab_verify.schemas import CategorySchema
from builderlab_verify.storage import RunDirectory

LOGGER = logging.getLogger(__name__)
MODEL_ID = "gemini-3.5-flash-lite"


class GeminiExtractionError(RuntimeError):
    """Raised when Gemini cannot produce a valid structured OCR extraction."""


def _field_response_schema() -> dict[str, Any]:
    return {
        "type": "OBJECT",
        "properties": {
            "value": {"type": "STRING", "nullable": True},
            "unit": {"type": "STRING", "nullable": True},
            "page": {"type": "INTEGER", "nullable": True},
        },
        "required": [],
    }


def build_response_schema(
    category: CategorySchema, branch: ExtractionBranch = ExtractionBranch.OCR
) -> dict[str, Any]:
    """Build the Gemini JSON schema with only the supplied category fields."""

    field_names = [field.name for field in category.fields]
    category_schema = {
        "type": "OBJECT",
        "properties": {
            "name": {"type": "STRING", "enum": [category.name]},
            "fields": {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "name": {"type": "STRING"},
                        "description": {"type": "STRING", "nullable": True},
                        "required": {"type": "BOOLEAN"},
                    },
                    "required": ["name", "description", "required"],
                },
            },
        },
        "required": ["name", "fields"],
    }
    return {
        "type": "OBJECT",
        "properties": {
            "run_id": {"type": "STRING"},
            "branch": {"type": "STRING", "enum": [branch.value]},
            "category": category_schema,
            "fields": {
                "type": "OBJECT",
                "properties": {name: _field_response_schema() for name in field_names},
                "required": [],
            },
        },
        "required": ["run_id", "branch", "category", "fields"],
    }


def _api_key_from_settings(api_key: str | None) -> str | None:
    if api_key is not None:
        return api_key.strip() or None
    configured = Settings().gemini_api_key
    return configured.get_secret_value() if configured is not None else None


def _usage_value(usage: object, name: str) -> object:
    return getattr(usage, name, None) if usage is not None else None


def _safe_error(error: Exception, api_key: str) -> str:
    return str(error).replace(api_key, "[redacted]") or error.__class__.__name__


def extract_ocr_to_json(
    run: RunDirectory,
    category: CategorySchema,
    *,
    client: Any | None = None,
    api_key: str | None = None,
) -> Path:
    """Extract structured OCR data and write it to ``run/ocr.json``."""

    resolved_key = _api_key_from_settings(api_key)
    if not resolved_key:
        raise GeminiExtractionError(
            "Gemini extraction requires GEMINI_API_KEY in the environment or .env."
        )
    if not run.ocr_text.exists():
        raise GeminiExtractionError(f"OCR text file does not exist: {run.ocr_text}")

    ocr_text = run.ocr_text.read_text(encoding="utf-8")
    prompt = (
        "Extract only fields defined by the supplied category schema from this OCR text. "
        "Do not infer, normalize, or guess unreadable or missing values; use null. "
        "Preserve page numbers and other provenance only when explicitly supported by the text. "
        f"Return run_id {run.root.name!r}, branch 'ocr', and this exact category.\n\n"
        f"CATEGORY SCHEMA:\n{category.model_dump_json()}\n\n"
        f"OCR TEXT:\n{ocr_text}"
    )
    request_config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=build_response_schema(category),
        thinking_config=types.ThinkingConfig(
            thinking_level=types.ThinkingLevel.MINIMAL
        ),
    )

    try:
        gemini_client = client or genai.Client(api_key=resolved_key)
        response = gemini_client.models.generate_content(
            model=MODEL_ID,
            contents=prompt,
            config=request_config,
        )
    except Exception as error:
        raise GeminiExtractionError(
            f"Gemini request failed: {_safe_error(error, resolved_key)}"
        ) from error

    usage = getattr(response, "usage_metadata", None)
    LOGGER.info(
        "Gemini OCR extraction usage model=%s prompt_token_count=%s "
        "candidates_token_count=%s total_token_count=%s",
        MODEL_ID,
        _usage_value(usage, "prompt_token_count"),
        _usage_value(usage, "candidates_token_count"),
        _usage_value(usage, "total_token_count"),
    )

    parsed = getattr(response, "parsed", None)
    if parsed is None:
        raise GeminiExtractionError("Gemini did not return structured ExtractionResult JSON.")
    try:
        result = ExtractionResult.model_validate(parsed)
    except ValidationError as error:
        raise GeminiExtractionError(
            f"Gemini returned invalid structured ExtractionResult: {error}"
        ) from error

    expected_fields = {field.name for field in category.fields}
    actual_fields = set(result.fields)
    if not actual_fields <= expected_fields:
        extras = ", ".join(sorted(actual_fields - expected_fields))
        raise GeminiExtractionError(
            f"Gemini returned fields outside supplied category: {extras}"
        )
    if result.run_id != run.root.name or result.branch is not ExtractionBranch.OCR:
        raise GeminiExtractionError("Gemini returned an unexpected run_id or extraction branch.")
    if result.category != category:
        raise GeminiExtractionError("Gemini returned a category different from the supplied schema.")

    run.ocr_json.write_text(result.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return run.ocr_json
