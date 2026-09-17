"""Gemini native-PDF structured extraction."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types
from pydantic import ValidationError

from builderlab_verify.models import ExtractionBranch, ExtractionResult
from builderlab_verify.ocr import (
    _api_key_from_settings,
    _safe_error,
    _usage_value,
    build_response_schema,
)
from builderlab_verify.schemas import CategorySchema
from builderlab_verify.storage import RunDirectory

LOGGER = logging.getLogger(__name__)
VISION_MODEL_ID = "gemini-3.8-flash"


class GeminiVisionError(RuntimeError):
    """Raised when Gemini cannot produce a valid structured Vision extraction."""


def extract_vision_to_json(
    run: RunDirectory,
    category: CategorySchema,
    *,
    client: Any | None = None,
    api_key: str | None = None,
) -> Path:
    """Extract structured data directly from ``run/source.pdf``."""

    resolved_key = _api_key_from_settings(api_key)
    if not resolved_key:
        raise GeminiVisionError(
            "Gemini Vision extraction requires GEMINI_API_KEY in the environment or .env."
        )
    if not run.source_pdf.exists():
        raise GeminiVisionError(f"Vision extraction requires the source PDF: {run.source_pdf}")

    prompt = (
        "Read the supplied PDF directly and extract only fields defined by the supplied "
        "category schema. Do not infer, normalize, or guess unreadable or missing values; "
        "use null or omit the field. Preserve page numbers when explicitly supported by "
        f"the PDF. Return run_id {run.root.name!r}, branch 'vision', and this exact category.\n\n"
        f"CATEGORY SCHEMA:\n{category.model_dump_json()}"
    )
    request_config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=build_response_schema(category, ExtractionBranch.VISION),
        thinking_config=types.ThinkingConfig(thinking_level=types.ThinkingLevel.LOW),
    )
    contents = [
        types.Part.from_bytes(data=run.source_pdf.read_bytes(), mime_type="application/pdf"),
        prompt,
    ]

    try:
        gemini_client = client or genai.Client(api_key=resolved_key)
        response = gemini_client.models.generate_content(
            model=VISION_MODEL_ID,
            contents=contents,
            config=request_config,
        )
    except Exception as error:
        raise GeminiVisionError(
            f"Gemini Vision request failed: {_safe_error(error, resolved_key)}"
        ) from error

    usage = getattr(response, "usage_metadata", None)
    LOGGER.info(
        "Gemini Vision extraction usage model=%s prompt_token_count=%s "
        "candidates_token_count=%s total_token_count=%s",
        VISION_MODEL_ID,
        _usage_value(usage, "prompt_token_count"),
        _usage_value(usage, "candidates_token_count"),
        _usage_value(usage, "total_token_count"),
    )

    parsed = getattr(response, "parsed", None)
    if parsed is None:
        raise GeminiVisionError("Gemini did not return structured ExtractionResult JSON.")
    try:
        result = ExtractionResult.model_validate(parsed)
    except ValidationError as error:
        raise GeminiVisionError(
            f"Gemini returned invalid structured ExtractionResult: {error}"
        ) from error

    expected_fields = {field.name for field in category.fields}
    actual_fields = set(result.fields)
    if not actual_fields <= expected_fields:
        extras = ", ".join(sorted(actual_fields - expected_fields))
        raise GeminiVisionError(f"Gemini returned fields outside supplied category: {extras}")
    if result.run_id != run.root.name or result.branch is not ExtractionBranch.VISION:
        raise GeminiVisionError("Gemini returned an unexpected run_id or extraction branch.")
    if result.category != category:
        raise GeminiVisionError("Gemini returned a category different from the supplied schema.")

    run.vision_json.write_text(result.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return run.vision_json
