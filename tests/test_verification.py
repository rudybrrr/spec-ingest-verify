import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from builderlab_verify.models import AlignmentStatus, RunStatus
from builderlab_verify.schemas import CategoryField, CategorySchema
from builderlab_verify.storage import RunDirectory
from builderlab_verify.verification import (
    GeminiVerificationError,
    verify_ocr_vision,
)


class FakeClient:
    def __init__(self, parsed: dict, usage: dict | None = None) -> None:
        self.response = SimpleNamespace(
            parsed=parsed,
            usage_metadata=SimpleNamespace(**(usage or {})),
        )
        self.models = self
        self.calls: list[dict] = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


def extraction(run_id: str, branch: str, category: CategorySchema, values: dict) -> dict:
    return {
        "run_id": run_id,
        "branch": branch,
        "category": category.model_dump(),
        "fields": {
            name: {"value": value, "unit": None, "page": None}
            for name, value in values.items()
        },
    }


def checker_output(run_id: str, decisions: dict[str, dict]) -> dict:
    return {"run_id": run_id, "fields": decisions}


def category() -> CategorySchema:
    return CategorySchema(
        name="part",
        fields=[
            CategoryField(name="required_value", required=True),
            CategoryField(name="optional_value", required=False),
            CategoryField(name="semantic_value", required=True),
        ],
    )


def seed_run(tmp_path: Path, cat: CategorySchema, ocr: dict, vision: dict) -> RunDirectory:
    run = RunDirectory.create(tmp_path, "run-001")
    run.ocr_json.write_text(json.dumps(ocr), encoding="utf-8")
    run.vision_json.write_text(json.dumps(vision), encoding="utf-8")
    return run


def test_verification_applies_deterministic_null_rules_without_calling_gemini(tmp_path: Path) -> None:
    cat = category()
    run = seed_run(
        tmp_path,
        cat,
        extraction("run-001", "ocr", cat, {"required_value": None, "optional_value": None, "semantic_value": "15 mm"}),
        extraction("run-001", "vision", cat, {"required_value": "x", "optional_value": None, "semantic_value": "15 millimeters"}),
    )
    client = FakeClient(checker_output("run-001", {"semantic_value": {"status": "match"}}))

    output = verify_ocr_vision(run, cat, client=client, api_key="test-key")
    result = json.loads(output.read_text(encoding="utf-8"))

    assert result["fields"]["required_value"]["status"] == AlignmentStatus.MISMATCH
    assert result["fields"]["optional_value"]["status"] == AlignmentStatus.MATCH
    assert result["fields"]["semantic_value"]["status"] == AlignmentStatus.MATCH
    assert result["status"] == RunStatus.HUMAN_REVIEW
    assert len(client.calls) == 1


def test_verification_preserves_snapshots_and_only_sends_schema_fields(tmp_path: Path) -> None:
    cat = CategorySchema(name="part", fields=[CategoryField(name="drawing_number")])
    ocr = extraction("run-001", "ocr", cat, {"drawing_number": "A-1"})
    vision = extraction("run-001", "vision", cat, {"drawing_number": "A 1"})
    ocr["fields"]["extra"] = {"value": "must not compare", "unit": None, "page": 1}
    run = seed_run(tmp_path, cat, ocr, vision)
    client = FakeClient(checker_output("run-001", {"drawing_number": {"status": "match", "note": "Equivalent formatting."}}), {"prompt_token_count": 10, "candidates_token_count": 4, "total_token_count": 14})

    verify_ocr_vision(run, cat, client=client, api_key="test-key")
    result = json.loads(run.verification_json.read_text(encoding="utf-8"))
    prompt = client.calls[0]["contents"]
    response_schema = client.calls[0]["config"].response_schema

    assert set(result["fields"]) == {"drawing_number"}
    assert result["fields"]["drawing_number"]["ocr"]["value"] == "A-1"
    assert result["fields"]["drawing_number"]["vision"]["value"] == "A 1"
    assert "must not compare" not in prompt
    assert "run-001" in prompt
    assert set(response_schema["properties"]["fields"]["properties"]) == {"drawing_number"}
    assert result["token_usage"] == {"prompt": 10, "candidates": 4, "thoughts": None, "total": 14}


def test_checker_requires_structured_output_and_rejects_corrected_values(tmp_path: Path) -> None:
    cat = CategorySchema(name="part", fields=[CategoryField(name="drawing_number")])
    run = seed_run(
        tmp_path,
        cat,
        extraction("run-001", "ocr", cat, {"drawing_number": "A-1"}),
        extraction("run-001", "vision", cat, {"drawing_number": "B-2"}),
    )
    client = FakeClient(
        checker_output("run-001", {"drawing_number": {"status": "mismatch", "corrected_value": "A-1"}})
    )

    with pytest.raises(GeminiVerificationError, match="corrected value"):
        verify_ocr_vision(run, cat, client=client, api_key="test-key")

    config = client.calls[0]["config"]
    assert config.response_mime_type == "application/json"
    assert config.thinking_config.thinking_level.value == "MINIMAL"


def test_both_null_required_field_is_uncertain_and_prevents_verified_status(tmp_path: Path) -> None:
    cat = CategorySchema(name="part", fields=[CategoryField(name="drawing_number")])
    run = seed_run(
        tmp_path,
        cat,
        extraction("run-001", "ocr", cat, {"drawing_number": None}),
        extraction("run-001", "vision", cat, {"drawing_number": None}),
    )
    client = FakeClient(checker_output("run-001", {}))

    verify_ocr_vision(run, cat, client=client, api_key="test-key")
    result = json.loads(run.verification_json.read_text(encoding="utf-8"))

    assert result["fields"]["drawing_number"]["status"] == AlignmentStatus.UNCERTAIN
    assert result["status"] == RunStatus.HUMAN_REVIEW
    assert client.calls == []


def test_all_matching_fields_produce_verified_status(tmp_path: Path) -> None:
    cat = CategorySchema(name="part", fields=[CategoryField(name="drawing_number")])
    run = seed_run(
        tmp_path,
        cat,
        extraction("run-001", "ocr", cat, {"drawing_number": "A-1"}),
        extraction("run-001", "vision", cat, {"drawing_number": "A-1"}),
    )
    client = FakeClient(checker_output("run-001", {"drawing_number": {"status": "match"}}))

    verify_ocr_vision(run, cat, client=client, api_key="test-key")
    result = json.loads(run.verification_json.read_text(encoding="utf-8"))

    assert result["fields"]["drawing_number"]["status"] == AlignmentStatus.MATCH
    assert result["status"] == RunStatus.VERIFIED
