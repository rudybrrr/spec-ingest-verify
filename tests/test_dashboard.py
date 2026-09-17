import json
from pathlib import Path

from builderlab_verify.dashboard import (
    build_final_result,
    parse_category_json,
    parse_json_scalar,
    parse_schema_mapping_json,
    write_final_result,
)
from builderlab_verify.models import FieldProvenance, RunStatus, VerificationResult


def verification(status: str = "human_review") -> VerificationResult:
    return VerificationResult.model_validate(
        {
            "run_id": "run-001",
            "status": status,
            "fields": {
                "part_number": {
                    "status": "mismatch",
                    "ocr": {"value": "A-1", "unit": None, "page": 1},
                    "vision": {"value": "B-2", "unit": None, "page": 1},
                }
            },
        }
    )


def test_dashboard_parses_manual_schema_and_scalar_values() -> None:
    category = parse_category_json('{"name":"part","fields":[{"name":"part_number"}]}')
    assert category.name == "part"
    assert parse_json_scalar("12.5") == 12.5
    assert parse_json_scalar('"A-1"') == "A-1"
    assert parse_json_scalar("plain text") == "plain text"
    mapping = parse_schema_mapping_json('{"one.pdf":{"name":"part","fields":[{"name":"part_number"}]}}')
    assert mapping["one.pdf"].name == "part"


def test_dashboard_builds_and_writes_human_resolved_final_result(tmp_path: Path) -> None:
    result = build_final_result(
        verification(),
        {"part_number": FieldProvenance(value="A-1", page=1)},
        reviewer=" analyst ",
        note="Confirmed from drawing.",
    )
    output = write_final_result(tmp_path / "final.json", result)
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert saved["status"] == RunStatus.VERIFIED
    assert saved["resolution"]["reviewer"] == "analyst"
    assert saved["fields"]["part_number"]["value"] == "A-1"
