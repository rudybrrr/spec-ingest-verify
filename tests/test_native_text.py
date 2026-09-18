import json
from pathlib import Path
from types import SimpleNamespace

import fitz

from builderlab_verify.native_text import NativeTextResult, extract_native_to_json, write_native_artifacts
from builderlab_verify.schemas import CategoryField, CategorySchema
from builderlab_verify.storage import RunDirectory


class FakeClient:
    def __init__(self, parsed: dict) -> None:
        self.models = self
        self.response = SimpleNamespace(
            parsed=parsed,
            usage_metadata=SimpleNamespace(),
        )
        self.calls: list[dict] = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


def category() -> CategorySchema:
    return CategorySchema(name="part", fields=[CategoryField(name="part_number")])


def test_native_structured_extraction_writes_native_json_without_overwriting_ocr_json(tmp_path: Path) -> None:
    run = RunDirectory.create(tmp_path, "run-001")
    run.ocr_json.write_text('{"branch":"ocr"}', encoding="utf-8")
    run.native_text.write_text("Part number: ABC-123\n", encoding="utf-8")
    client = FakeClient({
        "run_id": "run-001",
        "branch": "native_text",
        "category": category().model_dump(),
        "fields": {"part_number": {"value": "ABC-123", "unit": None, "page": 1}},
    })

    output = extract_native_to_json(run, category(), client=client, api_key="test-key")

    assert output == run.native_json
    assert json.loads(run.native_json.read_text(encoding="utf-8"))["branch"] == "native_text"
    assert run.ocr_json.read_text(encoding="utf-8") == '{"branch":"ocr"}'
    assert client.calls[0]["config"].response_schema["properties"]["branch"]["enum"] == ["native_text"]


def test_native_artifact_marks_unusable_text_as_unavailable(tmp_path: Path) -> None:
    run = RunDirectory.create(tmp_path, "run-001")
    result = NativeTextResult(pages=[], text="", word_count=0, char_count=0, block_count=0)

    write_native_artifacts(result, run.root)

    saved = json.loads(run.native_text_json.read_text(encoding="utf-8"))
    assert saved["status"] == "unavailable"
    assert saved["reason"] == "insufficient_native_text"
    assert saved["usable"] is False
