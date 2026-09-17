import json
import logging
from pathlib import Path

import pytest

from builderlab_verify.vision import GeminiVisionError, extract_vision_to_json
from builderlab_verify.schemas import CategoryField, CategorySchema
from builderlab_verify.storage import RunDirectory


class FakeResponse:
    def __init__(self, parsed: object, usage_metadata: object = None) -> None:
        self.parsed = parsed
        self.usage_metadata = usage_metadata


class FakeModels:
    def __init__(self, response: FakeResponse) -> None:
        self.response = response
        self.calls: list[dict[str, object]] = []

    def generate_content(self, **kwargs: object) -> FakeResponse:
        self.calls.append(kwargs)
        return self.response


class FakeClient:
    def __init__(self, response: FakeResponse) -> None:
        self.models = FakeModels(response)


def category() -> CategorySchema:
    return CategorySchema(
        name="laptop",
        fields=[
            CategoryField(name="screen_size", description="Display diagonal", required=True),
            CategoryField(name="color", required=False),
        ],
    )


def result_payload(run_id: str) -> dict[str, object]:
    return {
        "run_id": run_id,
        "branch": "vision",
        "category": category().model_dump(),
        "fields": {
            "screen_size": {"value": 15.6, "unit": "in", "page": 2},
            "color": {"value": None, "unit": None, "page": None},
        },
    }


def test_extract_vision_sends_native_pdf_and_writes_schema_constrained_result_and_usage_log(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    run = RunDirectory.create(tmp_path, "run-001")
    pdf_bytes = b"%PDF-native-test"
    run.source_pdf.write_bytes(pdf_bytes)
    usage = type(
        "Usage",
        (),
        {"prompt_token_count": 42, "candidates_token_count": 17, "total_token_count": 59},
    )()
    client = FakeClient(FakeResponse(result_payload("run-001"), usage))

    with caplog.at_level(logging.INFO):
        output = extract_vision_to_json(run, category(), client=client, api_key="test-key")

    assert output == run.vision_json
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert saved["branch"] == "vision"
    assert saved["fields"]["screen_size"] == {"value": 15.6, "unit": "in", "page": 2}
    request = client.models.calls[0]
    assert request["model"] == "gemini-3.8-flash"
    contents = request["contents"]
    assert len(contents) == 2
    assert contents[0].inline_data.mime_type == "application/pdf"
    assert contents[0].inline_data.data == pdf_bytes
    assert "ocr" not in str(contents[1]).lower()
    config = request["config"]
    assert config.response_mime_type == "application/json"
    assert config.thinking_config.thinking_level.value == "LOW"
    assert set(config.response_schema["properties"]["fields"]["properties"]) == {
        "screen_size",
        "color",
    }
    assert "total_token_count=59" in caplog.text
    assert "test-key" not in caplog.text


def test_extract_vision_fails_clearly_without_pdf(tmp_path: Path) -> None:
    run = RunDirectory.create(tmp_path, "run-001")

    with pytest.raises(GeminiVisionError, match="source PDF"):
        extract_vision_to_json(run, category(), api_key="test-key")


def test_extract_vision_fails_clearly_without_api_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = RunDirectory.create(tmp_path, "run-001")
    run.source_pdf.write_bytes(b"%PDF-native-test")
    monkeypatch.chdir(tmp_path)

    with pytest.raises(GeminiVisionError, match="GEMINI_API_KEY"):
        extract_vision_to_json(run, category(), api_key=None)


def test_extract_vision_rejects_invalid_structured_output(tmp_path: Path) -> None:
    run = RunDirectory.create(tmp_path, "run-001")
    run.source_pdf.write_bytes(b"%PDF-native-test")
    client = FakeClient(FakeResponse(None))

    with pytest.raises(GeminiVisionError, match="structured ExtractionResult"):
        extract_vision_to_json(run, category(), client=client, api_key="test-key")
