import json
import logging
from pathlib import Path

import pytest

from builderlab_verify.ocr import (
    GeminiExtractionError,
    extract_ocr_to_json,
)
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


class SequenceModels:
    def __init__(self, outcomes: list[object]) -> None:
        self.outcomes = outcomes
        self.calls: list[dict[str, object]] = []

    def generate_content(self, **kwargs: object) -> FakeResponse:
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class FakeGeminiError(Exception):
    def __init__(self, status_code: int) -> None:
        self.status_code = status_code
        super().__init__(f"{status_code} transient provider error")


class SequenceClient:
    def __init__(self, outcomes: list[object]) -> None:
        self.models = SequenceModels(outcomes)


def category() -> CategorySchema:
    return CategorySchema(
        name="laptop",
        fields=[
            CategoryField(name="screen_size", description="Display diagonal", required=True),
            CategoryField(name="color", required=False),
        ],
    )


def test_extract_ocr_writes_schema_constrained_result_and_usage_log(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    run = RunDirectory.create(tmp_path, "run-001")
    run.ocr_text.write_text(
        "[Page 2]\nScreen size: 15.6 in\nColor: unreadable\n", encoding="utf-8"
    )
    usage = type(
        "Usage",
        (),
        {"prompt_token_count": 42, "candidates_token_count": 17, "total_token_count": 59},
    )()
    client = FakeClient(
        FakeResponse(
            {
                "run_id": "run-001",
                "branch": "ocr",
                "category": category().model_dump(),
                "fields": {
                    "screen_size": {"value": 15.6, "unit": "in", "page": 2},
                    "color": {"value": None, "unit": None, "page": 2},
                },
            },
            usage,
        )
    )

    with caplog.at_level(logging.INFO):
        output = extract_ocr_to_json(run, category(), client=client, api_key="test-key")

    assert output == run.ocr_json
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert saved["fields"]["screen_size"] == {"value": 15.6, "unit": "in", "page": 2}
    assert saved["fields"]["color"]["value"] is None
    request = client.models.calls[0]
    assert request["model"] == "gemini-3.5-flash-lite"
    config = request["config"]
    assert config.response_mime_type == "application/json"
    assert config.thinking_config.thinking_level.value == "MINIMAL"
    assert set(config.response_schema["properties"]["fields"]["properties"]) == {
        "screen_size",
        "color",
    }
    assert "total_token_count=59" in caplog.text
    assert "test-key" not in caplog.text


def test_extract_ocr_rejects_fields_outside_supplied_category(tmp_path: Path) -> None:
    run = RunDirectory.create(tmp_path, "run-001")
    run.ocr_text.write_text("Screen size: 15.6 in\n", encoding="utf-8")
    client = FakeClient(
        FakeResponse(
            {
                "run_id": "run-001",
                "branch": "ocr",
                "category": category().model_dump(),
                "fields": {
                    "screen_size": {"value": 15.6, "unit": "in", "page": 1},
                    "secret_field": {"value": "guessed", "page": 1},
                },
            }
        )
    )

    with pytest.raises(GeminiExtractionError, match="outside supplied category"):
        extract_ocr_to_json(run, category(), client=client, api_key="test-key")

    assert len(client.models.calls) == 1
    metrics = json.loads(run.metrics_json.read_text(encoding="utf-8"))
    assert metrics["stages"]["ocr"]["error_category"] == "schema"
    assert metrics["stages"]["ocr"]["retry_count"] == 0


def test_extract_ocr_fails_clearly_without_api_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = RunDirectory.create(tmp_path, "run-001")
    run.ocr_text.write_text("Screen size: 15.6 in\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    with pytest.raises(GeminiExtractionError, match="GEMINI_API_KEY"):
        extract_ocr_to_json(run, category(), api_key=None)


def test_extract_ocr_fails_when_gemini_returns_unparseable_output(tmp_path: Path) -> None:
    run = RunDirectory.create(tmp_path, "run-001")
    run.ocr_text.write_text("Screen size: 15.6 in\n", encoding="utf-8")
    client = FakeClient(FakeResponse(None))

    with pytest.raises(GeminiExtractionError, match="structured ExtractionResult"):
        extract_ocr_to_json(run, category(), client=client, api_key="test-key")


def test_extract_ocr_retries_transient_503_then_succeeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = RunDirectory.create(tmp_path, "run-001")
    run.ocr_text.write_text("Screen size: 15.6 in\n", encoding="utf-8")
    response = FakeResponse(
        {
            "run_id": "run-001",
            "branch": "ocr",
            "category": category().model_dump(),
            "fields": {"screen_size": {"value": 15.6, "unit": "in", "page": 1}},
        }
    )
    client = SequenceClient([FakeGeminiError(503), response])
    monkeypatch.setattr("builderlab_verify.retry.time.sleep", lambda _: None)
    monkeypatch.setattr("builderlab_verify.retry.random.random", lambda: 0.5)

    extract_ocr_to_json(run, category(), client=client, api_key="test-key")

    assert len(client.models.calls) == 2
    metrics = json.loads(run.metrics_json.read_text(encoding="utf-8"))
    assert metrics["stages"]["ocr"]["attempts"] == 2
    assert metrics["stages"]["ocr"]["retry_count"] == 1
    assert metrics["stages"]["ocr"]["errors"][0]["status_code"] == 503


def test_extract_ocr_persists_exhausted_transient_retries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = RunDirectory.create(tmp_path, "run-001")
    run.ocr_text.write_text("Screen size: 15.6 in\n", encoding="utf-8")
    client = SequenceClient([FakeGeminiError(503) for _ in range(4)])
    monkeypatch.setattr("builderlab_verify.retry.time.sleep", lambda _: None)
    monkeypatch.setattr("builderlab_verify.retry.random.random", lambda: 0.5)

    with pytest.raises(GeminiExtractionError, match="503 transient provider error"):
        extract_ocr_to_json(run, category(), client=client, api_key="test-key")

    metrics = json.loads(run.metrics_json.read_text(encoding="utf-8"))
    assert len(client.models.calls) == 4
    assert metrics["stages"]["ocr"]["attempts"] == 4
    assert metrics["stages"]["ocr"]["retry_count"] == 3
    assert len(metrics["stages"]["ocr"]["errors"]) == 4
