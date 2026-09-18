import json
from pathlib import Path

from builderlab_verify.batch import BatchRunner, DocumentStatus
from builderlab_verify.ingestion import TesseractTimeoutError
from builderlab_verify.storage import RunDirectory
from builderlab_verify.schemas import CategoryField, CategorySchema
import pytest


def category() -> CategorySchema:
    return CategorySchema(name="part", fields=[CategoryField(name="part_number")])


def _provenance_index(path: Path, sku: str, contents: bytes) -> None:
    import hashlib

    digest = hashlib.sha256(contents).hexdigest().upper()
    path.write_text(json.dumps({"records": [{
        "expected_sku": sku,
        "remote_pdf_sha256": digest,
        "local_pdf_sha256": digest,
        "exact_hash_match": True,
        "page_explicitly_identifies_expected_sku": True,
        "page_explicitly_links_pdf": True,
        "namespace": "catalog_sku",
    }]}), encoding="utf-8")


def _extraction_payload(run_id: str, branch: str, value: str | None) -> dict[str, object]:
    return {
        "run_id": run_id,
        "branch": branch,
        "category": category().model_dump(),
        "fields": {"part_number": {"value": value, "unit": None, "page": 1}},
    }


def test_batch_is_sequential_resumable_and_persists_status(tmp_path: Path, monkeypatch) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "one.pdf").write_bytes(b"one")
    batch = BatchRunner(tmp_path / "batch")
    calls: list[str] = []

    def fake_run(source, run, selected_category):
        calls.append(source.name)
        run.source_pdf.write_bytes(source.read_bytes())
        run.ocr_json.write_text("{}", encoding="utf-8")
        run.vision_json.write_text("{}", encoding="utf-8")
        run.verification_json.write_text(json.dumps({"status": "verified"}), encoding="utf-8")

    monkeypatch.setattr(batch, "_run_document", fake_run)
    first = batch.run(input_dir, ["one.pdf"], category())
    second = batch.run(input_dir, ["one.pdf"], category())

    assert first[0].status is DocumentStatus.VERIFIED
    assert second[0].status is DocumentStatus.VERIFIED
    assert calls == ["one.pdf"]
    saved = json.loads(batch.manifest_path.read_text(encoding="utf-8"))
    assert saved["documents"]["one.pdf"]["status"] == "verified"


def test_batch_surfaces_failed_document_without_stopping_other_files(tmp_path: Path, monkeypatch) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    for name in ("bad.pdf", "good.pdf"):
        (input_dir / name).write_bytes(b"pdf")
    batch = BatchRunner(tmp_path / "batch")

    def fake_run(source, run, selected_category):
        if source.name == "bad.pdf":
            raise RuntimeError("bad extraction")
        run.verification_json.write_text(json.dumps({"status": "human_review"}), encoding="utf-8")

    monkeypatch.setattr(batch, "_run_document", fake_run)
    results = batch.run(input_dir, ["bad.pdf", "good.pdf"], category())

    assert [result.status for result in results] == [DocumentStatus.FAILED, DocumentStatus.HUMAN_REVIEW]
    assert "bad extraction" in (results[0].error or "")


def test_batch_requires_explicit_schema_for_every_mapped_file(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "one.pdf").write_bytes(b"pdf")

    with pytest.raises(ValueError, match="explicit schema mapping"):
        BatchRunner(tmp_path / "batch").run(
            input_dir,
            ["one.pdf"],
            category(),
            schema_by_filename={},
        )


def test_batch_classifies_tesseract_timeout_as_ocr(tmp_path: Path, monkeypatch) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "one.pdf").write_bytes(b"pdf")
    batch = BatchRunner(tmp_path / "batch")

    def fake_run(source, run, selected_category):
        raise TesseractTimeoutError("Tesseract timed out for page-001.png after 10 seconds")

    monkeypatch.setattr(batch, "_run_document", fake_run)
    result = batch.run(input_dir, ["one.pdf"], category())[0]

    assert result.status is DocumentStatus.FAILED
    metrics = json.loads((batch.runs_root / result.run_id / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["failure_category"] == "ocr"


def test_batch_preserves_terminal_failure_error_on_restart(tmp_path: Path, monkeypatch) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "one.pdf").write_bytes(b"pdf")
    batch = BatchRunner(tmp_path / "batch")
    error = "Tesseract timed out for page-001.png after 10 seconds"

    def fake_run(source, run, selected_category):
        raise TesseractTimeoutError(error)

    monkeypatch.setattr(batch, "_run_document", fake_run)
    first = batch.run(input_dir, ["one.pdf"], category())[0]
    second = batch.run(input_dir, ["one.pdf"], category())[0]

    assert first.error == f"TesseractTimeoutError: {error}"
    assert second.error == first.error


def test_batch_reingests_source_only_partial_run(tmp_path: Path, monkeypatch) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "one.pdf").write_bytes(b"pdf")
    batch = BatchRunner(tmp_path / "batch")
    calls: list[str] = []

    def fake_ingest(source, run, **kwargs):
        calls.append(source.name)
        run.ocr_text.write_text("ocr", encoding="utf-8")

    monkeypatch.setattr("builderlab_verify.batch.ingest_pdf", fake_ingest)
    monkeypatch.setattr("builderlab_verify.batch.extract_ocr_to_json", lambda run, category: run.ocr_json.write_text("{}", encoding="utf-8"))
    monkeypatch.setattr("builderlab_verify.batch.extract_vision_to_json", lambda run, category: run.vision_json.write_text("{}", encoding="utf-8"))
    monkeypatch.setattr("builderlab_verify.batch.verify_ocr_vision", lambda run, category: run.verification_json.write_text(json.dumps({"status": "verified"}), encoding="utf-8"))
    run = RunDirectory.create(batch.runs_root, "one")
    run.source_pdf.write_bytes(b"partial")
    batch._run_document(input_dir / "one.pdf", run, category())

    assert calls == ["one.pdf"]


def test_run_document_skips_native_extraction_when_ocr_vision_is_verified(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "one.pdf"
    source.write_bytes(b"pdf")
    run = RunDirectory.create(tmp_path / "batch", "one")
    native_calls: list[str] = []

    def fake_ingest(source_path, target_run, **kwargs):
        target_run.source_pdf.write_bytes(source_path.read_bytes())
        target_run.ocr_text.write_text("ocr", encoding="utf-8")
        target_run.native_text.write_text("native", encoding="utf-8")
        target_run.native_text_json.write_text(
            json.dumps({"status": "available", "usable": True}), encoding="utf-8"
        )

    def fake_native(target_run, selected_category):
        native_calls.append("native")
        target_run.native_json.write_text("{}", encoding="utf-8")

    monkeypatch.setattr("builderlab_verify.batch.ingest_pdf", fake_ingest)
    monkeypatch.setattr(
        "builderlab_verify.batch.extract_ocr_to_json",
        lambda target_run, selected_category: target_run.ocr_json.write_text("{}", encoding="utf-8"),
    )
    monkeypatch.setattr("builderlab_verify.batch.extract_native_to_json", fake_native)
    monkeypatch.setattr(
        "builderlab_verify.batch.extract_vision_to_json",
        lambda target_run, selected_category: target_run.vision_json.write_text("{}", encoding="utf-8"),
    )
    monkeypatch.setattr(
        "builderlab_verify.batch.verify_ocr_vision",
        lambda target_run, selected_category: target_run.verification_json.write_text(
            json.dumps({"status": "verified"}), encoding="utf-8"
        ),
    )

    BatchRunner._run_document(source, run, category())

    assert native_calls == []
    assert run.ocr_json.exists()
    assert not run.native_json.exists()


def test_native_structured_extraction_failure_does_not_fail_document(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "one.pdf"
    source.write_bytes(b"pdf")
    run = RunDirectory.create(tmp_path / "batch", "one")

    def fake_ingest(source_path, target_run, **kwargs):
        target_run.source_pdf.write_bytes(source_path.read_bytes())
        target_run.ocr_text.write_text("ocr", encoding="utf-8")
        target_run.native_text.write_text("native", encoding="utf-8")
        target_run.native_text_json.write_text(
            json.dumps({"status": "available", "usable": True}), encoding="utf-8"
        )

    monkeypatch.setattr("builderlab_verify.batch.ingest_pdf", fake_ingest)
    monkeypatch.setattr(
        "builderlab_verify.batch.extract_ocr_to_json",
        lambda target_run, selected_category: target_run.ocr_json.write_text("{}", encoding="utf-8"),
    )
    monkeypatch.setattr(
        "builderlab_verify.batch.extract_native_to_json",
        lambda target_run, selected_category: (_ for _ in ()).throw(RuntimeError("native Gemini failed")),
    )
    monkeypatch.setattr(
        "builderlab_verify.batch.extract_vision_to_json",
        lambda target_run, selected_category: target_run.vision_json.write_text("{}", encoding="utf-8"),
    )
    monkeypatch.setattr(
        "builderlab_verify.batch.verify_ocr_vision",
        lambda target_run, selected_category: target_run.verification_json.write_text(
            json.dumps({
                "status": "human_review",
                "fields": {"part_number": {"status": "mismatch"}},
            }),
            encoding="utf-8",
        ),
    )

    BatchRunner._run_document(source, run, category())

    saved = json.loads(run.native_text_json.read_text(encoding="utf-8"))
    assert saved["structured_extraction_status"] == "failed"
    assert saved["structured_extraction_error"] == "RuntimeError: native Gemini failed"
    assert json.loads(run.verification_json.read_text(encoding="utf-8"))["status"] == "human_review"


def test_vision_provider_failure_can_be_rescued_by_exact_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    source = input_dir / "one.pdf"
    contents = b"vision-failure-pdf"
    source.write_bytes(contents)
    index = tmp_path / "provenance.json"
    _provenance_index(index, "SKU-1", contents)
    batch = BatchRunner(tmp_path / "batch")

    def fake_ingest(source_path, target_run, **kwargs):
        target_run.source_pdf.write_bytes(source_path.read_bytes())
        target_run.ocr_text.write_text("", encoding="utf-8")

    def fake_ocr(target_run, selected_category):
        target_run.ocr_json.write_text(
            json.dumps(_extraction_payload(target_run.root.name, "ocr", None)), encoding="utf-8"
        )

    monkeypatch.setattr("builderlab_verify.batch.ingest_pdf", fake_ingest)
    monkeypatch.setattr("builderlab_verify.batch.extract_ocr_to_json", fake_ocr)
    monkeypatch.setattr(
        "builderlab_verify.batch.extract_vision_to_json",
        lambda target_run, selected_category: (_ for _ in ()).throw(RuntimeError("503 UNAVAILABLE")),
    )

    result = batch.run(
        input_dir,
        [source.name],
        category(),
        upstream_sku_by_filename={source.name: "SKU-1"},
        external_provenance_path=index,
    )[0]

    assert result.status is DocumentStatus.VERIFIED
    run = batch.runs_root / result.run_id
    saved = json.loads((run / "verification.json").read_text(encoding="utf-8"))
    assert saved["external_provenance"]["rescued"] is True
    assert saved["provider_failures"]["vision"]["error"] == "RuntimeError: 503 UNAVAILABLE"


def test_ocr_provider_failure_can_be_rescued_by_exact_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    source = input_dir / "one.pdf"
    contents = b"ocr-failure-pdf"
    source.write_bytes(contents)
    index = tmp_path / "provenance.json"
    _provenance_index(index, "SKU-1", contents)
    batch = BatchRunner(tmp_path / "batch")

    def fake_ingest(source_path, target_run, **kwargs):
        target_run.source_pdf.write_bytes(source_path.read_bytes())
        target_run.ocr_text.write_text("", encoding="utf-8")

    def fake_vision(target_run, selected_category):
        target_run.vision_json.write_text(
            json.dumps(_extraction_payload(target_run.root.name, "vision", None)), encoding="utf-8"
        )

    monkeypatch.setattr("builderlab_verify.batch.ingest_pdf", fake_ingest)
    monkeypatch.setattr(
        "builderlab_verify.batch.extract_ocr_to_json",
        lambda target_run, selected_category: (_ for _ in ()).throw(RuntimeError("503 UNAVAILABLE")),
    )
    monkeypatch.setattr("builderlab_verify.batch.extract_vision_to_json", fake_vision)

    result = batch.run(
        input_dir,
        [source.name],
        category(),
        upstream_sku_by_filename={source.name: "SKU-1"},
        external_provenance_path=index,
    )[0]

    assert result.status is DocumentStatus.VERIFIED
    run = batch.runs_root / result.run_id
    saved = json.loads((run / "verification.json").read_text(encoding="utf-8"))
    assert saved["external_provenance"]["rescued"] is True
    assert saved["provider_failures"]["ocr"]["error"] == "RuntimeError: 503 UNAVAILABLE"


def test_provider_failure_without_deterministic_rescue_remains_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    source = input_dir / "one.pdf"
    source.write_bytes(b"no-rescue-pdf")
    batch = BatchRunner(tmp_path / "batch")

    def fake_ingest(source_path, target_run, **kwargs):
        target_run.source_pdf.write_bytes(source_path.read_bytes())
        target_run.ocr_text.write_text("", encoding="utf-8")

    monkeypatch.setattr("builderlab_verify.batch.ingest_pdf", fake_ingest)
    monkeypatch.setattr(
        "builderlab_verify.batch.extract_ocr_to_json",
        lambda target_run, selected_category: (_ for _ in ()).throw(RuntimeError("503 UNAVAILABLE")),
    )
    monkeypatch.setattr(
        "builderlab_verify.batch.extract_vision_to_json",
        lambda target_run, selected_category: (_ for _ in ()).throw(RuntimeError("503 UNAVAILABLE")),
    )

    result = batch.run(input_dir, [source.name], category())[0]

    assert result.status is DocumentStatus.FAILED
    assert "provider failure" in (result.error or "").lower()
    run = batch.runs_root / result.run_id
    saved = json.loads((run / "verification.json").read_text(encoding="utf-8"))
    assert saved["status"] == "human_review"
    assert set(saved["provider_failures"]) == {"ocr", "vision"}
    assert json.loads((run / "metrics.json").read_text(encoding="utf-8"))["failure_category"] == "provider_failure"
