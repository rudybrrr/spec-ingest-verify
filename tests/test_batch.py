import json
from pathlib import Path

from builderlab_verify.batch import BatchRunner, DocumentStatus
from builderlab_verify.schemas import CategoryField, CategorySchema
import pytest


def category() -> CategorySchema:
    return CategorySchema(name="part", fields=[CategoryField(name="part_number")])


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
    assert calls == ["one.pdf", "one.pdf"]
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
