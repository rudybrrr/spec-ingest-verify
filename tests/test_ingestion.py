import json
from pathlib import Path

import fitz
import pytest
from PIL import Image

from builderlab_verify.ingestion import (
    InvalidPDFError,
    TesseractUnavailableError,
    TesseractTimeoutError,
    ingest_pdf,
    run_tesseract,
)
from builderlab_verify.storage import RunDirectory


def make_pdf(path: Path, *texts: str) -> None:
    document = fitz.open()
    for text in texts:
        page = document.new_page()
        page.insert_text((72, 72), text)
    document.save(path)
    document.close()


def test_ingest_copies_pdf_renders_lossless_pages_and_combines_ocr(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "input.pdf"
    make_pdf(source, "First page", "Second page")
    run = RunDirectory.create(tmp_path, "run-001")

    calls: list[Path] = []

    def fake_tesseract(
        image_path: Path, command: str | None = None, psm: int | None = None
    ) -> str:
        calls.append(image_path)
        return f"OCR for {image_path.name}"

    monkeypatch.setattr("builderlab_verify.ingestion._resolve_tesseract", lambda command: command)
    monkeypatch.setattr("builderlab_verify.ingestion.run_tesseract", fake_tesseract)

    output = ingest_pdf(source, run, tesseract_cmd="fake-tesseract")

    assert run.source_pdf.read_bytes() == source.read_bytes()
    assert sorted(run.root.joinpath("pages").glob("*.png")) == calls
    assert all(page.stat().st_size > 0 for page in calls)
    with Image.open(calls[0]) as image:
        assert image.size == (2480, 3509)
    assert output == run.ocr_text
    assert output.read_text(encoding="utf-8") == (
        "OCR for page-001.png\n\nOCR for page-002.png\n"
    )
    native_text = run.native_text.read_text(encoding="utf-8")
    native_metadata = json.loads(run.native_text_json.read_text(encoding="utf-8"))
    assert "First page" in native_text and "Second page" in native_text
    assert native_metadata["status"] == "available"
    assert native_metadata["usable"] is False


def test_native_text_failure_is_non_fatal_and_explicitly_persisted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "input.pdf"
    make_pdf(source, "A page with enough text to keep the OCR path running")
    run = RunDirectory.create(tmp_path, "run-001")

    monkeypatch.setattr("builderlab_verify.ingestion._resolve_tesseract", lambda command: command)
    monkeypatch.setattr(
        "builderlab_verify.ingestion.extract_native_text",
        lambda path: (_ for _ in ()).throw(RuntimeError("native parser failed")),
    )
    monkeypatch.setattr(
        "builderlab_verify.ingestion.run_tesseract",
        lambda image_path, command=None, psm=None, timeout_seconds=None: "OCR survived",
    )

    ingest_pdf(source, run, tesseract_cmd="fake-tesseract")

    saved = json.loads(run.native_text_json.read_text(encoding="utf-8"))
    assert saved["status"] == "failed"
    assert saved["error"] == "RuntimeError: native parser failed"
    assert run.ocr_text.read_text(encoding="utf-8") == "OCR survived\n"


def test_run_tesseract_uses_configured_psm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image = tmp_path / "page.png"
    image.write_bytes(b"png")
    calls: list[list[str]] = []

    class Result:
        returncode = 0
        stdout = "recognized"
        stderr = ""

    monkeypatch.setattr("builderlab_verify.ingestion._resolve_tesseract", lambda command: command)
    monkeypatch.setattr(
        "builderlab_verify.ingestion.subprocess.run",
        lambda command, **kwargs: calls.append(command) or Result(),
    )

    assert run_tesseract(image, command="fake-tesseract", psm=11) == "recognized"
    assert calls == [["fake-tesseract", str(image), "stdout", "--psm", "11"]]


def test_run_tesseract_applies_configured_timeout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    image = tmp_path / "page.png"
    image.write_bytes(b"png")
    monkeypatch.setattr("builderlab_verify.ingestion._resolve_tesseract", lambda command: command)
    def timed_out(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], kwargs["timeout"])
    import subprocess
    monkeypatch.setattr("builderlab_verify.ingestion.subprocess.run", timed_out)
    with pytest.raises(TesseractTimeoutError, match="timed out"):
        run_tesseract(image, command="fake-tesseract", timeout_seconds=3)


def test_ingest_persists_partial_progress_on_timeout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "input.pdf"
    make_pdf(source, "First page", "Second page")
    run = RunDirectory.create(tmp_path, "run-001")
    monkeypatch.setattr("builderlab_verify.ingestion._resolve_tesseract", lambda command: command)
    def fake_tesseract(image_path, command=None, psm=None, timeout_seconds=None):
        if image_path.name == "page-002.png":
            raise TesseractTimeoutError("Tesseract timed out on page-002.png after 1 seconds")
        return "first page"
    monkeypatch.setattr("builderlab_verify.ingestion.run_tesseract", fake_tesseract)
    with pytest.raises(TesseractTimeoutError):
        ingest_pdf(source, run, tesseract_cmd="fake-tesseract", tesseract_timeout_seconds=1)
    progress = run.root.joinpath("ocr-progress.json")
    assert progress.exists()
    assert progress.read_text(encoding="utf-8").find('"page": 2') >= 0
    assert run.ocr_text.exists()


def test_ingest_rejects_invalid_pdf_without_creating_source_copy(tmp_path: Path) -> None:
    source = tmp_path / "not-a-pdf.pdf"
    source.write_bytes(b"not a PDF")
    run = RunDirectory.create(tmp_path, "run-001")

    with pytest.raises(InvalidPDFError, match="valid PDF"):
        ingest_pdf(source, run)

    assert not run.source_pdf.exists()


def test_ingest_reports_missing_tesseract(tmp_path: Path) -> None:
    source = tmp_path / "input.pdf"
    make_pdf(source, "Page")
    run = RunDirectory.create(tmp_path, "run-001")

    with pytest.raises(TesseractUnavailableError, match="Tesseract"):
        ingest_pdf(source, run, tesseract_cmd="definitely-not-installed-tesseract")
