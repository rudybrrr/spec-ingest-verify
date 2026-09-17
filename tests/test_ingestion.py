from pathlib import Path

import fitz
import pytest
from PIL import Image

from builderlab_verify.ingestion import (
    InvalidPDFError,
    TesseractUnavailableError,
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
