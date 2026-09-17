"""PDF rendering and raw per-page Tesseract OCR."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pymupdf

from builderlab_verify.config import Settings
from builderlab_verify.storage import RunDirectory


class InvalidPDFError(ValueError):
    """Raised when the input cannot be opened as a non-empty PDF."""


class TesseractUnavailableError(RuntimeError):
    """Raised when the configured Tesseract executable is not available."""


def _validate_pdf(pdf_path: Path) -> None:
    try:
        document = pymupdf.open(str(pdf_path))
        page_count = document.page_count
        document.close()
    except (OSError, pymupdf.FileDataError, RuntimeError) as error:
        raise InvalidPDFError(f"Input is not a valid PDF: {pdf_path}") from error

    if page_count == 0:
        raise InvalidPDFError(f"Input is not a valid PDF: {pdf_path}")


def _resolve_tesseract(command: str) -> str:
    resolved = shutil.which(command)
    if resolved is None:
        raise TesseractUnavailableError(
            f"Tesseract executable was not found: {command!r}. "
            "Install Tesseract and add it to PATH, or provide tesseract_cmd."
        )
    return resolved


def run_tesseract(
    image_path: Path, command: str | None = None, psm: int | None = None
) -> str:
    """Run Tesseract on one image and return its raw text."""

    settings = Settings()
    executable = _resolve_tesseract(command or settings.tesseract_cmd)
    result = subprocess.run(
        [
            executable,
            str(image_path),
            "stdout",
            "--psm",
            str(settings.tesseract_psm if psm is None else psm),
        ],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or "no diagnostic output"
        raise RuntimeError(f"Tesseract failed for {image_path.name}: {detail}")
    return result.stdout


def ingest_pdf(
    input_pdf: Path,
    run: RunDirectory,
    *,
    tesseract_cmd: str | None = None,
) -> Path:
    """Copy, render, OCR, and persist raw text for one PDF run."""

    input_pdf = Path(input_pdf)
    _validate_pdf(input_pdf)
    settings = Settings()
    _resolve_tesseract(tesseract_cmd or settings.tesseract_cmd)

    shutil.copyfile(input_pdf, run.source_pdf)
    pages_dir = run.root / "pages"
    pages_dir.mkdir(exist_ok=True)

    document = pymupdf.open(str(run.source_pdf))
    try:
        scale = 300 / 72
        matrix = pymupdf.Matrix(scale, scale)
        page_text: list[str] = []
        for page_number, page in enumerate(document, start=1):
            image_path = pages_dir / f"page-{page_number:03d}.png"
            pixmap = page.get_pixmap(matrix=matrix, alpha=False)
            pixmap.save(str(image_path), output="png")
            page_text.append(
                run_tesseract(image_path, tesseract_cmd, settings.tesseract_psm).rstrip()
            )
    finally:
        document.close()

    run.ocr_text.write_text("\n\n".join(page_text) + "\n", encoding="utf-8")
    return run.ocr_text
