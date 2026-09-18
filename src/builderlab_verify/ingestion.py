"""PDF rendering and raw per-page Tesseract OCR."""

from __future__ import annotations

import shutil
import subprocess
import json
from collections.abc import Callable
from pathlib import Path

import pymupdf

from builderlab_verify.config import Settings
from builderlab_verify.native_text import (
    extract_native_text,
    write_native_artifacts,
    write_native_failure_artifact,
)
from builderlab_verify.storage import RunDirectory


class InvalidPDFError(ValueError):
    """Raised when the input cannot be opened as a non-empty PDF."""


class TesseractUnavailableError(RuntimeError):
    """Raised when the configured Tesseract executable is not available."""


class TesseractTimeoutError(RuntimeError):
    """Raised when Tesseract exceeds the configured per-page timeout."""


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
    image_path: Path, command: str | None = None, psm: int | None = None,
    timeout_seconds: float | None = None,
) -> str:
    """Run Tesseract on one image and return its raw text."""

    settings = Settings()
    executable = _resolve_tesseract(command or settings.tesseract_cmd)
    try:
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
            timeout=timeout_seconds if timeout_seconds is not None else settings.tesseract_timeout_seconds,
        )
    except subprocess.TimeoutExpired as error:
        timeout = timeout_seconds if timeout_seconds is not None else settings.tesseract_timeout_seconds
        raise TesseractTimeoutError(
            f"Tesseract timed out for {image_path.name} after {timeout:g} seconds"
        ) from error
    if result.returncode != 0:
        detail = result.stderr.strip() or "no diagnostic output"
        raise RuntimeError(f"Tesseract failed for {image_path.name}: {detail}")
    return result.stdout


def ingest_pdf(
    input_pdf: Path,
    run: RunDirectory,
    *,
    tesseract_cmd: str | None = None,
    tesseract_timeout_seconds: float | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> Path:
    """Copy, render, OCR, and persist raw text for one PDF run."""

    input_pdf = Path(input_pdf)
    _validate_pdf(input_pdf)
    settings = Settings()
    _resolve_tesseract(tesseract_cmd or settings.tesseract_cmd)

    shutil.copyfile(input_pdf, run.source_pdf)
    try:
        write_native_artifacts(extract_native_text(run.source_pdf), run.root)
    except Exception as error:
        write_native_failure_artifact(error, run.root)
    pages_dir = run.root / "pages"
    pages_dir.mkdir(exist_ok=True)

    document = pymupdf.open(str(run.source_pdf))
    try:
        scale = 300 / 72
        matrix = pymupdf.Matrix(scale, scale)
        page_text: list[str] = []
        total_pages = document.page_count
        for page_number, page in enumerate(document, start=1):
            image_path = pages_dir / f"page-{page_number:03d}.png"
            pixmap = page.get_pixmap(matrix=matrix, alpha=False)
            pixmap.save(str(image_path), output="png")
            try:
                if tesseract_timeout_seconds is None:
                    text = run_tesseract(image_path, tesseract_cmd, settings.tesseract_psm)
                else:
                    text = run_tesseract(
                        image_path, tesseract_cmd, settings.tesseract_psm,
                        tesseract_timeout_seconds,
                    )
                page_text.append(text.rstrip())
                run.ocr_progress_json.write_text(json.dumps({
                    "page": page_number, "total_pages": total_pages,
                    "pages_completed": page_number, "status": "in_progress",
                }, indent=2) + "\n", encoding="utf-8")
                if progress_callback:
                    progress_callback(page_number, total_pages)
            except Exception as error:
                run.ocr_text.write_text("\n\n".join(page_text) + ("\n" if page_text else ""), encoding="utf-8")
                run.ocr_progress_json.write_text(json.dumps({
                    "page": page_number, "total_pages": total_pages,
                    "pages_completed": len(page_text), "status": "failed",
                    "error": f"{type(error).__name__}: {error}",
                }, indent=2) + "\n", encoding="utf-8")
                raise
    finally:
        document.close()

    run.ocr_text.write_text("\n\n".join(page_text) + "\n", encoding="utf-8")
    run.ocr_progress_json.write_text(json.dumps({
        "page": len(page_text), "total_pages": len(page_text),
        "pages_completed": len(page_text), "status": "complete",
    }, indent=2) + "\n", encoding="utf-8")
    return run.ocr_text
