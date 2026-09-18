"""Offline native-PDF-text experiment helpers.

This module is intentionally kept outside the production pipeline. It extracts
PyMuPDF text plus coordinates, serializes the text as an OCR-shaped input, and
uses the existing OCR-side Gemini extractor for comparison only.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import fitz

from builderlab_verify.models import AlignmentStatus
from builderlab_verify.ocr import extract_ocr_to_json
from builderlab_verify.schemas import CategorySchema
from builderlab_verify.storage import RunDirectory


MIN_USABLE_WORDS = 10
MIN_USABLE_CHARS = 50


@dataclass(frozen=True)
class NativeTextResult:
    pages: list[dict[str, Any]]
    text: str
    word_count: int
    char_count: int
    block_count: int

    @property
    def usable(self) -> bool:
        return self.word_count >= MIN_USABLE_WORDS and self.char_count >= MIN_USABLE_CHARS


def _word_payload(word: tuple[Any, ...]) -> dict[str, Any]:
    x0, y0, x1, y1, text, block_no, line_no, word_no = word[:8]
    return {
        "text": text,
        "bbox": [round(float(x0), 3), round(float(y0), 3), round(float(x1), 3), round(float(y1), 3)],
        "block_no": int(block_no),
        "line_no": int(line_no),
        "word_no": int(word_no),
    }


def _block_payload(block: tuple[Any, ...], block_no: int) -> dict[str, Any]:
    x0, y0, x1, y1, text = block[:5]
    return {
        "block_no": block_no,
        "bbox": [round(float(x0), 3), round(float(y0), 3), round(float(x1), 3), round(float(y1), 3)],
        "text": str(text).strip(),
        "type": int(block[6]) if len(block) > 6 else 0,
    }


def extract_native_text(pdf_path: Path) -> NativeTextResult:
    """Extract page text, words, and blocks without consulting the filename."""

    pages: list[dict[str, Any]] = []
    page_texts: list[str] = []
    word_count = 0
    char_count = 0
    block_count = 0
    with fitz.open(pdf_path) as document:
        for page_number, page in enumerate(document, start=1):
            words = [_word_payload(word) for word in page.get_text("words", sort=True)]
            blocks = [
                _block_payload(block, block_no)
                for block_no, block in enumerate(page.get_text("blocks", sort=True), start=1)
                if str(block[4]).strip()
            ]
            text = page.get_text("text", sort=True).strip()
            pages.append({
                "page": page_number,
                "width": round(float(page.rect.width), 3),
                "height": round(float(page.rect.height), 3),
                "text": text,
                "words": words,
                "blocks": blocks,
            })
            page_texts.append(text)
            word_count += len(words)
            char_count += len(text)
            block_count += len(blocks)
    return NativeTextResult(
        pages=pages,
        text="\n\n".join(f"[Page {i}]\n{text}" for i, text in enumerate(page_texts, start=1) if text),
        word_count=word_count,
        char_count=char_count,
        block_count=block_count,
    )


def write_native_artifacts(result: NativeTextResult, run_root: Path) -> None:
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "native-text.txt").write_text(result.text + "\n", encoding="utf-8")
    (run_root / "native-text.json").write_text(
        json.dumps({
            "usable": result.usable,
            "word_count": result.word_count,
            "char_count": result.char_count,
            "block_count": result.block_count,
            "pages": result.pages,
        }, indent=2) + "\n",
        encoding="utf-8",
    )


def normalize_part_number(value: str | None) -> str:
    return re.sub(r"[^a-z0-9]", "", (value or "").lower())


def part_number_matches(left: str | None, right: str | None) -> bool:
    normalized_left = normalize_part_number(left)
    normalized_right = normalize_part_number(right)
    return bool(normalized_left) and normalized_left == normalized_right


def alignment(value: str | None, vision_value: str | None) -> AlignmentStatus:
    if value is None and vision_value is None:
        return AlignmentStatus.UNCERTAIN
    if value is None or vision_value is None:
        return AlignmentStatus.MISMATCH
    return AlignmentStatus.MATCH if part_number_matches(value, vision_value) else AlignmentStatus.MISMATCH


def run_native_gemini_extraction(
    result: NativeTextResult,
    run_root: Path,
    category: CategorySchema,
    *,
    client: Any | None = None,
    api_key: str | None = None,
) -> Path:
    """Run the existing OCR-side extractor against native text in an isolated run."""

    run = RunDirectory(run_root)
    run_root.mkdir(parents=True, exist_ok=True)
    run.ocr_text.write_text(result.text + "\n", encoding="utf-8")
    return extract_ocr_to_json(run, category, client=client, api_key=api_key)


def selected_value(native_value: str | None, tesseract_value: str | None, usable: bool) -> str | None:
    return native_value if usable else tesseract_value


def summarize_changes(
    rows: Iterable[dict[str, Any]],
) -> dict[str, int]:
    counts = {"improved": 0, "regressed": 0, "unchanged": 0}
    for row in rows:
        native = row["native_part_status"]
        tesseract = row["tesseract_part_status"]
        if native == tesseract:
            counts["unchanged"] += 1
        elif native == AlignmentStatus.MATCH.value:
            counts["improved"] += 1
        elif tesseract == AlignmentStatus.MATCH.value:
            counts["regressed"] += 1
        else:
            counts["unchanged"] += 1
    return counts
