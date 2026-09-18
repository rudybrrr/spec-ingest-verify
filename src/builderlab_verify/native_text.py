"""Conservative native PDF text extraction used as a supplementary source."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pymupdf

from builderlab_verify.models import ExtractionBranch
from builderlab_verify.ocr import _extract_text_to_json
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
    """Extract native text and provenance without consulting the filename."""

    pages: list[dict[str, Any]] = []
    page_texts: list[str] = []
    word_count = 0
    char_count = 0
    block_count = 0
    with pymupdf.open(str(pdf_path)) as document:
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
    """Persist native text and an explicit native-layer status artifact."""

    run_root = Path(run_root)
    run_root.mkdir(parents=True, exist_ok=True)
    usable = result.usable
    status = "available" if result.text.strip() else "unavailable"
    payload = {
        "status": status,
        "reason": None if status == "available" else "insufficient_native_text",
        "usable": usable,
        "word_count": result.word_count,
        "char_count": result.char_count,
        "block_count": result.block_count,
        "structured_extraction_status": "not_attempted",
        "structured_extraction_error": None,
        "pages": result.pages,
    }
    (run_root / "native-text.txt").write_text(result.text + "\n", encoding="utf-8")
    (run_root / "native-text.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_native_failure_artifact(error: Exception, run_root: Path) -> None:
    """Persist native parsing failure while leaving the primary pipeline usable."""

    run_root = Path(run_root)
    run_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "failed",
        "reason": "native_text_extraction_failed",
        "usable": False,
        "word_count": 0,
        "char_count": 0,
        "block_count": 0,
        "structured_extraction_status": "not_attempted",
        "structured_extraction_error": None,
        "error": f"{type(error).__name__}: {error}",
        "pages": [],
    }
    (run_root / "native-text.txt").write_text("\n", encoding="utf-8")
    (run_root / "native-text.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def update_native_structured_status(run: RunDirectory, *, status: str, error: str | None = None) -> None:
    payload = json.loads(run.native_text_json.read_text(encoding="utf-8"))
    payload["structured_extraction_status"] = status
    payload["structured_extraction_error"] = error
    run.native_text_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def extract_native_to_json(
    run: RunDirectory,
    category: CategorySchema,
    *,
    client: Any | None = None,
    api_key: str | None = None,
) -> Path:
    """Run the existing structured text extractor into independent ``native.json``."""

    return _extract_text_to_json(
        run,
        category,
        text_path=run.native_text,
        output_path=run.native_json,
        branch=ExtractionBranch.NATIVE_TEXT,
        source_label="native PDF text",
        stage="native_text",
        client=client,
        api_key=api_key,
    )
