"""Isolated PaddleOCR vs Tesseract benchmark for ``part_number`` only."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from builderlab_verify.ocr import extract_ocr_to_json
from builderlab_verify.schemas import CategoryField, CategorySchema
from builderlab_verify.storage import RunDirectory
from experiments.anchor_ocr_experiment import (
    _cheap_profile,
    discover_human_review,
    select_representative,
)


DEFAULT_EVAL_ROOT = Path(".runs/evaluation-100-spec-sheet-schema-comparison")
DEFAULT_OUTPUT_ROOT = Path(".runs/paddle-ocr-experiment")
DEFAULT_REPORT = Path("experiments/paddle-ocr-experiment.md")
PADDLE_CACHE = Path(".paddlex")

CATEGORY = CategorySchema(
    name="spec_sheet",
    fields=[
        CategoryField(
            name="part_number",
            description=(
                "Unique manufacturer/orderable part identifier for the specific product represented by the document. "
                "Do not use product-family names, document titles, drawing labels, stock labels, or arbitrary variant identifiers. "
                "Return null for multi-product/portfolio documents or when no single unique part number is clearly identifiable."
            ),
            required=True,
        ),
        CategoryField(name="description", description="Short product description", required=False),
    ],
)


def normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.casefold()).strip()


def aligned_part_number(value: str | None, reference: str | None) -> bool:
    return value is not None and reference is not None and normalize(value) == normalize(reference)


def normalize_paddle_result(result: Any) -> str:
    """Convert a PaddleOCR result payload into raw reading-order text.

    The small helper accepts both the compact test shape and PaddleOCR 3.x
    result objects/dicts so the benchmark never feeds filenames or other
    metadata into Gemini.
    """
    if hasattr(result, "json"):
        result = result.json
    if callable(result):
        result = result()
    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        if isinstance(result.get("res"), dict):
            result = result["res"]
        texts = result.get("rec_texts") or result.get("texts") or []
        return "\n".join(str(text) for text in texts if str(text).strip())
    if isinstance(result, (list, tuple)):
        lines: list[str] = []
        if len(result) % 2 == 0 and all(
            isinstance(result[index], (list, tuple))
            and len(result[index]) == 2
            and isinstance(result[index][0], str)
            for index in range(1, len(result), 2)
        ):
            return "\n".join(str(result[index][0]) for index in range(1, len(result), 2))
        for item in result:
            if isinstance(item, (list, tuple)) and len(item) == 2 and isinstance(item[1], (list, tuple)):
                candidate = item[1][0] if item[1] else ""
                if isinstance(candidate, str):
                    lines.append(candidate)
            elif isinstance(item, (list, tuple)):
                nested = normalize_paddle_result(item)
                if nested:
                    lines.append(nested)
        return "\n".join(lines)
    return ""


def _field_value(payload: dict[str, Any], field: str = "part_number") -> str | None:
    value = payload.get("fields", {}).get(field, {}).get("value")
    return value if value is None or isinstance(value, str) else str(value)


def _new_paddle_ocr() -> Any:
    os.environ.setdefault("PADDLE_PDX_CACHE_HOME", str(PADDLE_CACHE.resolve()))
    from paddleocr import PaddleOCR

    return PaddleOCR(
        lang="en",
        enable_mkldnn=False,
        text_detection_model_name="PP-OCRv5_mobile_det",
        text_recognition_model_name="en_PP-OCRv5_mobile_rec",
        text_det_limit_side_len=960,
        text_det_limit_type="max",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )


def paddle_page_text(ocr: Any, image_path: Path) -> str:
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        image.thumbnail((1600, 1600), Image.Resampling.LANCZOS)
        results = ocr.predict(np.asarray(image))
    return "\n".join(text for result in results if (text := normalize_paddle_result(result)))


def _gemini_extract(ocr_text: str, run_root: Path, api_key: str | None) -> dict[str, Any]:
    run_root.mkdir(parents=True, exist_ok=True)
    run = RunDirectory(run_root)
    run.ocr_text.write_text(ocr_text, encoding="utf-8")
    extract_ocr_to_json(run, CATEGORY, api_key=api_key)
    return json.loads(run.ocr_json.read_text(encoding="utf-8"))


def run_experiment(
    eval_root: Path = DEFAULT_EVAL_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    report_path: Path = DEFAULT_REPORT,
    *,
    limit: int = 20,
) -> dict[str, Any]:
    docs = discover_human_review(eval_root)
    if len(docs) != 47:
        raise RuntimeError(f"Expected 47 human-review PDFs, found {len(docs)}")
    selected = select_representative([_cheap_profile(eval_root, doc) for doc in docs], limit)
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        from builderlab_verify.config import Settings

        configured = Settings().gemini_api_key
        api_key = configured.get_secret_value() if configured is not None else None
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is required for OCR-side extraction")

    paddle = _new_paddle_ocr()
    cases: list[dict[str, Any]] = []
    for doc in selected:
        source = eval_root / "runs" / doc["run_id"]
        out = output_root / "runs" / doc["run_id"]
        raw_path = out / "paddle-ocr.txt"
        if raw_path.exists():
            raw_text = raw_path.read_text(encoding="utf-8")
        else:
            pages = sorted((source / "pages").glob("page-*.png"))
            raw_text = "\n\n".join(paddle_page_text(paddle, page) for page in pages) + "\n"
            out.mkdir(parents=True, exist_ok=True)
            raw_path.write_text(raw_text, encoding="utf-8")
        extraction_path = out / "paddle-ocr.json"
        if extraction_path.exists():
            alternative = json.loads(extraction_path.read_text(encoding="utf-8"))
        else:
            alternative = _gemini_extract(raw_text, out, api_key)
            extraction_path.write_text(json.dumps(alternative, indent=2) + "\n", encoding="utf-8")
        baseline = json.loads((source / "ocr.json").read_text(encoding="utf-8"))
        vision = json.loads((source / "vision.json").read_text(encoding="utf-8"))
        tesseract_value = _field_value(baseline)
        paddle_value = _field_value(alternative)
        reference = _field_value(vision)
        tesseract_aligned = aligned_part_number(tesseract_value, reference)
        paddle_aligned = aligned_part_number(paddle_value, reference)
        cases.append({
            "filename": doc["filename"],
            "run_id": doc["run_id"],
            "tesseract": tesseract_value,
            "paddleocr": paddle_value,
            "vision_reference": reference,
            "tesseract_aligned": tesseract_aligned,
            "paddleocr_aligned": paddle_aligned,
            "status": (
                "improved" if paddle_aligned and not tesseract_aligned else
                "regressed" if tesseract_aligned and not paddle_aligned else
                "unchanged"
            ),
        })
    aggregate = {
        "improved": sum(case["status"] == "improved" for case in cases),
        "regressed": sum(case["status"] == "regressed" for case in cases),
        "unchanged": sum(case["status"] == "unchanged" for case in cases),
        "integration_threshold_met": sum(case["status"] == "improved" for case in cases) >= 3
        and sum(case["status"] == "regressed" for case in cases) <= 1,
    }
    result = {
        "experiment": "paddleocr_part_number",
        "documents_tested": len(cases),
        "selected": [case["filename"] for case in cases],
        "aggregate": aggregate,
        "cases": cases,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    report_path.write_text(render_report(result), encoding="utf-8")
    return result


def render_report(result: dict[str, Any]) -> str:
    agg = result["aggregate"]
    lines = [
        "# PaddleOCR vs Tesseract `part_number` benchmark", "",
        "PaddleOCR raw OCR was passed through the existing OCR-side Gemini extraction. Vision, schemas, verification, and filename handling were unchanged; the frozen Vision result is comparison-only.", "",
        f"- PDFs tested: {result['documents_tested']}",
        f"- Improved: {agg['improved']}", f"- Regressed: {agg['regressed']}", f"- Unchanged: {agg['unchanged']}",
        f"- Threshold met (>=3 improvements and <=1 regression): `{agg['integration_threshold_met']}`", "",
        "| PDF | Tesseract | PaddleOCR | frozen reference | status |", "|---|---|---|---|---|",
    ]
    for case in result["cases"]:
        lines.append(f"| `{case['filename']}` | `{case['tesseract']}` | `{case['paddleocr']}` | `{case['vision_reference']}` | {case['status']} |")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-root", type=Path, default=DEFAULT_EVAL_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()
    result = run_experiment(args.eval_root, args.output_root, args.report, limit=args.limit)
    print(json.dumps(result["aggregate"], indent=2))


if __name__ == "__main__":
    main()
