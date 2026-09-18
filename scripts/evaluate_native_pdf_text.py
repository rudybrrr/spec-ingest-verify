"""Evaluate native PDF text against the frozen 100-PDF comparison run."""

from __future__ import annotations

import json
import os
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

from builderlab_verify.models import AlignmentStatus
from builderlab_verify.schemas import CategoryField, CategorySchema

try:
    from experiments.native_pdf_text_experiment import (
        alignment,
        extract_native_text,
        run_native_gemini_extraction,
        summarize_changes,
        write_native_artifacts,
    )
except ModuleNotFoundError as error:  # support ``python scripts/...py`` from the repo root
    if error.name != "experiments":
        raise
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from experiments.native_pdf_text_experiment import (
        alignment,
        extract_native_text,
        run_native_gemini_extraction,
        summarize_changes,
        write_native_artifacts,
    )


BASELINE_ROOT = Path(".runs/evaluation-100-spec-sheet-schema-comparison")
OUTPUT_ROOT = Path(".runs/native-pdf-text-evaluation")
CORPUS = Path(os.environ.get("BUILDERLAB_CORPUS", "input"))
CATEGORY = CategorySchema(
    name="spec_sheet",
    fields=[
        CategoryField(name="part_number", description="Unique manufacturer/orderable part identifier", required=True),
        CategoryField(name="description", description="Short product description", required=False),
    ],
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _status(value: str | None) -> str:
    return value or AlignmentStatus.UNCERTAIN.value


def main() -> None:
    baseline = _read_json(BASELINE_ROOT / "manifest.json")["documents"]
    review = [name for name, entry in baseline.items() if entry["status"] == "human_review"]
    if len(review) != 47:
        raise RuntimeError(f"Expected 47 human-review PDFs in frozen baseline, found {len(review)}")
    (OUTPUT_ROOT / "runs").mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for filename in sorted(review):
        run_id = baseline[filename]["run_id"]
        baseline_run = BASELINE_ROOT / "runs" / run_id
        output_run = OUTPUT_ROOT / "runs" / run_id
        pdf_path = CORPUS / filename
        result = extract_native_text(pdf_path)
        write_native_artifacts(result, output_run)
        native_json = output_run / "native.json"
        extraction_error = None
        if result.usable and not native_json.exists():
            try:
                extracted = run_native_gemini_extraction(result, output_run, CATEGORY)
                shutil.copyfile(extracted, native_json)
            except Exception as error:  # preserve per-document progress and continue
                extraction_error = f"{type(error).__name__}: {error}"
        native_payload = _read_json(native_json) if native_json.exists() else {"fields": {}}
        tesseract_payload = _read_json(baseline_run / "ocr.json")
        vision_payload = _read_json(baseline_run / "vision.json")
        native_fields = native_payload.get("fields", {})
        tesseract_fields = tesseract_payload.get("fields", {})
        vision_fields = vision_payload.get("fields", {})
        native_pn = native_fields.get("part_number", {}).get("value")
        tesseract_pn = tesseract_fields.get("part_number", {}).get("value")
        vision_pn = vision_fields.get("part_number", {}).get("value")
        native_desc = native_fields.get("description", {}).get("value")
        tesseract_desc = tesseract_fields.get("description", {}).get("value")
        vision_desc = vision_fields.get("description", {}).get("value")
        row = {
            "filename": filename,
            "run_id": run_id,
            "usable_native_text": result.usable,
            "native_word_count": result.word_count,
            "native_char_count": result.char_count,
            "native_block_count": result.block_count,
            "native_extraction_completed": native_json.exists(),
            "extraction_error": extraction_error,
            "native_part_number": native_pn,
            "tesseract_part_number": tesseract_pn,
            "vision_part_number": vision_pn,
            "native_description": native_desc,
            "tesseract_description": tesseract_desc,
            "vision_description": vision_desc,
            "native_part_status": alignment(native_pn, vision_pn).value,
            "tesseract_part_status": _status(
                _read_json(baseline_run / "verification.json")["fields"].get("part_number", {}).get("status")
            ),
            "native_description_status": _status(
                alignment(native_desc, vision_desc).value if native_json.exists() else None
            ),
            "tesseract_description_status": _status(
                _read_json(baseline_run / "verification.json")["fields"].get("description", {}).get("status")
            ),
        }
        rows.append(row)

    changes = summarize_changes(rows)
    native_completed = [row for row in rows if row["native_extraction_completed"]]
    native_usable = [row for row in rows if row["usable_native_text"]]
    native_part_counts = Counter(row["native_part_status"] for row in native_completed)
    tesseract_part_counts = Counter(row["tesseract_part_status"] for row in rows)
    native_desc_counts = Counter(row["native_description_status"] for row in native_completed)

    baseline_verified = sum(entry["status"] == "verified" for entry in baseline.values())
    hybrid_current_verified = baseline_verified
    hybrid_optional_verified = baseline_verified
    hybrid_current_rows: list[dict[str, Any]] = []
    for row in rows:
        use_native = row["usable_native_text"] and row["native_extraction_completed"]
        part_status = row["native_part_status"] if use_native else row["tesseract_part_status"]
        desc_status = row["native_description_status"] if use_native else row["tesseract_description_status"]
        current_verified = part_status == AlignmentStatus.MATCH.value and desc_status == AlignmentStatus.MATCH.value
        optional_verified = part_status == AlignmentStatus.MATCH.value
        hybrid_current_verified += int(current_verified)
        hybrid_optional_verified += int(optional_verified)
        hybrid_current_rows.append({"filename": row["filename"], "use_native": use_native, "part_status": part_status, "description_status": desc_status, "verified": current_verified})

    recovered = [row for row in rows if row["native_part_status"] == AlignmentStatus.MATCH.value and row["tesseract_part_status"] != AlignmentStatus.MATCH.value]
    report = {
        "experiment": "native_pdf_text_part_number",
        "production_changes": False,
        "baseline": {
            "root": str(BASELINE_ROOT),
            "total_pdfs": len(baseline),
            "verified": baseline_verified,
            "human_review": len(review),
        },
        "native_text": {
            "usable_definition": {"min_words": 10, "min_chars": 50},
            "review_pdf_count": len(review),
            "usable_count": len(native_usable),
            "usable_percentage": round(len(native_usable) * 100 / len(review), 2),
            "gemini_completed_count": len(native_completed),
            "part_number_alignment": dict(native_part_counts),
            "description_alignment_recorded": dict(native_desc_counts),
        },
        "comparison": {
            "tesseract_part_number_alignment": dict(tesseract_part_counts),
            "native_part_number_alignment": dict(native_part_counts),
            "improved_regressed_unchanged": changes,
            "recovered_examples": recovered[:20],
        },
        "simulation": {
            "hybrid_native_when_usable_current_gating": {
                "verified": hybrid_current_verified,
                "human_review": len(baseline) - hybrid_current_verified,
            },
            "hybrid_native_when_usable_optional_description_non_gating": {
                "verified": hybrid_optional_verified,
                "human_review": len(baseline) - hybrid_optional_verified,
            },
            "per_document": hybrid_current_rows,
        },
        "rows": rows,
    }
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
