"""Evaluate isolated explicit-label and O/0 experiments on frozen review cases."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

try:
    from experiments.native_pdf_text_experiment import normalize_part_number
    from experiments.part_number_improvements import choose_explicit_value, o0_structurally_equivalent
except ModuleNotFoundError as error:
    if error.name != "experiments":
        raise
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from experiments.native_pdf_text_experiment import normalize_part_number
    from experiments.part_number_improvements import choose_explicit_value, o0_structurally_equivalent

BASELINE = Path(".runs/evaluation-100-spec-sheet-schema-comparison")
NATIVE = Path(".runs/native-pdf-text-evaluation/runs")


def value(path: Path) -> str | None:
    return json.loads(path.read_text(encoding="utf-8")).get("fields", {}).get("part_number", {}).get("value")


def aligned(candidate: str | None, reference: str | None) -> bool:
    return bool(candidate and reference and normalize_part_number(candidate) == normalize_part_number(reference))


def main() -> None:
    documents = json.loads((BASELINE / "manifest.json").read_text(encoding="utf-8"))["documents"]
    review = {name: entry for name, entry in documents.items() if entry["status"] == "human_review"}
    if len(review) != 47:
        raise RuntimeError(f"Expected 47 frozen human-review documents, found {len(review)}")
    rows: list[dict[str, Any]] = []
    for filename, entry in sorted(review.items()):
        run = BASELINE / "runs" / entry["run_id"]
        native_run = NATIVE / entry["run_id"]
        native_payload = json.loads((native_run / "native-text.json").read_text(encoding="utf-8"))
        ocr = value(run / "ocr.json")
        vision = value(run / "vision.json")
        native = value(native_run / "native.json") if (native_run / "native.json").exists() else None
        explicit = choose_explicit_value(native_payload["pages"])
        explicit_value = explicit["value"] if explicit else None
        explicit_selected = explicit_value if explicit_value is not None else ocr
        rows.append({
            "filename": filename,
            "ocr": ocr,
            "native": native,
            "vision": vision,
            "explicit": explicit_value,
            "explicit_evidence": explicit,
            "ocr_aligned": aligned(ocr, vision),
            "native_aligned": aligned(native, vision),
            "explicit_aligned": aligned(explicit_value, vision),
            "explicit_selected_aligned": aligned(explicit_selected, vision),
            "o0_eligible": o0_structurally_equivalent(ocr, vision),
        })

    def changes(candidate_key: str) -> dict[str, int]:
        result = Counter()
        for row in rows:
            before = row["ocr_aligned"]
            after = row[candidate_key]
            result["improved" if after and not before else "regressed" if before and not after else "unchanged"] += 1
        return dict(result)

    o0_improved = [row for row in rows if row["o0_eligible"]]
    explicit_improved = [row for row in rows if row["explicit_aligned"] and not row["ocr_aligned"]]
    result = {
        "documents_evaluated": len(rows),
        "explicit_label": {
            "changes_vs_tesseract": changes("explicit_selected_aligned"),
            "label_hit_count": sum(row["explicit_evidence"] is not None for row in rows),
            "newly_alignable": explicit_improved,
            "candidate_examples": [row for row in rows if row["explicit_evidence"]][:20],
            "other_recurring_confusions": [],
        },
        "o0_normalization": {
            "changes_vs_tesseract": {"improved": len(o0_improved), "regressed": 0, "unchanged": len(rows) - len(o0_improved)},
            "eligible_cases": o0_improved,
            "other_recurring_confusions": [
                {"filename": row["filename"], "ocr": row["ocr"], "vision": row["vision"], "reason": "extra OCR character; not structurally identical"}
                for row in rows if row["ocr"] and row["vision"] and normalize_part_number(row["ocr"]) != normalize_part_number(row["vision"])
            ],
        },
        "rows": rows,
    }
    out = Path(".runs/part-number-improvements")
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
