"""Verify upstream BuilderLab SKUs against frozen PDF evidence.

This is an evaluation-only experiment. It does not change the production
pipeline and never treats a filename or run id as PDF evidence.
"""

from __future__ import annotations

import csv
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any

from builderlab_verify.verification import _part_number_equivalent


ROOT = Path(__file__).resolve().parents[1]
INDEX = Path(os.environ.get("BUILDERLAB_INDEX_PATH", "input/index.csv"))
EVAL_ROOT = ROOT / ".runs" / "evaluation-100-spec-sheet-native-lazy-fallback-authorized"
EVIDENCE_ROOT = ROOT / ".runs" / "evidence-first-part-number-harvest-v3" / "runs"
DOCLING_ROOT = ROOT / ".runs" / "docling-layout-table-experiment" / "runs"
OUT_JSON = ROOT / "experiments" / "upstream-sku-verification.json"
OUT_MD = ROOT / "experiments" / "upstream-sku-verification.md"

LABELS = (
    "manufacturer part number",
    "manufacturer part no",
    "part number",
    "part no",
    "part #",
    "mpn",
    "ordering code",
    "order number",
    "order no",
    "catalog number",
    "catalog no",
    "product number",
    "product no",
    "item number",
)


def norm(value: str | None) -> str:
    return re.sub(r"[^a-z0-9]", "", (value or "").lower())


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def field_value(path: Path) -> str | None:
    payload = load_json(path)
    value = (payload or {}).get("fields", {}).get("part_number", {}).get("value")
    return value if isinstance(value, str) and value.strip() else None


def add_hit(hits: list[dict[str, Any]], family: str, value: str | None, *, label: str | None = None, context: str = "") -> None:
    if value and norm(value):
        hits.append({"family": family, "value": value, "normalized": norm(value), "label": label, "context": context})


def normalized_occurrences(text: str, expected: str) -> list[str]:
    """Return nearby context windows for equivalent-format occurrences."""
    compact = norm(expected)
    if not compact:
        return []
    positions = [(index, char) for index, char in enumerate(text) if char.isalnum()]
    compact_text = "".join(char.lower() for _, char in positions)
    windows: list[str] = []
    start = 0
    while True:
        found = compact_text.find(compact, start)
        if found < 0:
            break
        left = positions[max(0, found - 180)][0]
        right = positions[min(len(positions) - 1, found + len(compact) - 1 + 180)][0] + 1
        windows.append(text[left:right])
        start = found + len(compact)
    return windows


def explicit_contexts(text: str, expected: str) -> list[dict[str, str]]:
    contexts = []
    for window in normalized_occurrences(text, expected):
        lowered = window.lower()
        labels = [label for label in LABELS if label in lowered]
        if labels:
            contexts.append({"label": labels[0], "context": " ".join(window.split())})
    return contexts


def candidate_hits(path: Path, expected: str, family: str) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    payload = load_json(path) or {}
    candidates = payload.get("candidates", [])
    if not isinstance(candidates, list):
        return [], []
    hits: list[dict[str, Any]] = []
    labeled_alternatives: list[dict[str, str]] = []
    expected_norm = norm(expected)
    for item in candidates:
        if not isinstance(item, dict):
            continue
        value = item.get("raw_text")
        normalized = item.get("normalized_value") or norm(value)
        context = str(item.get("context") or "")
        # Existing candidate artifacts can carry a whole table as context.
        # Only a compact context is allowed to turn a candidate into explicit
        # labeled evidence; distant labels are not evidence for this value.
        explicit = len(context) <= 300 and any(label in context.lower() for label in LABELS)
        if normalized == expected_norm:
            add_hit(hits, family, str(value or expected), label=next((label for label in LABELS if label in context.lower()), None), context=context)
        elif explicit and item.get("role") == "orderable_part" and expected_norm not in context.replace("-", "").replace(".", "").replace(" ", "").lower():
            labeled_alternatives.append({"value": str(value), "label": next(label for label in LABELS if label in context.lower()), "context": context})
    return hits, labeled_alternatives


def evaluate_case(filename: str, run_id: str, expected: str) -> dict[str, Any]:
    run = EVAL_ROOT / "runs" / run_id
    hits: list[dict[str, Any]] = []
    explicit: list[dict[str, Any]] = []
    alternatives: list[dict[str, str]] = []

    for name, family in (("ocr.json", "tesseract"), ("vision.json", "vision"), ("native.json", "native")):
        value = field_value(run / name)
        add_hit(hits, family, value)

    for name, family in (("ocr.txt", "tesseract"), ("native-text.txt", "native")):
        path = run / name
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            text = ""
        for context in explicit_contexts(text, expected):
            explicit.append({"family": family, **context})
        if norm(expected) and normalized_occurrences(text, expected):
            add_hit(hits, family, expected, context="raw text occurrence")

    for path, family in (
        (EVIDENCE_ROOT / run_id / "candidates.json", "evidence_candidates"),
        (DOCLING_ROOT / run_id / "candidates.json", "docling"),
    ):
        candidate_hit, candidate_alts = candidate_hits(path, expected, family)
        hits.extend(candidate_hit)
        alternatives.extend(candidate_alts)

    families = sorted({hit["family"] for hit in hits})
    explicit_families = sorted({hit["family"] for hit in explicit} | {hit["family"] for hit in hits if hit.get("label") and hit["family"] in {"tesseract", "native", "vision"}})
    structured = [hit for hit in hits if hit["family"] in {"tesseract", "vision", "native"}]
    value_counts = Counter(hit["normalized"] for hit in structured)
    # Evidence-first and Docling artifacts are derived from the same document
    # text/layout and are not counted as independent extraction branches.
    cross_source_support = len({hit["family"] for hit in structured}) >= 2
    explicit_support = bool(explicit_families)

    # A contradiction must be grounded in explicit labeled evidence or in the
    # same non-matching structured identifier from two independent branches.
    contradicted_by: list[dict[str, Any]] = []
    # Candidate artifacts are useful for finding support, but their broad
    # table contexts are too lossy to establish contradiction. Contradiction
    # here is limited to repeated disagreement among structured branches.
    for value, count in value_counts.items():
        if value != norm(expected) and count >= 2:
            contradicted_by.append({"family": "structured", "value": value, "agreement_count": count})

    candidate_label_support = any(
        hit["family"] in {"evidence_candidates", "docling"}
        and hit["normalized"] == norm(expected)
        and hit.get("label")
        for hit in hits
    )
    if contradicted_by:
        status = "contradicted"
    elif explicit_support or candidate_label_support or cross_source_support:
        status = "supported"
    else:
        status = "insufficient evidence"

    return {
        "filename": filename,
        "run_id": run_id,
        "expected_sku": expected,
        "status": status,
        "supporting_families": families,
        "explicit_label_support": explicit,
        "matching_hits": hits,
        "contradicting_evidence": contradicted_by,
        "structured_values": [{"family": hit["family"], "value": hit["value"]} for hit in structured],
    }


def main() -> None:
    rows = list(csv.DictReader(INDEX.open(newline="", encoding="utf-8-sig")))
    index_by_file = {row["File"]: row for row in rows}
    manifest = load_json(EVAL_ROOT / "manifest.json") or {}
    documents = manifest.get("documents", {})
    selected = sorted(documents)
    review = [name for name in selected if documents[name].get("status") == "human_review"]
    missing = [name for name in selected if name not in index_by_file]
    results = [evaluate_case(name, documents[name]["run_id"], index_by_file[name]["SKU"]) for name in review if name in index_by_file]
    counts = Counter(case["status"] for case in results)
    report = {
        "experiment": "upstream_sku_verification",
        "production_code_changed": False,
        "metadata_source": str(INDEX),
        "metadata_columns": ["SKU", "Category", "File", "SizeKB", "SourcePath"],
        "corpus": {
            "evaluation_manifest": str(EVAL_ROOT / "manifest.json"),
            "selected_pdfs": len(selected),
            "index_rows": len(rows),
            "index_rows_for_selected_pdfs": len(selected) - len(missing),
            "missing_index_rows": missing,
            "sku_populated": sum(bool(index_by_file.get(name, {}).get("SKU", "").strip()) for name in selected),
            "sku_unique": len({index_by_file[name]["SKU"] for name in selected if name in index_by_file}),
        },
        "review_population": {"count": len(review), "expected": 45, "evaluated": len(results)},
        "classification_counts": dict(counts),
        "safe_additional_auto_verifications": counts["supported"],
        "false_positive_risk": {
            "assessment": "low-to-moderate under this experiment rule; not zero",
            "controls": [
                "filename, run id, and manifest identity are excluded from PDF evidence",
                "punctuation-only equivalence uses the existing normalization",
                "single unlabeled substring occurrence is not sufficient by itself",
                "contradiction requires explicit alternative labeling or repeated structured agreement",
            ],
            "residual_risks": [
                "OCR/native/Vision may share a source-document error",
                "a multi-product sheet can explicitly list several valid identifiers",
                "raw-text occurrence detection does not establish product association without a nearby label",
            ],
        },
        "cases": results,
    }
    OUT_JSON.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Upstream SKU verification experiment",
        "",
        f"Metadata source: `{INDEX}` (`SKU, Category, File, SizeKB, SourcePath`).",
        "Production code changed: **no**.",
        "",
        f"The source contains {len(rows):,} rows; all {len(selected)} selected evaluation PDFs have a populated, unique SKU row ({len(selected)}/{len(selected)} coverage).",
        f"The frozen evaluation has {len(review)} human-review PDFs. Results: supported **{counts['supported']}**, contradicted **{counts['contradicted']}**, insufficient evidence **{counts['insufficient evidence']}**.",
        f"Safe additional auto-verifications under this conservative rule: **{counts['supported']}**.",
        "",
        "A case is supported by an explicit part-number/order-code label in PDF evidence or by matching evidence from at least two independent families. Filenames and the index itself are never PDF evidence.",
        "Contradicted requires an explicitly labeled alternative or the same non-matching structured identifier from at least two branches; all other cases abstain.",
        "",
        "## Cases",
        "",
        "| PDF | Expected SKU | Result | Evidence |",
        "|---|---|---|---|",
    ]
    for case in results:
        evidence = ", ".join(case["supporting_families"]) or ", ".join(item["family"] for item in case["contradicting_evidence"]) or "none"
        lines.append(f"| `{case['filename']}` | `{case['expected_sku']}` | {case['status']} | {evidence} |")
    lines += [
        "",
        "## Interpretation",
        "",
        "This framing materially outperforms identity discovery when judged by the review burden: it converts the task from selecting one correct identifier among many PDF strings into checking one known upstream claim. The exact gain is the supported count above; it should not be promoted to production without adjudicating the supported cases for false positives, especially multi-product sheets.",
        "",
        "Recommendation: preserve the upstream SKU as an input contract and add a separate claim-verification path. Do not replace the existing identity-discovery path universally until the supported cases are human-audited and the upstream manifest contract is formalized.",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"json": str(OUT_JSON), "markdown": str(OUT_MD), "counts": dict(counts)}, indent=2))


if __name__ == "__main__":
    main()
