"""Read-only audit of expected-SKU evidence for the remaining human reviews.

This intentionally writes only a new diagnostic report. It does not call the
production verifier or mutate any existing run artifact.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pymupdf

from builderlab_verify.upstream import normalize_identifier


WORKSPACE = Path(__file__).resolve().parents[1]
ROOT = WORKSPACE / ".runs" / "evaluation-100-spec-sheet-upstream-sku-authoritative-v8"
OUT = WORKSPACE / ".runs" / "expected-sku-recovery-35"
REPORT_JSON = OUT / "report.json"
REPORT_MD = OUT / "report.md"


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def clean(value: Any) -> str:
    return " ".join(str(value or "").split())


def page_texts(run: Path, filename: str) -> dict[int, str]:
    payload = read_json(run / "native-text.json")
    pages = payload.get("pages") or []
    if pages:
        return {int(page.get("page", 0)): clean(page.get("text")) for page in pages if page.get("page")}
    text = (run / "native-text.txt").read_text(encoding="utf-8", errors="replace") if (run / "native-text.txt").exists() else ""
    result: dict[int, str] = {}
    current = 1
    for chunk in re.split(r"\[Page\s+(\d+)\]", text):
        if chunk.isdigit():
            current = int(chunk)
        elif chunk.strip():
            result[current] = clean(chunk)
    return result


def normalized_hits(text: str, expected: str) -> list[dict[str, Any]]:
    target = normalize_identifier(expected)
    if not target:
        return []
    positions = [(index, char) for index, char in enumerate(text) if char.isalnum()]
    compact = "".join(char.lower() for _, char in positions)
    hits: list[dict[str, Any]] = []
    start = 0
    while True:
        at = compact.find(target, start)
        if at < 0:
            break
        left = positions[max(0, at - 140)][0]
        right = positions[min(len(positions) - 1, at + len(target) - 1 + 140)][0] + 1
        evidence = clean(text[left:right])
        exact_spelling = normalize_identifier(text[left:right]) != "" and expected.lower() in text[left:right].lower()
        hits.append({"value": expected, "exact_spelling": exact_spelling, "evidence": evidence})
        start = at + len(target)
    return hits


def context_strength(evidence: str, *, label: str | None = None, role: str | None = None) -> str:
    lowered = evidence.lower()
    strong_labels = (
        "sku", "catalog number", "catalog no", "product number", "product no",
        "item number", "ordering code", "order number", "order no", "orderable part",
    )
    if role == "orderable_part" or label in strong_labels or any(item in lowered for item in strong_labels):
        return "strong_product_context"
    if "part number" in lowered or "part no" in lowered or "part #" in lowered:
        return "ambiguous_part_label"
    return "ambiguous_context"


def structured_hits(run: Path, expected: str) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    target = normalize_identifier(expected)
    for filename, source in (("ocr.json", "tesseract_structured"), ("vision.json", "vision_structured"), ("native.json", "native_structured")):
        payload = read_json(run / filename)
        field = (payload.get("fields") or {}).get("part_number") or {}
        value = field.get("value")
        if isinstance(value, str) and normalize_identifier(value) == target:
            result.append({"source": source, "page": field.get("page"), "exact_evidence": value, "context": "structured part_number field", "strength": "strong_product_context"})
    for filename, source in (("candidates.json", "docling"), ("docling-layout-table-experiment-candidates.json", "docling"), ("evidence-first-part-number-harvest-v3-candidates.json", "ocr_tsv_word_data")):
        payload = read_json(run / filename)
        for item in payload.get("candidates", []):
            if not isinstance(item, dict) or normalize_identifier(str(item.get("normalized_value", ""))) != target:
                continue
            context = clean(item.get("context") or item.get("row_text") or item.get("evidence_text"))
            result.append({
                "source": source,
                "page": item.get("page"),
                "exact_evidence": clean(item.get("raw_text") or expected),
                "context": context,
                "role": item.get("role"),
                "label": item.get("element_label") or item.get("column_header"),
                "strength": context_strength(context, label=item.get("element_label") or item.get("column_header"), role=item.get("role")),
            })
    return result


def pdf_links(pdf: Path, expected: str) -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    target = normalize_identifier(expected)
    try:
        document = pymupdf.open(str(pdf))
    except Exception:
        return hits
    with document:
        for page_number, page in enumerate(document, start=1):
            for link in page.get_links() or []:
                uri = clean(link.get("uri"))
                if uri and target in normalize_identifier(uri):
                    hits.append({"source": "pdf_hyperlink", "page": page_number, "exact_evidence": uri, "context": "embedded product URL contains expected SKU", "strength": "strong_product_context"})
    return hits


def decode_codes(pdf: Path, expected: str) -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    target = normalize_identifier(expected)
    detector = cv2.QRCodeDetector()
    try:
        document = pymupdf.open(str(pdf))
    except Exception:
        return hits
    with document:
        for page_number, page in enumerate(document, start=1):
            try:
                pix = page.get_pixmap(matrix=pymupdf.Matrix(2.5, 2.5), alpha=False)
                image = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
                if pix.n == 4:
                    image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
                else:
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                decoded, _, _ = detector.detectAndDecode(image)
            except Exception:
                decoded = ""
            if decoded and target in normalize_identifier(decoded):
                hits.append({"source": "qr_code", "page": page_number, "exact_evidence": decoded, "context": "locally decoded QR payload contains expected SKU", "strength": "strong_product_context"})
    return hits


def audit_case(filename: str, run_id: str) -> dict[str, Any]:
    run = ROOT / "runs" / run_id
    verification = read_json(run / "verification.json")
    expected = ((verification.get("upstream_sku") or {}).get("expected") or "").strip()
    raw: list[dict[str, Any]] = []
    for source, path in (("native_pdf_text", run / "native-text.txt"), ("tesseract_ocr_text", run / "ocr.txt")):
        text = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
        for hit in normalized_hits(text, expected):
            raw.append({"source": source, "page": None, "exact_evidence": expected if hit["exact_spelling"] else "normalized match", "context": hit["evidence"], "strength": context_strength(hit["evidence"])})
    for page, text in page_texts(run, filename).items():
        for hit in normalized_hits(text, expected):
            raw.append({"source": "native_pdf_text_page", "page": page, "exact_evidence": expected if hit["exact_spelling"] else "normalized match", "context": hit["evidence"], "strength": context_strength(hit["evidence"])})
    structured = structured_hits(run, expected)
    links = pdf_links(run / "source.pdf", expected)
    codes = decode_codes(run / "source.pdf", expected)
    all_hits = raw + structured + links + codes
    strong = [hit for hit in all_hits if hit["strength"] == "strong_product_context"]
    ambiguous = [hit for hit in all_hits if hit["strength"] != "strong_product_context"]
    namespace_sources = (verification.get("identifier_namespaces") or {}).get("manufacturer_part_number_sources") or {}
    structured_non_expected = [
        value for value in namespace_sources.values()
        if isinstance(value, str) and normalize_identifier(value) and normalize_identifier(value) != normalize_identifier(expected)
    ]
    mpn = (verification.get("identifier_namespaces") or {}).get("manufacturer_part_number")
    namespace_issue = bool(structured_non_expected and any(
        normalize_identifier(str(hit.get("exact_evidence"))) == normalize_identifier(expected)
        for hit in all_hits
    ))
    if namespace_issue:
        classification = "namespace_issue"
    elif strong:
        classification = "exact_expected_sku_present_strong_context" if any(hit.get("exact_evidence") == expected for hit in strong) else "normalized_expected_sku_present_strong_context"
    elif ambiguous:
        classification = "expected_sku_present_context_ambiguous"
    else:
        classification = "expected_sku_absent_all_representations"
    safe_support = bool(strong) and not namespace_issue
    return {
        "filename": filename,
        "run_id": run_id,
        "expected_sku": expected,
        "classification": classification,
        "safe_support": safe_support,
        "evidence": all_hits,
        "source_presence": sorted({hit["source"] for hit in all_hits}),
        "hyperlink_recoveries": links,
        "qr_barcode_recoveries": codes,
        "manufacturer_part_number": mpn,
    }


def main() -> None:
    manifest = read_json(ROOT / "manifest.json")
    cases = [item for item in (manifest.get("documents") or {}).items() if item[1].get("status") == "human_review"]
    results = [audit_case(filename, meta["run_id"]) for filename, meta in cases]
    counts = Counter(item["classification"] for item in results)
    report = {
        "experiment": "expected_sku_evidence_recovery_35",
        "source_run": str(ROOT),
        "production_behavior_changed": False,
        "filename_used_as_evidence": False,
        "case_count": len(results),
        "classification_counts": dict(sorted(counts.items())),
        "expected_sku_present_somewhere": sum(bool(item["evidence"]) for item in results),
        "expected_sku_absent_everywhere": sum(item["classification"] == "expected_sku_absent_all_representations" for item in results),
        "additional_conservative_safe_supports": sum(item["safe_support"] for item in results),
        "qr_barcode_recoveries": sum(bool(item["qr_barcode_recoveries"]) for item in results),
        "hyperlink_recoveries": sum(bool(item["hyperlink_recoveries"]) for item in results),
        "cases": results,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Expected-SKU evidence recovery audit",
        "",
        f"Cases: {report['case_count']}",
        f"Expected SKU present somewhere: {report['expected_sku_present_somewhere']}",
        f"Absent everywhere: {report['expected_sku_absent_everywhere']}",
        f"Additional conservative safe supports: {report['additional_conservative_safe_supports']}",
        f"QR/barcode recoveries: {report['qr_barcode_recoveries']}",
        f"Hyperlink recoveries: {report['hyperlink_recoveries']}",
        "",
        "| Case | Expected SKU | Classification | Safe support | Evidence |",
        "|---|---|---|---:|---|",
    ]
    for item in results:
        evidence = " <br> ".join(f"{hit['source']} p{hit.get('page') or '?'}: {clean(hit.get('exact_evidence'))}; {clean(hit.get('context'))[:220]}" for hit in item["evidence"][:4]) or "—"
        lines.append(f"| {item['filename']} | {item['expected_sku']} | {item['classification']} | {'yes' if item['safe_support'] else 'no'} | {evidence} |")
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("case_count", "classification_counts", "expected_sku_present_somewhere", "expected_sku_absent_everywhere", "additional_conservative_safe_supports", "qr_barcode_recoveries", "hyperlink_recoveries")}, indent=2))


if __name__ == "__main__":
    main()
