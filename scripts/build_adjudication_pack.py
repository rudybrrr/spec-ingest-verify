from __future__ import annotations

import csv
import html
import json
import re
from collections import Counter
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parents[1]
AUTHORITATIVE = WORKSPACE / ".runs" / "evaluation-100-spec-sheet-native-lazy-fallback-authorized"
EVIDENCE_ROOT = WORKSPACE / ".runs" / "evidence-first-part-number-harvest-v3"
DOCLING_ROOT = WORKSPACE / ".runs" / "docling-layout-table-experiment"
OUTPUT = WORKSPACE / "adjudication_pack"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def field_value(payload: dict, name: str) -> dict:
    field = payload.get("fields", {}).get(name, {})
    return {
        "value": field.get("value") or None,
        "page": field.get("page") or None,
        "unit": field.get("unit"),
    }


def native_text_part_number(run_dir: Path) -> dict:
    path = run_dir / "native-text.json"
    if not path.exists():
        return {"value": None, "page": None, "unit": None}
    payload = read_json(path)
    pattern = re.compile(
        r"(?:manufacturer\s+part\s+number|part\s+number|part\s+no\.?|model\s+number)\s*[:|]?\s*([A-Z0-9][A-Z0-9./_–—-]{2,})",
        re.I,
    )
    for page in payload.get("pages", []):
        match = pattern.search(page.get("text") or "")
        if match:
            return {"value": match.group(1).strip(), "page": page.get("page"), "unit": None}
    return {"value": None, "page": None, "unit": None}


def clean_text(value, limit=900):
    if value is None:
        return None
    value = re.sub(r"\s+", " ", str(value)).strip()
    return value[:limit] + ("…" if len(value) > limit else "")


def unique_candidates(payload: dict, source: str, limit=5):
    candidates = payload.get("candidates", [])
    accepted = [c for c in candidates if c.get("accepted")]
    orderable = [c for c in candidates if c.get("role") == "orderable_part"]
    selected = accepted + orderable
    seen = set()
    result = []
    for c in sorted(selected, key=lambda x: (-float(x.get("score") or 0), str(x.get("raw_text") or ""))):
        raw = clean_text(c.get("raw_text"), 180)
        normalized = clean_text(c.get("normalized_value"), 180)
        key = (normalized or raw or "").lower()
        if not key or key in seen:
            continue
        seen.add(key)
        evidence = c.get("evidence_text") or c.get("row_text") or c.get("context") or ""
        result.append(
            {
                "source": source,
                "value": raw,
                "normalized_value": normalized,
                "page": c.get("page"),
                "role": c.get("role"),
                "score": c.get("score"),
                "accepted": bool(c.get("accepted")),
                "supporting_text": clean_text(evidence),
                "context": clean_text(c.get("context")),
                "bbox": c.get("bbox"),
                "abstention_reason": c.get("abstention_reason") or None,
            }
        )
        if limit is not None and len(result) >= limit:
            break
    return result


def strong_suggestion(evidence_candidates, docling_candidates):
    candidates = evidence_candidates + docling_candidates
    explicit = []
    label_re = re.compile(r"manufacturer\s+part\s+number", re.I)
    for candidate in candidates:
        value = (candidate.get("value") or "").strip()
        support = " ".join(filter(None, [candidate.get("supporting_text"), candidate.get("context")]))
        if not candidate.get("accepted") or candidate.get("role") != "orderable_part":
            continue
        if len(value) < 3 or not label_re.search(support):
            continue
        compact_support = re.sub(r"\s+", " ", support)
        direct_label = re.search(
                r"manufacturer\s+part\s+number\s*[:|\-]\s*" + re.escape(value),
            compact_support,
            re.I,
        )
        if not direct_label:
            extracted = re.search(
                r"manufacturer\s+part\s+number\s*[:|\-]\s*([A-Z0-9][A-Z0-9./_–—-]{2,})",
                compact_support,
                re.I,
            )
            extracted_value = extracted.group(1) if extracted else ""
            if extracted_value.lower() in {"description", "number", "part", "code", "information"}:
                continue
            if not extracted_value:
                continue
            candidate = dict(candidate)
            candidate["value"] = extracted_value
            value = extracted_value
        if value.startswith(".") or " " in value and len(value.split()) > 3:
            continue
        explicit.append(candidate)
    by_norm = {}
    for candidate in explicit:
        by_norm.setdefault((candidate.get("normalized_value") or candidate.get("value") or "").lower(), []).append(candidate)
    if len(by_norm) != 1:
        return None
    only = next(iter(by_norm.values()))
    if len({c.get("source") for c in only}) < 1:
        return None
    return {
        "value": only[0]["value"],
        "basis": "single unique accepted orderable-part candidate with explicit label context",
        "evidence": only,
    }


def suggested_category(evidence_candidates, docling_candidates, descriptions):
    all_candidates = evidence_candidates + docling_candidates
    support = " ".join(filter(None, [c.get("supporting_text") for c in all_candidates])).lower()
    accepted = [c for c in all_candidates if c.get("accepted") and c.get("role") == "orderable_part"]
    unique = {c.get("normalized_value") or c.get("value") for c in accepted}
    if "manufacturer part number" in support:
        return "explicit manufacturer part number"
    if "ordering information" in support or "ordering code" in support or "order code" in support:
        return "ordering code"
    if len(unique) > 1 or "part number" in support and len(accepted) > 1:
        return "product table"
    if len(descriptions) > 1 or any(word in support for word in ("series", "family", "portfolio")):
        return "family/portfolio document"
    if accepted:
        return "product table"
    return "ambiguous"


def file_link(path: Path) -> str:
    return str(path.resolve()) if path.exists() else None


def main():
    report = read_json(AUTHORITATIVE / "report.json")
    review_docs = [d for d in report["documents"] if d.get("status") == "human_review"]
    if len(review_docs) != 45:
        raise SystemExit(f"Expected 45 human-review documents, found {len(review_docs)}")

    manifest = read_json(AUTHORITATIVE / "manifest.json")
    input_dir = Path(manifest.get("input_dir", ""))
    records = []
    for doc in review_docs:
        filename = doc["filename"]
        run_id = doc["run_id"]
        run_dir = AUTHORITATIVE / "runs" / run_id
        ocr = read_json(run_dir / "ocr.json")
        native = read_json(run_dir / "native.json") if (run_dir / "native.json").exists() else {}
        vision = read_json(run_dir / "vision.json")
        verification = read_json(run_dir / "verification.json")
        evidence = read_json(EVIDENCE_ROOT / "runs" / run_id / "candidates.json")
        docling = read_json(DOCLING_ROOT / "runs" / run_id / "candidates.json")
        evidence_candidates = unique_candidates(evidence, "evidence-first")
        docling_candidates = unique_candidates(docling, "docling")
        evidence_candidates_all = unique_candidates(evidence, "evidence-first", limit=None)
        docling_candidates_all = unique_candidates(docling, "docling", limit=None)
        native_value = field_value(native, "part_number") if native else native_text_part_number(run_dir)
        current = {
            "tesseract": field_value(ocr, "part_number"),
            "native_text": native_value,
            "vision": field_value(vision, "part_number"),
        }
        descriptions = [
            x["value"] for x in [field_value(ocr, "description"), field_value(native, "description"), field_value(vision, "description")]
            if x["value"]
        ]
        preview_root = run_dir
        previews = sorted(preview_root.glob("pages/page-*.png"))
        if not previews:
            fallback_dir = WORKSPACE / ".runs" / "evaluation-100-spec-sheet-native-integrated-authorized" / "runs" / run_id
            previews = sorted(fallback_dir.glob("pages/page-*.png"))
        preview_paths = [file_link(p) for p in previews[:3]]
        pdf_path = input_dir / filename
        if not pdf_path.exists() and (run_dir / "source.pdf").exists():
            pdf_path = run_dir / "source.pdf"
        suggestion = strong_suggestion(evidence_candidates_all, docling_candidates_all)
        records.append(
            {
                "filename": filename,
                "run_id": run_id,
                "source_pdf_path": file_link(pdf_path),
                "current_part_numbers": current,
                "evidence_first_candidates": evidence_candidates,
                "docling_candidates": docling_candidates,
                "preview_paths": preview_paths,
                "suggested_answer": suggestion,
                "suggested_reason_category": suggested_category(evidence_candidates_all, docling_candidates_all, descriptions),
                "human_ground_truth": {
                    "part_number": None,
                    "reason_category": None,
                    "notes": "",
                },
                "manual_inspection_needed": suggestion is None,
                "verification_status": verification.get("status"),
            }
        )

    OUTPUT.mkdir(exist_ok=True)
    json_path = OUTPUT / "ground_truth_template.json"
    json_path.write_text(json.dumps({
        "schema": {
            "part_number": "exact manufacturer/orderable part number, or null if no single unique part number exists",
            "reason_category": ["explicit manufacturer part number", "ordering code", "product table", "family/portfolio document", "ambiguous", "other"],
        },
        "source_run": str(AUTHORITATIVE.relative_to(WORKSPACE)),
        "case_count": len(records),
        "records": records,
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    columns = [
        "filename", "run_id", "tesseract_part_number", "native_text_part_number", "vision_part_number",
        "suggested_answer", "suggested_reason_category", "manual_inspection_needed", "evidence_first_candidates",
        "docling_candidates", "preview_paths", "ground_truth_part_number", "ground_truth_reason_category", "adjudicator_notes",
    ]
    csv_path = OUTPUT / "adjudication_records.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for r in records:
            def values(source):
                return [source.get("value") for source in r["current_part_numbers"].values() if source.get("value")]
            writer.writerow({
                "filename": r["filename"],
                "run_id": r["run_id"],
                "tesseract_part_number": r["current_part_numbers"]["tesseract"]["value"] or "",
                "native_text_part_number": r["current_part_numbers"]["native_text"]["value"] or "",
                "vision_part_number": r["current_part_numbers"]["vision"]["value"] or "",
                "suggested_answer": (r["suggested_answer"] or {}).get("value", ""),
                "suggested_reason_category": r["suggested_reason_category"],
                "manual_inspection_needed": str(r["manual_inspection_needed"]).lower(),
                "evidence_first_candidates": json.dumps(r["evidence_first_candidates"], ensure_ascii=False),
                "docling_candidates": json.dumps(r["docling_candidates"], ensure_ascii=False),
                "preview_paths": json.dumps(r["preview_paths"], ensure_ascii=False),
                "ground_truth_part_number": "",
                "ground_truth_reason_category": "",
                "adjudicator_notes": "",
            })

    def candidate_html(items):
        if not items:
            return '<span class="muted">No accepted/orderable candidates</span>'
        chunks = []
        for c in items:
            details = [
                f"page {html.escape(str(c.get('page') or '?'))}",
                html.escape(str(c.get("role") or "")),
                f"score {html.escape(str(c.get('score') or ''))}",
                "accepted" if c.get("accepted") else "ranked",
            ]
            chunks.append(
                f"<div class='candidate'><b>{html.escape(str(c.get('value') or ''))}</b> <span class='meta'>({' · '.join(details)})</span>"
                f"<div class='evidence'>{html.escape(str(c.get('supporting_text') or ''))}</div></div>"
            )
        return "".join(chunks)

    rows = []
    for r in records:
        current = r["current_part_numbers"]
        previews = "<br>".join(
            f"<a href='file:///{html.escape(p.replace(chr(92), '/'))}'>{html.escape(p)}</a>" for p in r["preview_paths"] if p
        ) or '<span class="muted">none</span>'
        suggestion = r["suggested_answer"]
        rows.append(
            "<tr>"
            f"<td><b>{html.escape(r['filename'])}</b><br><span class='meta'>{html.escape(r['run_id'])}</span></td>"
            f"<td>{html.escape(str(current['tesseract']['value'] or ''))}<br><span class='meta'>p{html.escape(str(current['tesseract']['page'] or ''))}</span></td>"
            f"<td>{html.escape(str(current['native_text']['value'] or ''))}<br><span class='meta'>p{html.escape(str(current['native_text']['page'] or ''))}</span></td>"
            f"<td>{html.escape(str(current['vision']['value'] or ''))}<br><span class='meta'>p{html.escape(str(current['vision']['page'] or ''))}</span></td>"
            f"<td>{candidate_html(r['evidence_first_candidates'])}</td>"
            f"<td>{candidate_html(r['docling_candidates'])}</td>"
            f"<td>{previews}</td>"
            f"<td class='suggestion'>{html.escape(str((suggestion or {}).get('value') or ''))}<br><span class='meta'>{html.escape(str((suggestion or {}).get('basis') or ''))}</span></td>"
            f"<td>{html.escape(r['suggested_reason_category'])}<br><span class='meta'>{'manual inspection' if r['manual_inspection_needed'] else 'strong suggestion'}</span></td>"
            "<td class='blank'></td><td class='blank'></td><td class='blank'></td>"
            "</tr>"
        )
    html_path = OUTPUT / "adjudication_pack.html"
    html_path.write_text(
        "<!doctype html><html><head><meta charset='utf-8'><title>45-case part-number adjudication pack</title>"
        "<style>body{font:13px system-ui,Segoe UI,sans-serif;margin:20px;color:#18212b}h1{font-size:22px}p{max-width:1100px}.summary{background:#f3f6f9;padding:10px;border:1px solid #d8e0e8;margin:12px 0}table{border-collapse:collapse;width:100%;table-layout:fixed}th,td{border:1px solid #cfd8e3;padding:7px;vertical-align:top;overflow-wrap:anywhere}th{position:sticky;top:0;background:#e8eef5;z-index:1}th:nth-child(1){width:9%}th:nth-child(2),th:nth-child(3),th:nth-child(4){width:7%}th:nth-child(5),th:nth-child(6){width:18%}th:nth-child(7){width:10%}th:nth-child(8){width:9%}th:nth-child(9){width:8%}th:nth-child(n+10){width:7%}.candidate{margin-bottom:8px}.candidate:last-child{margin-bottom:0}.meta,.muted{color:#5d6b78;font-size:11px}.evidence{margin-top:3px;background:#fafbfd;padding:4px;border-left:3px solid #b9c7d6}.suggestion{background:#eef8ee}.blank{min-height:40px;background:#fffdf2}.legend{font-size:12px}</style></head><body>"
        "<h1>Part-number adjudication pack</h1>"
        "<p>Review the 45 human-review PDFs. Suggested answers are separate from the blank human ground-truth columns. Filename text was not used as evidence.</p>"
        f"<div class='summary'><b>45 cases</b> · strong suggestions: {sum(bool(r['suggested_answer']) for r in records)} · genuinely ambiguous/no accepted orderable candidate: {sum(r['suggested_reason_category']=='ambiguous' for r in records)} · manual inspection: {sum(r['manual_inspection_needed'] for r in records)}</div>"
        "<p class='legend'>Candidate evidence includes page, role, score, and exact extracted supporting text/context. Preview links point to the existing rendered page PNGs.</p>"
        "<table><thead><tr><th>Filename</th><th>Tesseract part_number</th><th>Native-text part_number</th><th>Vision part_number</th><th>Evidence-first candidates</th><th>Docling candidates</th><th>Preview paths</th><th>Suggested answer</th><th>Suggested category</th><th>Human ground truth part_number</th><th>Human reason category</th><th>Adjudicator notes</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></body></html>\n",
        encoding="utf-8",
    )

    counts = Counter()
    counts["strong_suggestions"] = sum(bool(r["suggested_answer"]) for r in records)
    counts["genuinely_ambiguous"] = sum(r["suggested_reason_category"] == "ambiguous" for r in records)
    counts["manual_inspection_needed"] = sum(r["manual_inspection_needed"] for r in records)
    (OUTPUT / "summary.json").write_text(json.dumps(dict(counts), indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), "records": len(records), **counts}, indent=2))


if __name__ == "__main__":
    main()
