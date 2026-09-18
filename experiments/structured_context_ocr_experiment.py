"""Isolated structured-neighborhood OCR experiment for ``part_number``.

This module is deliberately separate from the production OCR, Vision, and
verification paths. Vision and the prior bounded experiment are read only
after OCR-side extraction, for comparison labels.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pytesseract
from google import genai
from google.genai import types
from PIL import Image

from builderlab_verify.config import Settings
from builderlab_verify.ocr import MODEL_ID, build_response_schema
from builderlab_verify.schemas import CategoryField, CategorySchema
from experiments.anchor_ocr_experiment import _cheap_profile, discover_human_review, select_representative


DEFAULT_EVAL_ROOT = Path(".runs/evaluation-100-spec-sheet-schema-comparison")
DEFAULT_OUTPUT_ROOT = Path(".runs/structured-context-ocr-experiment")
DEFAULT_REPORT = Path("experiments/structured-context-ocr-experiment.md")

CATEGORY = CategorySchema(
    name="spec_sheet",
    fields=[CategoryField(name="part_number", description=(
        "Unique manufacturer/orderable part identifier for the specific product represented by the document. "
        "Do not use product-family names, document titles, drawing labels, stock labels, or arbitrary variant identifiers. "
        "Return null for multi-product/portfolio documents or when no single unique part number is clearly identifiable."
    ), required=True)],
)
ANCHOR_TARGETS = {
    "manufacturer_part_number": ("manufacturer", "part", "number"),
    "part_number": ("part", "number"),
    "mpn": ("mpn",),
}


@dataclass(frozen=True)
class StructuredAnchor:
    label: str
    page: int
    word_indices: list[int]
    text: str
    bbox: dict[str, int]
    line_key: list[int]
    neighborhood_word_indices: list[int]
    neighborhood: list[dict[str, Any]]


def normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.casefold()).strip()


def word_rows(data: dict[str, list[Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for i, raw_text in enumerate(data.get("text", [])):
        text = str(raw_text).strip()
        if not text or float(data["conf"][i]) < 0:
            continue
        rows.append({
            "index": i, "text": text, "left": int(data["left"][i]),
            "top": int(data["top"][i]), "width": int(data["width"][i]),
            "height": int(data["height"][i]), "conf": float(data["conf"][i]),
            "block_num": int(data["block_num"][i]), "par_num": int(data["par_num"][i]),
            "line_num": int(data["line_num"][i]), "page_num": int(data["page_num"][i]),
        })
    return rows


def line_key(word: dict[str, Any]) -> tuple[int, int, int]:
    return (word["block_num"], word["par_num"], word["line_num"])


def _line_groups(words: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    groups: dict[tuple[int, int, int], list[dict[str, Any]]] = {}
    for word in words:
        groups.setdefault(line_key(word), []).append(word)
    return [sorted(group, key=lambda w: w["left"]) for group in sorted(groups.values(), key=lambda g: (min(w["top"] for w in g), min(w["left"] for w in g)))]


def structured_neighborhood(words: list[dict[str, Any]], anchor_indices: list[int], nearby_rows: int = 2) -> list[dict[str, Any]]:
    """Return anchor row ± nearby rows, retaining every nearby column and TSV geometry."""
    groups = _line_groups(words)
    anchor_keys = {line_key(word) for word in words if word["index"] in anchor_indices}
    positions = [i for i, group in enumerate(groups) if line_key(group[0]) in anchor_keys]
    if not positions:
        return []
    lo = max(0, min(positions) - nearby_rows)
    hi = min(len(groups), max(positions) + nearby_rows + 1)
    selected = [word for group in groups[lo:hi] for word in group]
    return [dict(word, row_offset=(i - positions[0])) for i, word in enumerate(selected)]


def detect_part_number_anchors(page: int, image_path: Path) -> tuple[list[dict[str, Any]], list[StructuredAnchor]]:
    configured = Settings().tesseract_cmd
    if configured:
        pytesseract.pytesseract.tesseract_cmd = configured
    data = pytesseract.image_to_data(str(image_path), config="--psm 3", output_type=pytesseract.Output.DICT)
    words = word_rows(data)
    normalized = [normalize(word["text"]) for word in words]
    hits: list[StructuredAnchor] = []
    for label, target in ANCHOR_TARGETS.items():
        for start in range(len(words) - len(target) + 1):
            if tuple(normalized[start:start + len(target)]) != target:
                continue
            anchor_words = words[start:start + len(target)]
            if len({line_key(word) for word in anchor_words}) != 1:
                continue
            indices = [word["index"] for word in anchor_words]
            neighborhood = structured_neighborhood(words, indices)
            hits.append(StructuredAnchor(
                label=label, page=page, word_indices=indices,
                text=" ".join(word["text"] for word in anchor_words),
                bbox={"left": min(w["left"] for w in anchor_words), "top": min(w["top"] for w in anchor_words),
                      "right": max(w["left"] + w["width"] for w in anchor_words), "bottom": max(w["top"] + w["height"] for w in anchor_words)},
                line_key=list(line_key(anchor_words[0])),
                neighborhood_word_indices=[word["index"] for word in neighborhood],
                neighborhood=neighborhood,
            ))
    return words, hits


def _field_value(payload: dict[str, Any], field: str = "part_number") -> str | None:
    value = payload.get("fields", {}).get(field, {}).get("value")
    return value if value is None or isinstance(value, str) else str(value)


def _same(left: str | None, right: str | None) -> bool:
    return left is None and right is None or left is not None and right is not None and normalize(left) == normalize(right)


def _aligned(value: str | None, reference: str | None) -> bool:
    """Only a non-null frozen reference can establish an alignment."""
    return reference is not None and value is not None and _same(value, reference)


def aggregate_metrics(cases: list[dict[str, Any]]) -> dict[str, Any]:
    """Canonical aggregation shared by JSON persistence, Markdown, and CLI output."""
    metric_names = (
        "improvement_over_full_page_ocr", "improvement_over_prior_bounded",
        "regression_vs_full_page_ocr", "regression_vs_prior_bounded", "still_ambiguous",
    )
    return {
        **{name: sum(bool(case["comparison"].get(name, False)) for case in cases) for name in metric_names},
        "anchor_status": dict(Counter(case["comparison"].get("status") for case in cases)),
    }


def _prompt(anchor: StructuredAnchor) -> str:
    rows: dict[int, list[dict[str, Any]]] = {}
    for word in anchor.neighborhood:
        rows.setdefault(int(word["row_offset"]), []).append(word)
    rendered = []
    for offset in sorted(rows):
        rendered.append("ROW offset=" + str(offset) + "\n" + " ".join(
            f"{word['text']} [x={word['left']},y={word['top']},w={word['width']},h={word['height']},conf={word['conf']:.1f}]"
            for word in sorted(rows[offset], key=lambda item: item["left"])))
    return ("Extract only the part_number field from the structured OCR neighborhood below. "
            "Use the text, row offsets, columns, and coordinates as evidence. Do not use filenames, "
            "the original PDF, external outputs, or outside context. Do not infer or guess; return null "
            "when no single unique part number is clearly identifiable.\n\nANCHOR=" + anchor.text + "\n" + "\n".join(rendered))


def structured_extract(client: Any, anchor: StructuredAnchor) -> dict[str, Any]:
    response = client.models.generate_content(model=MODEL_ID, contents=_prompt(anchor), config=types.GenerateContentConfig(
        response_mime_type="application/json", response_schema=build_response_schema(CATEGORY),
        thinking_config=types.ThinkingConfig(thinking_level=types.ThinkingLevel.MINIMAL)))
    parsed = getattr(response, "parsed", None)
    if parsed is None:
        raise RuntimeError("structured OCR extraction returned no structured result")
    return dict(parsed) if isinstance(parsed, dict) else parsed.model_dump()


def _prior_bounded_values(report_path: Path = Path("experiments/anchor-ocr-experiment.md")) -> dict[str, str | None]:
    """Read the prior run's report as historical comparison data, never prompt input."""
    values: dict[str, str | None] = {}
    if not report_path.exists():
        return values
    for line in report_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.startswith("| `") or " -> " not in line:
            continue
        columns = line.split("|")
        if len(columns) < 4:
            continue
        filename = columns[1].strip().strip("`")
        match = re.search(r"`([^`]*)`\s*->\s*`([^`]*)`", columns[3])
        if match:
            values[filename] = match.group(2) or None
    return values


def run_experiment(eval_root: Path = DEFAULT_EVAL_ROOT, output_root: Path = DEFAULT_OUTPUT_ROOT, report_path: Path = DEFAULT_REPORT, *, limit: int = 20) -> dict[str, Any]:
    docs = discover_human_review(eval_root)
    if len(docs) != 47:
        raise RuntimeError(f"Expected 47 human-review PDFs, found {len(docs)}")
    profiled = [_cheap_profile(eval_root, doc) for doc in docs]
    selected = select_representative(profiled, limit)
    key = Settings().gemini_api_key
    if key is None:
        raise RuntimeError("GEMINI_API_KEY is required for structured OCR extraction")
    client = genai.Client(api_key=key.get_secret_value(), http_options=types.HttpOptions(timeout=60000))
    output_root.joinpath("runs").mkdir(parents=True, exist_ok=True)
    prior_values = _prior_bounded_values()
    cases: list[dict[str, Any]] = []
    for doc in selected:
        source = eval_root / "runs" / doc["run_id"]
        out = output_root / "runs" / doc["run_id"]
        out.mkdir(parents=True, exist_ok=True)
        all_words: dict[str, list[dict[str, Any]]] = {}
        hits: list[StructuredAnchor] = []
        cached_tsv = out / "tsv-evidence.json"
        cached_anchors = out / "anchors.json"
        if cached_tsv.exists() and cached_anchors.exists():
            all_words = json.loads(cached_tsv.read_text(encoding="utf-8"))
            hits = [StructuredAnchor(**item) for item in json.loads(cached_anchors.read_text(encoding="utf-8"))]
        else:
            for page, image_path in enumerate(sorted((source / "pages").glob("page-*.png")), 1):
                words, page_hits = detect_part_number_anchors(page, image_path)
                all_words[str(page)] = words
                hits.extend(page_hits)
        # Choose the first semantic hit deterministically; duplicate hits are retained as ambiguity evidence.
        chosen = hits[0] if hits else None
        prompt = _prompt(chosen) if chosen else None
        structured_path = out / "structured-ocr.json"
        if structured_path.exists():
            structured = json.loads(structured_path.read_text(encoding="utf-8"))
        else:
            structured = structured_extract(client, chosen) if chosen else {"fields": {"part_number": {"value": None}}}
        baseline = json.loads((source / "ocr.json").read_text(encoding="utf-8"))
        vision = json.loads((source / "vision.json").read_text(encoding="utf-8"))
        full = _field_value(baseline)
        value = _field_value(structured)
        vision_value = _field_value(vision)
        prior = prior_values.get(doc["filename"])
        status = "no_anchor" if not hits else "anchor_found"
        comparison = {
            "full_page_ocr": full, "structured_context": value,
            "prior_bounded": prior, "vision_reference": vision_value,
            "status": status,
            "improvement_over_full_page_ocr": status == "anchor_found" and _aligned(value, vision_value) and not _aligned(full, vision_value),
            "improvement_over_prior_bounded": status == "anchor_found" and _aligned(value, vision_value) and not _aligned(prior, vision_value),
            "regression_vs_full_page_ocr": status == "anchor_found" and _aligned(full, vision_value) and not _aligned(value, vision_value),
            "regression_vs_prior_bounded": status == "anchor_found" and _aligned(prior, vision_value) and not _aligned(value, vision_value),
            "still_ambiguous": status == "anchor_found" and value is None,
        }
        out.joinpath("tsv-evidence.json").write_text(json.dumps(all_words, indent=2) + "\n", encoding="utf-8")
        out.joinpath("anchors.json").write_text(json.dumps([asdict(hit) for hit in hits], indent=2) + "\n", encoding="utf-8")
        out.joinpath("structured-context.json").write_text(json.dumps([asdict(hit) for hit in hits], indent=2) + "\n", encoding="utf-8")
        out.joinpath("structured-prompt.txt").write_text((prompt or "NO_ANCHOR\n") + "\n", encoding="utf-8")
        structured_path.write_text(json.dumps(structured, indent=2) + "\n", encoding="utf-8")
        cases.append({"filename": doc["filename"], "run_id": doc["run_id"], "anchors": [asdict(hit) for hit in hits], "comparison": comparison})
    aggregate = aggregate_metrics(cases)
    result = {"experiment": "structured_context_part_number_ocr", "documents_tested": len(cases), "selected": [case["filename"] for case in cases], "aggregate": aggregate, "cases": cases}
    output_root.joinpath("results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    report_path.write_text(render_report(result), encoding="utf-8")
    return result


def reconcile_existing(results_path: Path = DEFAULT_OUTPUT_ROOT / "results.json", report_path: Path = DEFAULT_REPORT) -> dict[str, Any]:
    """Rebuild aggregate/report from recorded cases without OCR or Gemini calls."""
    result = json.loads(results_path.read_text(encoding="utf-8"))
    result["aggregate"] = aggregate_metrics(result["cases"])
    results_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    report_path.write_text(render_report(result), encoding="utf-8")
    return result


def render_report(result: dict[str, Any]) -> str:
    agg = result["aggregate"]
    lines = ["# Structured local-context OCR experiment", "", "OCR-only experiment for `part_number`; the frozen Vision result is reference-only and is consulted only for comparison after extraction. The prior bounded result is also comparison-only.", "", "## Aggregate", "", f"- PDFs tested: {result['documents_tested']}", f"- Anchors: `{agg['anchor_status']}`", f"- Improvement over full-page OCR: {agg['improvement_over_full_page_ocr']}", f"- Improvement over prior bounded experiment: {agg['improvement_over_prior_bounded']}", f"- Regressions vs full-page OCR: {agg['regression_vs_full_page_ocr']}", f"- Regressions vs prior bounded experiment: {agg['regression_vs_prior_bounded']}", f"- Still ambiguous: {agg['still_ambiguous']}", "", "## Per-PDF results", "", "| PDF | anchors | full-page OCR | structured context | prior bounded | status |", "|---|---|---|---|---|---|"]
    for case in result["cases"]:
        c = case["comparison"]
        anchors = ", ".join(hit["label"] + f" (p{hit['page']})" for hit in case["anchors"]) or "no_anchor"
        lines.append(f"| `{case['filename']}` | {anchors} | `{c['full_page_ocr']}` | `{c['structured_context']}` | `{c['prior_bounded']}` | {c['status']} |")
    materially = agg["improvement_over_prior_bounded"] > agg["regression_vs_prior_bounded"]
    examples = [case for case in result["cases"] if case["comparison"]["status"] == "anchor_found"][:3]
    lines += ["", "## Representative examples", ""]
    for case in examples:
        c = case["comparison"]
        lines.append(f"- `{case['filename']}`: anchor(s) found; full-page `{c['full_page_ocr']}`, structured `{c['structured_context']}`, prior bounded `{c['prior_bounded']}`, reference `{c['vision_reference']}`.")
    lines += ["", "## Interpretation", "", f"On this cohort, structured local context {'materially outperforms' if materially else 'does not materially outperform'} simple bounded extraction by the reference-only comparison: {agg['improvement_over_prior_bounded']} improvement(s) versus {agg['regression_vs_prior_bounded']} regression(s). There are {agg['still_ambiguous']} genuinely ambiguous anchor-found cases and {agg['anchor_status'].get('no_anchor', 0)} explicit `no_anchor` cases. Recommendation: do not integrate based on this cohort. Inspect `tsv-evidence.json`, `anchors.json`, and `structured-prompt.txt` under the run artifacts for exact coordinate evidence. No production code or Vision behavior is changed by this experiment.", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-root", type=Path, default=DEFAULT_EVAL_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--reconcile-existing", action="store_true", help="rebuild metrics/report from cached results without OCR or Gemini")
    args = parser.parse_args()
    result = reconcile_existing(args.output_root / "results.json", args.report) if args.reconcile_existing else run_experiment(args.eval_root, args.output_root, args.report, limit=args.limit)
    print(json.dumps(result["aggregate"], indent=2))


if __name__ == "__main__":
    main()
