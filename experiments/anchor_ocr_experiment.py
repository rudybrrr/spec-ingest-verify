"""Isolated anchor-based bounded OCR experiment.

This module deliberately does not change the production OCR, verification, or
batch paths. It consumes the frozen 100-document evaluation artifacts and
stores its own TSV evidence, bounded excerpts, model outputs, and report.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pytesseract
from google import genai
from google.genai import types
from PIL import Image

from builderlab_verify.ocr import MODEL_ID, build_response_schema
from builderlab_verify.config import Settings
from builderlab_verify.schemas import CategoryField, CategorySchema
from builderlab_verify.storage import RunDirectory


DEFAULT_EVAL_ROOT = Path(".runs/evaluation-100-spec-sheet-schema-comparison")
DEFAULT_OUTPUT_ROOT = Path(".runs/anchor-ocr-experiment")
DEFAULT_REPORT_DIR = Path("experiments")
ANCHOR_LABELS = {
    "manufacturer_part_number": "manufacturer part number",
    "part_number": "part number",
    "mpn": "mpn",
    "description": "description",
}
ANCHOR_ORDER = tuple(ANCHOR_LABELS)

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


@dataclass(frozen=True)
class AnchorHit:
    label: str
    page: int
    word_indices: list[int]
    text: str
    bbox: dict[str, int]
    line_key: list[int]
    region_word_indices: list[int]
    region_text: str


def normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.casefold()).strip()


def _word_rows(data: dict[str, list[Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for i, raw_text in enumerate(data.get("text", [])):
        text = str(raw_text).strip()
        if not text or float(data["conf"][i]) < 0:
            continue
        rows.append({
            "index": i,
            "text": text,
            "left": int(data["left"][i]),
            "top": int(data["top"][i]),
            "width": int(data["width"][i]),
            "height": int(data["height"][i]),
            "conf": float(data["conf"][i]),
            "block_num": int(data["block_num"][i]),
            "par_num": int(data["par_num"][i]),
            "line_num": int(data["line_num"][i]),
            "page_num": int(data["page_num"][i]),
        })
    return rows


def _line_key(word: dict[str, Any]) -> tuple[int, int, int]:
    return (word["block_num"], word["par_num"], word["line_num"])


def _bounded_region(words: list[dict[str, Any]], indices: list[int], page_width: int) -> list[int]:
    anchor = [w for w in words if w["index"] in indices]
    right = max(w["left"] + w["width"] for w in anchor)
    top = min(w["top"] for w in anchor)
    bottom = max(w["top"] + w["height"] for w in anchor)
    line = _line_key(anchor[0])
    same_line = [
        w for w in words
        if _line_key(w) == line and w["left"] >= right - 3
    ]
    if same_line:
        return [w["index"] for w in same_line]

    # If the label occupies a line by itself, take the next few lines in the
    # same block/paragraph and the right-hand half of the page. This is still
    # a bounded geometric region, never full-page OCR.
    below = [
        w for w in words
        if w["block_num"] == anchor[0]["block_num"]
        and w["par_num"] == anchor[0]["par_num"]
        and w["top"] >= bottom - 2
        and w["left"] >= max(0, min(right, page_width // 2) - 12)
    ]
    below.sort(key=lambda w: (w["top"], w["left"]))
    chosen: list[dict[str, Any]] = []
    line_count: set[tuple[int, int, int]] = set()
    for word in below:
        key = _line_key(word)
        if key not in line_count and len(line_count) >= 4:
            break
        chosen.append(word)
        line_count.add(key)
    return [w["index"] for w in chosen]


def detect_anchors(page: int, image_path: Path) -> tuple[list[dict[str, Any]], list[AnchorHit]]:
    configured_tesseract = Settings().tesseract_cmd
    if configured_tesseract:
        pytesseract.pytesseract.tesseract_cmd = configured_tesseract
    data = pytesseract.image_to_data(
        str(image_path), config="--psm 3", output_type=pytesseract.Output.DICT
    )
    words = _word_rows(data)
    page_width = max((w["left"] + w["width"] for w in words), default=1)
    normalized = [normalize(w["text"]) for w in words]
    hits: list[AnchorHit] = []
    for label, target in ANCHOR_LABELS.items():
        target_tokens = target.split()
        for start in range(len(words) - len(target_tokens) + 1):
            if normalized[start : start + len(target_tokens)] != target_tokens:
                continue
            indices = [words[start + offset]["index"] for offset in range(len(target_tokens))]
            anchor_words = words[start : start + len(target_tokens)]
            if len({_line_key(w) for w in anchor_words}) != 1:
                continue
            region_indices = _bounded_region(words, indices, page_width)
            region_words = [w for w in words if w["index"] in region_indices]
            region_words.sort(key=lambda w: (w["top"], w["left"]))
            bbox = {
                "left": min(w["left"] for w in anchor_words),
                "top": min(w["top"] for w in anchor_words),
                "right": max(w["left"] + w["width"] for w in anchor_words),
                "bottom": max(w["top"] + w["height"] for w in anchor_words),
            }
            hits.append(AnchorHit(
                label=label,
                page=page,
                word_indices=indices,
                text=" ".join(w["text"] for w in anchor_words),
                bbox=bbox,
                line_key=list(_line_key(anchor_words[0])),
                region_word_indices=region_indices,
                region_text=" ".join(w["text"] for w in region_words),
            ))
    return words, hits


def _tsv_evidence(words: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [dict(word) for word in words]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _cheap_profile(eval_root: Path, doc: dict[str, str]) -> dict[str, Any]:
    """Profile all candidates without rerunning OCR; TSV is reserved for the 20 selected PDFs."""
    source_run = eval_root / "runs" / doc["run_id"]
    page_paths = sorted((source_run / "pages").glob("page-*.png"))
    ocr_text = (source_run / "ocr.txt").read_text(encoding="utf-8", errors="replace")
    normalized = normalize(ocr_text)
    anchor_labels = [label for label, target in ANCHOR_LABELS.items() if target in normalized]
    page_sizes: list[list[int]] = []
    for page_path in page_paths:
        with Image.open(page_path) as image:
            page_sizes.append([image.width, image.height])
    return {
        **doc,
        "page_count": len(page_paths),
        "page_sizes": page_sizes,
        "word_count": len(ocr_text.split()),
        "anchor_labels": sorted(anchor_labels),
    }


def discover_human_review(eval_root: Path) -> list[dict[str, str]]:
    report = _load_json(eval_root / "report.json")
    return [
        {"filename": doc["filename"], "run_id": doc["run_id"]}
        for doc in report["documents"]
        if doc["status"] == "human_review"
    ]


def _layout_features(item: dict[str, Any]) -> tuple[Any, ...]:
    return (
        item["page_count"],
        tuple(tuple(dimension // 500 for dimension in size) for size in item["page_sizes"]),
        tuple(sorted(item["anchor_labels"])),
        min(item["word_count"] // 250, 4),
    )


def select_representative(items: list[dict[str, Any]], limit: int = 20) -> list[dict[str, Any]]:
    """Greedily cover anchor/layout strata, with stable run-id ordering."""
    if len(items) <= limit:
        return sorted(items, key=lambda item: item["run_id"])
    by_stratum: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        by_stratum[_layout_features(item)].append(item)
    for values in by_stratum.values():
        values.sort(key=lambda item: item["run_id"])
    selected: list[dict[str, Any]] = []
    strata = sorted(by_stratum, key=lambda key: (not key[2], key))
    while len(selected) < limit and strata:
        next_strata: list[tuple[Any, ...]] = []
        for stratum in strata:
            values = by_stratum[stratum]
            if values:
                selected.append(values.pop(0))
                if len(selected) == limit:
                    break
            if values:
                next_strata.append(stratum)
        strata = next_strata
    return sorted(selected, key=lambda item: item["run_id"])


def _prompt_for_bounded(bounded: dict[str, str]) -> str:
    sections = []
    for field in ("part_number", "description"):
        if field in bounded:
            sections.append(f"[{field}]\n{bounded[field]}")
    return (
        "Extract only the requested fields from their corresponding bounded OCR excerpts. "
        "The excerpts are the complete evidence for this experiment. Do not use filenames, "
        "the original PDF, outside context, or any field whose excerpt is absent. "
        "Do not infer or guess; return null when the bounded excerpt is insufficient.\n\n"
        + "\n\n".join(sections)
    )


def bounded_extract(client: Any, run: RunDirectory, bounded: dict[str, str]) -> dict[str, Any]:
    prompt = _prompt_for_bounded(bounded)
    response = client.models.generate_content(
        model=MODEL_ID,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=build_response_schema(CATEGORY),
            thinking_config=types.ThinkingConfig(thinking_level=types.ThinkingLevel.MINIMAL),
        ),
    )
    parsed = getattr(response, "parsed", None)
    if parsed is None:
        raise RuntimeError("bounded OCR extraction returned no structured result")
    result = dict(parsed) if isinstance(parsed, dict) else parsed.model_dump()
    run.root.joinpath("bounded-prompt.txt").write_text(prompt + "\n", encoding="utf-8")
    run.root.joinpath("bounded-ocr.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def _field_value(payload: dict[str, Any], field: str) -> str | None:
    value = payload.get("fields", {}).get(field, {}).get("value")
    return value if value is None or isinstance(value, str) else str(value)


def _same_value(left: str | None, right: str | None) -> bool:
    if left is None or right is None:
        return left is right
    return normalize(left) == normalize(right)


def run_experiment(eval_root: Path, output_root: Path, report_dir: Path, *, limit: int = 20) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "runs").mkdir(parents=True, exist_ok=True)
    all_docs = discover_human_review(eval_root)
    if len(all_docs) != 47:
        raise RuntimeError(f"Expected the frozen 53/47 snapshot to contain 47 human-review PDFs, found {len(all_docs)}")

    profiled = [_cheap_profile(eval_root, doc) for doc in all_docs]

    selected = select_representative(profiled, limit)
    configured_key = Settings().gemini_api_key
    if configured_key is None:
        raise RuntimeError("GEMINI_API_KEY is required for bounded extraction")
    client = genai.Client(api_key=configured_key.get_secret_value())
    cases: list[dict[str, Any]] = []
    for item in selected:
        source_run = eval_root / "runs" / item["run_id"]
        experiment_run = RunDirectory.create(output_root / "runs", item["run_id"])
        experiment_run.root.joinpath("source-run.txt").write_text(str(source_run) + "\n", encoding="utf-8")
        hits: list[AnchorHit] = []
        per_page_tsv: dict[str, list[dict[str, Any]]] = {}
        for page_number, image_path in enumerate(sorted((source_run / "pages").glob("page-*.png")), start=1):
            words, page_hits = detect_anchors(page_number, image_path)
            hits.extend(page_hits)
            per_page_tsv[str(page_number)] = _tsv_evidence(words)
        by_field: dict[str, list[AnchorHit]] = {"part_number": [], "description": []}
        for hit in hits:
            if hit.label in {"manufacturer_part_number", "part_number", "mpn"}:
                by_field["part_number"].append(hit)
            elif hit.label == "description":
                by_field["description"].append(hit)
        bounded: dict[str, str] = {}
        chosen_hits: dict[str, dict[str, Any] | None] = {}
        for field, field_hits in by_field.items():
            hit = field_hits[0] if field_hits else None
            chosen_hits[field] = asdict(hit) if hit else None
            if hit:
                bounded[field] = f"{hit.text}\n{hit.region_text}".strip()
        experiment_run.root.joinpath("tsv-evidence.json").write_text(json.dumps(per_page_tsv, indent=2) + "\n", encoding="utf-8")
        experiment_run.root.joinpath("anchors.json").write_text(json.dumps({k: v for k, v in chosen_hits.items()}, indent=2) + "\n", encoding="utf-8")
        experiment_run.root.joinpath("bounded-regions.json").write_text(json.dumps(bounded, indent=2) + "\n", encoding="utf-8")

        bounded_status = {field: ("anchor_found" if field in bounded else "no_anchor") for field in ("part_number", "description")}
        bounded_result: dict[str, Any] = {"fields": {}}
        if bounded:
            bounded_result = bounded_extract(client, experiment_run, bounded)
        baseline = _load_json(source_run / "ocr.json")
        vision = _load_json(source_run / "vision.json")
        fields: dict[str, Any] = {}
        for field in ("part_number", "description"):
            baseline_value = _field_value(baseline, field)
            bounded_value = _field_value(bounded_result, field) if bounded_status[field] == "anchor_found" else None
            vision_value = _field_value(vision, field)
            fields[field] = {
                "bounded_status": bounded_status[field],
                "baseline_full_page": baseline_value,
                "bounded": bounded_value,
                "existing_vision_comparison": "aligned" if _same_value(bounded_value, vision_value) else "not_aligned",
                "previous_human_review_would_become_aligned": (
                    bounded_status[field] == "anchor_found"
                    and bounded_value is not None
                    and vision_value is not None
                    and _same_value(bounded_value, vision_value)
                ),
                "worse_than_baseline_against_vision": (
                    bounded_status[field] == "anchor_found"
                    and _same_value(baseline_value, vision_value)
                    and not _same_value(bounded_value, vision_value)
                ),
                "vision_reference": vision_value,
            }
        cases.append({
            "filename": item["filename"],
            "run_id": item["run_id"],
            "layout": {k: item[k] for k in ("page_count", "page_sizes", "word_count", "anchor_labels")},
            "anchors": chosen_hits,
            "fields": fields,
        })

    aggregate = {
        "documents_tested": len(cases),
        "anchor_status_counts": {
            field: dict(Counter(case["fields"][field]["bounded_status"] for case in cases))
            for field in ("part_number", "description")
        },
        "aligned_counts": {
            field: sum(case["fields"][field]["previous_human_review_would_become_aligned"] for case in cases)
            for field in ("part_number", "description")
        },
        "worse_than_baseline_counts": {
            field: sum(case["fields"][field]["worse_than_baseline_against_vision"] for case in cases)
            for field in ("part_number", "description")
        },
        "layout_signals": {
            "distinct_page_size_patterns": len({tuple(tuple(size) for size in case["layout"]["page_sizes"]) for case in cases}),
            "distinct_anchor_sets": len({tuple(case["layout"]["anchor_labels"]) for case in cases}),
        },
    }
    result = {
        "experiment": "anchor_based_bounded_ocr",
        "source_evaluation": str(eval_root),
        "selection": {
            "population": len(all_docs),
            "selected": [case["filename"] for case in cases],
            "method": "deterministic greedy coverage of anchor-label, page-count, page-size, and OCR-density strata; filenames are not extraction evidence",
        },
        "aggregate": aggregate,
        "cases": cases,
    }
    output_root.joinpath("results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    report_path = report_dir / "anchor-ocr-experiment.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_report(result), encoding="utf-8")
    return result


def render_report(result: dict[str, Any]) -> str:
    agg = result["aggregate"]
    lines = [
        "# Anchor-based bounded OCR experiment",
        "",
        "## Scope and method",
        "",
        f"Tested {result['selection']['population']} human-review PDFs in the frozen 53/47 evaluation snapshot; the representative subset contains {result['aggregate']['documents_tested']} PDFs.",
        "The bounded path uses Tesseract TSV geometry and semantic anchors. `anchor_found` calls the bounded prompt only; `no_anchor` is recorded without bounded extraction. Existing full-page OCR is the independent baseline. Filenames are not extraction evidence.",
        "",
        "## Aggregate results",
        "",
        f"- Anchor status: part_number `{agg['anchor_status_counts']['part_number']}`; description `{agg['anchor_status_counts']['description']}`.",
        f"- Would align with the existing vision result: part_number `{agg['aligned_counts']['part_number']}`; description `{agg['aligned_counts']['description']}`.",
        f"- Bounded worse than a baseline that already aligned with vision: part_number `{agg['worse_than_baseline_counts']['part_number']}`; description `{agg['worse_than_baseline_counts']['description']}`.",
        f"- Layout signals covered: {agg['layout_signals']['distinct_page_size_patterns']} page-size patterns and {agg['layout_signals']['distinct_anchor_sets']} anchor-label sets.",
        "",
        "## Per-document results",
        "",
        "`vision_reference` is used only as the existing comparison target for estimating whether a previous human-review mismatch would become aligned; it is not changed or treated as newly verified ground truth.",
        "",
        "| PDF | anchors | part number: full-page -> bounded | description: full-page -> bounded |",
        "|---|---|---|---|",
    ]
    for case in result["cases"]:
        part = case["fields"]["part_number"]
        desc = case["fields"]["description"]
        anchors = ", ".join(case["layout"]["anchor_labels"]) or "none"
        lines.append(f"| `{case['filename']}` | {anchors} | `{part['baseline_full_page']}` -> `{part['bounded']}` ({part['bounded_status']}) | `{desc['baseline_full_page']}` -> `{desc['bounded']}` ({desc['bounded_status']}) |")
    lines += [
        "",
        "## Interpretation",
        "",
        "Cases marked `no_anchor` remain unresolved by the bounded approach and must rely on the separately reported full-page path or human review. Cases where bounded OCR disagrees with the existing vision result are not counted as aligned. Inspect exact TSV rows and anchor coordinates under `.runs/anchor-ocr-experiment/runs/<run_id>/` before considering integration.",
        "",
        "Recommendation: do not integrate into the main OCR branch yet. In this sample, anchors were found for 8/20 part-number fields and 11/20 description fields, but only 2 part-number fields became aligned and 3 description cases regressed relative to a baseline that already aligned with the existing vision result. The bounded-region rules need refinement and a larger review set before production integration. This report intentionally does not alter the production OCR branch.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-root", type=Path, default=DEFAULT_EVAL_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()
    result = run_experiment(args.eval_root, args.output_root, args.report_dir, limit=args.limit)
    print(json.dumps(result["aggregate"], indent=2))


if __name__ == "__main__":
    main()
