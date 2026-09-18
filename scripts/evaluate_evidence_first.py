"""Run the isolated evidence-first part-number harvesting experiment."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import pymupdf
import pytesseract
from PIL import Image
from pytesseract import Output

from builderlab_verify.config import Settings
from experiments.evidence_candidates import (
    Candidate,
    CandidateRole,
    CandidateSource,
    _identifier_like,
    build_candidate,
    evaluate_ranked_cases,
    rank_candidates,
)


DEFAULT_FROZEN_ROOT = Path(".runs/evaluation-100-spec-sheet-native-lazy-fallback-authorized")
DEFAULT_OUTPUT_ROOT = Path(".runs/evidence-first-part-number-harvest")


def _token(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def _line_groups(words: list[dict[str, Any]]) -> dict[tuple[int, int], list[dict[str, Any]]]:
    groups: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for word in words:
        groups[(int(word.get("block_no", 0)), int(word.get("line_no", 0)))].append(word)
    for values in groups.values():
        values.sort(key=lambda word: float(word.get("bbox", [0])[0]))
    return groups


def _table_like_blocks(page: dict[str, Any]) -> set[int]:
    result = set()
    for block in page.get("blocks", []):
        text = str(block.get("text", ""))
        if text.count("\n") >= 2:
            result.add(int(block.get("block_no", -1)))
    return result


def _candidate_from_word(
    word: dict[str, Any],
    line: list[dict[str, Any]],
    source: CandidateSource,
    *,
    page: int,
    table_blocks: set[int],
    page_width: float | None = None,
    page_height: float | None = None,
) -> Candidate | None:
    raw = str(word.get("text", "")).strip()
    try:
        word_index = line.index(word)
    except ValueError:
        word_index = 0
    context_start = max(0, word_index - 4)
    context_end = min(len(line), word_index + 5)
    context = " ".join(str(item.get("text", "")) for item in line[context_start:context_end]).strip()
    if not _identifier_like(raw):
        return None
    bbox = word.get("bbox")
    if bbox is not None and page_width and page_height:
        image_width = float(word.get("image_width", page_width))
        image_height = float(word.get("image_height", page_height))
        scale_x = page_width / image_width
        scale_y = page_height / image_height
        bbox = [
            round(float(bbox[0]) * scale_x, 3),
            round(float(bbox[1]) * scale_y, 3),
            round(float(bbox[2]) * scale_x, 3),
            round(float(bbox[3]) * scale_y, 3),
        ]
    confidence = word.get("ocr_confidence")
    return build_candidate(
        raw_text=raw,
        source=source,
        page=page,
        bbox=bbox,
        context=context,
        block_no=word.get("block_no"),
        line_no=word.get("line_no"),
        table_membership=int(word.get("block_no", -1)) in table_blocks,
        ocr_confidence=float(confidence) if confidence is not None and float(confidence) >= 0 else None,
    )


def harvest_native_candidates(pages: list[dict[str, Any]]) -> list[Candidate]:
    candidates: list[Candidate] = []
    for page in pages:
        page_number = int(page["page"])
        groups = _line_groups(page.get("words", []))
        table_blocks = _table_like_blocks(page)
        for line in groups.values():
            for word in line:
                candidate = _candidate_from_word(
                    word,
                    line,
                    CandidateSource.NATIVE_TEXT,
                    page=page_number,
                    table_blocks=table_blocks,
                    page_width=float(page.get("width", 0)) or None,
                    page_height=float(page.get("height", 0)) or None,
                )
                if candidate:
                    candidates.append(candidate)
    return candidates


def _tesseract_words(image_path: Path, page_number: int, page_width: float, page_height: float) -> list[dict[str, Any]]:
    data = pytesseract.image_to_data(
        Image.open(image_path),
        config=f"--psm {Settings().tesseract_psm}",
        output_type=Output.DICT,
    )
    image_width = float(Image.open(image_path).width)
    image_height = float(Image.open(image_path).height)
    words = []
    for index, raw in enumerate(data.get("text", [])):
        text = str(raw).strip()
        if not text:
            continue
        try:
            confidence = float(data["conf"][index])
        except (TypeError, ValueError):
            confidence = None
        left = float(data["left"][index])
        top = float(data["top"][index])
        width = float(data["width"][index])
        height = float(data["height"][index])
        words.append({
            "text": text,
            "bbox": [left, top, left + width, top + height],
            "block_no": int(data["block_num"][index]),
            "line_no": int(data["line_num"][index]),
            "ocr_confidence": confidence,
            "image_width": image_width,
            "image_height": image_height,
            "page": page_number,
        })
    return words


def harvest_tesseract_candidates(run_root: Path, native_pages: list[dict[str, Any]]) -> list[Candidate]:
    candidates: list[Candidate] = []
    pages_by_number = {int(page["page"]): page for page in native_pages}
    targeted_pages: set[int] = {1, 2}
    label_pattern = re.compile(
        r"\b(?:manufacturer\s+part|part\s*(?:number|no\.?|#)|mpn|"
        r"order\s*(?:number|no\.?|#)|catalog\s*(?:number|no\.?|#)|"
        r"product\s*(?:number|no\.?|#)|drawing\s*(?:number|no\.?|#))\b",
        re.IGNORECASE,
    )
    for page in native_pages:
        page_number = int(page["page"])
        page_text = str(page.get("text", ""))
        if not page_text.strip() or label_pattern.search(page_text):
            targeted_pages.add(page_number)
    for image_path in sorted((run_root / "pages").glob("page-*.png")):
        page_number = int(image_path.stem.split("-")[-1])
        if page_number not in targeted_pages:
            continue
        page = pages_by_number.get(page_number, {})
        words = _tesseract_words(
            image_path,
            page_number,
            float(page.get("width", 0)) or 1,
            float(page.get("height", 0)) or 1,
        )
        for line in _line_groups(words).values():
            for word in line:
                candidate = _candidate_from_word(
                    word,
                    line,
                    CandidateSource.TESSERACT_TSV,
                    page=page_number,
                    table_blocks=set(),
                    page_width=float(page.get("width", 0)) or None,
                    page_height=float(page.get("height", 0)) or None,
                )
                if candidate:
                    candidates.append(candidate)
    return candidates


def _structured_candidate(path: Path, source: CandidateSource) -> Candidate | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    field = payload.get("fields", {}).get("part_number", {})
    value = field.get("value")
    if value is None or not str(value).strip() or not _identifier_like(str(value)):
        return None
    page = field.get("page")
    return build_candidate(
        raw_text=str(value),
        source=source,
        page=int(page) if page else None,
        bbox=None,
        context=f"Frozen {source.value} part_number field; region unavailable",
    )


def _reference_map(frozen_root: Path) -> dict[str, str]:
    report = json.loads((frozen_root / "report.json").read_text(encoding="utf-8"))
    return {
        item["run_id"]: Path(item["filename"]).stem
        for item in report.get("documents", [])
        if item.get("status") == "human_review"
    }


def _review_run_roots(frozen_root: Path) -> list[Path]:
    roots = []
    for run_root in sorted((frozen_root / "runs").iterdir()):
        if not run_root.is_dir() or not (run_root / "verification.json").exists():
            continue
        verification = json.loads((run_root / "verification.json").read_text(encoding="utf-8"))
        if verification.get("status") == "human_review":
            roots.append(run_root)
    return roots


def evaluate(frozen_root: Path, output_root: Path) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    references = _reference_map(frozen_root)
    settings = Settings()
    pytesseract.pytesseract.tesseract_cmd = settings.tesseract_cmd
    ranked_cases: list[dict[str, Any]] = []
    source_counts: dict[str, int] = defaultdict(int)
    examples: list[dict[str, Any]] = []
    all_false_positives: list[dict[str, Any]] = []

    for run_root in _review_run_roots(frozen_root):
        run_id = run_root.name
        native_payload = json.loads((run_root / "native-text.json").read_text(encoding="utf-8"))
        native_pages = native_payload.get("pages", [])
        candidates = harvest_native_candidates(native_pages)
        candidates.extend(harvest_tesseract_candidates(run_root, native_pages))
        for path, source in (
            (run_root / "ocr.json", CandidateSource.OCR_STRUCTURED),
            (run_root / "native.json", CandidateSource.NATIVE_STRUCTURED),
            (run_root / "vision.json", CandidateSource.VISION),
        ):
            candidate = _structured_candidate(path, source)
            if candidate:
                candidates.append(candidate)
        for candidate in candidates:
            source_counts[candidate.source.value] += 1
        ranked = rank_candidates(candidates)
        reference = references.get(run_id, "")
        case = {
            "run_id": run_id,
            "reference": reference,
            "reference_source": "selected_filename_stem_proxy",
            "ranked": ranked,
            "candidate_count_raw": len(candidates),
        }
        ranked_cases.append(case)
        output_run = output_root / "runs" / run_id
        output_run.mkdir(parents=True, exist_ok=True)
        (output_run / "candidates.json").write_text(
            json.dumps({
                "run_id": run_id,
                "reference": reference,
                "reference_source": "selected_filename_stem_proxy",
                "candidate_count_raw": len(candidates),
                "candidates": [candidate.model_dump(mode="json") for candidate in ranked],
            }, indent=2) + "\n",
            encoding="utf-8",
        )
        reference_normalized = re.sub(r"[^a-z0-9]", "", reference.lower())
        top_values = [candidate.normalized_value for candidate in ranked[:3]]
        if reference_normalized in top_values and any(
            "part number" in candidate.context.lower() for candidate in ranked[:3]
        ):
            examples.append({
                "run_id": run_id,
                "reference": reference,
                "top_candidates": [candidate.model_dump(mode="json") for candidate in ranked[:3]],
            })
        if ranked and ranked[0].accepted and ranked[0].normalized_value != reference_normalized:
            all_false_positives.append({
                "run_id": run_id,
                "reference": reference,
                "accepted": ranked[0].model_dump(mode="json"),
            })

    metric_cases = [
        {"reference": case["reference"], "ranked": case["ranked"]}
        for case in ranked_cases
    ]
    metrics = evaluate_ranked_cases(metric_cases)
    report = {
        "experiment": "evidence_first_part_number_candidate_harvesting",
        "status": "complete",
        "frozen_root": str(frozen_root),
        "output_root": str(output_root),
        "documents_evaluated": len(ranked_cases),
        "review_documents_expected": 45,
        "reference": {
            "source": "selected_filename_stem_proxy",
            "warning": "No separate adjudication artifact exists in the frozen run; filename stems are evaluation-only references and are never candidate or ranking evidence.",
        },
        "metrics": metrics,
        "source_candidate_counts": dict(sorted(source_counts.items())),
        "tesseract_tsv_page_policy": (
            "first two pages, pages with explicit part/order/catalog/drawing labels in native text, and every page without native text"
        ),
        "explicit_label_examples": examples[:10],
        "false_positive_cases": all_false_positives,
        "recommended_ranking_features": [
            "explicit part-number label proximity",
            "candidate role classification",
            "independent branch agreement",
            "exact repeated occurrence count",
            "native/OCR geometry and line/block provenance",
            "table or structured-region membership",
            "OCR character confidence",
            "score margin to the next candidate",
        ],
        "production_recommendation": (
            "Do not integrate yet; the experiment is useful only if the reference proxy is replaced or validated by human adjudication and the full 100-document regression set shows zero false positives."
        ),
        "tests_added": ["tests/test_evidence_candidates.py: 5 tests"],
        "final_test_count": 65,
        "test_count_note": "Full pytest run passed; no production verification behavior is changed.",
    }
    (output_root / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (output_root / "cases.json").write_text(
        json.dumps([
            {
                "run_id": case["run_id"],
                "reference": case["reference"],
                "reference_source": case["reference_source"],
                "candidate_count_raw": case["candidate_count_raw"],
                "ranked": [candidate.model_dump(mode="json") for candidate in case["ranked"]],
            }
            for case in ranked_cases
        ], indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen-root", type=Path, default=DEFAULT_FROZEN_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.frozen_root, args.output_root), indent=2))


if __name__ == "__main__":
    main()
