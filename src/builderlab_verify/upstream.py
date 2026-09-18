"""Conservative verification of an upstream/catalog SKU claim."""

from __future__ import annotations

import csv
import json
import re
from collections import Counter
from pathlib import Path
from time import perf_counter
from typing import Any

from builderlab_verify.metrics import record_stage
from builderlab_verify.models import RunStatus
from builderlab_verify.storage import RunDirectory


SKU_LABELS = (
    "sku",
    "catalog number",
    "catalog no",
    "product number",
    "product no",
    "item number",
    "ordering code",
    "order number",
    "order no",
    "orderable part",
    "part number",
    "part no",
    "part #",
)
MANUFACTURER_LABELS = ("manufacturer part number", "manufacturer part no", "mpn")


def normalize_identifier(value: str | None) -> str:
    return re.sub(r"[^a-z0-9]", "", (value or "").lower())


def load_upstream_sku_index(index_path: Path, filenames: list[str]) -> dict[str, str]:
    """Load and validate the explicit upstream SKU input contract."""

    with Path(index_path).open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    required = {"SKU", "File"}
    if not rows and required - set(rows[0].keys() if rows else []):
        raise ValueError(f"Upstream index must contain columns: {sorted(required)}")
    if rows and required - set(rows[0]):
        raise ValueError(f"Upstream index must contain columns: {sorted(required)}")
    selected = set(filenames)
    mapping: dict[str, str] = {}
    for row in rows:
        filename = (row.get("File") or "").strip()
        sku = (row.get("SKU") or "").strip()
        if filename in selected:
            if not sku:
                raise ValueError(f"Upstream SKU is empty for selected file: {filename}")
            if filename in mapping and normalize_identifier(mapping[filename]) != normalize_identifier(sku):
                raise ValueError(f"Duplicate selected file has conflicting SKUs: {filename}")
            mapping[filename] = sku
    missing = sorted(selected - set(mapping))
    if missing:
        raise ValueError(f"Selected files missing upstream SKU rows: {missing}")
    return mapping


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _contexts(text: str, expected: str) -> list[dict[str, str]]:
    compact = normalize_identifier(expected)
    if not compact:
        return []
    positions = [(i, char) for i, char in enumerate(text) if char.isalnum()]
    compact_text = "".join(char.lower() for _, char in positions)
    results: list[dict[str, str]] = []
    start = 0
    while True:
        found = compact_text.find(compact, start)
        if found < 0:
            break
        left = positions[max(0, found - 180)][0]
        right = positions[min(len(positions) - 1, found + len(compact) - 1 + 180)][0] + 1
        window = " ".join(text[left:right].split())
        lowered = window.lower()
        labels = sorted(
            (label for label in SKU_LABELS + MANUFACTURER_LABELS if label in lowered),
            key=len,
            reverse=True,
        )
        if labels and labels[0] == "sku" and normalize_identifier(expected).startswith("sku"):
            labels = []
        if labels:
            label = labels[0]
            results.append({
                "label": label,
                "namespace": "manufacturer_part_number" if label in MANUFACTURER_LABELS else "catalog_sku",
                "context": window,
            })
        start = found + len(compact)
    return results


def _labelled_alternatives(text: str, expected: str) -> list[dict[str, str]]:
    expected_norm = normalize_identifier(expected)
    results: list[dict[str, str]] = []
    # Only explicit catalog/orderable labels can contradict a catalog SKU.
    # Generic "part number" rows are intentionally excluded: they commonly
    # carry a manufacturer part number in a different namespace.
    labels = "|".join(re.escape(label) for label in SKU_LABELS if label not in {"part number", "part no", "part #"})
    pattern = re.compile(rf"(?i)^(?P<prefix>.*?)\b(?P<label>{labels})\s*[:#=]\s*(?P<value>[A-Za-z0-9][A-Za-z0-9./_-]{{1,80}})\s*$")
    for line in text.splitlines():
        match = pattern.search(line.strip())
        if not match:
            continue
        value = match.group("value").strip(" .,:;|\t\r\n")
        namespace = "catalog_sku"
        if normalize_identifier(value) and normalize_identifier(value) != expected_norm:
            results.append({
                "label": match.group("label"),
                "value": value,
                "namespace": namespace,
                "context": " ".join(match.group(0).split()),
            })
    return results


def _structured_values(run: RunDirectory) -> list[dict[str, str]]:
    values: list[dict[str, str]] = []
    for filename, source in (("ocr.json", "ocr"), ("vision.json", "vision"), ("native.json", "native")):
        try:
            payload = json.loads((run.root / filename).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        value = payload.get("fields", {}).get("part_number", {}).get("value")
        if isinstance(value, str) and value.strip():
            values.append({"source": source, "value": value})
    return values


def _layout_evidence(run: RunDirectory, expected: str) -> list[dict[str, Any]]:
    try:
        payload = json.loads((run.root / "candidates.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    matches: list[dict[str, Any]] = []
    for candidate in payload.get("candidates", []):
        if not isinstance(candidate, dict) or normalize_identifier(str(candidate.get("normalized_value", ""))) != normalize_identifier(expected):
            continue
        context = str(candidate.get("context") or candidate.get("row_text") or "")
        lowered = context.lower()
        if any(label in lowered for label in SKU_LABELS) or candidate.get("role") == "orderable_part":
            matches.append({
                "source": "structured_layout",
                "value": str(candidate.get("raw_text") or expected),
                "label": next((label for label in SKU_LABELS if label in lowered), None),
                "context": context,
                "page": candidate.get("page"),
                "role": candidate.get("role"),
            })
    return matches


def _manufacturer_snapshot(run: RunDirectory) -> tuple[str | None, dict[str, str]]:
    values: dict[str, str] = {}
    for item in _structured_values(run):
        values[item["source"]] = item["value"]
    normalized = {normalize_identifier(value) for value in values.values()}
    selected = next(iter(values.values())) if len(normalized) == 1 and values else None
    return selected, values


def apply_upstream_sku_claim(run: RunDirectory, expected_sku: str) -> bool:
    """Persist a conservative SKU decision and return whether it rescued review."""

    started = perf_counter()
    ocr_text = _read_text(run.ocr_text)
    native_text = _read_text(run.native_text)
    sources = (("ocr", ocr_text), ("native", native_text))
    evidence: list[dict[str, Any]] = []
    for source, text in sources:
        for context in _contexts(text, expected_sku):
            evidence.append({"source": source, "value": expected_sku, **context})

    structured = _structured_values(run)
    layout = _layout_evidence(run, expected_sku)
    expected_norm = normalize_identifier(expected_sku)
    structured_matches = [item for item in structured if normalize_identifier(item["value"]) == expected_norm]
    alternatives = [
        item for source, text in sources for item in _labelled_alternatives(text, expected_sku)
        if item["namespace"] == "catalog_sku"
    ]
    explicit_catalog_support = [item for item in evidence if item["namespace"] == "catalog_sku"]
    support_reasons: list[str] = []
    if explicit_catalog_support:
        support_reasons.append("explicitly_labelled_identifier")
    elif layout:
        support_reasons.append("structured_layout_evidence")
    elif len({item["source"] for item in structured_matches}) >= 2:
        support_reasons.append("matching_structured_evidence")

    if alternatives:
        status = "contradicted"
    elif support_reasons:
        status = "supported"
    else:
        status = "insufficient_evidence"

    manufacturer_part_number, manufacturer_sources = _manufacturer_snapshot(run)
    try:
        verification = json.loads(run.verification_json.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        verification = {"run_id": run.root.name, "status": RunStatus.HUMAN_REVIEW.value, "fields": {}, "notes": {}}
    verification["identifier_namespaces"] = {
        "manufacturer_part_number": manufacturer_part_number,
        "manufacturer_part_number_sources": manufacturer_sources,
        "catalog_sku": expected_sku,
    }
    verification["upstream_sku"] = {
        "expected": expected_sku,
        "normalized_expected": expected_norm,
        "status": status,
        "namespace": "catalog_sku",
        "evidence": evidence,
        "structured_evidence": structured,
        "layout_evidence": layout,
        "contradicting_evidence": alternatives,
        "reason": support_reasons[0] if support_reasons else ("same_namespace_conflict" if alternatives else "no_strong_document_evidence"),
        "filename_used_as_evidence": False,
    }
    rescued = status == "supported" and verification.get("status") == RunStatus.HUMAN_REVIEW.value
    if rescued:
        verification["status"] = RunStatus.VERIFIED.value
        verification.setdefault("notes", {})["upstream_sku"] = "Catalog SKU claim supported by strong document evidence. Manufacturer part number preserved separately."
    run.verification_json.write_text(json.dumps(verification, indent=2) + "\n", encoding="utf-8")
    record_stage(run.metrics_json, "upstream_sku", model="deterministic-claim-verifier", usage=None,
                 latency_ms=(perf_counter() - started) * 1000)
    return rescued
