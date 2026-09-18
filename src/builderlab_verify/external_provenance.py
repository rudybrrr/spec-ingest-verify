"""Offline, exact-hash manufacturer provenance verification."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from builderlab_verify.models import RunStatus
from builderlab_verify.storage import RunDirectory
from builderlab_verify.upstream import normalize_identifier


DEFAULT_INDEX_PATH = Path(__file__).resolve().parents[2] / "data" / "external_provenance.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def load_provenance_index(index_path: Path = DEFAULT_INDEX_PATH) -> dict[str, dict[str, Any]]:
    payload = json.loads(Path(index_path).read_text(encoding="utf-8"))
    records = payload.get("records")
    if not isinstance(records, list):
        raise ValueError("External provenance index must contain a records list")
    result: dict[str, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("External provenance records must be objects")
        expected = str(record.get("expected_sku", "")).strip()
        if not expected:
            raise ValueError("External provenance record is missing expected_sku")
        key = normalize_identifier(expected)
        if key in result:
            raise ValueError(f"Duplicate external provenance SKU: {expected}")
        result[key] = record
    return result


def _same_namespace_contradiction(verification: dict[str, Any]) -> bool:
    upstream = verification.get("upstream_sku", {})
    if upstream.get("status") == "contradicted":
        return True
    comparison = verification.get("fields", {}).get("part_number", {})
    if comparison.get("status") != "mismatch":
        return False
    # A null/non-null mismatch is missing evidence, not a contradiction.
    ocr = comparison.get("ocr", {}).get("value")
    vision = comparison.get("vision", {}).get("value")
    return ocr is not None and vision is not None and normalize_identifier(str(ocr)) != normalize_identifier(str(vision))


def _namespace_binding_matches(
    verification: dict[str, Any],
    record: dict[str, Any],
    expected_sku: str,
) -> tuple[bool, dict[str, str] | None]:
    if record.get("namespace_binding_explicit") is not True:
        return False, None
    binding = record.get("namespace_binding") or {}
    catalog_sku = str(binding.get("catalog_sku") or record.get("expected_sku") or "").strip()
    manufacturer_part_number = str(
        binding.get("manufacturer_part_number") or record.get("manufacturer_part_number") or ""
    ).strip()
    if not catalog_sku or not manufacturer_part_number:
        return False, None
    if normalize_identifier(catalog_sku) != normalize_identifier(expected_sku):
        return False, None
    observed: set[str] = set()
    fields = verification.get("fields", {}).get("part_number", {})
    for source in ("ocr", "vision"):
        value = fields.get(source, {}).get("value")
        if value:
            observed.add(normalize_identifier(str(value)))
    namespaces = verification.get("identifier_namespaces", {})
    for value in (
        namespaces.get("manufacturer_part_number"),
        namespaces.get("catalog_sku"),
    ):
        if value:
            observed.add(normalize_identifier(str(value)))
    expected_values = {
        normalize_identifier(catalog_sku),
        normalize_identifier(manufacturer_part_number),
    }
    if not expected_values.issubset(observed):
        return False, None
    return True, {
        "catalog_sku": catalog_sku,
        "manufacturer_part_number": manufacturer_part_number,
    }


def evaluate_external_provenance(
    run: RunDirectory,
    expected_sku: str,
    *,
    index_path: Path = DEFAULT_INDEX_PATH,
) -> dict[str, Any]:
    """Evaluate a local provenance record without making any network request."""

    verification = json.loads(run.verification_json.read_text(encoding="utf-8"))
    index = load_provenance_index(index_path)
    record = index.get(normalize_identifier(expected_sku))
    result: dict[str, Any] = {
        "expected_sku": expected_sku,
        "namespace": "catalog_sku",
        "status": "not_found" if record is None else "rejected",
        "accepted": False,
        "rescued": False,
        "record": record,
        "rejection_reasons": [],
        "filename_used_as_evidence": False,
    }
    if record is None:
        result["rejection_reasons"].append("no_local_record")
    else:
        source_hash = _sha256(run.source_pdf) if run.source_pdf.exists() else None
        remote_hash = str(record.get("remote_pdf_sha256", "")).upper()
        local_hash = str(record.get("local_pdf_sha256", "")).upper()
        checks = {
            "explicit_sku_or_revision_page_evidence": record.get("page_explicitly_identifies_expected_sku") is True,
            "page_explicitly_links_pdf": record.get("page_explicitly_links_pdf") is True,
            "exact_hash_match_recorded": record.get("exact_hash_match") is True,
            "remote_local_hash_equal": bool(remote_hash) and remote_hash == local_hash,
            "local_pdf_hash_matches_run": bool(source_hash) and source_hash == local_hash,
            "expected_sku_matches_record": normalize_identifier(str(record.get("expected_sku", ""))) == normalize_identifier(expected_sku),
            "catalog_sku_namespace": record.get("namespace") == "catalog_sku",
        }
        result["checks"] = checks
        result["local_pdf_sha256_observed"] = source_hash
        result["status"] = "accepted" if all(checks.values()) else "rejected"
        result["accepted"] = result["status"] == "accepted"
        result["rejection_reasons"] = [name for name, passed in checks.items() if not passed]
        binding_matches, binding = _namespace_binding_matches(verification, record, expected_sku)
        if binding:
            result["namespace_binding"] = binding
        if result["accepted"] and _same_namespace_contradiction(verification) and not binding_matches:
            result["accepted"] = False
            result["status"] = "rejected"
            result["rejection_reasons"].append(
                "namespace_binding_mismatch" if record.get("namespace_binding_explicit") is True else "same_namespace_contradiction"
            )

    eligible = verification.get("status") == RunStatus.HUMAN_REVIEW.value
    historical_rescue = verification.get("external_provenance", {}).get("rescued") is True
    result["rescued"] = bool(result["accepted"] and (eligible or historical_rescue))
    if result["accepted"] and not eligible and not historical_rescue:
        result["rejection_reasons"].append("not_human_review")
    return result


def apply_external_provenance(
    run: RunDirectory,
    expected_sku: str,
    *,
    index_path: Path = DEFAULT_INDEX_PATH,
) -> bool:
    """Attach provenance evidence and rescue only an existing human review."""

    verification = json.loads(run.verification_json.read_text(encoding="utf-8"))
    decision = evaluate_external_provenance(run, expected_sku, index_path=index_path)
    verification["external_provenance"] = decision
    if decision["rescued"]:
        verification["status"] = RunStatus.VERIFIED.value
        binding = decision.get("namespace_binding")
        if binding:
            namespaces = verification.setdefault("identifier_namespaces", {})
            namespaces["catalog_sku"] = binding["catalog_sku"]
            namespaces["manufacturer_part_number"] = binding["manufacturer_part_number"]
            namespaces.setdefault("manufacturer_part_number_sources", {})["external_provenance"] = binding[
                "manufacturer_part_number"
            ]
        verification.setdefault("notes", {})["external_provenance"] = (
            "Exact manufacturer page-to-linked-PDF provenance and SHA-256 equality "
            "rescued an existing human-review result."
        )
    run.verification_json.write_text(json.dumps(verification, indent=2) + "\n", encoding="utf-8")
    return bool(decision["rescued"])
