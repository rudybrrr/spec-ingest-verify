import json
from pathlib import Path

import pytest

from builderlab_verify.storage import RunDirectory
from builderlab_verify.upstream import (
    apply_upstream_sku_claim,
    load_upstream_sku_index,
)


def _seed_run(tmp_path: Path, *, ocr_text: str, native_text: str = "", part_number: str | None = None) -> RunDirectory:
    run = RunDirectory.create(tmp_path, "run-001")
    run.ocr_text.write_text(ocr_text, encoding="utf-8")
    run.native_text.write_text(native_text, encoding="utf-8")
    run.ocr_json.write_text(json.dumps({"fields": {"part_number": {"value": part_number}}}), encoding="utf-8")
    run.native_json.write_text(json.dumps({"fields": {"part_number": {"value": part_number}}}), encoding="utf-8")
    run.vision_json.write_text(json.dumps({"fields": {"part_number": {"value": part_number}}}), encoding="utf-8")
    run.verification_json.write_text(json.dumps({
        "run_id": "run-001",
        "status": "human_review",
        "fields": {"part_number": {"status": "mismatch"}},
        "notes": {},
    }), encoding="utf-8")
    return run


def test_upstream_index_is_an_explicit_filename_to_sku_contract(tmp_path: Path) -> None:
    index = tmp_path / "index.csv"
    index.write_text("SKU,Category,File\nSKU-1,spec,a.pdf\n", encoding="utf-8")

    assert load_upstream_sku_index(index, ["a.pdf"]) == {"a.pdf": "SKU-1"}


def test_ordering_code_support_rescues_review_and_preserves_mpn_namespace(tmp_path: Path) -> None:
    run = _seed_run(
        tmp_path,
        ocr_text="Ordering Code: 10-E1126TA035M7-L859F73Z",
        part_number="10-E1126TA035M7-L859F73Z",
    )

    rescued = apply_upstream_sku_claim(run, "10-E1126TA035M7-L859F73Z")
    result = json.loads(run.verification_json.read_text(encoding="utf-8"))

    assert rescued is True
    assert result["status"] == "verified"
    assert result["upstream_sku"]["status"] == "supported"
    assert result["upstream_sku"]["evidence"][0]["source"] == "ocr"
    assert result["identifier_namespaces"]["manufacturer_part_number"] == "10-E1126TA035M7-L859F73Z"
    assert result["identifier_namespaces"]["catalog_sku"] == "10-E1126TA035M7-L859F73Z"


@pytest.mark.parametrize(
    ("expected", "label"),
    [
        ("04A-B01", "Part Number"),
        ("06513.0-00", "Manufacturer Part Number"),
        ("06520.0-00", "Manufacturer Part Number"),
        ("1-1414631-0", "Part Number"),
        ("1-2186527-1", "Part Number"),
        ("1-2186528-1", "Part Number"),
        ("1-2186576-1", "Product Information Part Number"),
        ("1-2186577-1", "Product Information Part Number"),
        ("1-480700-0", "Part Number"),
        ("10-E1126TA035M7-L859F73Z", "Ordering Code"),
    ],
)
def test_all_ten_audited_supports_are_conservatively_rescued(
    tmp_path: Path, expected: str, label: str
) -> None:
    run = _seed_run(
        tmp_path,
        ocr_text=f"{label}: {expected}",
        native_text=f"{label}: {expected}",
        part_number=expected,
    )

    assert apply_upstream_sku_claim(run, expected) is True
    result = json.loads(run.verification_json.read_text(encoding="utf-8"))
    assert result["upstream_sku"]["status"] == "supported"
    assert result["identifier_namespaces"]["catalog_sku"] == expected


def test_catalog_sku_and_mpn_are_not_a_contradiction(tmp_path: Path) -> None:
    run = _seed_run(
        tmp_path,
        ocr_text=(
            "100818033524001 Cut Sheet\n"
            "Manufacturer Part Number: MVSC1-180-3E1-NC-AC110-L-NPT"
        ),
        native_text="100818033524001 Cut Sheet\nManufacturer Part Number: MVSC1-180-3E1-NC-AC110-L-NPT",
        part_number="MVSC1-180-3E1-NC-AC110-L-NPT",
    )

    rescued = apply_upstream_sku_claim(run, "100818033524001")
    result = json.loads(run.verification_json.read_text(encoding="utf-8"))

    assert rescued is False
    assert result["status"] == "human_review"
    assert result["upstream_sku"]["status"] == "insufficient_evidence"
    assert result["upstream_sku"]["contradicting_evidence"] == []
    assert result["identifier_namespaces"]["manufacturer_part_number"] == "MVSC1-180-3E1-NC-AC110-L-NPT"
    assert result["identifier_namespaces"]["catalog_sku"] == "100818033524001"


def test_unlabelled_filename_or_raw_occurrence_does_not_support_sku(tmp_path: Path) -> None:
    run = _seed_run(tmp_path, ocr_text="SKU-1", part_number="OTHER")

    assert apply_upstream_sku_claim(run, "SKU-1") is False
    result = json.loads(run.verification_json.read_text(encoding="utf-8"))
    assert result["upstream_sku"]["status"] == "insufficient_evidence"


def test_same_namespace_conflict_is_contradiction(tmp_path: Path) -> None:
    run = _seed_run(tmp_path, ocr_text="Catalog Number: WRONG-1", part_number="OTHER")

    assert apply_upstream_sku_claim(run, "SKU-1") is False
    result = json.loads(run.verification_json.read_text(encoding="utf-8"))
    assert result["upstream_sku"]["status"] == "contradicted"
    assert result["upstream_sku"]["contradicting_evidence"][0]["value"] == "WRONG-1"
