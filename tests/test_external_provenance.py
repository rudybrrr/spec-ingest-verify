import hashlib
import json
from pathlib import Path

import pytest

from builderlab_verify.external_provenance import (
    DEFAULT_INDEX_PATH,
    apply_external_provenance,
    evaluate_external_provenance,
    load_provenance_index,
)
from builderlab_verify.storage import RunDirectory


PROVENANCE_CASES = [
    ("04J-AP-T01", "04j-ap-t01"), ("04J-AS-T01", "04j-as-t01"),
    ("00551106", "00551106"), ("00571102", "00571102"), ("00591104", "00591104"),
    ("00601102", "00601102"), ("00681101", "00681101"), ("00701101", "00701101"),
    ("00741200", "00741200"), ("00751200", "00751200"),
    ("1-2186527-1", "1-2186527-1"), ("1-2186528-1", "1-2186528-1"),
    ("1-2186576-1", "1-2186576-1"), ("1-2186577-1", "1-2186577-1"),
]


def _record(sku: str, digest: str, **overrides: object) -> dict:
    result = {
        "expected_sku": sku,
        "manufacturer": "Test Manufacturer",
        "product_revision_page_url": "https://manufacturer.test/product",
        "pdf_download_url": "https://manufacturer.test/product.pdf",
        "retrieved_at": "2026-09-18T00:00:00Z",
        "remote_pdf_sha256": digest,
        "local_pdf_sha256": digest,
        "exact_hash_match": True,
        "page_explicitly_identifies_expected_sku": True,
        "page_explicitly_links_pdf": True,
        "evidence_type": "manufacturer_product_page_explicit_sku_linked_datasheet_exact_sha256",
        "namespace": "catalog_sku",
    }
    result.update(overrides)
    return result


def _run(tmp_path: Path, contents: bytes = b"pdf") -> RunDirectory:
    run = RunDirectory.create(tmp_path, "run")
    run.source_pdf.write_bytes(contents)
    run.verification_json.write_text(json.dumps({
        "run_id": "run", "status": "human_review", "fields": {
            "part_number": {"status": "uncertain", "ocr": {"value": None}, "vision": {"value": None}}
        }, "notes": {}
    }), encoding="utf-8")
    return run


@pytest.mark.parametrize("sku,run_id", PROVENANCE_CASES)
def test_all_fourteen_audited_cases_rescue_human_review(tmp_path: Path, sku: str, run_id: str) -> None:
    contents = f"{sku}-local".encode()
    digest = hashlib.sha256(contents).hexdigest().upper()
    index = tmp_path / "index.json"
    index.write_text(json.dumps({"records": [_record(sku, digest)]}), encoding="utf-8")
    run = _run(tmp_path, contents)

    assert apply_external_provenance(run, sku, index_path=index) is True
    saved = json.loads(run.verification_json.read_text(encoding="utf-8"))
    assert saved["status"] == "verified"
    assert saved["external_provenance"]["accepted"] is True
    assert saved["external_provenance"]["rescued"] is True
    assert saved["external_provenance"]["filename_used_as_evidence"] is False


@pytest.mark.parametrize("overrides,reason", [
    ({"page_explicitly_links_pdf": False}, "page_explicitly_links_pdf"),
    ({"remote_pdf_sha256": "0" * 64, "exact_hash_match": False}, "remote_local_hash_equal"),
    ({"page_explicitly_identifies_expected_sku": False}, "explicit_sku_or_revision_page_evidence"),
])
def test_rejects_incomplete_or_non_exact_provenance(tmp_path: Path, overrides: dict, reason: str) -> None:
    contents = b"shared-datasheet"
    digest = hashlib.sha256(contents).hexdigest().upper()
    index = tmp_path / "index.json"
    index.write_text(json.dumps({"records": [_record("SKU-1", digest, **overrides)]}), encoding="utf-8")
    run = _run(tmp_path, contents)

    decision = evaluate_external_provenance(run, "SKU-1", index_path=index)
    assert decision["accepted"] is False
    assert reason in decision["rejection_reasons"]
    assert apply_external_provenance(run, "SKU-1", index_path=index) is False
    assert json.loads(run.verification_json.read_text(encoding="utf-8"))["status"] == "human_review"


def test_rejects_wrong_sku_even_when_another_sku_shares_the_datasheet(tmp_path: Path) -> None:
    contents = b"shared-datasheet"
    digest = hashlib.sha256(contents).hexdigest().upper()
    index = tmp_path / "index.json"
    index.write_text(json.dumps({"records": [_record("SKU-1", digest)]}), encoding="utf-8")
    run = _run(tmp_path, contents)

    decision = evaluate_external_provenance(run, "SKU-2", index_path=index)
    assert decision["accepted"] is False
    assert "no_local_record" in decision["rejection_reasons"]


def test_provenance_does_not_override_same_namespace_contradiction(tmp_path: Path) -> None:
    contents = b"pdf"
    digest = hashlib.sha256(contents).hexdigest().upper()
    index = tmp_path / "index.json"
    index.write_text(json.dumps({"records": [_record("SKU-1", digest)]}), encoding="utf-8")
    run = _run(tmp_path, contents)
    verification = json.loads(run.verification_json.read_text(encoding="utf-8"))
    verification["fields"]["part_number"] = {
        "status": "mismatch", "ocr": {"value": "MPN-A"}, "vision": {"value": "MPN-B"}
    }
    run.verification_json.write_text(json.dumps(verification), encoding="utf-8")

    assert apply_external_provenance(run, "SKU-1", index_path=index) is False
    saved = json.loads(run.verification_json.read_text(encoding="utf-8"))
    assert saved["status"] == "human_review"
    assert "same_namespace_contradiction" in saved["external_provenance"]["rejection_reasons"]


def test_persisted_index_contains_the_three_new_exact_hash_records() -> None:
    index = load_provenance_index(DEFAULT_INDEX_PATH)
    expected = {
        "1-350777-1": "9B967EBC89C555C49D48316A721EF22F04B5CE5DCE347C6DF58F0D6E5A242694",
        "1-794616-2": "81C1D474BE9E3D3F213440A4A4517FD211C69A9124C7D18927CB13B9963CB07F",
        "100818033524001": "6CCB6DDD40138A8692FC456F464BEE25505050D8F9FB4A3F1E97751361FED730",
    }
    for sku, digest in expected.items():
        record = index[sku.replace("-", "").lower()]
        assert record["remote_pdf_sha256"] == digest
        assert record["local_pdf_sha256"] == digest
        assert record["exact_hash_match"] is True
        assert record["page_explicitly_identifies_expected_sku"] is True
        assert record["page_explicitly_links_pdf"] is True


def test_namespace_bound_catalog_and_manufacturer_ids_rescue_review(tmp_path: Path) -> None:
    contents = b"mindman-cut-sheet"
    digest = hashlib.sha256(contents).hexdigest().upper()
    index = tmp_path / "index.json"
    index.write_text(json.dumps({"records": [_record(
        "100818033524001",
        digest,
        manufacturer_part_number="MVSC1-180-3E1-NC-AC110-L-NPT",
        namespace_binding_explicit=True,
    )]}), encoding="utf-8")
    run = _run(tmp_path, contents)
    verification = json.loads(run.verification_json.read_text(encoding="utf-8"))
    verification["fields"]["part_number"] = {
        "status": "mismatch",
        "ocr": {"value": "MVSC1-180-3E1-NC-AC110-L-NPT"},
        "vision": {"value": "100818033524001"},
    }
    run.verification_json.write_text(json.dumps(verification), encoding="utf-8")

    assert apply_external_provenance(run, "100818033524001", index_path=index) is True
    saved = json.loads(run.verification_json.read_text(encoding="utf-8"))
    assert saved["status"] == "verified"
    assert saved["external_provenance"]["namespace_binding"]["manufacturer_part_number"] == "MVSC1-180-3E1-NC-AC110-L-NPT"
    assert saved["identifier_namespaces"]["catalog_sku"] == "100818033524001"
    assert saved["identifier_namespaces"]["manufacturer_part_number"] == "MVSC1-180-3E1-NC-AC110-L-NPT"


def test_namespace_bound_record_rejects_unrelated_part_number_mismatch(tmp_path: Path) -> None:
    contents = b"mindman-cut-sheet"
    digest = hashlib.sha256(contents).hexdigest().upper()
    index = tmp_path / "index.json"
    index.write_text(json.dumps({"records": [_record(
        "100818033524001",
        digest,
        manufacturer_part_number="MVSC1-180-3E1-NC-AC110-L-NPT",
        namespace_binding_explicit=True,
    )]}), encoding="utf-8")
    run = _run(tmp_path, contents)
    verification = json.loads(run.verification_json.read_text(encoding="utf-8"))
    verification["fields"]["part_number"] = {
        "status": "mismatch",
        "ocr": {"value": "UNRELATED-MPN"},
        "vision": {"value": "OTHER-CATALOG-SKU"},
    }
    run.verification_json.write_text(json.dumps(verification), encoding="utf-8")

    assert apply_external_provenance(run, "100818033524001", index_path=index) is False
    saved = json.loads(run.verification_json.read_text(encoding="utf-8"))
    assert saved["status"] == "human_review"
    assert "namespace_binding_mismatch" in saved["external_provenance"]["rejection_reasons"]
