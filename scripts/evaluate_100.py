from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path

from builderlab_verify.batch import BatchRunner
from builderlab_verify.schemas import CategoryField, CategorySchema
from builderlab_verify.verification import _part_number_equivalent
from builderlab_verify.upstream import load_upstream_sku_index


CORPUS = Path(os.environ.get("BUILDERLAB_CORPUS", "input"))
BATCH_ROOT = Path(os.environ.get("BUILDERLAB_EVALUATION_ROOT", ".runs/evaluation-100-spec-sheet-upstream-sku"))
INDEX = Path(os.environ.get("BUILDERLAB_INDEX_PATH", str(CORPUS / "index.csv")))
BASELINE_ROOT = Path(".runs") / "evaluation-100-spec-sheet-schema-comparison"
EAGER_NATIVE_ROOT = Path(".runs") / "evaluation-100-spec-sheet-native-integrated-authorized"
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


def main() -> None:
    filenames = sorted(path.name for path in CORPUS.glob("*.pdf"))[:100]
    if len(filenames) != 100:
        raise RuntimeError(f"Expected 100 PDFs, found {len(filenames)}")
    upstream_sku_by_filename = load_upstream_sku_index(INDEX, filenames)
    resume = os.environ.get("BUILDERLAB_RESUME_EVALUATION") == "1"
    if BATCH_ROOT.exists() and not resume:
        raise RuntimeError(
            f"Clean evaluation requires an unused output directory: {BATCH_ROOT}. "
            "Set BUILDERLAB_EVALUATION_ROOT to a new path, or explicitly set "
            "BUILDERLAB_RESUME_EVALUATION=1 for an interrupted clean run."
        )
    runner = BatchRunner(BATCH_ROOT)
    runner.run(CORPUS, filenames, CATEGORY, upstream_sku_by_filename=upstream_sku_by_filename)
    first_manifest = runner._read_manifest()
    before = {}
    completed_names = [
        name for name in filenames
        if first_manifest.get(name, {}).get("status") in ("verified", "human_review")
    ]
    for name in completed_names:
        run_dir = BATCH_ROOT / "runs" / first_manifest[name]["run_id"]
        before[name] = {
            str(path.relative_to(run_dir)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in run_dir.rglob("*") if path.is_file()
        }
    runner.run(CORPUS, filenames, CATEGORY, upstream_sku_by_filename=upstream_sku_by_filename)
    final_manifest = runner._read_manifest()
    report = json.loads((BATCH_ROOT / "report.json").read_text(encoding="utf-8"))
    unchanged = True
    for name, files in before.items():
        run_dir = BATCH_ROOT / "runs" / final_manifest[name]["run_id"]
        after = {str(path.relative_to(run_dir)): hashlib.sha256(path.read_bytes()).hexdigest()
                 for path in run_dir.rglob("*") if path.is_file()}
        unchanged = unchanged and files == after
    report["resumability"] = {
        "restart_completed_count": len(before),
        "restart_completed_unchanged_count": len(before) if unchanged else 0,
        "result": "passed" if unchanged else "failed",
    }

    baseline_manifest = json.loads((BASELINE_ROOT / "manifest.json").read_text(encoding="utf-8"))["documents"]
    baseline_report_path = BASELINE_ROOT / "report.json"
    baseline_report = json.loads(baseline_report_path.read_text(encoding="utf-8")) if baseline_report_path.exists() else {}
    eager_report_path = EAGER_NATIVE_ROOT / "report.json"
    eager_report = json.loads(eager_report_path.read_text(encoding="utf-8")) if eager_report_path.exists() else {}
    rescued: list[str] = []
    regressions: list[str] = []
    native_attributable_regressions: list[str] = []
    native_statuses: dict[str, int] = {}
    native_structured_statuses: dict[str, int] = {}
    native_usable_count = 0
    native_failure_documents: list[str] = []
    native_fallback_invoked_documents: list[str] = []
    upstream_invoked_documents: list[str] = []
    upstream_supported_documents: list[str] = []
    upstream_rescued_documents: list[str] = []
    namespace_cases: list[str] = []
    for filename in filenames:
        final_entry = final_manifest[filename]
        final_run = BATCH_ROOT / "runs" / final_entry["run_id"]
        native_metadata_path = final_run / "native-text.json"
        native_payload_path = final_run / "native.json"
        if native_metadata_path.exists():
            native_metadata = json.loads(native_metadata_path.read_text(encoding="utf-8"))
            native_statuses[native_metadata.get("status", "unknown")] = native_statuses.get(native_metadata.get("status", "unknown"), 0) + 1
            structured_status = native_metadata.get("structured_extraction_status", "unknown")
            native_structured_statuses[structured_status] = native_structured_statuses.get(structured_status, 0) + 1
            native_usable_count += int(native_metadata.get("usable") is True)
            if structured_status in {"complete", "failed"}:
                native_fallback_invoked_documents.append(filename)
            if structured_status == "failed":
                native_failure_documents.append(filename)
        metrics = json.loads((final_run / "metrics.json").read_text(encoding="utf-8")) if (final_run / "metrics.json").exists() else {}
        if "upstream_sku" in metrics.get("stages", {}):
            upstream_invoked_documents.append(filename)
        baseline_entry = baseline_manifest.get(filename, {})
        baseline_run = BASELINE_ROOT / "runs" / baseline_entry.get("run_id", "")
        baseline_verification_path = baseline_run / "verification.json"
        final_verification_path = final_run / "verification.json"
        if not baseline_verification_path.exists() or not final_verification_path.exists():
            continue
        baseline_verification = json.loads(baseline_verification_path.read_text(encoding="utf-8"))
        final_verification = json.loads(final_verification_path.read_text(encoding="utf-8"))
        upstream = final_verification.get("upstream_sku", {})
        if upstream.get("status") == "supported":
            upstream_supported_documents.append(filename)
        namespaces = final_verification.get("identifier_namespaces", {})
        if (
            namespaces.get("manufacturer_part_number")
            and namespaces.get("catalog_sku")
            and _part_number_equivalent(
                str(namespaces["manufacturer_part_number"]), str(namespaces["catalog_sku"])
            ) is False
        ):
            namespace_cases.append(filename)
        baseline_part_status = baseline_verification.get("fields", {}).get("part_number", {}).get("status")
        final_part_status = final_verification.get("fields", {}).get("part_number", {}).get("status")
        if baseline_entry.get("status") == "verified" and final_entry.get("status") != "verified":
            regressions.append(filename)
            if any(
                "native pdf text aligned" in note.lower()
                for note in final_verification.get("notes", {}).values()
            ):
                native_attributable_regressions.append(filename)
        if baseline_entry.get("status") == "human_review" and final_entry.get("status") == "verified" and upstream.get("status") == "supported":
            upstream_rescued_documents.append(filename)
        native_note = final_verification.get("notes", {}).get("part_number", "")
        if native_note.lower().startswith("native pdf text aligned") and native_payload_path.exists():
            native_fields = json.loads(native_payload_path.read_text(encoding="utf-8")).get("fields", {})
            vision_fields = json.loads((final_run / "vision.json").read_text(encoding="utf-8")).get("fields", {})
            native_part = native_fields.get("part_number", {}).get("value")
            vision_part = vision_fields.get("part_number", {}).get("value")
            if _part_number_equivalent(native_part, vision_part):
                rescued.append(filename)

    baseline_verified = sum(entry.get("status") == "verified" for entry in baseline_manifest.values())
    final_verified = report["counts"]["verified"]["count"]
    final_counts = {name: report["counts"][name]["count"] for name in ("verified", "human_review", "failed")}
    baseline_counts = {
        "verified": baseline_verified,
        "human_review": sum(entry.get("status") == "human_review" for entry in baseline_manifest.values()),
        "failed": sum(entry.get("status") == "failed" for entry in baseline_manifest.values()),
    }
    eager_counts = {
        name: eager_report.get("counts", {}).get(name, {}).get("count")
        for name in ("verified", "human_review", "failed")
    }

    def comparison(reference_counts: dict[str, int | None], reference_report: dict) -> dict:
        reference_cost = reference_report.get("total_api_cost_usd")
        reference_latency = reference_report.get("processing_time_ms", {}).get("average")
        return {
            "counts": reference_counts,
            "count_delta": {
                name: final_counts[name] - reference_counts[name]
                if reference_counts.get(name) is not None else None
                for name in final_counts
            },
            "total_api_cost_usd": reference_cost,
            "api_cost_delta_usd": round(report["total_api_cost_usd"] - reference_cost, 8)
            if reference_cost is not None else None,
            "average_latency_ms": reference_latency,
            "average_latency_delta_ms": round(report["processing_time_ms"]["average"] - reference_latency, 2)
            if reference_latency is not None else None,
        }

    report["baseline_comparison"] = {
        "baseline_root": str(BASELINE_ROOT),
        **comparison(baseline_counts, baseline_report),
        "verification_rate_change_percentage_points": round(
            (final_verified - baseline_verified) * 100 / len(filenames), 2
        ),
        "rescued_documents": rescued,
        "regressions": regressions,
        "native_attributable_regressions": native_attributable_regressions,
    }
    report["eager_native_comparison"] = {
        "eager_native_root": str(EAGER_NATIVE_ROOT),
        **comparison(eager_counts, eager_report),
    }
    report["native_fallback"] = {
        "usable_definition": {"min_words": 10, "min_chars": 50},
        "status_counts": native_statuses,
        "structured_extraction_status_counts": native_structured_statuses,
        "usable_count": native_usable_count,
        "invoked_count": len(native_fallback_invoked_documents),
        "invoked_documents": native_fallback_invoked_documents,
        "rescue_count": len(rescued),
        "rescued_documents": rescued,
        "structured_failure_documents": native_failure_documents,
    }
    report["upstream_sku_claim_verification"] = {
        "metadata_source": str(INDEX),
        "contract_coverage": len(upstream_sku_by_filename),
        "invocation_count": len(upstream_invoked_documents),
        "invoked_documents": upstream_invoked_documents,
        "supported_count": len(upstream_supported_documents),
        "rescued_count": len(upstream_rescued_documents),
        "rescued_documents": upstream_rescued_documents,
        "namespace_cases": namespace_cases,
        "filename_used_as_evidence": False,
        "additional_documents_rescued": len(upstream_rescued_documents),
    }
    report["cost_latency_impact"] = {
        "final_api_cost_usd": report.get("total_api_cost_usd"),
        "final_processing_time_ms": report.get("processing_time_ms"),
        "native_text_stage_cost_usd": report.get("cost_usd_by_stage", {}).get("native_text", 0),
        "native_text_stage_latency_ms": report.get("latency_ms_by_stage", {}).get("native_text", 0),
        "api_cost_per_document_usd": report.get("average_api_cost_usd"),
        "baseline": comparison(baseline_counts, baseline_report),
        "eager_native": comparison(eager_counts, eager_report),
        "upstream_sku_stage_cost_usd": report.get("cost_usd_by_stage", {}).get("upstream_sku", 0),
        "upstream_sku_stage_latency_ms": report.get("latency_ms_by_stage", {}).get("upstream_sku", 0),
    }
    report["projection_65_35_0"] = final_counts == {"verified": 65, "human_review": 35, "failed": 0}
    (BATCH_ROOT / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
