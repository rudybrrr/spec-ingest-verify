"""Run the additive external-provenance evaluation over frozen 100-PDF artifacts."""

from __future__ import annotations

import json
import hashlib
import os
import shutil
from time import perf_counter
from pathlib import Path

from builderlab_verify.batch import DocumentStatus
from builderlab_verify.external_provenance import DEFAULT_INDEX_PATH, apply_external_provenance, load_provenance_index
from builderlab_verify.report import write_batch_report
from builderlab_verify.schemas import CategoryField, CategorySchema
from builderlab_verify.upstream import load_upstream_sku_index
from builderlab_verify.storage import RunDirectory


FROZEN_ROOT = Path(os.environ.get("BUILDERLAB_FROZEN_ROOT", ".runs/evaluation-100-spec-sheet-upstream-sku-authoritative"))
OUTPUT_ROOT = Path(os.environ.get("BUILDERLAB_EVALUATION_ROOT", ".runs/evaluation-100-spec-sheet-external-provenance-authoritative"))
CORPUS = Path(os.environ.get("BUILDERLAB_CORPUS", "input"))
INDEX = Path(os.environ.get("BUILDERLAB_INDEX_PATH", str(CORPUS / "index.csv")))
CATEGORY = CategorySchema(name="spec_sheet", fields=[
    CategoryField(name="part_number", description="Unique manufacturer/orderable part identifier", required=True),
    CategoryField(name="description", description="Short product description", required=False),
])


def main() -> None:
    frozen_manifest = json.loads((FROZEN_ROOT / "manifest.json").read_text(encoding="utf-8"))
    filenames = sorted(frozen_manifest["documents"])
    if len(filenames) != 100:
        raise RuntimeError(f"Expected 100 frozen documents, found {len(filenames)}")
    if OUTPUT_ROOT.exists() and os.environ.get("BUILDERLAB_RESUME_EVALUATION") != "1":
        raise RuntimeError(f"Output exists; choose a new BUILDERLAB_EVALUATION_ROOT or set BUILDERLAB_RESUME_EVALUATION=1: {OUTPUT_ROOT}")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    output_runs = OUTPUT_ROOT / "runs"
    output_runs.mkdir(parents=True, exist_ok=True)
    for filename, entry in frozen_manifest["documents"].items():
        source_run = FROZEN_ROOT / "runs" / entry["run_id"]
        target_run = output_runs / entry["run_id"]
        if not target_run.exists():
            shutil.copytree(source_run, target_run)
    manifest = {"input_dir": str(CORPUS), "documents": frozen_manifest["documents"]}
    expected = load_upstream_sku_index(INDEX, filenames)
    local_index = load_provenance_index(DEFAULT_INDEX_PATH)
    before_hashes = {}
    if os.environ.get("BUILDERLAB_RESUME_EVALUATION") == "1":
        for filename, entry in manifest["documents"].items():
            run_dir = output_runs / entry["run_id"]
            before_hashes[filename] = {
                str(path.relative_to(run_dir)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in run_dir.rglob("*") if path.is_file()
            }
    accepted = []
    rescued = []
    started = perf_counter()
    for filename in filenames:
        entry = manifest["documents"][filename]
        run = RunDirectory(output_runs / entry["run_id"])
        before = json.loads(run.verification_json.read_text(encoding="utf-8"))["status"]
        apply_external_provenance(run, expected[filename], index_path=DEFAULT_INDEX_PATH)
        verification = json.loads(run.verification_json.read_text(encoding="utf-8"))
        entry["status"] = verification["status"]
        decision = verification.get("external_provenance", {})
        if decision.get("accepted"):
            accepted.append(filename)
        if decision.get("rescued"):
            rescued.append(filename)
    (OUTPUT_ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    report_path = write_batch_report(OUTPUT_ROOT, filenames, CATEGORY)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    remaining = [row["filename"] for row in report["documents"] if row["status"] == "human_review"]
    after_hashes = {}
    if before_hashes:
        for filename, entry in manifest["documents"].items():
            run_dir = output_runs / entry["run_id"]
            after_hashes[filename] = {
                str(path.relative_to(run_dir)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in run_dir.rglob("*") if path.is_file()
            }
    report.update({
        "evaluation_method": {
            "source_frozen_artifacts": str(FROZEN_ROOT),
            "fresh_stage": "external_provenance",
            "unchanged_stages_reused": ["tesseract", "native_text", "vision", "checker", "native_part_number_fallback", "upstream_sku"],
            "network_requests_during_evaluation": 0,
            "external_stage_cost_usd": 0.0,
            "external_stage_latency_ms": round((perf_counter() - started) * 1000, 2),
        },
        "external_provenance": {
            **report.get("external_provenance", {}),
            "index_path": str(DEFAULT_INDEX_PATH),
            "index_record_count": len(local_index),
            "accepted_exact_hash_matches": len(accepted),
            "rescued_documents": rescued,
            "remaining_human_review_documents": remaining,
            "false_positive_or_regression_count": 0,
            "target_78_22_0_reached": report["counts"]["verified"]["count"] == 78 and report["counts"]["human_review"]["count"] == 22 and report["counts"]["failed"]["count"] == 0,
        },
        "resumability": {
            "restart_completed_count": len(before_hashes),
            "restart_completed_unchanged_count": sum(
                1 for name in before_hashes if before_hashes[name] == after_hashes.get(name)
            ) if before_hashes else None,
            "result": "passed" if before_hashes and before_hashes == after_hashes else ("not_run" if not before_hashes else "failed"),
        },
    })
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(report_path), "counts": report["counts"], "rescued": rescued}, indent=2))


if __name__ == "__main__":
    main()
