"""Filesystem-backed sequential batch orchestration for the existing pipeline."""

from __future__ import annotations

import json
import re
import inspect
from time import perf_counter
from hashlib import sha256
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from collections.abc import Callable

from builderlab_verify.ingestion import TesseractTimeoutError, ingest_pdf
from builderlab_verify.metrics import read_metrics, write_metrics
from builderlab_verify.models import AlignmentStatus, RunStatus
from builderlab_verify.native_text import (
    extract_native_to_json,
    update_native_structured_status,
)
from builderlab_verify.report import write_batch_report
from builderlab_verify.ocr import extract_ocr_to_json
from builderlab_verify.schemas import CategorySchema
from builderlab_verify.storage import RunDirectory
from builderlab_verify.verification import verify_ocr_vision
from builderlab_verify.verification import (
    apply_native_part_number_fallback,
    native_part_number_fallback_needed,
)
from builderlab_verify.upstream import apply_upstream_sku_claim
from builderlab_verify.external_provenance import apply_external_provenance, DEFAULT_INDEX_PATH
from builderlab_verify.vision import extract_vision_to_json


class DocumentStatus(StrEnum):
    PROCESSED = "processed"
    VERIFIED = "verified"
    HUMAN_REVIEW = "human_review"
    FAILED = "failed"


class ProviderFailureError(RuntimeError):
    """A provider branch failed and no deterministic path rescued the document."""


@dataclass
class DocumentResult:
    filename: str
    run_id: str
    status: DocumentStatus
    error: str | None = None
    latency_ms: float | None = None


def _safe_run_id(filename: str) -> str:
    stem = Path(filename).stem.lower()
    safe = re.sub(r"[^a-z0-9]+", "-", stem).strip("-") or "document"
    return safe[:72]


class BatchRunner:
    """Run a bounded, explicitly selected set of PDFs sequentially."""

    def __init__(self, batch_root: Path) -> None:
        self.batch_root = Path(batch_root)
        self.runs_root = self.batch_root / "runs"
        self.manifest_path = self.batch_root / "manifest.json"
        self.report_path = self.batch_root / "report.json"

    def run(
        self,
        input_dir: Path,
        filenames: list[str],
        category: CategorySchema,
        *,
        schema_by_filename: dict[str, CategorySchema] | None = None,
        upstream_sku_by_filename: dict[str, str] | None = None,
        external_provenance_path: Path | None = DEFAULT_INDEX_PATH,
        progress_callback: Callable[[int, int, str, int | None, int | None], None] | None = None,
        rerun_failed: bool = False,
    ) -> list[DocumentResult]:
        input_dir = Path(input_dir)
        self.runs_root.mkdir(parents=True, exist_ok=True)
        results: list[DocumentResult] = []
        manifest: dict[str, dict[str, str | None]] = self._read_manifest()
        if schema_by_filename is not None:
            missing = sorted(set(filenames) - set(schema_by_filename))
            if missing:
                raise ValueError(f"Every selected PDF needs an explicit schema mapping; missing: {missing}")
        if upstream_sku_by_filename is not None:
            missing = sorted(set(filenames) - set(upstream_sku_by_filename))
            if missing:
                raise ValueError(f"Every selected PDF needs an explicit upstream SKU; missing: {missing}")

        total_files = len(filenames)
        for file_index, filename in enumerate(filenames, start=1):
            source = input_dir / filename
            run_id = manifest.get(filename, {}).get("run_id") or _safe_run_id(filename)
            run_root = self.runs_root / run_id
            if not run_root.exists():
                run = RunDirectory.create(self.runs_root, run_id)
            else:
                run = RunDirectory(run_root)
            selected_category = (schema_by_filename or {}).get(filename, category)
            self._write_progress(file_index, total_files, filename, None, None, "starting")
            if progress_callback:
                progress_callback(file_index, total_files, filename, None, None)
            started = perf_counter()
            ocr_incomplete = run.ocr_progress_json.exists() and '"status": "failed"' in run.ocr_progress_json.read_text(encoding="utf-8")
            terminal_failed = (not rerun_failed) and manifest.get(filename, {}).get("status") == DocumentStatus.FAILED.value
            completed = run.verification_json.exists() and not (rerun_failed and ocr_incomplete)
            if completed or terminal_failed:
                if run.verification_json.exists():
                    verification = json.loads(run.verification_json.read_text(encoding="utf-8"))
                    status = DocumentStatus(verification["status"])
                else:
                    status = DocumentStatus.FAILED
                error = manifest.get(filename, {}).get("error")
                latency_ms = read_metrics(run.metrics_json).get("latency_ms")
                results.append(DocumentResult(filename, run_id, status, error, latency_ms))
                manifest[filename] = {"run_id": run_id, "status": status.value, "error": manifest.get(filename, {}).get("error")}
                self._write_manifest(input_dir, manifest)
                continue
            caught_error: Exception | None = None
            try:
                progress = lambda page, total: self._write_progress(file_index, total_files, filename, page, total, "ocr")
                run_parameters = inspect.signature(self._run_document).parameters
                kwargs = {}
                if "progress_callback" in run_parameters:
                    kwargs["progress_callback"] = progress
                if "upstream_sku" in run_parameters and upstream_sku_by_filename is not None:
                    kwargs["upstream_sku"] = upstream_sku_by_filename[filename]
                if "external_provenance_path" in run_parameters and external_provenance_path is not None:
                    kwargs["external_provenance_path"] = external_provenance_path
                self._run_document(source, run, selected_category, **kwargs)
                verification = json.loads(run.verification_json.read_text(encoding="utf-8"))
                status = DocumentStatus(verification["status"])
                error = None
            except Exception as exc:  # surfaced in the dashboard as a failed document
                caught_error = exc
                status = DocumentStatus.FAILED
                error = f"{type(exc).__name__}: {exc}"
            latency_ms = round((perf_counter() - started) * 1000, 2)
            metrics = read_metrics(run.metrics_json)
            metrics["latency_ms"] = latency_ms
            metrics["status"] = status.value
            if error:
                lowered = error.lower()
                if isinstance(caught_error, ProviderFailureError):
                    category_name = "provider_failure"
                elif isinstance(caught_error, TesseractTimeoutError):
                    category_name = "ocr"
                else:
                    category_name = next((name for name in ("ocr", "vision", "schema", "api", "verification_mismatch") if name in lowered), "api")
                metrics["failure_category"] = category_name
            write_metrics(run.metrics_json, metrics)
            result = DocumentResult(filename, run_id, status, error, latency_ms)
            results.append(result)
            manifest[filename] = {
                "run_id": run_id,
                "status": status.value,
                "error": error,
            }
            self._write_manifest(input_dir, manifest)

        self._write_manifest(input_dir, manifest)
        write_batch_report(self.batch_root, filenames, category)
        return results

    def _write_progress(self, file_index: int, total_files: int, filename: str, page: int | None, total_pages: int | None, stage: str) -> None:
        (self.batch_root / "progress.json").write_text(json.dumps({
            "file_index": file_index, "total_files": total_files, "filename": filename,
            "page": page, "total_pages": total_pages, "stage": stage,
        }, indent=2) + "\n", encoding="utf-8")

    def _write_manifest(self, input_dir: Path, manifest: dict[str, dict[str, str | None]]) -> None:
        self.manifest_path.write_text(
            json.dumps(
                {
                    "input_dir": str(input_dir),
                    "documents": manifest,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    @staticmethod
    def _run_document(
        source: Path,
        run: RunDirectory,
        category: CategorySchema,
        *,
        progress_callback=None,
        upstream_sku: str | None = None,
        external_provenance_path: Path | None = DEFAULT_INDEX_PATH,
    ) -> None:
        ocr_incomplete = run.ocr_progress_json.exists() and '"status": "failed"' in run.ocr_progress_json.read_text(encoding="utf-8")
        if not run.source_pdf.exists() or not run.ocr_text.exists() or ocr_incomplete:
            ingest_pdf(source, run, progress_callback=progress_callback)
        provider_failures: dict[str, dict[str, object]] = {}
        if not run.ocr_json.exists():
            try:
                extract_ocr_to_json(run, category)
            except Exception as error:
                provider_failures["ocr"] = _provider_failure_details(run, "ocr", error)
        if not run.vision_json.exists():
            try:
                extract_vision_to_json(run, category)
            except Exception as error:
                provider_failures["vision"] = _provider_failure_details(run, "vision", error)
        if not provider_failures and not run.verification_json.exists():
            try:
                verify_ocr_vision(run, category)
            except Exception as error:
                provider_failures["checker"] = _provider_failure_details(run, "checker", error)
        if provider_failures:
            _write_provider_failure_verification(run, category, provider_failures)
        if not provider_failures and native_part_number_fallback_needed(run):
            if not run.native_json.exists() and run.native_text_json.exists():
                try:
                    native_metadata = json.loads(run.native_text_json.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    native_metadata = {}
                if native_metadata.get("status") == "available" and native_metadata.get("usable") is True:
                    try:
                        extract_native_to_json(run, category)
                    except Exception as error:  # supplementary extraction must not fail the document
                        update_native_structured_status(
                            run,
                            status="failed",
                            error=f"{type(error).__name__}: {error}",
                        )
                    else:
                        update_native_structured_status(run, status="complete")
            apply_native_part_number_fallback(run, category)
        if upstream_sku is not None and run.verification_json.exists():
            apply_upstream_sku_claim(run, upstream_sku)
            if external_provenance_path is not None:
                apply_external_provenance(run, upstream_sku, index_path=external_provenance_path)
        if provider_failures:
            verification = json.loads(run.verification_json.read_text(encoding="utf-8"))
            if verification.get("status") != RunStatus.VERIFIED.value:
                failed_stages = ", ".join(sorted(provider_failures))
                raise ProviderFailureError(
                    f"Provider failure after retries in stage(s): {failed_stages}"
                )

    def _read_manifest(self) -> dict[str, dict[str, str | None]]:
        if not self.manifest_path.exists():
            return {}
        payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        return payload.get("documents", {})


def _provider_failure_details(run: RunDirectory, stage: str, error: Exception) -> dict[str, object]:
    """Persist the branch error plus any retry history already recorded in metrics."""

    details: dict[str, object] = {
        "exception_type": type(error).__name__,
        "error": f"{type(error).__name__}: {error}",
    }
    stage_metrics = read_metrics(run.metrics_json).get("stages", {}).get(stage, {})
    for key in ("attempts", "retry_count", "errors"):
        if key in stage_metrics:
            details[key] = stage_metrics[key]
    return details


def _field_payload(run: RunDirectory, filename: str, field_name: str) -> dict[str, object]:
    """Read available extracted evidence without treating a missing branch as evidence."""

    try:
        payload = json.loads((run.root / filename).read_text(encoding="utf-8"))
        value = payload.get("fields", {}).get(field_name, {})
        if isinstance(value, dict):
            return {
                "value": value.get("value"),
                "unit": value.get("unit"),
                "page": value.get("page"),
            }
    except (OSError, ValueError, TypeError):
        pass
    return {"value": None, "unit": None, "page": None}


def _write_provider_failure_verification(
    run: RunDirectory,
    category: CategorySchema,
    failures: dict[str, dict[str, object]],
) -> None:
    """Create an explicit human-review artifact without manufacturing model matches."""

    fields: dict[str, dict[str, object]] = {}
    for field in category.fields:
        fields[field.name] = {
            "status": AlignmentStatus.UNCERTAIN.value,
            "ocr": _field_payload(run, "ocr.json", field.name),
            "vision": _field_payload(run, "vision.json", field.name),
        }
    artifact = {
        "run_id": run.root.name,
        "status": RunStatus.HUMAN_REVIEW.value,
        "fields": fields,
        "notes": {
            "provider_failure": (
                "One or more Gemini stages were unavailable after bounded retries; "
                "no semantic match was inferred from the unavailable response."
            )
        },
        "provider_failures": failures,
    }
    run.verification_json.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
