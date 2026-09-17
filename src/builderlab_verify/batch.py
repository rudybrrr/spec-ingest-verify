"""Filesystem-backed sequential batch orchestration for the existing pipeline."""

from __future__ import annotations

import json
import re
from hashlib import sha256
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from builderlab_verify.ingestion import ingest_pdf
from builderlab_verify.ocr import extract_ocr_to_json
from builderlab_verify.schemas import CategorySchema
from builderlab_verify.storage import RunDirectory
from builderlab_verify.verification import verify_ocr_vision
from builderlab_verify.vision import extract_vision_to_json


class DocumentStatus(StrEnum):
    PROCESSED = "processed"
    VERIFIED = "verified"
    HUMAN_REVIEW = "human_review"
    FAILED = "failed"


@dataclass
class DocumentResult:
    filename: str
    run_id: str
    status: DocumentStatus
    error: str | None = None


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

    def run(
        self,
        input_dir: Path,
        filenames: list[str],
        category: CategorySchema,
        *,
        schema_by_filename: dict[str, CategorySchema] | None = None,
    ) -> list[DocumentResult]:
        input_dir = Path(input_dir)
        self.runs_root.mkdir(parents=True, exist_ok=True)
        results: list[DocumentResult] = []
        manifest: dict[str, dict[str, str | None]] = self._read_manifest()
        if schema_by_filename is not None:
            missing = sorted(set(filenames) - set(schema_by_filename))
            if missing:
                raise ValueError(f"Every selected PDF needs an explicit schema mapping; missing: {missing}")

        for filename in filenames:
            source = input_dir / filename
            run_id = manifest.get(filename, {}).get("run_id") or _safe_run_id(filename)
            run_root = self.runs_root / run_id
            if not run_root.exists():
                run = RunDirectory.create(self.runs_root, run_id)
            else:
                run = RunDirectory(run_root)
            selected_category = (schema_by_filename or {}).get(filename, category)
            try:
                self._run_document(source, run, selected_category)
                verification = json.loads(run.verification_json.read_text(encoding="utf-8"))
                status = DocumentStatus(verification["status"])
                error = None
            except Exception as exc:  # surfaced in the dashboard as a failed document
                status = DocumentStatus.FAILED
                error = f"{type(exc).__name__}: {exc}"
            result = DocumentResult(filename, run_id, status, error)
            results.append(result)
            manifest[filename] = {
                "run_id": run_id,
                "status": status.value,
                "error": error,
            }

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
        return results

    @staticmethod
    def _run_document(source: Path, run: RunDirectory, category: CategorySchema) -> None:
        if not run.source_pdf.exists():
            ingest_pdf(source, run)
        if not run.ocr_json.exists():
            extract_ocr_to_json(run, category)
        if not run.vision_json.exists():
            extract_vision_to_json(run, category)
        if not run.verification_json.exists():
            verify_ocr_vision(run, category)

    def _read_manifest(self) -> dict[str, dict[str, str | None]]:
        if not self.manifest_path.exists():
            return {}
        payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        return payload.get("documents", {})
