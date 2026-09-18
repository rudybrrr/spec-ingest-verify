"""Rerun the four documents that failed only because Gemini was unavailable."""

from __future__ import annotations

import json
import os
from pathlib import Path

from builderlab_verify.batch import BatchRunner
from builderlab_verify.schemas import CategoryField, CategorySchema
from builderlab_verify.upstream import load_upstream_sku_index


CORPUS = Path(os.environ.get("BUILDERLAB_CORPUS", "input"))
INDEX = Path(os.environ.get("BUILDERLAB_INDEX_PATH", CORPUS / "index.csv"))
BATCH_ROOT = Path(os.environ.get("BUILDERLAB_RECOVERY_ROOT", ".runs/provider-failure-recovery-20260918"))
FILENAMES = ["00601102.pdf", "1-794616-2.pdf", "10-AT3.pdf", "100-ATN10K6.pdf"]
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
    if BATCH_ROOT.exists():
        raise RuntimeError(f"Recovery root already exists; choose a new root: {BATCH_ROOT}")
    upstream = load_upstream_sku_index(INDEX, FILENAMES)
    runner = BatchRunner(BATCH_ROOT)
    results = runner.run(
        CORPUS,
        FILENAMES,
        CATEGORY,
        upstream_sku_by_filename=upstream,
    )
    payload = []
    for result in results:
        run_root = BATCH_ROOT / "runs" / result.run_id
        metrics = json.loads((run_root / "metrics.json").read_text(encoding="utf-8"))
        verification = json.loads((run_root / "verification.json").read_text(encoding="utf-8")) if (run_root / "verification.json").exists() else {}
        payload.append({
            "filename": result.filename,
            "status": result.status.value,
            "error": result.error,
            "retry_counts": {
                stage: data.get("retry_count", 0)
                for stage, data in metrics.get("stages", {}).items()
            },
            "provider_failures": verification.get("provider_failures", {}),
            "external_provenance": verification.get("external_provenance", {}),
        })
    print(json.dumps({"batch_root": str(BATCH_ROOT), "documents": payload}, indent=2))


if __name__ == "__main__":
    main()
