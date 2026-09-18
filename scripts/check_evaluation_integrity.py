from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from builderlab_verify.batch import BatchRunner
from builderlab_verify.report import write_batch_report
from builderlab_verify.schemas import CategoryField, CategorySchema

root = Path(".runs") / "evaluation-100-spec-sheet"
corpus = Path(os.environ.get("BUILDERLAB_CORPUS", "input"))
category = CategorySchema(name="spec_sheet", fields=[
    CategoryField(name="part_number", description=(
        "Unique manufacturer/orderable part identifier for the specific product represented by the document. "
        "Do not use product-family names, document titles, drawing labels, stock labels, or arbitrary variant identifiers. "
        "Return null for multi-product/portfolio documents or when no single unique part number is clearly identifiable."
    ), required=True),
    CategoryField(name="description", description="Short product description", required=False),
])
filenames = sorted(path.name for path in corpus.glob("*.pdf"))[:100]
runner = BatchRunner(root)
manifest = runner._read_manifest()
successes = [name for name in filenames if manifest.get(name, {}).get("status") in ("verified", "human_review")]

def hashes(name: str) -> dict[str, str]:
    run = root / "runs" / manifest[name]["run_id"]
    return {str(p.relative_to(run)): hashlib.sha256(p.read_bytes()).hexdigest() for p in run.rglob("*") if p.is_file()}

before = {name: hashes(name) for name in successes}
runner.run(corpus, filenames, category)
after = {name: hashes(name) for name in successes}
report_path = write_batch_report(root, filenames, category)
report = json.loads(report_path.read_text(encoding="utf-8"))
report["resumability"] = {
    "restart_completed_count": len(successes),
    "restart_completed_unchanged_count": sum(before[name] == after[name] for name in successes),
    "result": "passed" if before == after else "failed",
}
report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print(json.dumps(report["resumability"]))
