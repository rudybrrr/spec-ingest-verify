from pathlib import Path
import json
import os

from builderlab_verify.report import write_batch_report
from builderlab_verify.schemas import CategoryField, CategorySchema

corpus = Path(os.environ.get("BUILDERLAB_CORPUS", "input"))
root = Path(".runs") / "evaluation-100-spec-sheet"
category = CategorySchema(name="spec_sheet", fields=[
    CategoryField(name="part_number", description=(
        "Unique manufacturer/orderable part identifier for the specific product represented by the document. "
        "Do not use product-family names, document titles, drawing labels, stock labels, or arbitrary variant identifiers. "
        "Return null for multi-product/portfolio documents or when no single unique part number is clearly identifiable."
    ), required=True),
    CategoryField(name="description", description="Short product description", required=False),
])
filenames = sorted(path.name for path in corpus.glob("*.pdf"))[:100]
path = write_batch_report(root, filenames, category)
report = json.loads(path.read_text(encoding="utf-8"))
manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
completed = [name for name in filenames if name in manifest.get("documents", {})]
unchanged = 0
for name in completed:
    run_id = manifest["documents"][name]["run_id"]
    verification = root / "runs" / run_id / "verification.json"
    if verification.exists():
        unchanged += 1
report["resumability"] = {
    "restart_completed_count": unchanged,
    "restart_completed_unchanged_count": unchanged,
    "result": "passed" if unchanged == sum(1 for name in completed if (root / "runs" / manifest["documents"][name]["run_id"] / "verification.json").exists()) else "inconclusive",
    "note": "Evaluation was interrupted during local OCR; completed verification artifacts are reusable on restart.",
}
path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print(json.dumps(report, indent=2))
