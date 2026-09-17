from pathlib import Path

from builderlab_verify.storage import RunDirectory


def test_run_directory_creates_the_fixed_artifact_layout(tmp_path: Path) -> None:
    run = RunDirectory.create(tmp_path, "run-001")

    assert run.root == tmp_path / "run-001"
    assert run.root.is_dir()
    assert run.source_pdf == run.root / "source.pdf"
    assert run.ocr_text == run.root / "ocr.txt"
    assert run.ocr_json == run.root / "ocr.json"
    assert run.vision_json == run.root / "vision.json"
    assert run.verification_json == run.root / "verification.json"
    assert run.final_json == run.root / "final.json"
