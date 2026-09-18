"""Filesystem layout for per-run pipeline artifacts."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RunDirectory:
    """Paths for the fixed artifact set belonging to one pipeline run."""

    root: Path

    @classmethod
    def create(cls, base_dir: Path, run_id: str) -> "RunDirectory":
        root = base_dir / run_id
        root.mkdir(parents=True, exist_ok=False)
        return cls(root=root)

    @property
    def source_pdf(self) -> Path:
        return self.root / "source.pdf"

    @property
    def ocr_text(self) -> Path:
        return self.root / "ocr.txt"

    @property
    def ocr_json(self) -> Path:
        return self.root / "ocr.json"

    @property
    def native_text(self) -> Path:
        return self.root / "native-text.txt"

    @property
    def native_text_json(self) -> Path:
        return self.root / "native-text.json"

    @property
    def native_json(self) -> Path:
        return self.root / "native.json"

    @property
    def vision_json(self) -> Path:
        return self.root / "vision.json"

    @property
    def verification_json(self) -> Path:
        return self.root / "verification.json"

    @property
    def final_json(self) -> Path:
        return self.root / "final.json"

    @property
    def metrics_json(self) -> Path:
        return self.root / "metrics.json"

    @property
    def ocr_progress_json(self) -> Path:
        return self.root / "ocr-progress.json"
