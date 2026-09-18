from pathlib import Path

from experiments.native_pdf_text_experiment import (
    MIN_USABLE_CHARS,
    MIN_USABLE_WORDS,
    NativeTextResult,
    alignment,
    normalize_part_number,
    selected_value,
    summarize_changes,
)
from builderlab_verify.models import AlignmentStatus


def test_part_number_normalization_accepts_punctuation_only_variants():
    assert normalize_part_number("AB-12 / C") == "ab12c"
    assert alignment("AB-12", "ab12") is AlignmentStatus.MATCH


def test_native_usability_requires_text_volume():
    result = NativeTextResult(pages=[], text="", word_count=MIN_USABLE_WORDS, char_count=MIN_USABLE_CHARS, block_count=1)
    assert result.usable
    assert not NativeTextResult(pages=[], text="", word_count=MIN_USABLE_WORDS - 1, char_count=MIN_USABLE_CHARS, block_count=1).usable
    assert not NativeTextResult(pages=[], text="", word_count=MIN_USABLE_WORDS, char_count=MIN_USABLE_CHARS - 1, block_count=1).usable


def test_native_value_is_selected_only_when_layer_and_extraction_are_usable():
    assert selected_value("native", "ocr", True) == "native"
    assert selected_value(None, "ocr", False) == "ocr"


def test_change_summary_counts_only_match_recovery_as_improvement():
    rows = [
        {"native_part_status": "match", "tesseract_part_status": "mismatch"},
        {"native_part_status": "mismatch", "tesseract_part_status": "match"},
        {"native_part_status": "uncertain", "tesseract_part_status": "uncertain"},
        {"native_part_status": "mismatch", "tesseract_part_status": "uncertain"},
    ]
    assert summarize_changes(rows) == {"improved": 1, "regressed": 1, "unchanged": 2}
