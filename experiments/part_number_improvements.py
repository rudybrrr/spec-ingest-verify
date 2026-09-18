"""Isolated, offline ``part_number`` improvement experiments.

This module reads only frozen PyMuPDF word artifacts and frozen extraction
outputs. It does not alter production extraction, OCR, Vision, schemas, or
verification behavior.
"""

from __future__ import annotations

import re
from typing import Any, Iterable

from experiments.native_pdf_text_experiment import normalize_part_number

LABELS = (
    ("manufacturer_part_number", ("manufacturer", "part", "number")),
    ("part_number", ("part", "number")),
    ("mpn", ("mpn",)),
)


def _token(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def _identifier(value: str) -> bool:
    normalized = normalize_part_number(value)
    return len(normalized) >= 2 and any(char.isdigit() for char in normalized)


def _same_row(left: dict[str, Any], right: dict[str, Any]) -> bool:
    ly = (left["bbox"][1] + left["bbox"][3]) / 2
    ry = (right["bbox"][1] + right["bbox"][3]) / 2
    height = max(left["bbox"][3] - left["bbox"][1], right["bbox"][3] - right["bbox"][1], 1)
    return abs(ly - ry) <= height * 0.65


def explicit_label_candidates(pages: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return deterministic nearest values for preferred explicit labels.

    A value must be to the right on the same native-text line, or be the first
    identifier on the immediately following line in the same text block. The
    filename is never consulted.
    """
    hits: list[dict[str, Any]] = []
    for page in pages:
        words = page.get("words", [])
        normalized = [_token(str(word.get("text", ""))) for word in words]
        for label, parts in LABELS:
            for index in range(len(words) - len(parts) + 1):
                if tuple(normalized[index:index + len(parts)]) != parts:
                    continue
                end = index + len(parts) - 1
                label_word = words[end]
                label_block = label_word.get("block_no")
                candidates = []
                for candidate_index in range(end + 1, len(words)):
                    candidate = words[candidate_index]
                    if candidate.get("block_no") != label_block:
                        if candidate_index > end + 8:
                            break
                        continue
                    if _same_row(label_word, candidate):
                        if candidate["bbox"][0] >= label_word["bbox"][2] and _identifier(str(candidate.get("text", ""))):
                            candidates.append((0, candidate_index, candidate))
                    elif candidate["bbox"][1] >= label_word["bbox"][3] and candidate["bbox"][1] - label_word["bbox"][3] <= (label_word["bbox"][3] - label_word["bbox"][1]) * 1.75 and _identifier(str(candidate.get("text", ""))):
                        candidates.append((1, candidate_index, candidate))
                if candidates:
                    rank, candidate_index, candidate = min(candidates, key=lambda item: (item[0], item[1]))
                    hits.append({
                        "page": page.get("page"),
                        "label": label,
                        "label_text": " ".join(str(w.get("text", "")) for w in words[index:end + 1]),
                        "value": str(candidate.get("text", "")),
                        "value_word_index": candidate_index,
                        "relation": "same_row" if rank == 0 else "adjacent_line",
                    })
    return hits


def choose_explicit_value(pages: Iterable[dict[str, Any]]) -> dict[str, Any] | None:
    hits = explicit_label_candidates(pages)
    for label, _ in LABELS:
        for hit in hits:
            if hit["label"] == label:
                return hit
    return None


def o0_structurally_equivalent(left: str | None, right: str | None) -> bool:
    """Allow only same-length punctuation-normalized O/0 substitutions."""
    left_normalized = normalize_part_number(left)
    right_normalized = normalize_part_number(right)
    if not left_normalized or len(left_normalized) != len(right_normalized):
        return False
    differences = [(a, b) for a, b in zip(left_normalized, right_normalized) if a != b]
    return bool(differences) and all({a, b} == {"o", "0"} for a, b in differences)
