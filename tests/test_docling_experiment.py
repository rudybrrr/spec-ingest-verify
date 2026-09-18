from types import SimpleNamespace

from experiments.docling_layout_experiment import (
    candidate_from_evidence,
    harvest_docling_candidates,
    rank_docling_candidates,
)


def provenance(page: int = 2):
    return SimpleNamespace(page_no=page, bbox=SimpleNamespace(l=1, t=2, r=30, b=12))


def test_candidate_preserves_docling_provenance_and_role() -> None:
    candidate = candidate_from_evidence(
        raw_text="ABC-123",
        page=2,
        bbox=[1.0, 2.0, 30.0, 12.0],
        context="Manufacturer Part Number ABC-123",
        source_ref="text-4",
    )

    assert candidate.normalized_value == "abc123"
    assert candidate.role == "orderable_part"
    assert candidate.page == 2
    assert candidate.bbox == [1.0, 2.0, 30.0, 12.0]
    assert candidate.source_ref == "text-4"


def test_harvest_docling_candidates_keeps_text_provenance() -> None:
    document = SimpleNamespace(
        groups=[],
        tables=[],
        texts=[
            SimpleNamespace(
                text="Part Number ABC-123",
                prov=[provenance()],
                label=SimpleNamespace(value="text"),
                parent=None,
            )
        ],
    )

    candidates, structure = harvest_docling_candidates(document)

    assert structure == {"text_elements": 1, "tables": 0, "elements_seen": 1}
    assert len(candidates) == 1
    assert candidates[0].normalized_value == "partnumberabc123"
    assert candidates[0].page == 2
    assert candidates[0].source_ref == "text-0"


def test_rank_requires_explicit_labeled_evidence_for_acceptance() -> None:
    labeled = candidate_from_evidence(raw_text="ABC-123", page=1, context="Part Number")
    unlabeled = candidate_from_evidence(raw_text="999-XYZ", page=1, context="Product family")

    ranked = rank_docling_candidates([unlabeled, labeled])

    assert ranked[0].normalized_value == "abc123"
    assert ranked[0].accepted is True
    assert ranked[1].accepted is False
