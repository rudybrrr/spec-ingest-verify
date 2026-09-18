from experiments.evidence_candidates import (
    Candidate,
    CandidateRole,
    CandidateSource,
    build_candidate,
    evaluate_ranked_cases,
    rank_candidates,
)


def test_candidate_preserves_provenance_and_excludes_filename_evidence() -> None:
    candidate = build_candidate(
        raw_text="ABC-123",
        source=CandidateSource.NATIVE_TEXT,
        page=2,
        bbox=[10, 20, 80, 35],
        context="Manufacturer Part Number: ABC-123",
        block_no=4,
        line_no=2,
        filename="ABC-123.pdf",
    )

    assert candidate.raw_text == "ABC-123"
    assert candidate.page == 2
    assert candidate.bbox == [10, 20, 80, 35]
    assert candidate.context == "Manufacturer Part Number: ABC-123"
    assert candidate.role is CandidateRole.ORDERABLE_PART
    assert "ABC-123.pdf" not in candidate.evidence_text


def test_explicit_part_label_ranks_above_title_and_keeps_occurrence_count() -> None:
    candidates = [
        build_candidate(
            raw_text="Widget Series",
            source=CandidateSource.NATIVE_TEXT,
            page=1,
            bbox=[10, 10, 100, 20],
            context="Widget Series Datasheet",
        ),
        build_candidate(
            raw_text="ABC-123",
            source=CandidateSource.NATIVE_TEXT,
            page=1,
            bbox=[10, 100, 80, 112],
            context="Manufacturer Part Number: ABC-123",
        ),
        build_candidate(
            raw_text="ABC-123",
            source=CandidateSource.TESSERACT_TSV,
            page=1,
            bbox=[10, 100, 80, 112],
            context="Manufacturer Part Number: ABC-123",
            ocr_confidence=96.0,
        ),
    ]

    ranked = rank_candidates(candidates)

    assert ranked[0].normalized_value == "abc123"
    assert ranked[0].occurrence_count == 2
    assert ranked[0].role is CandidateRole.ORDERABLE_PART
    assert ranked[0].score > ranked[1].score


def test_conservative_ranker_abstains_without_explicit_or_cross_branch_evidence() -> None:
    candidate = build_candidate(
        raw_text="12345",
        source=CandidateSource.NATIVE_TEXT,
        page=1,
        bbox=[10, 10, 50, 20],
        context="Revision 1.2",
    )

    ranked = rank_candidates([candidate])

    assert ranked[0].accepted is False
    assert ranked[0].abstention_reason


def test_measurement_like_values_do_not_inherit_a_distant_part_label() -> None:
    candidate = build_candidate(
        raw_text="1.62",
        source=CandidateSource.TESSERACT_TSV,
        page=2,
        bbox=[10, 10, 30, 20],
        context="COLOR PART NUMBER 1.62",
        ocr_confidence=45.0,
    )

    ranked = rank_candidates([candidate])

    assert ranked[0].role is CandidateRole.UNKNOWN
    assert ranked[0].accepted is False


def test_evaluation_separates_absent_from_misranked_cases() -> None:
    cases = [
        {"reference": "ABC-123", "ranked": [Candidate.model_validate({"raw_text": "ABC-123", "normalized_value": "abc123", "source": "native_text", "page": 1, "role": "orderable_part", "score": 10, "accepted": True})]},
        {"reference": "DEF-456", "ranked": [Candidate.model_validate({"raw_text": "TITLE", "normalized_value": "title", "source": "native_text", "page": 1, "role": "document_title", "score": 8, "accepted": False})]},
        {"reference": "GHI-789", "ranked": []},
    ]

    result = evaluate_ranked_cases(cases)

    assert result["cases"] == 3
    assert result["recall_at_1"] == 1 / 3
    assert result["correct_value_present"] == 1
    assert result["misranked"] == 0
    assert result["absent"] == 2
