from experiments.structured_context_ocr_experiment import (
    StructuredAnchor,
    _aligned,
    aggregate_metrics,
    _prompt,
    structured_neighborhood,
)


def word(index, text, top, left, line):
    return {"index": index, "text": text, "left": left, "top": top, "width": 20, "height": 10,
            "conf": 90.0, "block_num": 1, "par_num": 1, "line_num": line, "page_num": 1}


def test_neighborhood_keeps_same_row_nearby_rows_and_columns():
    words = [word(0, "Manufacturer", 10, 10, 1), word(1, "Part", 10, 90, 1), word(2, "Number", 10, 140, 1),
             word(3, "ABC-123", 10, 300, 1), word(4, "Description", 30, 10, 2), word(5, "Widget", 30, 300, 2),
             word(6, "Notes", 50, 10, 3)]
    result = structured_neighborhood(words, [0, 1, 2], nearby_rows=1)
    assert [item["text"] for item in result] == ["Manufacturer", "Part", "Number", "ABC-123", "Description", "Widget"]


def test_prompt_contains_coordinates_and_no_vision_reference():
    anchor = StructuredAnchor("part_number", 1, [1], "Part Number", {"left": 1, "top": 2, "right": 3, "bottom": 4}, [1, 1, 1], [1], [
        {"index": 1, "text": "Part", "left": 10, "top": 20, "width": 30, "height": 10, "conf": 88.5, "row_offset": 0},
    ])
    prompt = _prompt(anchor)
    assert "Part [x=10,y=20,w=30,h=10,conf=88.5]" in prompt
    assert "Vision" not in prompt


def test_null_reference_cannot_count_as_alignment():
    assert not _aligned(None, None)
    assert _aligned("ABC-123", "ABC-123")


def test_aggregate_metrics_is_canonical_for_recorded_case_flags():
    cases = [
        {"comparison": {"status": "anchor_found", "improvement_over_full_page_ocr": True, "improvement_over_prior_bounded": True, "regression_vs_full_page_ocr": False, "regression_vs_prior_bounded": False, "still_ambiguous": False}},
        {"comparison": {"status": "anchor_found", "improvement_over_full_page_ocr": False, "improvement_over_prior_bounded": True, "regression_vs_full_page_ocr": True, "regression_vs_prior_bounded": True, "still_ambiguous": True}},
        {"comparison": {"status": "no_anchor"}},
    ]
    assert aggregate_metrics(cases) == {
        "improvement_over_full_page_ocr": 1,
        "improvement_over_prior_bounded": 2,
        "regression_vs_full_page_ocr": 1,
        "regression_vs_prior_bounded": 1,
        "still_ambiguous": 1,
        "anchor_status": {"anchor_found": 2, "no_anchor": 1},
    }
