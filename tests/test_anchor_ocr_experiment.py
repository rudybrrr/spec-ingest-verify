from experiments.anchor_ocr_experiment import normalize, select_representative


def test_normalize_matches_semantic_anchor_spacing_and_case():
    assert normalize("Manufacturer\nPart Number:") == "manufacturer part number"


def test_selection_is_deterministic_and_covers_strata():
    items = []
    for index in range(8):
        items.append({
            "run_id": f"run-{index}",
            "page_count": 1 + index % 2,
            "page_sizes": [[100 + index, 200 + index]],
            "anchor_labels": ["description"] if index % 2 else ["mpn"],
            "word_count": 100 + index * 100,
        })
    selected = select_representative(items, limit=4)
    assert len(selected) == 4
    assert {item["anchor_labels"][0] for item in selected} == {"description", "mpn"}
    assert [item["run_id"] for item in selected] == sorted(item["run_id"] for item in selected)
