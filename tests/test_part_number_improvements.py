from experiments.part_number_improvements import choose_explicit_value, o0_structurally_equivalent


def word(text, x0, y0, x1, y1, block=1, line=1):
    return {"text": text, "bbox": [x0, y0, x1, y1], "block_no": block, "line_no": line}


def test_explicit_label_prefers_manufacturer_part_number_and_nearest_value():
    pages = [{"page": 1, "words": [
        word("Part", 10, 10, 30, 20), word("Number", 32, 10, 70, 20), word("FAMILY", 75, 10, 125, 20),
        word("Manufacturer", 10, 30, 80, 40), word("Part", 82, 30, 102, 40), word("Number", 104, 30, 144, 40),
        word("ABC-0", 150, 30, 190, 40),
    ]}]
    hit = choose_explicit_value(pages)
    assert hit and hit["label"] == "manufacturer_part_number" and hit["value"] == "ABC-0"


def test_o0_requires_structural_identity():
    assert o0_structurally_equivalent("ABO-12", "AB0-12")
    assert not o0_structurally_equivalent("ABO-12", "AB0-123")
    assert not o0_structurally_equivalent("MVSC1-180", "100818033524001")
