from experiments.paddle_ocr_experiment import aligned_part_number, normalize_paddle_result


def test_normalize_paddle_result_keeps_text_in_reading_order() -> None:
    result = [
        [[10, 10], [100, 30]],
        ("PART NUMBER 123-ABC", 0.98),
        [[110, 10], [200, 30]],
        ("ignored", 0.50),
    ]

    assert normalize_paddle_result(result) == "PART NUMBER 123-ABC\nignored"
    assert normalize_paddle_result({"res": {"rec_texts": ["PART NUMBER", "123-ABC"]}}) == "PART NUMBER\n123-ABC"


def test_aligned_part_number_requires_non_null_equal_values() -> None:
    assert aligned_part_number("123-ABC", "123 abc") is True
    assert aligned_part_number(None, "123-ABC") is False
    assert aligned_part_number("123-ABC", None) is False
    assert aligned_part_number("123-ABC", "999-XYZ") is False
