from builderlab_verify.config import Settings


def test_ocr_settings_default_to_datasheet_safe_values() -> None:
    settings = Settings(_env_file=None)

    assert settings.tesseract_cmd == "tesseract"
    assert settings.tesseract_psm == 3


def test_ocr_settings_can_be_overridden_by_environment(monkeypatch) -> None:
    monkeypatch.setenv("TESSERACT_CMD", "C:/tools/tesseract.exe")
    monkeypatch.setenv("TESSERACT_PSM", "11")

    settings = Settings()

    assert settings.tesseract_cmd == "C:/tools/tesseract.exe"
    assert settings.tesseract_psm == 11
