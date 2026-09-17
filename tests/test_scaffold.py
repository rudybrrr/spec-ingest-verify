from builderlab_verify.config import Settings
from builderlab_verify.schemas import CategoryField, CategorySchema


def test_settings_can_load_without_a_key(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    settings = Settings()
    assert settings.gemini_api_key is None


def test_category_schema_round_trips() -> None:
    category = CategorySchema(
        name="laptop",
        fields=[CategoryField(name="screen_size", required=True)],
    )

    assert category.model_dump() == {
        "name": "laptop",
        "fields": [{"name": "screen_size", "description": None, "required": True}],
    }
