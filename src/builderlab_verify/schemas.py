"""Pydantic schema placeholders for the v0 category input."""

from pydantic import BaseModel, Field


class CategoryField(BaseModel):
    """A field expected in a category-specific spec sheet."""

    name: str = Field(min_length=1)
    description: str | None = None
    required: bool = True


class CategorySchema(BaseModel):
    """Category name and fields used to interpret a spec sheet."""

    name: str = Field(min_length=1)
    fields: list[CategoryField] = Field(default_factory=list)
