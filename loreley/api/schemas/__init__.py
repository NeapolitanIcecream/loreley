"""Pydantic schemas returned by the UI API."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

__all__ = ["OrmOutModel"]


class OrmOutModel(BaseModel):
    """Base model for UI API responses validated from objects (e.g., ORM rows).

    FastAPI response models frequently receive SQLAlchemy ORM instances or other
    attribute-based objects. Enabling `from_attributes` allows Pydantic to read
    values via `getattr()` rather than requiring dict inputs.
    """

    model_config = ConfigDict(from_attributes=True)


