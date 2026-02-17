"""Gaia Predictive — Database Declarative Base.

Shared SQLAlchemy declarative base for all ORM models.
"""

from typing import Any

from sqlalchemy.orm import DeclarativeBase, declared_attr

class Base(DeclarativeBase):
    """Base class for all SQLAlchemy models."""
    id: Any
    __name__: str

    # Generate __tablename__ automatically
    @declared_attr
    def __tablename__(cls) -> str:
        return cls.__name__.lower() + "s"
