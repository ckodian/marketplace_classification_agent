from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, String, func, text
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy_utils import Ltree, LtreeType

from src.config import settings
from src.database import Base


class Category(Base):
    __tablename__ = "categories"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    # FK targets the TABLE name, and the type must match categories.id (text)
    parent_id: Mapped[str | None] = mapped_column(ForeignKey("categories.id"))
    path: Mapped[Ltree] = mapped_column(LtreeType, nullable=False, unique=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    # Mapped[str] alone means NOT NULL in SQLAlchemy 2.0; descriptions start empty
    description: Mapped[str | None] = mapped_column(String)
    description_generated: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=text("false")
    )
    # The Google taxonomy has no logos, so this must be nullable and not unique
    logo_url: Mapped[str | None] = mapped_column(String)
    depth: Mapped[int] = mapped_column(Integer, nullable=False)
    is_leaf: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text("false"))
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text("true"))
    taxonomy_version: Mapped[int] = mapped_column(Integer, nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    # The DB trigger owns updated_at; no ORM onupdate, so there's one source of truth
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    parent: Mapped["Category | None"] = relationship(
        back_populates="children",
        remote_side="Category.id",
    )
    children: Mapped[list["Category"]] = relationship(back_populates="parent")


class CategoryEmbedding(Base):
    __tablename__ = "category_embeddings"

    category_id: Mapped[str] = mapped_column(
        ForeignKey("categories.id", ondelete="CASCADE"), primary_key=True
    )
    embed_model: Mapped[str] = mapped_column(String, primary_key=True)
    source_text: Mapped[str] = mapped_column(String, nullable=False)
    source_text_hash: Mapped[str] = mapped_column(String, nullable=False)
    embedding: Mapped[list[float]] = mapped_column(
        Vector(settings.ollama_embedding_dimension), nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    # Not in the original DDL. Add it:
    #   ALTER TABLE category_embeddings
    #     ADD COLUMN updated_at timestamptz NOT NULL DEFAULT now();
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )