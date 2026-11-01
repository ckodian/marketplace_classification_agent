"""Embed categories into category_embeddings.

Source text per category (built in SQL from the ltree path):

    Bird Cage Bird Baths
    Animals & Pet Supplies > Pet Supplies > Bird Supplies > Bird Cage Accessories > Bird Cage Bird Baths
    <description, when there is one>

The work list is derived from the database on every batch: a category is
pending when it has no embedding for this model, or when the hash of its
current source text differs from the stored one. So a run can stop at any
point and the next run resumes, and generating descriptions later
automatically marks those categories for re-embedding.
"""

import hashlib
import logging
from dataclasses import dataclass, field

from sqlalchemy import func, text
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from src.models.categories import CategoryEmbedding  # adjust to where models.py lives
from src.services.rag.embedder import EmbeddingError, OllamaEmbedder

logger = logging.getLogger(__name__)

# One SQL definition of the source text, so the selection query and the
# stored text can never disagree. Changing it changes every hash, which
# correctly triggers a full re-embed.
SOURCE_TEXT_SQL = """
    c.name
    || CASE WHEN nlevel(c.path) > 1 THEN E'\\n' || (    -- roots: path = name
        SELECT string_agg(a.name, ' > ' ORDER BY nlevel(a.path))
          FROM categories a
         WHERE a.path @> c.path
    ) ELSE '' END
    || COALESCE(E'\\n' || NULLIF(btrim(c.description), ''), '')
"""

PENDING_SQL = text(f"""
    WITH src AS (
        SELECT c.id, {SOURCE_TEXT_SQL} AS source_text
          FROM categories c
         WHERE c.is_active
           AND c.id > :after
    )
    SELECT s.id, s.source_text
      FROM src s
      LEFT JOIN category_embeddings e
             ON e.category_id = s.id AND e.embed_model = :model
     WHERE :force
        OR e.category_id IS NULL
        OR e.source_text_hash <> encode(sha256(convert_to(s.source_text, 'UTF8')), 'hex')
     ORDER BY s.id
     LIMIT :limit
""")

COUNT_SQL = text(f"""
    WITH src AS (
        SELECT c.id, {SOURCE_TEXT_SQL} AS source_text
          FROM categories c
         WHERE c.is_active
    )
    SELECT count(*) AS active,
           count(*) FILTER (
               WHERE e.category_id IS NOT NULL
                 AND e.source_text_hash =
                     encode(sha256(convert_to(s.source_text, 'UTF8')), 'hex')
           ) AS current
      FROM src s
      LEFT JOIN category_embeddings e
             ON e.category_id = s.id AND e.embed_model = :model
""")


def text_hash(source_text: str) -> str:
    """Must match encode(sha256(convert_to(x, 'UTF8')), 'hex') in PENDING_SQL."""
    return hashlib.sha256(source_text.encode("utf-8")).hexdigest()


@dataclass
class EmbeddingRunResult:
    embedded: int = 0
    failed: int = 0
    batches: int = 0
    errors: list[str] = field(default_factory=list)


async def embedding_status(session: AsyncSession, model: str) -> dict:
    row = (await session.execute(COUNT_SQL, {"model": model})).one()
    return {
        "embed_model": model,
        "active_categories": row.active,
        "embedded_current": row.current,
        "pending": row.active - row.current,
    }


async def select_pending(
        session: AsyncSession, model: str, after: str, limit: int, force: bool
) -> list[tuple[str, str]]:
    result = await session.execute(
        PENDING_SQL, {"model": model, "after": after, "limit": limit, "force": force}
    )
    return [(r.id, r.source_text) for r in result]


async def upsert_embeddings(
        session: AsyncSession,
        model: str,
        rows: list[tuple[str, str]],
        vectors: list[list[float]],
) -> None:
    values = [
        {
            "category_id": cat_id,
            "embed_model": model,
            "source_text": source_text,
            "source_text_hash": text_hash(source_text),
            "embedding": vec,
        }
        for (cat_id, source_text), vec in zip(rows, vectors, strict=True)
    ]
    stmt = insert(CategoryEmbedding).values(values)
    stmt = stmt.on_conflict_do_update(
        index_elements=[CategoryEmbedding.category_id, CategoryEmbedding.embed_model],
        set_={
            "source_text": stmt.excluded.source_text,
            "source_text_hash": stmt.excluded.source_text_hash,
            "embedding": stmt.excluded.embedding,
            "updated_at": func.now(),
        },
    )
    await session.execute(stmt)


async def embed_categories(
        session_factory: async_sessionmaker[AsyncSession],
        embedder: OllamaEmbedder,
        force: bool = False,
        batch_size: int = 64,
) -> EmbeddingRunResult:
    """Embed every pending category, one committed batch at a time.

    Paging is keyset-based (id > last id seen), not "select pending again".
    That guarantees the loop ends: with force=True every row stays
    selectable, and a batch that keeps failing would otherwise be picked
    forever. Failed rows stay pending and are retried on the next run.
    """
    model = embedder.model
    result = EmbeddingRunResult()
    after = ""

    while True:
        async with session_factory() as session:
            rows = await select_pending(session, model, after, batch_size, force)
            if not rows:
                break
            after = rows[-1][0]
            result.batches += 1

            try:
                vectors = await embedder.embed_documents([t for _, t in rows])
            except EmbeddingError as e:
                result.failed += len(rows)
                result.errors.append(f"batch after {rows[0][0]}: {e}")
                logger.warning("embedding batch failed (%d rows): %s", len(rows), e)
                continue

            await upsert_embeddings(session, model, rows, vectors)
            await session.commit()
            result.embedded += len(rows)
            logger.info("embedded batch %d: %d rows (last id %s)",
                        result.batches, len(rows), after)

    return result