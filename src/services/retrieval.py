"""Candidate retrieval: semantic search over category embeddings, plus the
categories already linked to the product's base product and siblings."""

import logging

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from src.services.embedder import OllamaEmbedder   # adjust import path
from src.models.products import Candidate, ProductContext

logger = logging.getLogger(__name__)

FULL_PATH_SQL = """(
    SELECT string_agg(a.name, ' > ' ORDER BY nlevel(a.path))
      FROM categories a
     WHERE a.path @> c.path
)"""

# The query vector is bound as text and cast in SQL. That works whether or
# not the pgvector asyncpg codec is registered on the connection.
QVEC = "CAST(CAST(:qvec AS text) AS vector)"

VECTOR_SEARCH_SQL = text(f"""
    SELECT c.id, c.name, c.description, {FULL_PATH_SQL} AS full_path,
           1 - (e.embedding <=> {QVEC}) AS similarity
      FROM category_embeddings e
      JOIN categories c ON c.id = e.category_id
     WHERE e.embed_model = :model
       AND c.is_active
       AND c.is_leaf
     ORDER BY e.embedding <=> {QVEC}
     LIMIT :k
""")
# Note: with the HNSW index, Postgres fetches hnsw.ef_search (default 40)
# nearest rows and THEN applies the WHERE filters, so it can return fewer
# than k. At ~5.5k rows a sequential scan is fast; if results come back
# short, run `SET LOCAL hnsw.ef_search = 200` in the same transaction.

RELATED_SQL = text(f"""
    SELECT c.id, c.name, c.description, {FULL_PATH_SQL} AS full_path,
           1 - (e.embedding <=> {QVEC}) AS similarity
      FROM categories c
      JOIN category_embeddings e
        ON e.category_id = c.id AND e.embed_model = :model
     WHERE c.id = ANY(:ids) AND c.is_active
""")

RELATED_IDS_SQL = text("""
                       SELECT DISTINCT category_id
                       FROM category_links
                       WHERE product_id = ANY(:product_ids)
                       """)


def _vec_literal(vec: list[float]) -> str:
    return "[" + ",".join(f"{x:.7f}" for x in vec) + "]"


async def related_category_ids(session: AsyncSession, ctx: ProductContext) -> list[str]:
    """Categories already linked to the base product or any sibling."""
    product_ids = [p for p in [ctx.base_product_id, *ctx.sibling_ids] if p]
    if not product_ids:
        return []
    result = await session.execute(RELATED_IDS_SQL, {"product_ids": product_ids})
    return [r.category_id for r in result]


async def retrieve_candidates(
        session: AsyncSession,
        embedder: OllamaEmbedder,
        ctx: ProductContext,
        k: int = 15,
) -> list[Candidate]:
    qvec = _vec_literal(await embedder.embed_query(ctx.to_query_text()))
    params = {"qvec": qvec, "model": embedder.model}

    rows = (await session.execute(VECTOR_SEARCH_SQL, {**params, "k": k})).all()
    candidates = [
        Candidate(category_id=r.id, name=r.name, full_path=r.full_path,
                  description=r.description, similarity=float(r.similarity),
                  rank=i, source="vector")
        for i, r in enumerate(rows, start=1)
    ]

    # Union in sibling/base categories the vector search missed
    seen = {c.category_id for c in candidates}
    extra = [cid for cid in ctx.related_category_ids if cid not in seen]
    if extra:
        rows = (await session.execute(RELATED_SQL, {**params, "ids": extra})).all()
        candidates += [
            Candidate(category_id=r.id, name=r.name, full_path=r.full_path,
                      description=r.description, similarity=float(r.similarity),
                      rank=None, source="sibling")
            for r in rows
        ]

    logger.info("retrieved %d candidates for %s (top: %s %.3f)", len(candidates),
                ctx.product_id, candidates[0].full_path if candidates else "-",
                candidates[0].similarity if candidates else 0)
    return candidates