import asyncio
import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from src.config import settings
from src.database import get_db, async_session_factory, get_db_session

from src.services.category_embedding_service import embed_categories, embedding_status
from src.services.embedder import OllamaEmbedder

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/categories", tags=["categories"])

_lock = asyncio.Lock()
_task: asyncio.Task | None = None
_last_run: dict = {"status": "never_run"}


class EmbedRequest(BaseModel):
    force: bool = Field(False, description="Re-embed every active category, ignoring hashes.")
    batch_size: int = Field(64, ge=1, le=512)


async def _run(req: EmbedRequest) -> None:
    global _last_run
    _last_run = {"status": "running", "started_at": datetime.now(timezone.utc).isoformat(),
                 "force": req.force}
    try:
        async with OllamaEmbedder() as embedder:
            result = await embed_categories(
                async_session_factory, embedder, force=req.force, batch_size=req.batch_size
            )
        _last_run |= {
            "status": "completed" if result.failed == 0 else "completed_with_errors",
            "embedded": result.embedded,
            "failed": result.failed,
            "batches": result.batches,
            "errors": result.errors[:10],
        }
    except Exception as e:  # report it on the status endpoint instead of losing it
        logger.exception("category embedding run crashed")
        _last_run |= {"status": "failed", "error": repr(e)}
    finally:
        _last_run["finished_at"] = datetime.now(timezone.utc).isoformat()
        _lock.release()


@router.post("/embed", status_code=status.HTTP_202_ACCEPTED)
async def start_embedding(
        req: EmbedRequest | None = None,
        session: AsyncSession = Depends(get_db),
):
    """Start embedding pending categories in the background."""
    global _task
    req = req or EmbedRequest()
    model = settings.ollama_default_embedder

    current = await embedding_status(session, model)
    if current["pending"] == 0 and not req.force:
        return {"status": "nothing_to_do", **current}

    if _lock.locked():
        raise HTTPException(status.HTTP_409_CONFLICT,
                            detail={"status": "already_running", **_last_run})
    await _lock.acquire()  # released by _run when it finishes

    # Keep a reference: asyncio only holds weak references to tasks.
    _task = asyncio.create_task(_run(req))
    return {"status": "started", "force": req.force, **current}


@router.get("/embed/status")
async def get_embedding_status(session: AsyncSession = Depends(get_db_session)):
    """Progress is read from the database, so it's accurate even mid-run."""
    current = await embedding_status(session, settings.ollama_default_embedder)
    return {**current, "last_run": _last_run}