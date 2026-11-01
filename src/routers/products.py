from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from src.database import get_db   # adjust to your project

from src.services.llm import OllamaChat
from src.services.embedder import OllamaEmbedder
from src.agents.pipeline import classify_product
from src.models.products import ClassificationResult, ProductIn

router = APIRouter(prefix="/products", tags=["products"])


@router.post("/classify", response_model=ClassificationResult)
async def classify(product: ProductIn, session: AsyncSession = Depends(get_db)):
    """Synchronous for the prototype: a few LLM calls, so expect 5-30 s on Ollama."""
    async with OllamaEmbedder() as embedder, OllamaChat() as llm:
        return await classify_product(session, embedder, llm, product)