from src.config import settings
import logging

import httpx

from src.config import settings

logger = logging.getLogger(__name__)

# Some embedding models are trained with task prefixes and lose quality
# without them. nomic-embed-text expects these two.
TASK_PREFIXES: dict[str, tuple[str, str]] = {
    "nomic-embed-text": ("search_document: ", "search_query: "),
}


class EmbeddingError(Exception):
    """Ollama failed or returned something unusable. Never store a fallback vector."""


class OllamaEmbedder:
    def __init__(self, client: httpx.AsyncClient | None = None):
        self.model = settings.ollama_default_embedder
        self.base_url = settings.ollama_base_url.rstrip("/")
        self.dimension = settings.ollama_embedding_dimension
        self.batch_size = settings.ollama_max_batch_embedding
        self._client = client or httpx.AsyncClient(
            base_url=self.base_url,
            timeout=settings.timeout_external_http_seconds,
        )
        base_name = self.model.split(":")[0]
        self._doc_prefix, self._query_prefix = TASK_PREFIXES.get(base_name, ("", ""))

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> "OllamaEmbedder":
        return self

    async def __aexit__(self, *exc) -> None:
        await self.aclose()

    async def _embed(self, inputs: list[str]) -> list[list[float]]:
        """One call to /api/embed with a list of inputs."""
        try:
            res = await self._client.post(
                "/api/embed",
                json={"model": self.model, "input": inputs},
            )
            res.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise EmbeddingError(
                f"Ollama {e.response.status_code}: {e.response.text[:300]}"
            ) from e
        except httpx.HTTPError as e:
            raise EmbeddingError(f"Ollama request failed: {e!r}") from e

        embeddings = res.json().get("embeddings")
        if not isinstance(embeddings, list) or len(embeddings) != len(inputs):
            raise EmbeddingError(
                f"expected {len(inputs)} embeddings, got "
                f"{len(embeddings) if isinstance(embeddings, list) else embeddings!r}"
            )
        for vec in embeddings:
            if len(vec) != self.dimension:
                raise EmbeddingError(
                    f"model {self.model!r} returned {len(vec)} dimensions, expected "
                    f"{self.dimension}. Check the model name or the configured dimension."
                )
        return embeddings

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed texts that will be stored and searched (categories, products)."""
        results: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = [self._doc_prefix + t for t in texts[i:i + self.batch_size]]
            results.extend(await self._embed(batch))
        return results

    async def embed_query(self, text: str) -> list[float]:
        """Embed a search query (a product looking for its category)."""
        (vec,) = await self._embed([self._query_prefix + text])
        return vec
