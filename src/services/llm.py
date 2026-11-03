import logging
import time
from typing import Protocol, TypeVar

import httpx
from pydantic import BaseModel, ValidationError
from src.config import settings

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

class LLMError(Exception):
    def __init__(self, message: str, raw: str | None = None):
        super().__init__(message)
        self.raw = raw;

class LLMClient(Protocol):
    model:str
    async def complete_json(self, system: str, user: str, output: type[T], temperature: float) -> T: ...


class OllamaChat:
    def __init__(self, model: str | None = None, client: httpx.AsyncClient | None = None):
        self.model = model or settings.ollama_text_model
        self._client = client or httpx.AsyncClient(
            base_url=settings.ollama_base_url.rstrip("/"),
            timeout=settings.timeout_external_http_seconds
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> "OllamaChat":
        return self

    async def __aexit__(self, *exc) -> None:
        await self.aclose()

    async def complete_json(self, system: str, user: str, output: type[T], temperature: float = 0.0) -> T:
        started = time.perf_counter()

        try:

            req = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "format": output.model_json_schema(),
                "stream": False,
                "options": {"temperature": temperature}
            }
            print(f"{req}")

            res = await self._client.post("/api/chat",
                json = {
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    "format": output.model_json_schema(),
                    "stream": False,
                    "options": {"temperature": temperature},
            })
            print(f"{res}")
            res.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise LLMError(f"Ollama {e.response.status_code}: {e.response.text[:300]}") from e
        except httpx.HTTPError as e:
            raise LLMError(f"Ollama request failed: {e!r}") from e
        except Exception as ex:
            raise LLMError(f"An exception occured: {ex!r}") from ex

        raw = res.json().get("message", {}).get("content", "")

        try:
            parsed = output.model_validate_json(raw)
        except ValidationError as e:
            raise LLMError(f"{output.__name__} failed validation: {e.errors()[:3]}", raw=raw) from e

        logger.debug("%s via %s in %.0f ms", output.__name__, self.model,
                     (time.perf_counter() - started) * 1000)
        return parsed

class OllamaCriticChat(OllamaChat):
    def __init__(self, model:str| None = None, client: httpx.AsyncClient | None = None):
        super().__init__(model, client)
        self.model = model or settings.ollama_critic_model