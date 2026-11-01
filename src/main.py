from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.config import settings
from src.routers import category_backend_router, agent_router


def create_app() -> FastAPI:
    app = FastAPI(
        title="Product Information Processor Agent",
        version="0.1.0",
        docs_url="/docs" if settings.env != "production" else None,
        redoc_url="/redoc" if settings.env != "production" else None,
        # lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["Authorization", "Content-Type"],
    )
    # Public — needed before login so the frontend knows its timeouts.
    app.include_router(category_backend_router)
    app.include_router(agent_router)

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    return app

app = create_app()
