from contextlib import asynccontextmanager
from fastapi import FastAPI
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from app.memory.db import get_pool, close_pool
from app.core.config import settings
from app.api.routes import router  # <-- Imported our new routes

@asynccontextmanager
async def lifespan(app: FastAPI):
    pool = await get_pool()
    checkpointer = AsyncPostgresSaver(pool)
    await checkpointer.setup()
    yield
    await close_pool()

app = FastAPI(title=settings.PROJECT_NAME, lifespan=lifespan)

# <-- Mount the router to the application
app.include_router(router, prefix="/api")

@app.get("/health")
async def health_check():
    return {"status": "ok", "message": "API and DB Pool are running smoothly."}
