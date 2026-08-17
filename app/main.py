from contextlib import asynccontextmanager
from fastapi import FastAPI
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from app.memory.db import get_pool, close_pool
from app.core.config import settings

@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Startup ---
    pool = await get_pool()
    
    # Initialize LangGraph checkpointer tables in PostgreSQL automatically
    checkpointer = AsyncPostgresSaver(pool)
    await checkpointer.setup()
    
    yield
    
    # --- Shutdown ---
    await close_pool()

app = FastAPI(title=settings.PROJECT_NAME, lifespan=lifespan)

@app.get("/health")
async def health_check():
    return {"status": "ok", "message": "API and DB Pool are running smoothly."}
