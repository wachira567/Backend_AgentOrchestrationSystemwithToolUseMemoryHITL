from psycopg_pool import AsyncConnectionPool
from app.core.config import settings

# Global pool instance
_pool: AsyncConnectionPool | None = None

async def get_pool() -> AsyncConnectionPool:
    global _pool
    if _pool is None:
        _pool = AsyncConnectionPool(
            conninfo=settings.POSTGRES_URI,
            max_size=20,
            kwargs={"autocommit": True}
        )
    return _pool

async def close_pool():
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None
