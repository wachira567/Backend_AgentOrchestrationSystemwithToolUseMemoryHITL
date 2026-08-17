import json
from typing import Any, Dict, List, Optional
import redis.asyncio as aioredis
from app.core.config import settings

class RedisWorkingMemory:
    """
    Short-Term Working Memory for Agents.
    Stores agent execution scratchpad, intermediate context, and active conversation state.
    """
    def __init__(self):
        self.redis_url = settings.REDIS_URL or "redis://:redispassword@localhost:6379/0"
        self._client: Optional[aioredis.Redis] = None

    async def get_client(self) -> aioredis.Redis:
        if self._client is None:
            self._client = aioredis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True
            )
        return self._client

    async def set_session_state(self, session_id: str, key: str, value: Any, expire_seconds: int = 3600):
        client = await self.get_client()
        redis_key = f"session:{session_id}:{key}"
        serialized = json.dumps(value)
        await client.set(redis_key, serialized, ex=expire_seconds)

    async def get_session_state(self, session_id: str, key: str) -> Optional[Any]:
        client = await self.get_client()
        redis_key = f"session:{session_id}:{key}"
        data = await client.get(redis_key)
        if data:
            return json.loads(data)
        return None

    async def push_scratchpad_log(self, session_id: str, log_entry: Dict[str, Any]):
        client = await self.get_client()
        redis_key = f"scratchpad:{session_id}"
        await client.rpush(redis_key, json.dumps(log_entry))
        await client.expire(redis_key, 86400) # 24h retention

    async def get_scratchpad_logs(self, session_id: str) -> List[Dict[str, Any]]:
        client = await self.get_client()
        redis_key = f"scratchpad:{session_id}"
        items = await client.lrange(redis_key, 0, -1)
        return [json.loads(i) for i in items]

working_memory = RedisWorkingMemory()
