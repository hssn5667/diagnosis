"""
Shared Redis client for caching across agents
Used by: Patient Context Agent, Drug Safety Agent
"""
import os
import redis.asyncio as redis
from typing import Optional
import json


class RedisClient:
    """Async Redis client wrapper"""
    
    def __init__(self):
        # Support both full URL (Railway) and separate host/port (local dev)
        self.redis_url = os.getenv("REDIS_URL")
        if not self.redis_url:
            self.host = os.getenv("REDIS_HOST", "localhost")
            self.port = int(os.getenv("REDIS_PORT", "6379"))
            self.password = os.getenv("REDIS_PASSWORD")
            
            # Build URL from components
            if self.password:
                self.redis_url = f"redis://:{self.password}@{self.host}:{self.port}"
            else:
                self.redis_url = f"redis://{self.host}:{self.port}"
        
        self._client: Optional[redis.Redis] = None
    
    async def connect(self):
        """Establish Redis connection"""
        if not self._client:
            self._client = await redis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True
            )
    
    async def disconnect(self):
        """Close Redis connection"""
        if self._client:
            await self._client.close()
    
    async def get(self, key: str) -> Optional[str]:
        """Get value from Redis"""
        if not self._client:
            await self.connect()
        return await self._client.get(key)
    
    async def set(self, key: str, value: str, ttl: int = 600):
        """Set value in Redis with TTL (default 10 minutes)"""
        if not self._client:
            await self.connect()
        await self._client.setex(key, ttl, value)
    
    async def delete(self, key: str):
        """Delete key from Redis"""
        if not self._client:
            await self.connect()
        await self._client.delete(key)
    
    async def get_json(self, key: str) -> Optional[dict]:
        """Get and parse JSON from Redis"""
        value = await self.get(key)
        if value:
            return json.loads(value)
        return None
    
    async def set_json(self, key: str, value: dict, ttl: int = 600):
        """Set JSON value in Redis"""
        await self.set(key, json.dumps(value), ttl)

    async def delete(self, key: str) -> bool:
        """Delete key from Redis"""
        if not self._client:
            await self.connect()
        result = await self._client.delete(key)
        return result > 0


# Singleton instance
redis_client = RedisClient()