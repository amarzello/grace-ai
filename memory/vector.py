"""Vector & graph memory interface for Grace (offline)."""

import asyncio
from functools import cached_property
from typing import List

from mem0 import AsyncMemory
from grace.config.mem0_config import DEFAULT_CONFIG as MEM0_CFG

# The old VectorStorage class has been deprecated.

class _Mem0Manager:
    @cached_property
    def mem(self) -> AsyncMemory:
        return AsyncMemory.from_config(MEM0_CFG)

    async def add(self, text: str, user_id: str = "default") -> None:
        await self.mem.add(messages=[{"role": "user", "content": text}], user_id=user_id)

    async def search(self, query: str, user_id: str = "default", k: int = 5) -> List[str]:
        res = await self.mem.search(query, user_id=user_id, limit=k)
        return [m["content"] for m in res]


_mgr = _Mem0Manager()


def add_memory(text: str, user_id: str = "default") -> None:
    asyncio.run(_mgr.add(text, user_id))


def search_memories(query: str, user_id: str = "default", k: int = 5) -> List[str]:
    return asyncio.run(_mgr.search(query, user_id, k))
