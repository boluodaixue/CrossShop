"""Process-local serialization for stateful shopping sessions."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass


@dataclass
class _LockEntry:
    lock: asyncio.Lock
    waiters: int = 0


class SessionLockRegistry:
    """Serialize one session while allowing unrelated sessions to overlap."""

    def __init__(self) -> None:
        self._guard = asyncio.Lock()
        self._entries: dict[str, _LockEntry] = {}

    @asynccontextmanager
    async def acquire(self, session_key: str) -> AsyncIterator[None]:
        async with self._guard:
            entry = self._entries.setdefault(
                session_key,
                _LockEntry(lock=asyncio.Lock()),
            )
            entry.waiters += 1
        acquired = False
        try:
            await entry.lock.acquire()
            acquired = True
            yield
        finally:
            if acquired:
                entry.lock.release()
            async with self._guard:
                entry.waiters -= 1
                if entry.waiters == 0 and not entry.lock.locked():
                    self._entries.pop(session_key, None)

    @property
    def size(self) -> int:
        return len(self._entries)
