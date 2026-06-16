import asyncio

from .config import image_gen_config

image_gen_semaphore = asyncio.Semaphore(image_gen_config.max_concurrency)
_semaphore_limit = image_gen_config.max_concurrency
_draw_pending = 0
_draw_pending_lock = asyncio.Lock()


async def acquire_draw_pending_slot() -> bool:
    """进程内画画任务占位；draw_max_pending<=0 时不限制。"""
    global _draw_pending
    limit = image_gen_config.draw_max_pending
    if limit <= 0:
        return True
    async with _draw_pending_lock:
        if _draw_pending >= limit:
            return False
        _draw_pending += 1
        return True


async def release_draw_pending_slot() -> None:
    global _draw_pending
    limit = image_gen_config.draw_max_pending
    if limit <= 0:
        return
    async with _draw_pending_lock:
        if _draw_pending > 0:
            _draw_pending -= 1


def sync_image_gen_semaphore(max_concurrency: int) -> None:
    global image_gen_semaphore, _semaphore_limit
    limit = max(1, min(int(max_concurrency), 32))
    if limit == _semaphore_limit:
        return
    image_gen_semaphore = asyncio.Semaphore(limit)
    _semaphore_limit = limit
