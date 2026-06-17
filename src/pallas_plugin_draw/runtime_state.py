import asyncio
import time
from dataclasses import dataclass

from .config import image_gen_config

image_gen_semaphore = asyncio.Semaphore(image_gen_config.max_concurrency)
_semaphore_limit = image_gen_config.max_concurrency
_draw_pending = 0
_draw_pending_lock = asyncio.Lock()


@dataclass(slots=True)
class AiRuntimeCircuitState:
    consecutive_failures: int = 0
    last_failure_at: float = 0.0
    last_success_at: float = 0.0
    circuit_open_until: float = 0.0
    recent_failure_reason: str = ""


_ai_runtime_circuit = AiRuntimeCircuitState()


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


def ai_runtime_circuit_is_open(now: float | None = None) -> bool:
    current = time.time() if now is None else now
    return _ai_runtime_circuit.circuit_open_until > current


def record_ai_runtime_success() -> None:
    _ai_runtime_circuit.consecutive_failures = 0
    _ai_runtime_circuit.last_success_at = time.time()
    _ai_runtime_circuit.circuit_open_until = 0.0
    _ai_runtime_circuit.recent_failure_reason = ""


def record_ai_runtime_failure(reason: str) -> None:
    now = time.time()
    _ai_runtime_circuit.consecutive_failures += 1
    _ai_runtime_circuit.last_failure_at = now
    _ai_runtime_circuit.recent_failure_reason = (reason or "").strip()
    if (
        _ai_runtime_circuit.consecutive_failures
        >= image_gen_config.ai_runtime_open_circuit_failures
    ):
        _ai_runtime_circuit.circuit_open_until = (
            now + image_gen_config.ai_runtime_circuit_cooldown_sec
        )


def ai_runtime_circuit_status() -> AiRuntimeCircuitState:
    return AiRuntimeCircuitState(
        consecutive_failures=_ai_runtime_circuit.consecutive_failures,
        last_failure_at=_ai_runtime_circuit.last_failure_at,
        last_success_at=_ai_runtime_circuit.last_success_at,
        circuit_open_until=_ai_runtime_circuit.circuit_open_until,
        recent_failure_reason=_ai_runtime_circuit.recent_failure_reason,
    )
