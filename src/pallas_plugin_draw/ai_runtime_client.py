from __future__ import annotations

import base64
import time
import uuid
from dataclasses import dataclass

import httpx

from src.features.llm.config import get_llm_config, llm_server_base_url

from .replies import DRAW_VAGUE_REPLY

_IMAGE_GENERATE_ENDPOINT = "/api/images/generate"


@dataclass(frozen=True)
class AiImageResult:
    ok: bool
    image_bytes: bytes | None = None
    reply_text: str = DRAW_VAGUE_REPLY
    provider_id: str | None = None
    backend_id: str | None = None


def image_generate_endpoint() -> str:
    cfg = get_llm_config()
    return llm_server_base_url(cfg) + _IMAGE_GENERATE_ENDPOINT


async def generate_image_via_ai_service(
    client: httpx.AsyncClient,
    *,
    bot_id: int,
    group_id: int,
    user_id: int,
    prompt: str,
    ref_urls: list[str],
    timeout_sec: float,
) -> AiImageResult:
    request_id = f"draw-{bot_id}-{group_id}-{user_id}-{uuid.uuid4().hex[:8]}"
    payload = {
        "request_id": request_id,
        "capability": "image.generate",
        "caller": {
            "source": "bot",
            "bot_id": bot_id,
            "plugin": "pallas_plugin_draw",
        },
        "context": {
            "group_id": group_id,
            "user_id": user_id,
            "metadata": {
                "submitted_at": int(time.time()),
            },
        },
        "policy": {
            "mode": "default",
            "timeout_sec": timeout_sec,
            "allow_fallback": True,
            "prefer_local": False,
            "force_task_mode": False,
        },
        "payload": {
            "prompt": prompt,
            "reference_urls": ref_urls,
        },
    }
    response = await client.post(
        image_generate_endpoint(),
        json=payload,
        timeout=httpx.Timeout(timeout_sec, connect=min(15.0, timeout_sec)),
    )
    response.raise_for_status()
    body = response.json()
    if str(body.get("result_state") or "").strip().lower() != "success":
        return AiImageResult(
            ok=False,
            reply_text=str(
                (
                    ((body.get("error") or {}) if isinstance(body, dict) else {}).get(
                        "message"
                    )
                )
                or DRAW_VAGUE_REPLY
            ),
            provider_id=str(body.get("provider_id") or "") or None,
            backend_id=str(body.get("backend_id") or "") or None,
        )
    data = body.get("data") if isinstance(body, dict) else None
    if not isinstance(data, dict):
        return AiImageResult(ok=False)
    raw_b64 = str(data.get("b64_data") or "").strip()
    if not raw_b64:
        return AiImageResult(ok=False)
    try:
        blob = base64.b64decode(raw_b64, validate=True)
    except (ValueError, TypeError):
        return AiImageResult(ok=False)
    return AiImageResult(
        ok=True,
        image_bytes=blob,
        provider_id=str(body.get("provider_id") or "") or None,
        backend_id=str(body.get("backend_id") or "") or None,
    )
