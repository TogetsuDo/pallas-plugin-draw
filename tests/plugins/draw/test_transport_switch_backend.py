"""传输错误应优先切备线，而不是先耗尽主网关参数重试。"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from pallas_plugin_draw.config import ImageApiBackend
from pallas_plugin_draw.draw_attempts import DrawDeadline, run_backend_param_attempts
from pallas_plugin_draw.image_request_options import ImageGenRequestOptions


def _backend(label: str) -> ImageApiBackend:
    return ImageApiBackend(
        base_url=f"https://{label}.example/",
        api_key="k",
        model="m",
        label=label,
    )


@pytest.mark.asyncio
async def test_transport_error_switches_to_next_backend_before_param_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "pallas_plugin_draw.draw_attempts.capped_param_attempts",
        lambda **_kwargs: [
            ImageGenRequestOptions(response_format="b64_json"),
            ImageGenRequestOptions(response_format="url"),
            ImageGenRequestOptions(quality="auto", response_format="b64_json"),
        ],
    )
    monkeypatch.setattr(
        "pallas_plugin_draw.draw_attempts.request_timeout_for_backend_attempt",
        lambda *_a, **_k: 30.0,
    )

    calls: list[str] = []

    async def post_request(
        backend: ImageApiBackend,
        _opts: ImageGenRequestOptions,
        _timeout: float,
    ) -> tuple[int, str]:
        calls.append(backend.label)
        if backend.label == "primary":
            raise httpx.ReadTimeout("")
        return 200, '{"data":[{"b64_json":"YQ=="}]}'

    monkeypatch.setattr(
        "pallas_plugin_draw.draw_attempts.reply_from_image_api_json",
        AsyncMock(return_value=True),
    )
    monkeypatch.setattr(
        "pallas_plugin_draw.draw_attempts.bump_pallas_draw_usage",
        MagicMock(),
    )

    class _Sem:
        async def __aenter__(self):
            return None

        async def __aexit__(self, *_args):
            return None

    monkeypatch.setattr("pallas_plugin_draw.draw_attempts.image_gen_semaphore", _Sem())

    ok = await run_backend_param_attempts(
        MagicMock(),
        MagicMock(),
        bot_id=1,
        group_id=2,
        user_id=3,
        usage_key=(2, 3),
        count_usage=False,
        deadline=DrawDeadline(120.0),
        op="generations",
        backends=[_backend("primary"), _backend("fallback-0")],
        with_ref_urls=False,
        post_request=post_request,
        last_body_holder=[],
        last_status_holder=[],
    )
    assert ok is True
    assert calls == ["primary", "fallback-0"]
