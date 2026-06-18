from __future__ import annotations

import base64

import httpx
import pytest

from pallas.api.media import decode_inline_image_reference
from pallas_plugin_draw.image_api import resolve_reference_urls_for_upstream

PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADU0lEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


@pytest.mark.asyncio
async def test_resolve_reference_urls_for_upstream_delegates_to_platform(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_resolve(client, ref_urls, *, options, download_timeout=None):
        assert ref_urls == ["https://gchat.qpic.cn/download/abc"]
        assert download_timeout == 30.0
        from pallas.api.media import ReferenceResolveResult, bytes_to_data_reference_url

        return ReferenceResolveResult(inline_urls=[bytes_to_data_reference_url(PNG_BYTES)])

    monkeypatch.setattr(
        "pallas_plugin_draw.image_api.resolve_reference_inline_urls",
        fake_resolve,
    )

    async with httpx.AsyncClient() as client:
        result = await resolve_reference_urls_for_upstream(
            client,
            ["https://gchat.qpic.cn/download/abc"],
            download_timeout=30.0,
        )

    assert len(result.inline_urls) == 1
    assert decode_inline_image_reference(result.inline_urls[0]) == PNG_BYTES
