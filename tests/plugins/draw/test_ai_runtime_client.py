from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from pallas_plugin_draw.ai_runtime_client import (
    build_image_request_payload,
    generate_image_via_ai_service,
    reference_urls_for_payload,
    should_use_task_mode,
)


def test_reference_urls_for_payload_keeps_http_and_data_urls() -> None:
    data_url = "data:image/png;base64,abc"
    http_url = "https://gchat.qpic.cn/x"
    assert reference_urls_for_payload([data_url, http_url]) == [data_url, http_url]


def test_should_use_task_mode_for_any_reference_or_long_timeout() -> None:
    assert should_use_task_mode(ref_urls=["https://example.com/a.png"], timeout_sec=30.0) is True
    assert should_use_task_mode(ref_urls=[], timeout_sec=30.0) is False
    assert should_use_task_mode(ref_urls=[], timeout_sec=90.0) is True


def test_build_image_request_payload_sends_all_reference_urls() -> None:
    data_url = "data:image/png;base64,abc"
    http_url = "https://gchat.qpic.cn/x"
    payload = build_image_request_payload(
        request_id="req-1",
        bot_id=1,
        group_id=2,
        user_id=3,
        prompt="test",
        ref_urls=[data_url, http_url],
        timeout_sec=60.0,
        force_task_mode=True,
    )
    assert payload["payload"]["reference_urls"] == [data_url, http_url]


def test_build_image_request_payload_includes_gateway_backends() -> None:
    from pallas_plugin_draw.ai_runtime_client import gateway_payload_from_backends
    from pallas_plugin_draw.config import ImageApiBackend

    gateway = gateway_payload_from_backends(
        [
            ImageApiBackend(
                base_url="https://aigateway.example/",
                api_key="sk-a",
                model="gpt-image-2",
                label="primary",
            ),
            ImageApiBackend(
                base_url="https://packy.example/",
                api_key="sk-b",
                model="gpt-image-2",
                label="fallback-0",
                omit_response_format=True,
            ),
        ]
    )
    payload = build_image_request_payload(
        request_id="req-gw",
        bot_id=1,
        group_id=2,
        user_id=3,
        prompt="test",
        ref_urls=[],
        timeout_sec=30.0,
        force_task_mode=False,
        gateway=gateway,
    )
    assert payload["payload"]["gateway"]["backends"][0]["base_url"] == "https://aigateway.example/"
    assert payload["payload"]["gateway"]["backends"][1]["omit_response_format"] is True
    assert "api_key" in payload["payload"]["gateway"]["backends"][0]


@pytest.mark.asyncio
async def test_generate_image_task_mode_registers_callback_task(monkeypatch: pytest.MonkeyPatch) -> None:
    add_task = AsyncMock()
    monkeypatch.setattr("src.foundation.config.TaskManager.add_task", add_task)
    monkeypatch.setattr("src.foundation.config.TaskManager.remove_task", AsyncMock())

    response = MagicMock()
    response.raise_for_status = MagicMock()
    response.json.return_value = {
        "result_state": "accepted",
        "task_id": "task-1",
        "provider_id": "p",
        "backend_id": "b",
    }
    client = AsyncMock(spec=httpx.AsyncClient)
    client.post = AsyncMock(return_value=response)

    result = await generate_image_via_ai_service(
        client,
        bot_id=1,
        group_id=2,
        user_id=3,
        prompt="test",
        ref_urls=["https://example.com/a.png"],
        timeout_sec=120.0,
        count_usage=True,
    )

    assert result.pending_callback is True
    add_task.assert_awaited_once()
    task_id = add_task.await_args.args[0]
    assert str(task_id).startswith("draw-1-2-3-")
    payload = add_task.await_args.args[1]
    assert payload["task_type"] == "draw"
    assert payload["count_usage"] is True


@pytest.mark.asyncio
async def test_generate_image_task_mode_submit_failed_removes_task(monkeypatch: pytest.MonkeyPatch) -> None:
    add_task = AsyncMock()
    remove_task = AsyncMock()
    monkeypatch.setattr("src.foundation.config.TaskManager.add_task", add_task)
    monkeypatch.setattr("src.foundation.config.TaskManager.remove_task", remove_task)

    response = MagicMock()
    response.raise_for_status = MagicMock()
    response.json.return_value = {
        "result_state": "failed",
        "error": {"message": "rejected"},
    }
    client = AsyncMock(spec=httpx.AsyncClient)
    client.post = AsyncMock(return_value=response)

    result = await generate_image_via_ai_service(
        client,
        bot_id=1,
        group_id=2,
        user_id=3,
        prompt="test",
        ref_urls=["https://example.com/a.png"],
        timeout_sec=120.0,
    )

    assert result.ok is False
    assert result.pending_callback is False
    remove_task.assert_awaited_once()
