from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from nonebot.adapters.onebot.v11 import GroupMessageEvent, Message
from nonebot.exception import FinishedException


def _make_event(
    *,
    body: str = "牛牛画画",
    self_id: int = 123,
    time: int = 1,
    group_id: int = 1,
    user_id: int = 2,
    message_id: int = 3,
) -> GroupMessageEvent:
    return GroupMessageEvent.model_construct(
        time=time,
        self_id=self_id,
        post_type="message",
        message_type="group",
        sub_type="normal",
        user_id=user_id,
        group_id=group_id,
        message_id=message_id,
        message=Message(body),
        raw_message=body,
        reply=None,
    )


@pytest.mark.asyncio
async def test_draw_handle_skips_claim_when_backend_missing(monkeypatch):
    from pallas_plugin_draw import draw as mod

    called = False

    async def fake_claim(*_args, **_kwargs) -> bool:
        nonlocal called
        called = True
        return True

    async def fake_finish(*_args, **_kwargs):
        raise FinishedException

    monkeypatch.setattr(mod, "draw_group_allowed", lambda _gid: True)
    monkeypatch.setattr(mod, "draw_group_cooldown_ready", AsyncMock(return_value=True))
    monkeypatch.setattr(
        mod, "active_image_gen_settings", lambda: SimpleNamespace(api_backends=list)
    )
    monkeypatch.setattr(mod, "try_claim_group_message_once", fake_claim)
    monkeypatch.setattr(mod.pallas_draw, "finish", fake_finish)

    with pytest.raises(FinishedException):
        await mod.pallas_draw_handle(
            SimpleNamespace(self_id="123"), _make_event(), Message("")
        )

    assert called is False


@pytest.mark.asyncio
async def test_draw_handle_skips_claim_when_prompt_and_refs_empty(monkeypatch):
    from pallas_plugin_draw import draw as mod
    from pallas_plugin_draw.config import ImageApiBackend

    called = False

    async def fake_claim(*_args, **_kwargs) -> bool:
        nonlocal called
        called = True
        return True

    async def fake_finish(*_args, **_kwargs):
        raise FinishedException

    backend = ImageApiBackend(
        base_url="https://api.example.com/",
        api_key="sk-test",
        model="m",
        label="primary",
    )

    monkeypatch.setattr(mod, "draw_group_allowed", lambda _gid: True)
    monkeypatch.setattr(mod, "draw_group_cooldown_ready", AsyncMock(return_value=True))
    monkeypatch.setattr(
        mod,
        "active_image_gen_settings",
        lambda: SimpleNamespace(api_backends=lambda: [backend]),
    )
    monkeypatch.setattr(mod, "draw_should_count_usage", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(mod, "SUPERUSER", AsyncMock(return_value=False))
    monkeypatch.setattr(mod, "try_claim_group_message_once", fake_claim)
    monkeypatch.setattr(mod.pallas_draw, "finish", fake_finish)

    with pytest.raises(FinishedException):
        await mod.pallas_draw_handle(
            SimpleNamespace(self_id="123"), _make_event(), Message("")
        )

    assert called is False


@pytest.mark.asyncio
async def test_draw_handle_skips_entire_command_when_message_scrub_blocked(monkeypatch):
    from pallas_plugin_draw import draw as mod
    from pallas_plugin_draw.config import ImageApiBackend

    called = False
    sent = False

    async def fake_claim(*_args, **_kwargs) -> bool:
        nonlocal called
        called = True
        return True

    async def fake_send(*_args, **_kwargs):
        nonlocal sent
        sent = True

    backend = ImageApiBackend(
        base_url="https://api.example.com/",
        api_key="sk-test",
        model="m",
        label="primary",
    )

    monkeypatch.setattr(mod, "draw_group_allowed", lambda _gid: True)
    monkeypatch.setattr(mod, "draw_group_cooldown_ready", AsyncMock(return_value=True))
    monkeypatch.setattr(
        mod,
        "active_image_gen_settings",
        lambda: SimpleNamespace(api_backends=lambda: [backend]),
    )
    monkeypatch.setattr(mod, "draw_should_count_usage", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(mod, "SUPERUSER", AsyncMock(return_value=False))
    monkeypatch.setattr(
        mod, "is_message_scrub_blocked_async", AsyncMock(return_value=True)
    )
    monkeypatch.setattr(mod, "try_claim_group_message_once", fake_claim)
    monkeypatch.setattr(mod.pallas_draw, "send", fake_send)

    await mod.pallas_draw_handle(
        SimpleNamespace(self_id="123"),
        _make_event(body="牛牛画画 badword"),
        Message("badword"),
    )

    assert called is False
    assert sent is False


@pytest.mark.asyncio
async def test_draw_handle_same_bot_duplicate_event_only_cheers_once(monkeypatch):
    from pallas.core.platform.shard.registry import config as shard_cfg
    from pallas_plugin_draw import draw as mod
    from pallas_plugin_draw.config import ImageApiBackend

    backend = ImageApiBackend(
        base_url="https://api.example.com/",
        api_key="sk-test",
        model="m",
        label="primary",
    )
    event = _make_event(
        body="牛牛画画 小羊",
        self_id=2927116873,
        time=1780560720,
        group_id=934988865,
        user_id=1581408000,
        message_id=1001,
    )
    sent: list[str] = []
    queued: list[str] = []

    async def fake_send(message):
        sent.append(str(message))

    async def fake_run_queued(*_args, **_kwargs):
        queued.append("queued")

    def fake_create_task(coro, *, name=None):
        coro.close()
        queued.append(name or "task")
        return SimpleNamespace()

    monkeypatch.setattr(shard_cfg, "is_sharding_active", lambda: False)
    monkeypatch.setattr(mod, "draw_group_allowed", lambda _gid: True)
    monkeypatch.setattr(mod, "draw_group_cooldown_ready", AsyncMock(return_value=True))
    monkeypatch.setattr(
        mod,
        "active_image_gen_settings",
        lambda: SimpleNamespace(api_backends=lambda: [backend]),
    )
    monkeypatch.setattr(mod, "draw_should_count_usage", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(mod, "SUPERUSER", AsyncMock(return_value=False))
    monkeypatch.setattr(mod, "acquire_draw_pending_slot", AsyncMock(return_value=True))
    monkeypatch.setattr(mod, "consume_draw_group_cooldown", AsyncMock())
    monkeypatch.setattr(mod, "run_pallas_draw_queued", fake_run_queued)
    monkeypatch.setattr(mod.asyncio, "create_task", fake_create_task)
    monkeypatch.setattr(mod.pallas_draw, "send", fake_send)

    await mod.pallas_draw_handle(
        SimpleNamespace(self_id="2927116873"), event, Message("小羊")
    )
    await mod.pallas_draw_handle(
        SimpleNamespace(self_id="2927116873"), event, Message("小羊")
    )

    assert sent == ["欢呼吧！"]
    assert queued == ["pallas_draw:934988865:1581408000"]


@pytest.mark.asyncio
async def test_draw_handle_same_body_different_time_cheers_twice(monkeypatch):
    from pallas.core.platform.shard.registry import config as shard_cfg
    from pallas_plugin_draw import draw as mod
    from pallas_plugin_draw.config import ImageApiBackend

    backend = ImageApiBackend(
        base_url="https://api.example.com/",
        api_key="sk-test",
        model="m",
        label="primary",
    )
    sent: list[str] = []
    queued: list[str] = []

    async def fake_send(message):
        sent.append(str(message))

    async def fake_run_queued(*_args, **_kwargs):
        queued.append("queued")

    def fake_create_task(coro, *, name=None):
        coro.close()
        queued.append(name or "task")
        return SimpleNamespace()

    monkeypatch.setattr(shard_cfg, "is_sharding_active", lambda: False)
    monkeypatch.setattr(mod, "draw_group_allowed", lambda _gid: True)
    monkeypatch.setattr(mod, "draw_group_cooldown_ready", AsyncMock(return_value=True))
    monkeypatch.setattr(
        mod,
        "active_image_gen_settings",
        lambda: SimpleNamespace(api_backends=lambda: [backend]),
    )
    monkeypatch.setattr(mod, "draw_should_count_usage", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(mod, "SUPERUSER", AsyncMock(return_value=False))
    monkeypatch.setattr(mod, "acquire_draw_pending_slot", AsyncMock(return_value=True))
    monkeypatch.setattr(mod, "consume_draw_group_cooldown", AsyncMock())
    monkeypatch.setattr(mod, "run_pallas_draw_queued", fake_run_queued)
    monkeypatch.setattr(mod.asyncio, "create_task", fake_create_task)
    monkeypatch.setattr(mod.pallas_draw, "send", fake_send)

    event_a = _make_event(
        body="牛牛画画 小羊",
        self_id=2927116873,
        time=1780560720,
        group_id=934988865,
        user_id=1581408000,
        message_id=1001,
    )
    event_b = _make_event(
        body="牛牛画画 小羊",
        self_id=2927116873,
        time=1780560780,
        group_id=934988865,
        user_id=1581408000,
        message_id=1002,
    )

    await mod.pallas_draw_handle(
        SimpleNamespace(self_id="2927116873"), event_a, Message("小羊")
    )
    await mod.pallas_draw_handle(
        SimpleNamespace(self_id="2927116873"), event_b, Message("小羊")
    )

    assert sent == ["欢呼吧！", "欢呼吧！"]
    assert len(queued) == 2
