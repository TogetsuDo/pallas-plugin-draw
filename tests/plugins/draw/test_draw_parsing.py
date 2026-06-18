from __future__ import annotations

from nonebot.adapters.onebot.v11 import Message, MessageSegment

from pallas_plugin_draw.draw import (
    extract_at_user_ids_from_messages,
    extract_image_urls_from_messages,
)


def test_extract_image_urls_from_messages_dedupes_and_keeps_order() -> None:
    a = Message()
    a.append(MessageSegment("image", {"url": "http://x/a.png"}))
    a.append(MessageSegment("image", {"file": "http://x/b.png"}))
    b = Message()
    b.append(MessageSegment("image", {"url": "http://x/a.png"}))
    b.append(MessageSegment("image", {"file": "http://x/c.png"}))

    assert extract_image_urls_from_messages(a, b) == ["http://x/a.png", "http://x/b.png", "http://x/c.png"]


def test_extract_at_user_ids_from_messages_accumulates() -> None:
    a = Message()
    a.append(MessageSegment.at("100"))
    b = Message()
    b.append(MessageSegment.at("200"))
    b.append(MessageSegment("at", {"qq": "bad"}))

    assert extract_at_user_ids_from_messages(a, b) == [100, 200]
