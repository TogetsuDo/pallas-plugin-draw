import pytest

from pallas_plugin_draw import config as cfg_mod
from pallas_plugin_draw.runtime_state import acquire_draw_pending_slot, release_draw_pending_slot


@pytest.mark.asyncio
async def test_draw_pending_slot_limit() -> None:
    cfg = cfg_mod.get_draw_config()
    cfg_mod.image_gen_config.reload(cfg.model_copy(update={"pallas_image_draw_max_pending": 1}))
    try:
        assert await acquire_draw_pending_slot()
        assert not await acquire_draw_pending_slot()
        await release_draw_pending_slot()
        assert await acquire_draw_pending_slot()
        await release_draw_pending_slot()
    finally:
        cfg_mod.image_gen_config.reload(cfg)
