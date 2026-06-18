from src.platform.ai_callback.media_task_hooks import register_media_task_hooks
from src.platform.ai_callback.task_types import DRAW_IMAGE_TASK_TYPE
from src.features.plugin_storage.startup import register_plugin_storage_startup_hook

from .media_callback import on_draw_media_task_failed, on_draw_media_task_success

register_plugin_storage_startup_hook()
register_media_task_hooks(
    DRAW_IMAGE_TASK_TYPE,
    on_failure=on_draw_media_task_failed,
    on_success=on_draw_media_task_success,
)
