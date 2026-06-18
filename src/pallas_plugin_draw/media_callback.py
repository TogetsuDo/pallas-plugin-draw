"""draw 媒体任务 AI callback 收尾。"""

from __future__ import annotations

from typing import Any

from src.platform.plugin_runtime.resolve import import_plugin_submodule


def on_draw_media_task_failed(task: dict[str, Any]) -> None:
    runtime_state = import_plugin_submodule("draw", "runtime_state")
    runtime_state.record_ai_runtime_failure("draw_callback_failed")


def on_draw_media_task_success(
    task: dict[str, Any], image_bytes: bytes, group_id: int
) -> None:
    runtime_state = import_plugin_submodule("draw", "runtime_state")
    usage_store = import_plugin_submodule("draw", "draw_usage_store")
    image_api = import_plugin_submodule("draw", "image_api")

    runtime_state.record_ai_runtime_success()
    persist_user = task.get("user_id")
    if persist_user is not None:
        image_api.schedule_persist_generated_draw(
            image_bytes, int(group_id), int(persist_user)
        )
    if task.get("count_usage"):
        usage_user = task.get("user_id")
        if usage_user is not None:
            usage_store.bump_pallas_draw_usage((int(group_id), int(usage_user)), True)
