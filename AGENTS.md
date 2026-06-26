# AGENTS.md

## 项目

- **名称**：pallas-plugin-draw
- **类型**：Pallas-Bot 4.0 官方扩展（NoneBot 插件包）
- **Python**：3.12+
- **依赖**：`uv` · 运行时依赖 [Pallas-Bot](https://github.com/PallasBot/Pallas-Bot) `>=4.0`

## 本地开发

```bash
# 扩展仓内（需 sibling Pallas-Bot）
uv sync --group dev
uv pip install -e ../Pallas-Bot
uv pip install -e .
uv run ruff check src/
uv run pytest tests/ -q

# 或在本体仓联调（推荐）
cd ../Pallas-Bot
uv pip install -e ../Pallas-Plugin-Draw
uv run pytest ../Pallas-Plugin-Draw/tests/ -q
```

与本体联调：在本体仓库执行 `uv pip install -e ../Pallas-Plugin-Draw`，或在扩展仓根目录 `uv.toml` 中配置 `pallas-bot` 的 path 源。

## 约定

- 仅改 `src/`；通过 `pallas-bot` 公开 API（`src.features` / `src.platform`）访问内核，勿反向依赖本体 `src/plugins`。
- 分片协调在插件 `on_startup` 调用 `register_*_coord()`（见本体 `src/features/plugin_coord/`）。
- 提交前 `ruff check` + `ruff format --check` 通过。
- 维护 `CHANGELOG.md`（[Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)）：日常改动记入 `## [Unreleased]`，发布版本时归档到 `## [X.Y.Z]` 并打 `vX.Y.Z` git tag（触发 PyPI 发布）。
