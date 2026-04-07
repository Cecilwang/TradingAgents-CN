# LESSONS

## 规则

- 本地 CLI 厂家的可用状态必须来自运行时探测，不要从静态配置、导出包或 API Key 字段推断。
- 导入脚本只负责恢复导出快照；当前默认模型和厂家 seed 应直接进入安装导出包，不要在导入路径里隐式补种。
- 需要真实外部 CLI 或联网环境的 smoke 脚本不要放在 `tests/` 下，避免 pytest discovery 在 collection 阶段触发真实调用。
- 给 Codex CLI 生成 `response_format` schema 时，对象字段要走严格 schema：`additionalProperties: false`，全部属性进入 `required`，可选参数改成 nullable。
- Codex 模型测试应走本地 CLI 检查路径，不应要求 `api_key` 或 `api_base`。
- Codex 的连接/schema错误不能归类成 API Key 错误；先区分协议错误，再区分凭证错误。

## 事件

- 事件：曾尝试在 `scripts/import_config_and_create_user.py` 中自动补 `codex/gpt-5.4`。操作：撤回导入期补种，恢复脚本为纯导入，并把 Codex seed 直接写入安装包 `install/database_export_config_2025-11-13.json`。总结：快照导入和默认配置初始化是两件事，不能混在同一条导入路径里。
- 事件：曾把 Codex smoke 用例放进 `tests/`，会在 pytest collection 阶段触发真实 `codex exec`。操作：删除 `tests/test_codex_cli_simple.py`，改为手动脚本 `scripts/codex_cli_smoke.py`。总结：依赖外部命令和真实环境的验证应该是手动 smoke，不应该进入默认测试发现路径。
- 事件：Codex tool-call 输出最初沿用了普通 JSON Schema，可选参数没有进入 `required`，导致 `invalid_json_schema`。操作：在 `ChatCodexCLI._build_output_schema()` 中为 tool args 生成严格对象 schema，并增加覆盖 `start_date/end_date/curr_date` 的测试。总结：面向 Codex structured output 时，要按 Codex 的严格 schema 约束生成，而不是照搬常规 OpenAPI/JSON Schema 习惯。
- 事件：Codex 配置测试和错误分类最初沿用了远程 HTTP 模型的逻辑，导致“本地 CLI 未测通”和“schema 错误”被误报成 API Key 问题。操作：`ConfigService.test_llm_config()` 对 Codex 走 `_test_codex_cli()`，`ErrorFormatter` 对 `invalid_json_schema` 归类为大模型调用错误。总结：本地 CLI provider 需要独立的探测和报错路径，不能套远程 API provider 的凭证模型。
