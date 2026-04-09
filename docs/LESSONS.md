# LESSONS

## 规则

- 本地 CLI 厂家的可用状态必须来自运行时探测，不要从静态配置、导出包或 API Key 字段推断。
- 导入脚本只负责恢复导出快照；当前默认模型和厂家 seed 应直接进入安装导出包，不要在导入路径里隐式补种。
- 需要真实外部 CLI 或联网环境的 smoke 脚本不要放在 `tests/` 下，避免 pytest discovery 在 collection 阶段触发真实调用。
- 给 Codex CLI 生成 `response_format` schema 时，对象字段要走严格 schema：`additionalProperties: false`，全部属性进入 `required`，可选参数改成 nullable。
- Codex 模型测试应走本地 CLI 检查路径，不应要求 `api_key` 或 `api_base`。
- Codex 的连接/schema错误不能归类成 API Key 错误；先区分协议错误，再区分凭证错误。
- 本地 CLI provider 的 `backend_url` 只表示推理入口，不能直接复用为 embedding 等 OpenAI-compatible HTTP 能力的端点。
- 向量化初始化优先按厂家默认分支处理；当前厂家没有默认 embedding 模型时直接禁用，等厂家真实支持后再补默认值。
- 同一套 MongoDB 连接信息只能有一个真相源；若同时保留 `MONGODB_CONNECTION_STRING` 和 `MONGODB_USERNAME/PASSWORD/...`，两者必须完全一致。
- 自选股当前的真实持久化集合是 `user_favorites`；`users.favorite_stocks` 只能视为旧模型兼容字段，排障和新逻辑都应先查 `user_favorites`。
- 分析默认模型变更要直接迁移 `system_configs` 和 `config/settings.json` 里的真相源，不要长期依赖前端对旧默认值做 normalize 兼容。
- 第三方接口出现解析错误时，先补原始响应诊断信息，再决定是否增加 fallback；日志至少要能看到请求目标和响应体预览。
- 回退或重做改动时，只撤目标行为，不要顺手删改用户已有注释。
- 统一现有组件样式时，先验证插槽和数据渲染语义是否一致；不要为了对齐样式把真实值替换成占位值。
- 统计口径切换时，字段名可以保留，但语义必须在采集、落库、聚合、展示四层同时统一；`input_tokens` 现在固定表示未命中输入，`cached_input_tokens` 单独表示缓存命中输入。
- 新增价格字段时，配置页、目录页和导出数据都要补“加载兜底 + 编辑回填 + 保存写回”三条链路，只改 schema 不够。
- 调度器运行状态不能只存在 APScheduler 内存里；凡是用户可修改且期望重启后保留的任务状态，都要持久化并在启动时恢复。
- 使用统计页当前展示的是落库时保存的 `cost`，不是前端或统计服务实时重算；变更 token 语义而不迁移历史记录时，历史成本会和新口径 token 不一致。
- 同一功能同时存在 Streamlit `web/` 和 Vue `frontend/` 两套入口时，改动前先定位用户实际访问的页面和路由，再修改对应展示层。
- 分页功能必须把“前端页码参数透传接口、后端返回 total、查询层实际做 skip/limit”三层同时打通；只改分页控件不会生效。

## 事件

- 事件：曾尝试在 `scripts/import_config_and_create_user.py` 中自动补 `codex/gpt-5.4`。操作：撤回导入期补种，恢复脚本为纯导入，并把 Codex seed 直接写入安装包 `install/database_export_config_2025-11-13.json`。总结：快照导入和默认配置初始化是两件事，不能混在同一条导入路径里。
- 事件：曾把 Codex smoke 用例放进 `tests/`，会在 pytest collection 阶段触发真实 `codex exec`。操作：删除 `tests/test_codex_cli_simple.py`，改为手动脚本 `scripts/codex_cli_smoke.py`。总结：依赖外部命令和真实环境的验证应该是手动 smoke，不应该进入默认测试发现路径。
- 事件：Codex tool-call 输出最初沿用了普通 JSON Schema，可选参数没有进入 `required`，导致 `invalid_json_schema`。操作：在 `ChatCodexCLI._build_output_schema()` 中为 tool args 生成严格对象 schema，并增加覆盖 `start_date/end_date/curr_date` 的测试。总结：面向 Codex structured output 时，要按 Codex 的严格 schema 约束生成，而不是照搬常规 OpenAPI/JSON Schema 习惯。
- 事件：Codex 配置测试和错误分类最初沿用了远程 HTTP 模型的逻辑，导致“本地 CLI 未测通”和“schema 错误”被误报成 API Key 问题。操作：`ConfigService.test_llm_config()` 对 Codex 走 `_test_codex_cli()`，`ErrorFormatter` 对 `invalid_json_schema` 归类为大模型调用错误。总结：本地 CLI provider 需要独立的探测和报错路径，不能套远程 API provider 的凭证模型。
- 事件：Codex 记忆功能曾沿用通用 OpenAI embedding 路径，把 `local://codex-cli` 当成向量化端点，触发 `Connection error` 后只能回退空向量。操作：撤回能力透传和跨厂家 fallback 方案，在 `FinancialSituationMemory` 中将 `llm_provider=codex` 直接禁用记忆；未来若 Codex 提供默认 embedding，再单独补分支。总结：当前厂家没有默认 embedding 模型时，不要为了“兼容”去猜别的端点或复用别家能力。
- 事件：MongoDB 配置里同时保留了 `MONGODB_USERNAME/PASSWORD/...` 和独立的 `MONGODB_CONNECTION_STRING`，且两套凭据不一致，导致主应用和 `mongodb_storage` 走到不同认证路径。操作：排障时分别核对 URI 拼装和 `MONGODB_CONNECTION_STRING` 的实际值，并要求后续统一成一套凭据。总结：同一基础设施一旦出现多入口配置，就必须在运行前做一致性校验。
- 事件：分析页默认模型最初试图在前端把 `qwen-flash/qwen-turbo/qwen-plus/qwen-max` 临时 normalize 成 `gpt-5.4`。操作：删除前端兼容层，改用 `scripts/update_analysis_models.py` 直接迁移 `system_configs` 与 `config/settings.json`，并验证 `unified_config` 读取结果为 `gpt-5.4`。总结：默认配置升级应改真相源并做一次性迁移，不要把历史值兼容长期留在展示层。
- 事件：AKShare 股票列表报 `JSONDecodeError` 时，最初直接往 `get_stock_list()` 里加了 fallback 取数逻辑。操作：撤回 fallback，只保留请求 `url/status/content_type/body_preview` 的诊断日志，并把 `logger.exception` 收回为 `logger.error`。总结：第三方接口响应异常时，先把原始证据打全，再决定是否需要降级路径。
- 事件：回退 AKShare 诊断改动时误删了用户原本保留的注释。操作：恢复原注释，并把“回退只撤目标行为”固定为规则。总结：回退代码时不要把用户已有注释和表达一起清理掉。
- 事件：统计页为了统一“总成本”和其他指标的视觉样式，曾把它改挂到 `el-statistic` 的 `value` 插槽上，结果页面实际显示成 `0`。操作：改回静态 DOM 承载真实金额，同时复用 `el-statistic` 的类名结构和样式，再单独补图标与金额同行布局。总结：样式统一应优先复用稳定的结构和样式层，不要假设组件插槽能无损承接现有数据渲染。
- 事件：worktree 分支里的修复最初直接用 `cherry-pick` 合回 `main`，随后才把“先 commit、再合并回主分支”的流程补进仓库约束。操作：先在 worktree 分支提交独立 commit，再把 worktree 合并流程写入 `AGENTS.md`。总结：使用 worktree 开发时，合回主分支前要先在 worktree 固化独立提交，让合并边界和回溯边界都清晰。
- 事件：为缓存命中计费扩展 token 统计时，曾尝试同时删除 `input_price_per_1k` 并引入未命中/命中双价格，改动面过大。操作：回退为保留 `input_price_per_1k`，只新增 `cached_input_price_per_1k`，并把 `input_tokens` 明确为未命中输入。总结：涉及持久化 schema 的口径调整，优先做增量语义收敛，不要一次性清理兼容字段。
- 事件：模型目录编辑页最初只补了保存时的 `cached_input_price_per_1k`，打开已有目录编辑时仍可能留空。操作：在目录加载和编辑回填两处统一做 `cached_input_price_per_1k ?? input_price_per_1k` 归一化。总结：新增可编辑字段时，读取路径和保存路径必须成对检查，尤其是旧数据回填。
- 事件：定时任务管理页的暂停/恢复原先只调用 APScheduler，服务重启后状态丢失。操作：把 `paused` 状态写入 `scheduler_metadata`，并在应用启动后重新应用到调度器。总结：用户通过管理界面修改的运行态，如果预期跨重启保留，就必须有持久化真相源。
- 事件：分析报告的 `codex session` 最初只加到了 Streamlit 报告页，用户在 Vue 报告详情页里看不到。操作：确认真实入口是 `frontend/src/views/Reports/ReportDetail.vue` 和 `frontend/src/views/Analysis/SingleAnalysis.vue`，再把 `codex_role_sessions` 映射抽到前端公共工具统一渲染。总结：同名功能有多套前端时，先锁定真实页面，再决定改哪一层。
- 事件：使用统计页最初只改了前端分页控件，但请求仍只传 `limit`，后端也没有 `page/page_size/total` 语义，导致“记录很多但不能翻页”。操作：在 Vue 页、`frontend/src/api/usage.ts`、`app/routers/usage_statistics.py` 和 `app/services/usage_statistics_service.py` 一起补齐分页参数和总数返回。总结：分页问题本质上是前后端契约问题，不能只在展示层修。
