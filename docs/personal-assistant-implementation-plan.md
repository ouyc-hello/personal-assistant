# 个人助理 Agent 实现方案：CLI + LangChain/LangGraph + 可信记忆

## 1. 方案定位

这是一个面向单用户个人助理的 MVP/生产规划，目标不是做一个通用 RAG 问答，而是把 `interview_docs` 中沉淀的三块能力做成可演示、可迭代、可面试讲的系统：

1. 可信长期记忆：区分用户事实、偏好、任务、临时上下文和模型推断。
2. 行动与状态：支持工具调用、人机确认、重试恢复、重复回调去重。
3. CLI：以终端交互为主入口，支持对话、记忆、检索、任务、工具执行和调试。
4. 框架分工：LangChain 负责文档、索引、Embedding、Retriever 和 RAG Tools；LangGraph 负责 Agent 状态、工具调用、审批中断、Checkpoint 和恢复；业务层负责可信性和状态规则。

一句话定义：

> 个人助理围绕“对话、记忆、行动”三个闭环运行；成熟框架负责通用能力，业务层先保证状态可信；记忆先验证再沉淀（非阻塞可撤回），行动先留痕再执行，状态先持久化再恢复。

## 2. 用户目标与核心场景

### 2.1 目标用户

- 个人用户：希望助理记住家庭/工作/生活信息，并代为安排提醒、纪要、报销、采购等动作。
- 进阶用户：希望看到记忆来源、执行轨迹，能确认、修改、删除任何系统记录。

### 2.2 首批场景

| 场景 | 助理要做的事 |
| --- | --- |
| 记住事实 | 用户说“我住上海，喜欢骑行”，系统自动发布到记忆，并在侧边栏提供 24 小时撤回/修正 |
| 记住偏好 | 用户说“以后报告都要中文、简洁”，写入 communication_style 偏好 |
| 短期任务 | 用户说“周五提醒我提交报销”，创建带过期时间的任务记忆 |
| 工具执行 | 用户说“帮我查杭州未来三天天气”，调用天气工具并显示 Trace |
| 人机确认 | 用户说“帮我提交一笔采购审批”，生成快照、确认后提交 |
| 审计回看 | 用户查“助理为什么记住我住上海”，查看来源、证据、状态和历史版本 |

## 3. 总体架构

```text
┌──────────────────────────────────────────────────────────────┐
│ Terminal CLI                                                  │
│ Typer + prompt_toolkit + Rich                                 │
└───────────────┬──────────────────────────────────────────────┘
                │
┌───────────────▼──────────────────────────────────────────────┐
│ LangGraph Agent Runtime                                      │
│ 路由 → 检索 → Agent 循环 → 工具 → 审批中断 → 恢复 → 回复      │
└───────┬──────────────────┬────────────────┬──────────────────┘
        │                  │                │
┌───────▼────────┐ ┌───────▼───────┐ ┌──────▼──────────┐
│ LangChain RAG  │ │ Tool Layer    │ │ Business Layer  │
│ Loader/Index/   │ │ Tool schemas  │ │ 可信记忆/任务/   │
│ Retriever/Tool  │ │ and executor  │ │ 审计/权限       │
└───────┬────────┘ └───────┬───────┘ └──────┬──────────┘
        │                  │                │
        └──────────────────┴────────────────┘
                        │
        PostgreSQL = 业务真源；pgvector = 可信记忆语义检索
        Milvus = 文档知识向量检索；Neo4j = 实体关系与 Graph RAG
```

职责边界：

```text
LangChain：文档加载、文本分块、Embedding、Milvus/pgvector Retriever、RAG Tools
LangGraph：路由、Agent 循环、工具调用、审批中断、Checkpoint、失败恢复
Pydantic：结构化输出、工具参数、Agent State、记忆候选和任务模型
业务层：可信长期记忆、记忆冲突、任务状态、审计、权限控制
```

LangChain 负责“能力组件”，LangGraph 负责“执行流程”，业务层负责“可信性和状态规则”。框架只提供通用能力，不替代记忆生命周期、工具权限、审批快照、幂等去重和审计等业务约束。

## 4. 框架边界与参考适配清单

### 4.1 框架原则：组件交给成熟框架，业务规则留在核心

“使用成熟框架”不等于把状态机、可信性和持久化全部交给框架；否则项目会退化成一个无法解释和恢复的黑盒 Agent。本方案按以下边界修订：

| 领域 | 保持核心 | 作为插件 |
| --- | --- | --- |
| LLM 对话/提取/Embedding | 结构化输出 Schema、source/confidence 契约 | LangChain model/embedding 接口 |
| 文档知识库 | 文档权限、文档版本、重建策略、来源审计 | LangChain loader/splitter/vector store/retriever |
| Agent 流程 | 业务路由、工具权限、最大步数、止损 | LangGraph StateGraph/checkpointer |
| 记忆 | 自动发布与撤回门禁、版本、冲突、过期、审计 | LangChain Retriever 只负责候选召回 |
| 工具 | 三分类状态机、Trace、幂等、熔断 | LangChain tools + LangGraph tool node |
| 审批 | 快照哈希、去重、状态轮询恢复 | LangGraph interrupt/resume + 业务状态表 |
| 用户体系 | 单用户权限、审计、敏感数据策略 | CLI 配置和业务层；多用户后再扩展 |

不交给框架的核心边界：记忆自动发布与撤回门禁、工具三分类状态机、审批快照与回调去重、数据库状态轮询恢复、审计和权限本身。若生产阶段启用事件溯源/Outbox，其状态机仍留在业务层。

业务组件统一契约：

```text
Pydantic model
  input schema / output schema / state schema / error schema
工具契约
  name / description / input_schema / permission / idempotency_policy
失败约定
  provider 报错 -> LangGraph 状态节点记录错误，不直接跳过业务状态机
```

MVP 不引入动态插件市场，也不把每个框架组件包装成插件。需要替换模型、Embedding、向量库或工具时，通过 LangChain 的统一接口和配置切换；业务状态只写核心数据库，错误必须进入 LangGraph state 和审计记录。

### 4.2 现有 RAG 代码：参考实现，不直接复制

`RAG/` 目录是学习和验证材料，不直接复制其中的应用脚本。正式实现统一使用 LangChain 的 loader、splitter、embedding、vector store 和 retriever；现有示例用于比较分块、混合检索、查询改写、重排和图 RAG 的效果。

| `fuyong` 组件 | 处理方式 | 本方案用途 |
| --- | --- | --- |
| 文档加载与分块示例 | 迁移验证 | 对照 LangChain loader/splitter 验证 Markdown、TXT、PDF 处理后写入 Milvus |
| 混合检索示例 | 参考实现 | 迁移为 LangChain Retriever 组合，保留 RRF、metadata filter 和 trace |
| 查询改写与路由 | 参考实现 | 迁移为 LangGraph 节点，不把查询改写结果直接当作工具参数 |
| 重排与评估 | 参考实现 | 在 LangChain Retriever 后增加 reranker 和离线评估 |
| Graph RAG | 后续扩展 | 只有出现多跳关系查询时再引入 Neo4j |

### 4.3 长期记忆：Retriever 可复用，生命周期自行建模

| 能力 | 适配结论 |
| --- | --- |
| LangChain Retriever | 只负责召回候选记忆，并附带 `memory_id/status/source/confidence` metadata |
| 向量库 | 可信记忆使用 PostgreSQL + pgvector；文档知识使用 Milvus |
| Memory Service | 负责候选提取、确认门禁、冲突、版本、过期、删除和审计 |
| LangGraph state | 保存当前会话和任务上下文，不作为长期记忆真源 |

注意：任何向量库或 Retriever 都不能成为可信记忆状态机的真源；业务关系表才保存 `PENDING/PUBLISHED/SUPERSEDED/EXPIRED/DELETED`、`source_chain`、确认和版本。

### 4.4 生产依赖

```text
langchain
langchain-core
langchain-community
langchain-openai
langchain-huggingface
langchain-text-splitters
langgraph
pydantic
pydantic-settings
typer
prompt-toolkit
rich
sqlalchemy
psycopg[binary]
pgvector
pymilvus
neo4j
httpx
python-dotenv
```

MVP 不再把 FAISS/Chroma 作为正式生产后端；它们可以保留为离线评估依赖，但公开实现的目标后端固定为 PostgreSQL/pgvector、Milvus 和 Neo4j。

### 4.5 三存储职责与数据边界

本项目同时引入 Milvus、Neo4j 和 PostgreSQL + pgvector，但三者不承担相同职责，也不互相复制业务真相：

```text
PostgreSQL       = 业务系统真源、事务、状态机、审计、LangGraph checkpoint 元数据
pgvector         = PostgreSQL 内的可信记忆语义索引，索引失效可由关系记录重建
Milvus           = 大规模文档分块向量检索，不保存记忆生命周期和审批状态
Neo4j            = 实体、关系、多跳路径和 Graph RAG，不替代任务/审批表
```

| 系统 | 首批数据 | 必须遵守的边界 |
| --- | --- | --- |
| PostgreSQL | `threads`、`messages`、`memory_records`、`memory_candidates`、`tasks`、`tool_runs`、`approval_requests`、`audit_events`、文档元数据 | 所有状态迁移、幂等约束、权限判断和审计在事务中完成 |
| pgvector | 已发布记忆和候选记忆的 embedding | 只做召回索引；以 `memory_records.status`、版本和权限过滤，不能单独决定“可信” |
| Milvus | Markdown/TXT/PDF 等文档 chunk embedding | 每条向量必须带 `document_id/chunk_id/version/scope/source/page`；删除和重建由 PostgreSQL 元数据驱动 |
| Neo4j | `Person`、`Project`、`Task`、`Document`、`Policy` 等节点及关系 | 只保存可解释的实体关系；任务状态和审批状态仍以 PostgreSQL 为准 |

LangGraph 路由按问题类型选择后端：记忆问题走 PostgreSQL 过滤 + pgvector，文档问题走 Milvus，关系/多跳问题走 Neo4j；无法确定时执行受权限约束的混合检索，并在 CLI trace 中显示实际命中的后端。MVP 先实现适配器接口和健康检查，再逐步接入真实索引，避免在业务层散落数据库 SDK 调用。

## 5. CLI 规划

### 5.1 交互入口

```text
assistant                 # 启动交互式终端
assistant chat "问题"     # 单次对话
assistant index ./docs     # 导入或重建知识库
assistant memories         # 查看长期记忆
assistant tasks            # 查看任务和提醒
assistant runs             # 查看工具执行轨迹
assistant debug           # 查看最近一次路由、检索和工具过程
```

交互式终端使用 `Typer + prompt_toolkit + Rich`：支持历史输入、补全、多行消息、Markdown 渲染、表格和可选调试输出。不引入 WebUI、REST、SSE 或浏览器依赖。

### 5.2 REPL 命令

```text
/help                         显示帮助
/memory <query>               搜索已确认的个人记忆
/memories                     列出记忆及状态
/remember <content>           手动创建记忆候选
/forget <memory_id>           撤回或删除记忆
/tasks                        查看任务和提醒
/runs                         查看工具执行轨迹
/reload <path>                导入或重建本地文档索引
/sources                      显示本轮回答引用来源
/debug                        显示路由、检索、节点和工具信息
/quit                         退出
```

### 5.3 终端输出原则

- 默认只显示用户可读的回答、必要的来源和待确认操作。
- 使用 `/debug` 查看 LangGraph 节点、检索命中、工具参数摘要和错误码。
- 记忆候选显示来源、置信度、状态和撤回期限；高风险操作必须在终端明确确认。
- 工具失败、审批中断和恢复操作必须显示当前状态，不把异常静默吞掉。

## 6. 后端模块设计

### 6.1 建议目录

```text
assistant_app/
├── pyproject.toml                 # 可安装 CLI 与成熟框架依赖
├── docker-compose.yml             # PostgreSQL/pgvector、Milvus、Neo4j 本地基础设施
├── .env.example                   # 仅配置模板，不含真实密钥
├── src/personal_assistant/
│   ├── cli.py                     # Typer + Rich 终端入口
│   ├── settings.py                # 环境变量配置（不使用根目录 config.py）
│   ├── schemas.py                 # Pydantic 契约与 Agent state
│   ├── agent/graph.py             # LangGraph 最小执行图
│   ├── rag/router.py              # 检索路由
│   └── storage/
│       ├── contracts.py           # 记忆/文档/图检索接口
│       ├── backends.py            # 后端选择与职责边界
│       ├── postgres.py            # PostgreSQL + pgvector 健康检查/适配器入口
│       ├── milvus.py              # Milvus 健康检查/适配器入口
│       ├── neo4j.py               # Neo4j 健康检查/适配器入口
│       └── migrations/            # PostgreSQL 业务表和 pgvector 扩展
└── tests/                         # Fake 模式和路由回归测试
```

### 6.2 Assistant Graph

```text
输入：user_message + user_id + thread_id + max_loop_steps = 5

0. 建立 ChatRun / Trace 根节点
   - chat_id、model_id、启用的工具集合、loop_step

1. ReAct / Plan-Execute 循环
   a. 构造当前上下文
      - 当前 Thread 消息
      - 已检索的 PUBLISHED 记忆
      - 本轮已执行工具结果（作为 tool result 回填，可再次触发检索）
      - 当前有权限的 LangChain Tool 定义

   b. 调用 LangChain ChatModel 输出结构化结果
      - content / tool_calls / memory_candidates / loop_decision
      - 同一模型可重试；若重试时工具参数变化，按 PARAM_ADJUSTMENT 处理

   c. 二次/多次记忆检索
      - 工具结果包含新实体、与已检索记忆冲突、或 LLM 判断需要补充上下文时，
        把工具结果/当前意图转成检索 query，再次经 LangChain Retriever 检索

   d. 执行工具（有 tool_calls 时）
      - 将工具调用绑定 tool_name 和 invocation_id；写入 ToolRun；按三分类状态机执行
      - 工具结果回填上下文；执行结果触发记忆候选时，本轮内生成候选并发布/待复核
      - loop_step += 1；若未结束则回到 1

   e. 生成最终回复
      - 无 tool_calls、达到 max_loop_steps 或 LLM 明确收尾时，输出最终回复

2. LangGraph state 事件
   - token / loop_step / memory_candidates / tool_run / done
   - 断线重放按 messages.event_seq + chat_id 关联的 run/memory 记录补齐

3. 持久化
   - messages、tool_runs、memory_records、audit
```

`AssistantGraph` 不直接依赖具体模型或向量库；它只通过 LangChain ChatModel、Retriever 和 Tool 接口工作，业务状态写入自己的 repositories。

### 6.3 Memory Service

写入流水线严格遵循 interview_docs，但 MVP 不做聊天内阻塞确认：

```text
原始消息/工具结果
  -> 提取候选记忆
  -> 类型分类：user_fact / preference / task / context
  -> 来源标注：USER / LLM_INFERENCE / TOOL / MANUAL
  -> 冲突检测：属性冲突 + 语义相似 + 过期判断
  -> MVP 发布策略
     - 高置信且低敏感：直接 PUBLISHED，写入 published_at + withdraw_deadline（24h）
     - 低置信/敏感/可推翻：写入 PENDING 待复核，只进侧边栏被动通知，不阻塞聊天
     - PENDING 设置过期归档，避免长期堆积成垃圾
  -> 长文本分块
  -> 已发布记忆写入 pgvector，PENDING 候选带 status 检索但默认过滤
  -> 撤回/修正窗口
     - 24h 内撤回：status -> DELETED / SUPERSEDED，写 memory_events 审计
     - 24h 后更正：走正式编辑流程，记录 evidence / reason / actor
  -> metadata/距离过滤检索，默认只返回 PUBLISHED
```

`MemoryService` 不把 Retriever 当作 CRUD：写入、检索过滤、候选提取、冲突和生命周期由业务层负责；LangChain Retriever 只提供候选召回，`memory_records` 仍是业务事实、版本和审计的真源。

### 6.4 Tool Executor

每个工具调用都复用“三分类状态机”：

```text
NOT_EXECUTED                # 未执行
  -> validated/acknowledged

EXECUTED_FAILED             # 已执行但失败
  -> retryable / terminal

EXECUTED_SUCCESS_UNACK      # 已成功但未回传
  -> reconcile_by_idempotency_key
  -> COMPLETED

SUPERSEDED                  # 被语义重试替换，仅 Trace 保留
```

每次转移都追加到 `tool_runs.trace`（有序 JSONB），`tool_runs` 同时保存当前状态和最新 Trace 字段。MVP 不单独建 `tool_events` 表；重试、断电恢复、人工接管直接读同一行有序 Trace 决策。生产阶段若引入事件溯源，再把 trace 投影到独立事件表或 Outbox。

语义重试与参数修正规则：

```text
同参重试：同一 invocation_id + attempt + 1，沿用原 idempotency_key
参数修正：retry_decision = PARAM_ADJUSTMENT
  - 旧 run 标记 SUPERSEDED，保留 side_effect_status，不再用旧 input_hash 重放
  - 新 run 使用新 invocation_id + 新 input_hash
  - 新 run.parent_run_id = 旧 run.id，Trace/UI 保持父子关系
  - 若旧 run 可能是 EXECUTED_SUCCESS_UNACK，先对账旧调用，再开启新调用
```

`ToolExecutor` 不硬编码工具。工具通过 LangChain Tool 注册，工具契约提供 `name / description / input_schema / idempotency_policy / runtime_permissions`；execute 前后仍使用核心三分类状态机，组件抛出异常只写入 `audit_events` 与 `tool_runs.trace`。

### 6.5 Approval Service

MVP 审批采用“数据库状态轮询 + 唯一索引去重”，不默认引入 `approval_events` 和 `outbox`：

```text
状态                                    触发条件
DRAFT -> READY_FOR_SUBMIT               confirmation token + snapshot_hash + workflow_version
READY_FOR_SUBMIT -> SUBMITTING          用户提交
SUBMITTING -> SUBMITTED                 callback/轮询确认外部已受理
SUBMITTED -> APPROVED / REJECTED        审批结果
APPROVED -> NOTIFIED / NEEDS_REVISION   通知结果或需要修正
```

状态迁移直接更新 `approval_requests`；回调写 `callbacks`，靠 `(request_id, source, external_event_id)` 唯一索引去重。提交/催办由后台 worker 扫描 `SUBMITTING` / `PENDING_APPROVAL`，按状态和快照查询外部系统。生产阶段需要可靠消息投递时，再加 `outbox`、`approval_events` 和 `OutboxPublisherPlugin`，插件接口本期保留但默认不启用。

`approval_requests.request_type / workflow_version` 绑定 `WorkflowPlugin`，外部提交通过 `IntegrationCallbackPlugin` 归一化为规范化回调；快照哈希、去重和恢复状态机仍由核心实现。

## 7. 数据模型

### 7.1 核心实体

```text
threads             对话线程和 LangGraph thread_id
messages            用户消息、助手消息、工具消息
memory_records      可信长期记忆及其生命周期
memory_candidates   待确认记忆候选
knowledge_documents 导入的文档
knowledge_chunks    文档分块和来源 metadata
tasks               提醒和短期任务
tool_runs           工具调用、重试、错误和结果摘要
approval_requests   需要用户确认的操作快照
audit_events        不可抵赖的操作和状态变更记录
```

LangChain 向量库只保存 `memory_record_id` 或 `chunk_id`、版本、来源和权限等 metadata；关系型存储仍是业务状态真源。LangGraph checkpoint 保存线程级状态，不替代 `memory_records`。

### 7.2 关键状态

```text
memory:    CANDIDATE → PUBLISHED → SUPERSEDED/EXPIRED/DELETED
tool_run:  CREATED → EXECUTING → FAILED/SUCCESS_UNACK → COMPLETED
approval:  DRAFT → PENDING_CONFIRMATION → SUBMITTING → ACCEPTED/REJECTED/UNKNOWN
```

所有状态迁移必须经过业务层校验并写入 `audit_events`；LangGraph 节点不能绕过状态迁移直接修改业务结果。

## 8. CLI 与 Agent 契约

### 8.1 对话执行

```text
assistant chat <message>
  → 创建/恢复 thread_id
  → LangGraph route 节点分析意图
  → memory_retriever / knowledge_retriever 检索
  → Agent 节点决定回答或调用工具
  → tool 节点执行并写入 tool_runs
  → 需要确认时 interrupt
  → 生成回答、来源和状态摘要
```

### 8.2 终端恢复

```text
assistant resume <thread_id>
  → 读取 LangGraph checkpoint
  → 展示待确认操作和快照摘要
  → 用户确认/拒绝
  → resume graph
```

### 8.3 调试契约

`/debug` 或 `assistant debug <thread_id>` 至少显示：`thread_id`、路由结果、命中的 memory/document ID、LangGraph 节点、工具名称、attempt、状态、错误码和最终来源。默认不显示完整敏感参数。

## 9. 记忆可信性设计

### 9.1 什么可以写

- 用户明确说出的个人事实。
- 高置信的模型推断（自动 PUBLISHED，24 小时内可撤回）。
- 工具返回值中的可验证事实。
- 用户偏好、常用上下文、长期任务。

### 9.2 什么不能直接写

- 模型对用户情绪/身份的猜测。
- 单次行为推导出的稳定偏好。
- 过期假设、临时工作状态、敏感凭据。
- 从“用户没反对”推断出的同意。

### 9.3 更新策略

```text
新候选 -> 检测冲突
  -> 无冲突且高置信、低敏感：自动 PUBLISHED + withdraw_deadline（24h）
  -> 低置信/敏感/可推翻：PENDING 待复核，只进侧边栏，不阻塞聊天
  -> 有冲突：展示新旧证据，先不强发布，让用户在侧边栏选择
  -> 撤回/更正后：旧版本 SUPERSEDED，新版本 version + 1，写 memory_events
```

任何修改、删除都要写 `memory_events`，保证可审计。

### 9.4 防止模型猜测伪装成事实

```text
1. source 必须区分 USER / LLM_INFERENCE / TOOL。
2. 提取器输出 JSON Schema，必须包含 source 和 confidence。
3. 默认不把 low confidence 当作事实。
4. 写入前用自动发布阈值和敏感规则拦截，低置信进入待复核。
5. UI 明确展示“已发布可撤回”或“待复核”，不做聊天阻塞式确认。
6. 用红队用例验证：模型把猜测写进记忆时，要么停在 PENDING，要么发布后 24 小时内被撤回。
```

## 10. 可靠性与容错设计

### 10.1 工具调用

```text
连续失败 4 次处理：
1. 查看 Trace：status / error / attempt / idempotency_key / side_effect
2. 分类：
   - NOT_EXECUTED：安全重试或参数修正
   - EXECUTED_FAILED：按错误码重试，超过阈值熔断
   - EXECUTED_SUCCESS_UNACK：先查询外部状态，再决定重试/补偿
3. 止损：
   - max_attempts = 4
   - 指数退避
   - 打开熔断器
   - 告知用户并转人工
4. 回归：
   - 幂等重放
   - 超时无回传
   - 服务重启恢复
   - schema 变化
   - LLM 语义重试修改参数时，旧 run 标 SUPERSEDED，新 run 使用新 invocation_id + parent_run_id，
     不把旧 input_hash 当作新调用幂等键
5. 组件兜底：
   - 调用前检查模型、Retriever 和工具可用性，不可用则不进入执行
   - 组件抛错只写 `audit_events` 与 `tool_runs.trace`，不绕过三分类状态机
```

### 10.2 审批确认

| 场景 | 方案 |
| --- | --- |
| 用户确认后改金额 | `snapshot_hash` 变化，原 `confirmation_token_hash` 不再匹配，拒绝自动提交 |
| 审批节点变化 | 比较 `workflow_version`，重新拉取流程定义并让用户确认 |
| 工作流版本变化 | `workflow_version` 不匹配时拒绝自动提交，先重新生成快照并让用户确认 |
| 服务重启 | 按 recovery index 限流扫描 `EXECUTED_SUCCESS_UNACK` / `SUBMITTING` / `PENDING_APPROVAL`；先查询外部状态，再按唯一索引恢复；MVP 不依赖 Outbox |
| 重复回调 | `(request_id, source, external_event_id)` 唯一索引去重，并按 `event_sequence` 判断乱序 |

### 10.3 长文本与记忆融合

```text
长文本 -> 按 section/paragraph 分块
       -> 每块单独 embedding
       -> 保留 memory_id 和 chunk_index
       -> 检索后按 memory_id 聚合
       -> 业务层只合并规范化事实，不直接吞并全文
```

`memory_records` 是业务事实、版本和审计的来源；FAISS/Chroma/pgvector 只是 LangChain Retriever 使用的向量索引，chunk 的 metadata 必须保存 `memory_record_id / version / chunk_index`，不允许让任何 vector store 直接成为状态机的真源。

## 11. CLI 技术要求

- 使用 `Typer` 暴露一次性命令，使用 `prompt_toolkit` 提供交互式 REPL。
- 使用 `Rich` 渲染 Markdown、表格、状态面板、来源和工具执行结果。
- 默认隐藏框架内部细节；`/debug` 显示 LangGraph 节点、检索结果、工具调用和错误码。
- 审批、记忆确认和删除操作必须在终端明确显示并等待确认。
- 长任务显示节点进度和当前状态；进程退出后根据 `thread_id` 和 checkpoint 恢复。

## 12. 监控与指标

### 12.1 Agent 指标

```text
chat_latency_p50/p95
memory_extraction_time
memory_withdraw_rate
memory_disabled_rate
```

### 12.2 工具指标

```text
tool_run_total
tool_run_failed_total
tool_run_unack_total
retry_attempts
circuit_breaker_state
```

### 12.3 审批/恢复指标

```text
approval_submit_success_rate
stale_run_age
stale_approval_age
callback_duplicate_total
manual_recovery_total
```

### 12.4 框架与组件指标

```text
retriever_empty_result_total
embedding_failure_total
graph_checkpoint_resume_total
graph_node_failure_total
security_denied_total
```

### 12.5 审计

- 所有记忆、审批、工具操作写 audit log。
- 模型、Embedding、Retriever、工具和 Graph 节点的失败、恢复、权限拒绝写入 audit log。
- 保留 source_chain、证据、版本和操作者。
- 删除用户数据时，同时清理会话、记忆、向量索引和审计里的关联引用。

## 13. 测试计划

### 13.1 记忆测试

| 用例 | 期望 |
| --- | --- |
| 用户明确说“我住杭州” | 提取为 user_fact，高置信自动 PUBLISHED，证据包含原始消息 |
| 模型猜测“用户喜欢旅游” | 低置信进入 PENDING 待复核，不阻塞聊天 |
| 24 小时内撤回 | 状态 DELETED/SUPERSEDED，写 memory_events，检索立即失效 |
| 用户更正 | 旧版本 SUPERSEDED，新版本 version+1 |
| 冲突记忆 | 不静默覆盖，展示新旧证据 |
| 过期 | 检索自动过滤，页面可见 EXPIRED |
| 删除 | 从检索和页面消失，审计保留 |

### 13.2 工具测试

| 用例 | 期望 |
| --- | --- |
| 参数校验失败 | NOT_EXECUTED，可重试 |
| 外部服务 500 | EXECUTED_FAILED，指数退避 |
| 响应超时 | EXECUTED_SUCCESS_UNACK，idempotency_key 查询 |
| 4 次连续失败 | 熔断打开，不再继续调用 |
| LLM 重试后修改参数 | 旧 run 标记 SUPERSEDED，新 run 使用新 invocation_id + parent_run_id，不重放旧 input_hash |
| 服务重启 | 按状态索引限流扫描并先做外部状态查询，再恢复 run |
| schema 变化 | 明确校验错误，不污染 Trace |
| Trace 重放 | 读 `tool_runs.trace` 重放 `VALIDATED/ACKED/STARTED/...`，恢复决策一致 |

### 13.3 审批测试

| 用例 | 期望 |
| --- | --- |
| 确认后改金额 | 提交被拒，要求重新确认 |
| 审批节点变化 | workflow_version 不一致，重新确认 |
| 服务重启 | 状态轮询恢复，重复提交被唯一索引与 token 校验拦截 |
| 回调重复 | 第二次回调返回首次处理结果 |
| 回调乱序 | 按事件序列号/时间戳判断接受或拒绝 |

### 13.4 框架与组件测试

| 用例 | 期望 |
| --- | --- |
| LangChain Retriever | 文档检索、metadata 过滤、来源信息和空结果行为符合预期 |
| LangGraph checkpoint | 中断后可恢复到正确节点，状态不会重复提交 |
| 工具权限 | 无权限工具不进入 Agent 上下文，记录 `SECURITY_DENIED` 事件 |
| 向量库替换 | FAISS -> Chroma/pgvector 后，记忆状态机和检索结果契约一致 |
| 确定性 Fake LLM/工具 | 不依赖真实服务即可回归路由、三分类状态机和审批流程 |
| 组件失败降级 | Retriever、Embedding 或工具失败时，状态机可记录错误并安全结束 |

### 13.5 当前落地状态（2026-09-08）

公开仓库当前处于 CLI-first 实现准备阶段：根目录 `.gitignore` 继续忽略 `RAG/`、`interview_docs/` 和 `config.py`，这些学习材料与敏感配置不作为运行时依赖，也不会上传。正式代码放在独立的 `assistant_app/` 目录，并只提交 `.env.example`、接口、迁移和测试。

本阶段先建立三存储适配器边界：PostgreSQL 保存业务真相，pgvector 保存可信记忆 embedding，Milvus 保存文档 chunk embedding，Neo4j 保存实体关系；LangChain/LangGraph 只依赖这些边界接口。真实数据库联调、Embedding 和外部工具在对应里程碑开启，Fake 模式用于无外部服务回归。

## 14. 里程碑

| 阶段 | 范围 | 验收 |
| --- | --- | --- |
| M0 | 基础设施与安全边界 | `assistant_app/` 可安装；`.env.example` 完整；真实 `.env`、本地数据和参考材料不会被提交 |
| M1 | CLI + LangChain/LangGraph 骨架 | `assistant` 可启动 REPL/单次对话；Fake LLM 可运行；产生 thread/run trace |
| M2 | PostgreSQL 业务层 | 完成 threads/messages/tasks/tool_runs/approvals/audit 迁移；状态迁移和幂等约束可测试 |
| M3 | Milvus 文档 RAG | Markdown/TXT/PDF 导入、分块、Embedding、Milvus 检索、来源展示和索引重建可用 |
| M4 | pgvector 可信记忆 | 候选提取、冲突、发布/撤回/过期/删除、关系真源与向量索引同步可测 |
| M5 | Neo4j Graph RAG | 实体关系抽取、幂等 upsert、实体查找、多跳检索和来源回链可用 |
| M6 | 统一路由与混合检索 | LangGraph 按意图选择记忆/Milvus/Neo4j，支持权限过滤、融合、重排和 debug trace |
| M7 | 可靠工具与审批恢复 | interrupt/resume、checkpoint、重试/止损、快照哈希、回调去重、重启对账可复现 |

## 15. 面试主讲逻辑

1. 先讲“这个方案不是纯聊天”：它把对话、记忆、工具、审批四件事统一到状态、Trace 和审计上。
2. 再讲框架边界：LangChain 负责能力组件，LangGraph 负责执行流程，Pydantic 负责结构化数据；可信记忆、状态机、权限和审计仍由业务层持有。
3. 再讲技术选型：CLI 使用 Typer/prompt_toolkit/Rich，RAG 使用 LangChain，Agent 编排使用 LangGraph；PostgreSQL 保存业务真相，pgvector 服务可信记忆，Milvus 服务文档知识，Neo4j 服务多跳关系。
4. 重点讲记忆：来源、置信度、高置信自动发布、24 小时撤回、冲突、过期、删除。
5. 重点讲工具：未执行/失败/成功未回传三分类、PARAM_ADJUSTMENT 语义重试、Trace 和止损。
6. 重点讲审批：快照哈希、数据库状态轮询、唯一索引去重、幂等回调、启动对账。
7. 最后收束：核心先可靠，外部能力才可插件化；状态先于智能、验证先于沉淀、对账先于重试。

## 16. 可直接使用的简历描述

> 基于 LangChain + LangGraph + PostgreSQL/pgvector/Milvus/Neo4j 构建终端个人助理：由 LangChain 提供文档加载、分块、Embedding、Retriever 和 RAG Tools，由 LangGraph 负责路由、Agent 循环、工具调用、审批中断、Checkpoint 和失败恢复，并通过 PostgreSQL 业务层实现可信长期记忆、来源证据、24 小时撤回、冲突检测、过期删除、工具 Trace 和审计；使用 Typer + prompt_toolkit + Rich 提供 CLI 交互，使用 pgvector 管理可信记忆、Milvus 管理文档向量、Neo4j 管理实体关系。
