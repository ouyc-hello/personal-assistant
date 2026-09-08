# Personal Assistant CLI

这是公开仓库中的**正式实现骨架**，与根目录被 `.gitignore` 忽略的 `RAG/`、`interview_docs/` 和 `config.py` 分离。不会在运行时导入或复制那些参考材料。

## 架构边界

- **LangChain**：文档加载、分块、Embedding、Retriever、RAG tools。
- **LangGraph**：路由、Agent loop、工具调用、审批 interrupt、checkpoint/resume。
- **PostgreSQL**：业务真源，保存对话、记忆生命周期、任务、工具运行、审批和审计。
- **pgvector**：PostgreSQL 中的可信记忆语义索引；索引不是事实真源。
- **Milvus**：文档 chunk 的大规模向量检索；每条向量保留文档、版本、页码和权限 scope metadata。
- **Neo4j**：实体、关系和多跳 Graph RAG。
- **Typer + Rich**：终端入口，不提供 WebUI。

## 快速开始

```bash
cd assistant_app
python -m venv .venv
. .venv/bin/activate
pip install -e '.[dev]'
cp .env.example .env
# 编辑 .env，为 PostgreSQL、MinIO、Neo4j 设置本地密码，并将 PA_DATABASE_URL 中的密码改成相同值。

# 不连接外部服务即可验证 CLI 和路由
python -m personal_assistant architecture
python -m personal_assistant route '我住在哪里？'
python -m personal_assistant chat '你好' --fake
python -m personal_assistant health

# PostgreSQL 开发库已启动后，创建 ORM 开发表（生产使用 SQL migration）
python -m personal_assistant db-init

# 导入单个 Markdown/TXT/PDF 文件（默认 fake embedding，需 Milvus 已启动）
python -m personal_assistant index ./notes/project.md

# 需要数据库时（仅本机开发）
docker compose up -d
```

`health` 会尝试连接配置的三个后端；未启动服务时应显示 `unavailable`，不会自动创建数据或上传内容。

## 数据安全约束

- `.env`、密钥、日志和本地运行目录由 `assistant_app/.gitignore` 排除。
- 真实文档通过 `assistant index /path/to/docs` 显式传入，不把个人资料放进仓库。
- Milvus、Neo4j 和 PostgreSQL 的账号/地址只从环境变量读取。
- 提交前执行 `git diff --check` 和 `git status --short`，确认没有敏感文件。

## 当前范围

当前已完成 M1-M7 的可回归实现：

- CLI、确定性路由和 LangGraph 风格的最小 Agent 骨架。
- PostgreSQL 业务模型、Repository、状态迁移、幂等约束和审计。
- LangChain 文档加载、递归分块、Embedding 工厂、Milvus 文档索引与来源 metadata。
- 可信长期记忆生命周期：候选、来源/置信度、敏感信息拦截、冲突解决、版本 supersede、24 小时撤回、过期归档、删除审计，以及 pgvector 语义检索（SQLite fallback 仅用于测试）。
- Neo4j Graph RAG：实体/关系幂等写入、用户隔离、实体查找、多跳邻域、来源回链和 Fake 图后端。
- 统一检索：按意图选择后端，执行权限二次过滤、RRF 融合、跨后端去重、单后端故障降级，并在 Agent state 中保留 trace。
- 可靠执行：工具状态先落库，外部调用携带幂等键，输入哈希防止复用冲突；支持 `EXECUTED_SUCCESS_UNACK` 重启对账，以及审批快照/工作流版本校验。

`memory_records` 是可信记忆的关系真源；pgvector 只用于语义召回，默认只返回 `PUBLISHED` 记录，不能绕过状态、版本和权限规则。Neo4j 只保存可解释实体关系，不替代 PostgreSQL 任务/审批状态。工具执行和审批恢复通过 `personal_assistant.agent.execution` 提供业务层封装；真实外部工具只需实现幂等执行与对账接口。Fake 模式用于没有 LLM/数据库时的确定性回归。

运行 M1-M7 回归测试：

```bash
PYTHONPATH=src pytest -q tests
```
