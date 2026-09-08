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

当前提交建立配置、路由、三数据库健康检查和 LangGraph 最小骨架；业务迁移、真实 RAG 索引、可信记忆生命周期、工具审批将在后续里程碑实现。Fake 模式用于没有 LLM/数据库时的确定性回归。
