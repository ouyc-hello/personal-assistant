# Personal Assistant CLI

这是公开仓库中的**正式实现骨架**，与根目录被 `.gitignore` 忽略的 `RAG/`、`interview_docs/` 和 `config.py` 分离。不会在运行时导入或复制那些参考材料。

## 架构边界

- **LangChain**：文档加载、分块、Embedding、Retriever、RAG tools。
- **LangGraph**：路由、Agent loop、工具调用、审批 interrupt、checkpoint/resume。
- **PostgreSQL**：业务真源，保存对话、记忆生命周期、任务、工具运行、审批和审计。
- **pgvector**：PostgreSQL 中的可信记忆语义索引；索引不是事实真源。
- **Milvus**：文档 chunk 的大规模向量检索；每条向量保留文档、版本、页码和权限 scope metadata。
- **Neo4j**：实体、关系和多跳 Graph RAG。
- **Typer + Rich**：终端入口，不提供 WebUI。`PA_LLM_PROVIDER=fake` 时使用确定性假回复；设置为 `openai` 时，终端对话会调用 `PA_OPENAI_BASE_URL` 指定的 OpenAI-compatible 接口。

## 快速开始

```bash
cd assistant_app
python -m venv .venv
. .venv/bin/activate
# GPU（NVIDIA）安装本地 BGE；若只用 fake/OpenAI，可跳过这两条 torch/transformers 安装。
python -m pip install \
  torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install 'transformers>=4.40,<5.0' safetensors
python -m pip install -e '.[dev,openai]'
cp .env.example .env
# 使用本地 BGE 时设置：PA_EMBEDDING_PROVIDER=local，
# PA_EMBEDDING_MODEL=../bge-small-zh-v1.5，PA_EMBEDDING_DEVICE=cuda，
# PA_EMBEDDING_DIMENSION=512。
# 编辑 .env：填写本机 PostgreSQL 的 PA_DATABASE_URL，以及 Compose 中 MinIO、Neo4j 的密码。

# 不连接外部服务即可验证 CLI 和路由
python -m personal_assistant architecture
python -m personal_assistant route '我住在哪里？'
python -m personal_assistant chat '你好' --fake  # 可选：强制 Fake 模式
python -m personal_assistant health

# Linux Bash 菜单界面
./bin/assistant-bash.sh

# PostgreSQL 开发库已启动后，创建 ORM 开发表（生产使用 SQL migration）
python -m personal_assistant db-init

# 导入单个 Markdown/TXT/PDF 文件（本地 BGE 使用 GPU，需 Milvus 已启动）
python -m personal_assistant index ./notes/project.md

# 启动 Milvus、Neo4j 及其依赖（PostgreSQL 使用主机本地安装，GPU 仅用于本地 BGE，不需要为模型启动容器）
docker compose --env-file .env up -d --pull never
```

`health` 会尝试连接配置的三个后端；PostgreSQL 使用主机本地服务，Milvus/Neo4j 使用 Docker Compose。未启动服务时应显示 `unavailable`，不会自动创建数据或上传内容。

PostgreSQL 不由本项目的 Compose 管理。请先确保主机 PostgreSQL 已启动，并在 `.env` 中配置 `PA_DATABASE_URL`；如果启用了 pgvector，还需要在目标数据库执行 `CREATE EXTENSION IF NOT EXISTS vector;`。可以运行 `python -m personal_assistant db-init` 创建应用表。

本地 BGE 模型可以直接通过 `model.safetensors` 加载，不需要 Docker 或 7078 端口的 Embedding HTTP 服务。
模型目录需要至少包含 `config.json`、`model.safetensors`（或 `pytorch_model.bin`）和 tokenizer 文件；
当前 `bge-small-zh-v1.5/` 已满足该结构。在 NVIDIA GPU 环境中将 `PA_EMBEDDING_DEVICE=cuda`；也可以保留 `cpu` 运行。启动前可用下面的命令确认 PyTorch 已识别 CUDA：

```bash
.venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

GPU 版 PyTorch 的 CUDA runtime 与主机 NVIDIA 驱动兼容即可，不要求另外安装 CUDA Toolkit。

## Linux Bash 界面

`bin/assistant-bash.sh` 是一个轻量 Bash 菜单层，不重复实现 Agent/RAG 逻辑；它只调用现有 Python CLI。支持单次对话、REPL、路由分析、文档导入、健康检查、数据库初始化和架构查看。

```bash
cd assistant_app
./bin/assistant-bash.sh
```

脚本会优先使用 `assistant_app/.venv/bin/python`，否则使用系统 `python3`，并自动加载本地 `.env`。`.env` 不会被提交到 Git。

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
