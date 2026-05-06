# iris-cli M1: 记忆系统 — 规格说明书

**项目**: iris-cli  
**里程碑**: M1 - Memory System  
**版本**: 1.0  
**日期**: 2026-05-06  
**作者**: beloved-iris  

---

## 一、项目概述

iris-cli 是一个 Python CLI 工具，目标是构建 Agent 的核心能力基础设施。M1 聚焦**记忆系统**——Agent 一切能力的基础，也是当前最弱的维度（EntroCamp 记忆维度得分67/B）。

### 设计原则
1. **先落地后优化**：M1 只做存储+检索+衰减，不做图数据库、不做多 Agent 协同
2. **CLI-first**：每个能力都通过 CLI 可达，同时也是可导入的 Python 库
3. **本地优先**：数据全部存本地，零外部依赖（不需要 Docker、不需要云服务）
4. **可观测**：所有操作有日志、有统计、可导出

---

## 二、技术栈

| 组件 | 技术 | 版本 | 理由 |
|------|------|------|------|
| 语言 | Python | 3.10+ | 生态丰富，异步支持好 |
| CLI框架 | Typer | 0.12+ | 类型安全，Rich集成好 |
| 终端美化 | Rich | 13.0+ | 表格/进度条/Markdown渲染 |
| 向量数据库 | ChromaDB | 0.4+ | 嵌入式，零配置，本地运行 |
| 元数据存储 | SQLite | 内置 | 零依赖，FTS5全文搜索 |
| 异步框架 | asyncio | 内置 | Python原生，开发阶段够用 |
| 包管理 | uv | 最新 | 快速，lockfile支持 |
| 测试 | pytest | 8.0+ | 标准测试框架 |

**不使用**：Redis（过重）、Neo4j（M1不需要）、PostgreSQL（过重）、Docker（本地优先）

---

## 三、核心数据模型

### 3.1 Memory（记忆条目）

```python
@dataclass
class Memory:
    id: str                    # UUID
    content: str               # 记忆内容（自然语言）
    summary: str               # 自动生成的摘要（≤100字）
    memory_type: MemoryType    # episodic / semantic / procedural
    tags: list[str]            # 用户标签 + 自动标签
    weight: float              # 当前权重 [0.0, 1.0]，初始1.0
    source: str                # 来源标识（如 "conversation", "research", "heartbeat"）
    embedding: list[float]     # 向量嵌入（ChromaDB管理）
    created_at: datetime       # 创建时间
    accessed_at: datetime      # 最近访问时间
    access_count: int          # 访问次数
    status: MemoryStatus       # active / archived / decayed
    metadata: dict             # 扩展元数据
```

### 3.2 枚举类型

```python
class MemoryType(str, Enum):
    EPISODIC = "episodic"      # 事件/经历（如"今天做了代码审查"）
    SEMANTIC = "semantic"      # 知识/事实（如"Python GIL影响多线程性能"）
    PROCEDURAL = "procedural"  # 技能/方法（如"用Typer创建CLI的步骤"）

class MemoryStatus(str, Enum):
    ACTIVE = "active"          # 活跃，权重>0.3
    ARCHIVED = "archived"      # 归档，权重0.1-0.3
    DECAYED = "decayed"        # 衰退，权重<0.1，待清理
```

---

## 四、模块架构

```
iris_cli/
├── __init__.py
├── __main__.py               # python -m iris_cli 入口
├── cli/
│   ├── __init__.py
│   ├── main.py               # Typer app, 根命令
│   ├── memory.py             # memory 子命令组
│   └── utils.py              # CLI工具函数（输出格式化等）
├── memory/
│   ├── __init__.py
│   ├── models.py             # 数据模型（Memory, MemoryType等）
│   ├── store.py              # 存储层（ChromaDB + SQLite）
│   ├── decay.py              # 衰减引擎（艾宾浩斯曲线）
│   ├── search.py             # 检索引擎（语义+关键词+混合）
│   ├── consolidate.py        # 记忆整合（合并相似记忆）
│   └── embedder.py           # 嵌入生成（本地模型/API）
├── config.py                 # 配置管理
└── utils.py                  # 通用工具
```

---

## 五、存储层设计

### 5.1 ChromaDB（向量存储）

- 用途：语义搜索、相似度计算
- 集合名：`iris_memories`
- 嵌入模型：先用 `all-MiniLM-L6-v2`（sentence-transformers，本地运行，无API调用）
- 存储：`~/.iris/data/chroma/`

```python
class VectorStore:
    """ChromaDB向量存储封装"""
    
    def add(self, memory_id: str, content: str, metadata: dict) -> None: ...
    def search(self, query: str, top_k: int = 5) -> list[SearchResult]: ...
    def update(self, memory_id: str, content: str) -> None: ...
    def delete(self, memory_id: str) -> None: ...
```

### 5.2 SQLite（元数据存储）

- 用途：结构化查询、全文搜索、权重管理、统计分析
- 位置：`~/.iris/data/iris.db`
- 核心表：

```sql
CREATE TABLE memories (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    summary TEXT,
    memory_type TEXT NOT NULL,
    tags TEXT,  -- JSON array
    weight REAL DEFAULT 1.0,
    source TEXT,
    created_at TEXT NOT NULL,
    accessed_at TEXT NOT NULL,
    access_count INTEGER DEFAULT 0,
    status TEXT DEFAULT 'active',
    metadata TEXT  -- JSON object
);

-- 全文搜索虚拟表
CREATE VIRTUAL TABLE memories_fts USING fts5(
    id, content, summary, tags,
    content='memories',
    content_rowid='rowid'
);
```

---

## 六、衰减引擎

### 6.1 艾宾浩斯遗忘曲线

```
weight(t) = weight_0 × R(t)
R(t) = e^(-t/S)
S = S_0 × (access_count + 1)^0.5    # 访问越多，遗忘越慢
```

| 时间段 | 衰减率 | 说明 |
|--------|--------|------|
| Day 1 | -10% | 新记忆快速衰减 |
| Day 2-7 | -5%/天 | 中期衰减 |
| Day 8-30 | -1%/天 | 长期记忆缓慢衰减 |
| weight < 0.1 | 标记decayed | 可清理 |
| weight 0.1-0.3 | 标记archived | 归档但可检索 |

### 6.2 强化机制（访问时）

- 每次检索命中：weight += 0.1（上限1.0），access_count += 1
- 用户手动强化：`iris memory reinforce <id>`
- 强化后重置衰减起点：accessed_at = now()

### 6.3 CLI接口

```bash
# 手动运行衰减
iris memory decay

# 查看衰减统计
iris memory decay --stats

# 清理decayed记忆
iris memory decay --cleanup
```

---

## 七、检索引擎

### 7.1 三种检索策略

| 策略 | 实现 | 适用场景 |
|------|------|----------|
| 语义搜索 | ChromaDB向量相似度 | 模糊查询、概念关联 |
| 关键词搜索 | SQLite FTS5 | 精确匹配、标签过滤 |
| 混合搜索 | 语义×0.6 + 关键词×0.4 + 权重×0.3 | 默认策略，最佳平衡 |

### 7.2 检索结果排序

```python
final_score = (
    semantic_score * 0.6 +      # 语义相似度
    keyword_score * 0.4 +       # 关键词匹配度
    weight * 0.3                # 记忆权重加成
)
```

### 7.3 CLI接口

```bash
# 搜索（默认混合）
iris memory search "Python异步编程"

# 仅语义搜索
iris memory search "异步编程" --mode semantic

# 仅关键词搜索
iris memory search "asyncio" --mode keyword

# 限定类型和标签
iris memory search "记忆系统" --type semantic --tags architecture,design

# 限定时间范围
iris memory search "调研" --from 2026-04-01 --to 2026-05-01
```

---

## 八、记忆整合

### 8.1 整合逻辑

当多条记忆相似度 > 0.85 时，自动合并为一条：

1. 保留权重最高的记忆作为主记忆
2. 将其他记忆的内容合并到主记忆的metadata中
3. 主记忆权重 = max(所有权重) + 0.05
4. 标记被合并记忆为 `consolidated_into: <main_id>`

### 8.2 CLI接口

```bash
# 查看可整合的记忆对
iris memory consolidate --dry-run

# 执行整合
iris memory consolidate

# 整合指定类型
iris memory consolidate --type semantic
```

---

## 九、完整CLI命令

```bash
# 基础操作
iris memory add "内容" [--type episodic] [--tags tag1,tag2] [--source conversation]
iris memory get <id>
iris memory update <id> [--content "新内容"] [--tags +tag3]
iris memory delete <id>

# 检索
iris memory search <query> [--mode hybrid|semantic|keyword] [--type TYPE] [--tags TAGS] [--limit 5]
iris memory list [--type TYPE] [--status active|archived] [--sort weight|date] [--limit 20]
iris memory recent [--days 7] [--limit 10]

# 衰减与维护
iris memory decay [--stats] [--cleanup]
iris memory reinforce <id>
iris memory consolidate [--dry-run]

# 导入导出
iris memory export [--format json|markdown] [--output memories.json]
iris memory import <file> [--format json]

# 统计
iris memory stats
```

---

## 十、配置管理

配置文件位置：`~/.iris/config.toml`

```toml
[memory]
# 存储路径
data_dir = "~/.iris/data"

# 衰减参数
decay_rate_day1 = 0.10
decay_rate_week1 = 0.05
decay_rate_month1 = 0.01
decay_cleanup_threshold = 0.10
decay_archive_threshold = 0.30

# 强化参数
reinforce_boost = 0.10
max_weight = 1.0

# 检索参数
search_default_mode = "hybrid"
search_semantic_weight = 0.6
search_keyword_weight = 0.4
search_weight_boost = 0.3
search_default_limit = 5

# 整合参数
consolidate_similarity_threshold = 0.85

# 嵌入模型
embed_model = "all-MiniLM-L6-v2"
```

---

## 十一、M1验收标准

| 验收项 | 标准 | 优先级 |
|--------|------|--------|
| 记忆写入 | `iris memory add` 可添加，返回ID | P0 |
| 记忆检索 | `iris memory search` 混合搜索可用，返回相关结果 | P0 |
| 衰减运行 | `iris memory decay` 按艾宾浩斯曲线衰减权重 | P0 |
| 状态流转 | weight<0.3→archived, weight<0.1→decayed | P0 |
| 访问强化 | 检索命中或手动reinforce时权重提升 | P1 |
| 记忆整合 | 相似度>0.85的记忆自动合并 | P1 |
| 导入导出 | JSON格式导入导出可用 | P1 |
| 统计面板 | `iris memory stats` 显示总数/类型分布/权重分布 | P2 |
| 列表过滤 | 按类型/状态/时间过滤 | P2 |

---

## 十二、项目初始化要求

1. 使用 `uv` 初始化项目（pyproject.toml）
2. 项目名：`iris-cli`
3. Python >= 3.10
4. 依赖：typer, rich, chromadb, sentence-transformers, pydantic
5. 开发依赖：pytest, pytest-asyncio, ruff
6. 入口点：`iris` 命令
7. 目录结构严格按第四节

---

## 十三、里程碑计划

| 阶段 | 日期 | 内容 |
|------|------|------|
| M1.1 | 5/7 | 项目骨架 + 数据模型 + 存储层 + 基础CLI |
| M1.2 | 5/9 | 衰减引擎 + 检索引擎(语义+关键词+混合) |
| M1.3 | 5/11 | 记忆整合 + 导入导出 + 统计面板 + 完整测试 |
