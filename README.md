# iris-cli - Agent 记忆系统

一个 Python CLI 工具，用于构建 Agent 的记忆能力基础设施。

## 特性

- **记忆存储**: SQLite + ChromaDB 双存储
- **语义检索**: 本地 sentence-transformers 向量模型
- **艾宾浩斯衰减**: 基于遗忘曲线的记忆权重管理
- **记忆整合**: 自动合并相似记忆
- **Rich 美化**: 终端友好输出

## 安装

```bash
# 使用 pip
pip install -e .

# 或使用 uv
uv pip install -e .
```

## 快速开始

```bash
# 初始化
iris init

# 添加记忆
iris memory add "用户喜欢蓝色的配色方案" --type semantic --tags design

# 列出记忆
iris memory list

# 搜索记忆
iris memory search "配色"

# 获取详情
iris memory get <memory-id>

# 查看统计
iris memory stats
```

## 项目结构

```
iris_cli/
├── __init__.py
├── __main__.py           # python -m iris_cli 入口
├── cli/
│   ├── __init__.py
│   ├── main.py           # Typer app, 根命令
│   ├── memory.py         # memory 子命令组
│   └── utils.py          # CLI工具函数
├── memory/
│   ├── __init__.py
│   ├── models.py         # 数据模型
│   ├── store.py          # 存储层
│   └── embedder.py       # 嵌入生成
└── config.py             # 配置管理
```

## CLI 命令

### 记忆管理

| 命令 | 说明 |
|------|------|
| `iris init` | 初始化配置 |
| `iris memory add <content>` | 添加记忆 |
| `iris memory get <id>` | 获取记忆详情 |
| `iris memory list` | 列出记忆 |
| `iris memory search <query>` | 搜索记忆 |
| `iris memory delete <id>` | 删除记忆 |
| `iris memory stats` | 显示统计 |

### 选项

```bash
# 添加记忆选项
iris memory add "内容" \
  --type episodic|semantic|procedural \
  --tags tag1,tag2 \
  --source conversation \
  --importance 1-10

# 搜索选项
iris memory search "查询" \
  --mode hybrid|semantic|keyword \
  --type semantic \
  --limit 5
```

## 配置

配置文件: `~/.iris/config.toml`

```toml
[memory]
data_dir = "~/.iris/data"
embed_model = "sentence-transformers/all-MiniLM-L6-v2"

# 衰减参数
decay_rate_day1 = 0.10
decay_rate_week1 = 0.05
decay_rate_month1 = 0.01

# 搜索参数
search_default_mode = "hybrid"
search_semantic_weight = 0.6
search_keyword_weight = 0.4
```

## 开发

```bash
# 安装依赖
pip install -e ".[dev]"

# 运行测试
pytest tests/

# 代码格式化
ruff check .
```

## 依赖

- Python >= 3.10
- typer >= 0.12
- rich >= 13.0
- chromadb >= 0.4
- sentence-transformers
- pydantic
