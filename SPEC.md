# Iris-CLI: Agent 记忆系统规格文档

## 1. 项目概述

**项目名称**: iris-cli
**项目类型**: Python CLI 工具
**核心功能**: 为 AI Agent 提供持久化记忆系统，支持语义存储、检索、遗忘曲线管理和记忆整合
**目标用户**: AI Agent 开发者、需要构建个性化 AI 记忆能力的应用

## 2. 技术栈

| 组件 | 技术选型 | 用途 |
|------|----------|------|
| CLI 框架 | Typer | 命令行参数解析与交互 |
| 终端美化 | Rich | 彩色输出、表格、进度条 |
| 向量数据库 | ChromaDB | 语义检索与向量存储 |
| 关系数据库 | SQLite | 元数据存储、衰减调度 |
| 向量化 | sentence-transformers | 文本向量嵌入 |

## 3. 功能模块

### 3.1 记忆存储 (store)
- **命令**: `iris store <content> [--tag <tag>] [--importance <1-10>]`
- **功能**:
  - 将文本内容存储到记忆系统
  - 自动生成向量嵌入存储到 ChromaDB
  - 记录元数据（时间戳、标签、重要度）
  - 计算并存储下次复习时间（基于艾宾浩斯曲线）

### 3.2 语义检索 (recall)
- **命令**: `iris recall <query> [--top-k <n>] [--filter-tag <tag>]`
- **功能**:
  - 将查询文本向量化
  - 在 ChromaDB 中进行相似度搜索
  - 返回最相关的记忆条目
  - 支持标签过滤和数量限制

### 3.3 艾宾浩斯衰减 (decay)
- **命令**: `iris decay [--dry-run]`
- **功能**:
  - 检查所有记忆的复习间隔
  - 对超过复习时间的记忆进行衰减处理
  - 降低重要度评分，延长复习间隔
  - 重要度过低的记忆自动归档或删除
  - 支持预览模式（--dry-run）

### 3.4 记忆整合 (consolidate)
- **命令**: `iris consolidate [--threshold <similarity>]`
- **功能**:
  - 找出高度相似的记忆条目
  - 将相似记忆整合为更简洁的表述
  - 保留原始记忆的元数据
  - 减少冗余，提升检索效率

### 3.5 记忆管理
- **命令**: `iris list [--tag <tag>] [--limit <n>]`
- **命令**: `iris delete <memory-id>`
- **命令**: `iris stats`
- **功能**:
  - 列出记忆（支持分页和标签过滤）
  - 删除指定记忆
  - 显示记忆系统统计信息

## 4. 数据模型

### 4.1 SQLite 表结构

```sql
-- 记忆元数据表
CREATE TABLE memories (
    id TEXT PRIMARY KEY,              -- UUID
    content TEXT NOT NULL,            -- 记忆内容原文
    summary TEXT,                     -- 整合后的摘要
    tag TEXT,                          -- 标签
    importance INTEGER DEFAULT 5,     -- 重要度 1-10
    review_count INTEGER DEFAULT 0,   -- 复习次数
    ease_factor REAL DEFAULT 2.5,     -- 艾宾浩斯易记因子
    interval_days INTEGER DEFAULT 1,   -- 下次复习间隔（天）
    next_review_at TEXT,               -- 下次复习时间 ISO8601
    created_at TEXT NOT NULL,          -- 创建时间
    updated_at TEXT NOT NULL,         -- 更新时间
    is_archived INTEGER DEFAULT 0      -- 是否已归档
);
```

### 4.2 ChromaDB Collection

```
Collection: memories
- id: UUID (与 SQLite 对应)
- embedding: 384维向量 (all-MiniLM-L6-v2)
- document: 记忆内容
- metadata: {tag, importance, created_at}
```

## 5. 艾宾浩斯遗忘曲线算法

```
初始参数:
- ease_factor (EF): 默认 2.5
- interval: 初始 1 天

复习后更新:
- 如果正确回忆:
  - EF = EF + 0.1 - (5 - rating) * (0.08 + (5 - rating) * 0.02)
  - interval = interval * EF
- 如果错误回忆:
  - EF = max(1.3, EF - 0.2)
  - interval = 1

简化实现（无交互评分）:
- 每次衰减: interval = interval * 1.2（最长30天）
- 每次衰减: importance = max(1, importance - 0.5)
- 低于重要度阈值(2)的记忆自动归档
```

## 6. 交互流程

### 6.1 首次使用
```bash
$ iris init
✓ 创建数据库: ~/.iris/memory.db
✓ 初始化向量库: ~/.iris/chroma
✓ 默认模型: all-MiniLM-L6-v2
```

### 6.2 存储记忆
```bash
$ iris store "用户喜欢蓝色的配色方案" --tag design
✓ 记忆已存储 [ID: abc123]
  标签: design | 重要度: 5
  下次复习: 1 天后
```

### 6.3 检索记忆
```bash
$ iris recall "用户偏好什么颜色"
┌─────────────────────────────────────┐
│ 🔍 语义检索结果                       │
├─────┬───────────────────┬───────────┤
│ 相似度 │ 内容               │ 标签      │
├─────┼───────────────────┼───────────┤
│ 0.92 │ 用户喜欢蓝色配色方案   │ design   │
└─────┴───────────────────┴───────────┘
```

### 6.4 统计数据
```bash
$ iris stats
┌─────────────────────────────────────┐
│ 📊 Iris 记忆系统统计                  │
├─────────────────┬─────────────────────┤
│ 总记忆数        │ 156                 │
│ 活跃记忆        │ 142                 │
│ 已归档          │ 14                  │
│ 待复习          │ 23                  │
│ 平均重要度      │ 5.3                 │
└─────────────────┴─────────────────────┘
```

## 7. 项目结构

```
iris-cli/
├── pyproject.toml
├── README.md
├── iris/
│   ├── __init__.py
│   ├── __main__.py
│   ├── cli.py              # Typer CLI 入口
│   ├── config.py           # 配置管理
│   ├── db.py               # SQLite 操作
│   ├── vector.py           # ChromaDB 操作
│   ├── ebbinghaus.py       # 艾宾浩斯算法
│   ├── consolidate.py      # 记忆整合
│   └── utils.py            # 工具函数
└── tests/
    ├── __init__.py
    ├── test_db.py
    ├── test_vector.py
    └── test_ebbinghaus.py
```

## 8. 验收标准

- [ ] `iris init` 成功初始化数据库和向量库
- [ ] `iris store` 能存储记忆并生成向量
- [ ] `iris recall` 能语义检索相关记忆
- [ ] `iris list` 能列出并过滤记忆
- [ ] `iris decay` 能执行艾宾浩斯衰减
- [ ] `iris consolidate` 能整合相似记忆
- [ ] `iris delete` 能删除记忆
- [ ] `iris stats` 能显示统计信息
- [ ] 所有输出使用 Rich 美化
- [ ] 单元测试覆盖核心功能

## 9. 依赖包

```
typer[all]>=0.12.0
rich>=13.7.0
chromadb>=0.5.0
sentence-transformers>=2.5.0
```

## 10. 命令速查

| 命令 | 说明 |
|------|------|
| `iris init` | 初始化记忆系统 |
| `iris store <content>` | 存储新记忆 |
| `iris recall <query>` | 语义检索记忆 |
| `iris list` | 列出所有记忆 |
| `iris delete <id>` | 删除记忆 |
| `iris decay` | 执行衰减调度 |
| `iris consolidate` | 整合相似记忆 |
| `iris stats` | 显示统计信息 |
