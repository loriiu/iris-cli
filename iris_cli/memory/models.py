"""数据模型模块.

定义记忆系统的核心数据结构。
"""

import json
import uuid
from datetime import datetime
from enum import Enum
from typing import Optional, Any

from pydantic import BaseModel, Field, field_validator


class MemoryType(str, Enum):
    """记忆类型枚举."""

    EPISODIC = "episodic"  # 事件/经历（如"今天做了代码审查"）
    SEMANTIC = "semantic"  # 知识/事实（如"Python GIL影响多线程性能"）
    PROCEDURAL = "procedural"  # 技能/方法（如"用Typer创建CLI的步骤"）


class MemoryStatus(str, Enum):
    """记忆状态枚举."""

    ACTIVE = "active"  # 活跃，权重 > 0.3
    ARCHIVED = "archived"  # 归档，权重 0.1-0.3
    DECAYED = "decayed"  # 衰退，权重 < 0.1，待清理
    CONSOLIDATED = "consolidated"  # 已整合，被合并到其他记忆


class SearchMode(str, Enum):
    """搜索模式枚举."""

    SEMANTIC = "semantic"  # 语义搜索
    KEYWORD = "keyword"  # 关键词搜索
    HYBRID = "hybrid"  # 混合搜索


class Memory(BaseModel):
    """记忆条目模型.

    Attributes:
        id: UUID 唯一标识
        content: 记忆内容（自然语言）
        summary: 自动生成的摘要（≤100字）
        memory_type: 记忆类型
        tags: 用户标签 + 自动标签列表
        weight: 当前权重 [0.0, 1.0]，初始 1.0
        source: 来源标识（如 "conversation", "research", "heartbeat"）
        created_at: 创建时间
        accessed_at: 最近访问时间
        access_count: 访问次数
        status: 记忆状态
        metadata: 扩展元数据
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content: str
    summary: str = ""
    memory_type: MemoryType = MemoryType.EPISODIC
    tags: list[str] = Field(default_factory=list)
    weight: float = 1.0
    source: str = "manual"
    created_at: datetime = Field(default_factory=datetime.now)
    accessed_at: datetime = Field(default_factory=datetime.now)
    access_count: int = 0
    status: MemoryStatus = MemoryStatus.ACTIVE
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("weight")
    @classmethod
    def validate_weight(cls, v: float) -> float:
        """验证权重范围."""
        return max(0.0, min(1.0, v))

    @field_validator("tags")
    @classmethod
    def validate_tags(cls, v: list[str]) -> list[str]:
        """验证和清理标签."""
        return [tag.strip() for tag in v if tag.strip()]

    def model_dump(self, **kwargs) -> dict[str, Any]:
        """序列化模型为字典."""
        data = super().model_dump(**kwargs)
        # 将枚举值转换为字符串
        data["memory_type"] = self.memory_type.value
        data["status"] = self.status.value
        # 序列化 datetime
        data["created_at"] = self.created_at.isoformat()
        data["accessed_at"] = self.accessed_at.isoformat()
        # 序列化 tags 和 metadata 为 JSON 字符串
        data["tags"] = json.dumps(self.tags)
        data["metadata"] = json.dumps(self.metadata)
        return data

    @classmethod
    def from_db_row(cls, row: dict[str, Any]) -> "Memory":
        """从数据库行创建模型.

        Args:
            row: 数据库行字典

        Returns:
            Memory 实例
        """
        # 解析 tags
        tags = row.get("tags", "[]")
        if isinstance(tags, str):
            tags = json.loads(tags) if tags else []

        # 解析 metadata
        metadata = row.get("metadata", "{}")
        if isinstance(metadata, str):
            metadata = json.loads(metadata) if metadata else {}

        # 解析 datetime
        created_at = row.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)

        accessed_at = row.get("accessed_at")
        if isinstance(accessed_at, str):
            accessed_at = datetime.fromisoformat(accessed_at)

        # 解析枚举
        memory_type = MemoryType(row.get("memory_type", "episodic"))
        status = MemoryStatus(row.get("status", "active"))

        return cls(
            id=row["id"],
            content=row["content"],
            summary=row.get("summary", "") or "",
            memory_type=memory_type,
            tags=tags,
            weight=float(row.get("weight", 1.0)),
            source=row.get("source", "manual"),
            created_at=created_at,
            accessed_at=accessed_at,
            access_count=int(row.get("access_count", 0)),
            status=status,
            metadata=metadata,
        )

    def to_db_dict(self) -> dict[str, Any]:
        """转换为数据库字典.

        Returns:
            数据库字段字典
        """
        return {
            "id": self.id,
            "content": self.content,
            "summary": self.summary,
            "memory_type": self.memory_type.value,
            "tags": json.dumps(self.tags),
            "weight": self.weight,
            "source": self.source,
            "created_at": self.created_at.isoformat(),
            "accessed_at": self.accessed_at.isoformat(),
            "access_count": self.access_count,
            "status": self.status.value,
            "metadata": json.dumps(self.metadata),
        }


class SearchResult(BaseModel):
    """搜索结果模型."""

    memory: Memory
    score: float
    match_type: SearchMode = SearchMode.SEMANTIC

    @classmethod
    def from_chroma_result(
        cls,
        memory: Memory,
        distance: float,
        match_type: SearchMode = SearchMode.SEMANTIC,
    ) -> "SearchResult":
        """从 ChromaDB 结果创建.

        Args:
            memory: 记忆对象
            distance: 距离（余弦距离）
            match_type: 匹配类型

        Returns:
            SearchResult 实例
        """
        # 余弦相似度 = 1 - 余弦距离
        similarity = 1.0 - distance
        return cls(memory=memory, score=similarity, match_type=match_type)


class MemoryStats(BaseModel):
    """记忆统计模型."""

    total: int = 0
    by_type: dict[str, int] = Field(default_factory=dict)
    by_status: dict[str, int] = Field(default_factory=dict)
    avg_weight: float = 0.0
    total_access_count: int = 0
    tags_distribution: dict[str, int] = Field(default_factory=dict)
