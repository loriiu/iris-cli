"""测试数据模型."""

import pytest
from datetime import datetime
from iris_cli.memory.models import Memory, MemoryType, MemoryStatus, SearchMode


class TestMemoryModel:
    """测试 Memory 模型."""

    def test_create_memory(self):
        """测试创建记忆."""
        now = datetime.now()
        memory = Memory(
            id="test-id-123",
            content="测试内容",
            summary="测试摘要",
            memory_type=MemoryType.EPISODIC,
            tags=["test", "demo"],
            weight=0.8,
            source="test",
            created_at=now,
            accessed_at=now,
            access_count=0,
            status=MemoryStatus.ACTIVE,
            metadata={"key": "value"},
        )

        assert memory.id == "test-id-123"
        assert memory.content == "测试内容"
        assert memory.memory_type == MemoryType.EPISODIC
        assert memory.tags == ["test", "demo"]
        assert memory.weight == 0.8
        assert memory.status == MemoryStatus.ACTIVE

    def test_memory_type_enum(self):
        """测试记忆类型枚举."""
        assert MemoryType.EPISODIC.value == "episodic"
        assert MemoryType.SEMANTIC.value == "semantic"
        assert MemoryType.PROCEDURAL.value == "procedural"

    def test_memory_status_enum(self):
        """测试记忆状态枚举."""
        assert MemoryStatus.ACTIVE.value == "active"
        assert MemoryStatus.ARCHIVED.value == "archived"
        assert MemoryStatus.DECAYED.value == "decayed"

    def test_to_db_dict(self):
        """测试转换为数据库字典."""
        now = datetime.now()
        memory = Memory(
            id="test-id",
            content="内容",
            summary="摘要",
            memory_type=MemoryType.SEMANTIC,
            tags=["tag1"],
            weight=0.9,
            source="manual",
            created_at=now,
            accessed_at=now,
            access_count=1,
            status=MemoryStatus.ACTIVE,
            metadata={"meta": "data"},
        )

        db_dict = memory.to_db_dict()

        assert db_dict["id"] == "test-id"
        assert db_dict["content"] == "内容"
        assert db_dict["memory_type"] == "semantic"
        assert db_dict["tags"] == '["tag1"]'
        assert db_dict["weight"] == 0.9
        assert db_dict["access_count"] == 1

    def test_from_db_row(self):
        """测试从数据库行创建对象."""
        now_str = "2026-05-06T12:00:00"
        row = {
            "id": "row-id",
            "content": "数据库内容",
            "summary": "数据库摘要",
            "memory_type": "semantic",
            "tags": '["db", "test"]',
            "weight": 0.75,
            "source": "import",
            "created_at": now_str,
            "accessed_at": now_str,
            "access_count": 5,
            "status": "active",
            "metadata": '{"imported": true}',
        }

        memory = Memory.from_db_row(row)

        assert memory.id == "row-id"
        assert memory.content == "数据库内容"
        assert memory.memory_type == MemoryType.SEMANTIC
        assert memory.tags == ["db", "test"]
        assert memory.weight == 0.75
        assert memory.access_count == 5

    def test_search_mode_enum(self):
        """测试搜索模式枚举."""
        assert SearchMode.SEMANTIC.value == "semantic"
        assert SearchMode.KEYWORD.value == "keyword"
        assert SearchMode.HYBRID.value == "hybrid"
