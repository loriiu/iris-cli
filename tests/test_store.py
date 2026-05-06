"""测试存储层."""

import pytest
import tempfile
import shutil
from datetime import datetime
from pathlib import Path

from iris_cli.memory.models import Memory, MemoryType, MemoryStatus
from iris_cli.memory.store import VectorStore, MetaStore


class TestMetaStore:
    """测试元数据存储."""

    @pytest.fixture
    def meta_store(self, temp_dir):
        """创建测试用 MetaStore."""
        db_path = temp_dir / "test.db"
        store = MetaStore(db_path)
        yield store
        store.close()

    def test_init_creates_schema(self, temp_dir):
        """测试初始化创建 schema."""
        db_path = temp_dir / "new.db"
        store = MetaStore(db_path)

        assert db_path.exists()

        import sqlite3
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # 检查表是否存在
        cursor.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table' AND name='memories'
        """)
        assert cursor.fetchone() is not None

        # 检查索引
        cursor.execute("""
            SELECT name FROM sqlite_master
            WHERE type='index' AND name='idx_memories_type'
        """)
        assert cursor.fetchone() is not None

        conn.close()
        store.close()

    def test_add_memory(self, meta_store):
        """测试添加记忆."""
        now = datetime.now()
        memory = Memory(
            id="test-add-1",
            content="测试内容",
            summary="测试摘要",
            memory_type=MemoryType.EPISODIC,
            tags=["test"],
            weight=0.8,
            source="test",
            created_at=now,
            accessed_at=now,
            access_count=0,
            status=MemoryStatus.ACTIVE,
            metadata={},
        )

        meta_store.add(memory)

        # 验证添加成功
        retrieved = meta_store.get("test-add-1")
        assert retrieved is not None
        assert retrieved.content == "测试内容"
        assert retrieved.id == "test-add-1"

    def test_get_memory(self, meta_store):
        """测试获取记忆."""
        now = datetime.now()
        memory = Memory(
            id="test-get-1",
            content="获取测试",
            summary="获取摘要",
            memory_type=MemoryType.SEMANTIC,
            tags=[],
            weight=0.9,
            source="test",
            created_at=now,
            accessed_at=now,
            access_count=0,
            status=MemoryStatus.ACTIVE,
            metadata={},
        )

        meta_store.add(memory)
        retrieved = meta_store.get("test-get-1")

        assert retrieved is not None
        assert retrieved.id == "test-get-1"
        assert retrieved.content == "获取测试"

    def test_get_nonexistent(self, meta_store):
        """测试获取不存在的记忆."""
        result = meta_store.get("nonexistent-id")
        assert result is None

    def test_update_memory(self, meta_store):
        """测试更新记忆."""
        now = datetime.now()
        memory = Memory(
            id="test-update-1",
            content="原始内容",
            summary="原始摘要",
            memory_type=MemoryType.PROCEDURAL,
            tags=["old"],
            weight=0.5,
            source="test",
            created_at=now,
            accessed_at=now,
            access_count=0,
            status=MemoryStatus.ACTIVE,
            metadata={},
        )

        meta_store.add(memory)

        # 更新
        memory.content = "更新后内容"
        memory.weight = 0.95
        memory.tags = ["new", "updated"]
        meta_store.update(memory)

        # 验证更新
        retrieved = meta_store.get("test-update-1")
        assert retrieved.content == "更新后内容"
        assert retrieved.weight == 0.95
        assert retrieved.tags == ["new", "updated"]

    def test_delete_memory(self, meta_store):
        """测试删除记忆."""
        now = datetime.now()
        memory = Memory(
            id="test-delete-1",
            content="待删除内容",
            summary="删除摘要",
            memory_type=MemoryType.EPISODIC,
            tags=[],
            weight=0.7,
            source="test",
            created_at=now,
            accessed_at=now,
            access_count=0,
            status=MemoryStatus.ACTIVE,
            metadata={},
        )

        meta_store.add(memory)
        result = meta_store.delete("test-delete-1")

        assert result is True
        assert meta_store.get("test-delete-1") is None

    def test_delete_nonexistent(self, meta_store):
        """测试删除不存在的记忆."""
        result = meta_store.delete("nonexistent-id")
        assert result is False

    def test_list_memories(self, meta_store):
        """测试列出记忆."""
        now = datetime.now()

        for i in range(5):
            memory = Memory(
                id=f"test-list-{i}",
                content=f"内容 {i}",
                summary=f"摘要 {i}",
                memory_type=MemoryType.EPISODIC,
                tags=[],
                weight=0.5 + i * 0.1,
                source="test",
                created_at=now,
                accessed_at=now,
                access_count=0,
                status=MemoryStatus.ACTIVE,
                metadata={},
            )
            meta_store.add(memory)

        memories = meta_store.list()
        assert len(memories) == 5

    def test_list_with_type_filter(self, meta_store):
        """测试按类型过滤."""
        now = datetime.now()

        # 添加不同类型的记忆
        for i, mem_type in enumerate([MemoryType.EPISODIC, MemoryType.SEMANTIC, MemoryType.SEMANTIC]):
            memory = Memory(
                id=f"test-filter-{i}",
                content=f"内容 {i}",
                summary=f"摘要 {i}",
                memory_type=mem_type,
                tags=[],
                weight=0.7,
                source="test",
                created_at=now,
                accessed_at=now,
                access_count=0,
                status=MemoryStatus.ACTIVE,
                metadata={},
            )
            meta_store.add(memory)

        episodic = meta_store.list(memory_type="episodic")
        semantic = meta_store.list(memory_type="semantic")

        assert len(episodic) == 1
        assert len(semantic) == 2

    def test_list_with_status_filter(self, meta_store):
        """测试按状态过滤."""
        now = datetime.now()

        # 添加不同状态的记忆
        for i, status in enumerate([MemoryStatus.ACTIVE, MemoryStatus.ARCHIVED, MemoryStatus.ACTIVE]):
            memory = Memory(
                id=f"test-status-{i}",
                content=f"内容 {i}",
                summary=f"摘要 {i}",
                memory_type=MemoryType.EPISODIC,
                tags=[],
                weight=0.7,
                source="test",
                created_at=now,
                accessed_at=now,
                access_count=0,
                status=status,
                metadata={},
            )
            meta_store.add(memory)

        active = meta_store.list(status="active")
        archived = meta_store.list(status="archived")

        assert len(active) == 2
        assert len(archived) == 1

    def test_count(self, meta_store):
        """测试计数功能."""
        now = datetime.now()

        for i in range(3):
            memory = Memory(
                id=f"test-count-{i}",
                content=f"内容 {i}",
                summary=f"摘要 {i}",
                memory_type=MemoryType.SEMANTIC,
                tags=[],
                weight=0.7,
                source="test",
                created_at=now,
                accessed_at=now,
                access_count=0,
                status=MemoryStatus.ACTIVE,
                metadata={},
            )
            meta_store.add(memory)

        assert meta_store.count() == 3
        assert meta_store.count(memory_type="semantic") == 3
        assert meta_store.count(memory_type="episodic") == 0


class TestVectorStore:
    """测试向量存储."""

    @pytest.fixture
    def vector_store(self, temp_dir):
        """创建测试用 VectorStore."""
        chroma_path = temp_dir / "chroma"
        store = VectorStore(chroma_path)
        yield store
        store.close()

    def test_init_creates_directory(self, temp_dir):
        """测试初始化创建目录."""
        chroma_path = temp_dir / "new_chroma"
        store = VectorStore(chroma_path)

        assert chroma_path.exists()
        store.close()

    def test_add_memory(self, vector_store):
        """测试添加记忆向量."""
        now = datetime.now()
        memory = Memory(
            id="vec-test-1",
            content="向量测试内容",
            summary="测试摘要",
            memory_type=MemoryType.SEMANTIC,
            tags=["test"],
            weight=0.8,
            source="test",
            created_at=now,
            accessed_at=now,
            access_count=0,
            status=MemoryStatus.ACTIVE,
            metadata={},
        )

        vector_store.add(memory)

        # 验证可以检索
        result = vector_store.get_by_id("vec-test-1")
        assert result is not None
        assert result["content"] == "向量测试内容"

    def test_search(self, vector_store):
        """测试搜索功能."""
        now = datetime.now()
        memory = Memory(
            id="vec-search-1",
            content="蓝色的设计风格",
            summary="蓝色设计",
            memory_type=MemoryType.SEMANTIC,
            tags=["design"],
            weight=0.9,
            source="test",
            created_at=now,
            accessed_at=now,
            access_count=0,
            status=MemoryStatus.ACTIVE,
            metadata={},
        )
        vector_store.add(memory)

        results = vector_store.search("蓝色配色", top_k=1)
        assert len(results) >= 1

    def test_update_memory(self, vector_store):
        """测试更新记忆向量."""
        now = datetime.now()
        memory = Memory(
            id="vec-update-1",
            content="原始向量内容",
            summary="原始摘要",
            memory_type=MemoryType.EPISODIC,
            tags=["old"],
            weight=0.6,
            source="test",
            created_at=now,
            accessed_at=now,
            access_count=0,
            status=MemoryStatus.ACTIVE,
            metadata={},
        )
        vector_store.add(memory)

        memory.content = "更新后向量内容"
        vector_store.update(memory)

        result = vector_store.get_by_id("vec-update-1")
        assert result["content"] == "更新后向量内容"

    def test_delete_memory(self, vector_store):
        """测试删除记忆向量."""
        now = datetime.now()
        memory = Memory(
            id="vec-delete-1",
            content="待删除内容",
            summary="删除摘要",
            memory_type=MemoryType.SEMANTIC,
            tags=[],
            weight=0.7,
            source="test",
            created_at=now,
            accessed_at=now,
            access_count=0,
            status=MemoryStatus.ACTIVE,
            metadata={},
        )
        vector_store.add(memory)

        vector_store.delete("vec-delete-1")

        result = vector_store.get_by_id("vec-delete-1")
        assert result is None

    def test_context_manager(self, temp_dir):
        """测试上下文管理器."""
        chroma_path = temp_dir / "context_chroma"

        with VectorStore(chroma_path) as store:
            assert store._client is not None

        # 退出后应该关闭
        assert store._client is None
