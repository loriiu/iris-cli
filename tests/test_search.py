"""检索引擎测试。"""

import pytest
from datetime import datetime
from unittest.mock import MagicMock, patch

from iris_cli.memory.models import Memory, MemoryType, MemoryStatus
from iris_cli.memory.search import SearchEngine, SearchMode, SearchOptions, SearchResult


@pytest.fixture
def mock_store():
    """创建模拟存储。"""
    store = MagicMock()
    store.list_memories.return_value = []
    store.get_memory.return_value = None
    store.update_memory.return_value = None
    store.semantic_search.return_value = []
    store.keyword_search.return_value = []
    return store


@pytest.fixture
def sample_memories():
    """创建示例记忆列表。"""
    return [
        Memory(
            id="mem-1",
            content="Python 是一种高级编程语言",
            summary="Python语言介绍",
            memory_type=MemoryType.SEMANTIC,
            tags=["python", "编程"],
            weight=1.0,
            source="test",
            created_at=datetime.now(),
            accessed_at=datetime.now(),
            access_count=5,
            status=MemoryStatus.ACTIVE,
            metadata={},
        ),
        Memory(
            id="mem-2",
            content="JavaScript 用于 Web 开发",
            summary="JavaScript介绍",
            memory_type=MemoryType.SEMANTIC,
            tags=["javascript", "web"],
            weight=0.8,
            source="test",
            created_at=datetime.now(),
            accessed_at=datetime.now(),
            access_count=3,
            status=MemoryStatus.ACTIVE,
            metadata={},
        ),
        Memory(
            id="mem-3",
            content="使用 Typer 框架创建 CLI 应用",
            summary="Typer CLI框架",
            memory_type=MemoryType.PROCEDURAL,
            tags=["typer", "cli", "工具"],
            weight=0.9,
            source="test",
            created_at=datetime.now(),
            accessed_at=datetime.now(),
            access_count=2,
            status=MemoryStatus.ACTIVE,
            metadata={},
        ),
    ]


def test_search_mode_enum():
    """测试搜索模式枚举。"""
    assert SearchMode.SEMANTIC.value == "semantic"
    assert SearchMode.KEYWORD.value == "keyword"
    assert SearchMode.HYBRID.value == "hybrid"


def test_search_options_defaults():
    """测试搜索选项默认值。"""
    options = SearchOptions()
    assert options.mode == SearchMode.HYBRID
    assert options.limit == 5
    assert options.memory_type is None
    assert options.tags is None


def test_search_result_creation():
    """测试搜索结果创建。"""
    memory = Memory(
        id="test-1",
        content="测试内容",
        summary="测试摘要",
        memory_type=MemoryType.SEMANTIC,
        tags=["test"],
        weight=0.9,
        source="test",
        created_at=datetime.now(),
        accessed_at=datetime.now(),
        access_count=1,
        status=MemoryStatus.ACTIVE,
        metadata={},
    )
    result = SearchResult(
        memory=memory,
        score=0.95,
        semantic_score=0.9,
        keyword_score=1.0,
    )
    assert result.memory.id == "test-1"
    assert result.score == 0.95
    assert result.semantic_score == 0.9
    assert result.keyword_score == 1.0


class TestSearchEngine:
    """搜索引擎测试类。"""

    def test_init_with_store(self, mock_store):
        """测试搜索引擎初始化。"""
        engine = SearchEngine(mock_store)
        assert engine._store is mock_store

    def test_search_empty_query_returns_empty(self, mock_store):
        """测试空查询返回空结果。"""
        mock_store.semantic_search.return_value = []
        mock_store.keyword_search.return_value = []
        engine = SearchEngine(mock_store)
        # 空查询应该返回空结果
        result = engine.search("", options=SearchOptions(mode=SearchMode.SEMANTIC))
        assert result == []

    def test_search_whitespace_query_returns_empty(self, mock_store):
        """测试空白查询返回空结果。"""
        mock_store.semantic_search.return_value = []
        mock_store.keyword_search.return_value = []
        engine = SearchEngine(mock_store)
        result = engine.search("   ", options=SearchOptions(mode=SearchMode.SEMANTIC))
        assert result == []

    def test_calculate_keyword_score(self, mock_store):
        """测试关键词分数计算。"""
        engine = SearchEngine(mock_store)
        # Positive case
        score = engine._calculate_keyword_score("Python 编程语言", "python")
        assert score > 0

        # Negative case
        score = engine._calculate_keyword_score("JavaScript 开发", "python")
        assert score == 0.0

    def test_calculate_keyword_score_case_insensitive(self, mock_store):
        """测试关键词搜索大小写不敏感。"""
        engine = SearchEngine(mock_store)
        assert engine._calculate_keyword_score("Python 编程", "PYTHON") > 0
        assert engine._calculate_keyword_score("Python 编程", "Python") > 0


class TestHybridSearch:
    """混合搜索测试类。"""

    def test_hybrid_search_combines_scores(self, mock_store, sample_memories):
        """测试混合搜索组合分数。"""
        # Mock semantic search to return results
        mock_store.semantic_search.return_value = [
            SearchResult(
                memory=sample_memories[0],
                score=0.8,
                semantic_score=0.8,
                keyword_score=0.5,
            )
        ]

        engine = SearchEngine(mock_store)
        # Note: Actual hybrid search requires ChromaDB, but we test the logic
