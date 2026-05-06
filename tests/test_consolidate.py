"""Memory consolidation module tests."""

from datetime import datetime
from unittest.mock import MagicMock

import pytest

from iris_cli.memory.models import Memory, MemoryStatus, MemoryType


class TestConsolidationPair:
    """Test ConsolidationPair dataclass."""

    def test_pair_creation(self):
        """Test creating a consolidation pair."""
        from iris_cli.memory.consolidate import ConsolidationPair

        memory_a = Memory(
            id="mem-a",
            content="Content A",
            summary="Summary A",
            memory_type=MemoryType.SEMANTIC,
            tags=["tag1"],
            weight=0.8,
            source="test",
            created_at=datetime.now(),
            accessed_at=datetime.now(),
            access_count=1,
            status=MemoryStatus.ACTIVE,
            metadata={},
        )
        memory_b = Memory(
            id="mem-b",
            content="Content B",
            summary="Summary B",
            memory_type=MemoryType.SEMANTIC,
            tags=["tag2"],
            weight=0.6,
            source="test",
            created_at=datetime.now(),
            accessed_at=datetime.now(),
            access_count=0,
            status=MemoryStatus.ACTIVE,
            metadata={},
        )

        pair = ConsolidationPair(
            memory_a=memory_a,
            memory_b=memory_b,
            similarity=0.9,
            suggested_action="merge_b_into_a",
        )

        assert pair.memory_a.id == "mem-a"
        assert pair.memory_b.id == "mem-b"
        assert pair.similarity == 0.9
        assert pair.suggested_action == "merge_b_into_a"


class TestConsolidationResult:
    """Test ConsolidationResult dataclass."""

    def test_result_defaults(self):
        """Test result default values."""
        from iris_cli.memory.consolidate import ConsolidationResult

        result = ConsolidationResult()

        assert result.pairs_found == 0
        assert result.memories_consolidated == 0
        assert result.memories_preserved == 0
        assert result.pairs == []
        assert result.errors == []


class TestConsolidator:
    """Test Consolidator class."""

    def test_init(self):
        """Test Consolidator initialization."""
        from iris_cli.memory.consolidate import Consolidator

        store = MagicMock()
        consolidator = Consolidator(store=store)

        assert consolidator.threshold == 0.85
        assert consolidator._store == store

    def test_custom_threshold(self):
        """Test custom threshold."""
        from iris_cli.memory.consolidate import Consolidator

        store = MagicMock()
        consolidator = Consolidator(store=store, threshold=0.9)

        assert consolidator.threshold == 0.9

    def test_threshold_property(self):
        """Test threshold property getter and setter."""
        from iris_cli.memory.consolidate import Consolidator

        store = MagicMock()
        consolidator = Consolidator(store=store)

        consolidator.threshold = 0.9
        assert consolidator.threshold == 0.9

    def test_find_consolidation_pairs_empty(self):
        """Test finding pairs with no memories."""
        from iris_cli.memory.consolidate import Consolidator

        store = MagicMock()
        store.list.return_value = []

        consolidator = Consolidator(store=store)
        pairs = consolidator.find_consolidation_pairs()

        assert pairs == []
        store.list.assert_called_once()

    def test_find_consolidation_pairs_with_memories(self):
        """Test finding pairs with multiple memories."""
        from iris_cli.memory.consolidate import Consolidator

        store = MagicMock()
        memory_a = Memory(
            id="mem-a",
            content="Python programming language",
            summary="Summary A",
            memory_type=MemoryType.SEMANTIC,
            tags=["python"],
            weight=0.8,
            source="test",
            created_at=datetime.now(),
            accessed_at=datetime.now(),
            access_count=1,
            status=MemoryStatus.ACTIVE,
            metadata={},
        )
        memory_b = Memory(
            id="mem-b",
            content="Python is a programming language",
            summary="Summary B",
            memory_type=MemoryType.SEMANTIC,
            tags=["python"],
            weight=0.6,
            source="test",
            created_at=datetime.now(),
            accessed_at=datetime.now(),
            access_count=0,
            status=MemoryStatus.ACTIVE,
            metadata={},
        )
        store.list.return_value = [memory_a, memory_b]

        # Mock embedder
        store._vector._embed.return_value = [[1.0, 0.0], [1.0, 0.0]]

        consolidator = Consolidator(store=store, threshold=0.85)
        pairs = consolidator.find_consolidation_pairs()

        # Two identical embeddings should have similarity 1.0
        assert len(pairs) == 1
        assert pairs[0].memory_a.id == "mem-a"
        assert pairs[0].memory_b.id == "mem-b"
        assert pairs[0].similarity == 1.0

    def test_consolidate_dry_run(self):
        """Test dry run consolidation."""
        from iris_cli.memory.consolidate import Consolidator

        store = MagicMock()
        memory_a = Memory(
            id="mem-a",
            content="Content A",
            summary="Summary A",
            memory_type=MemoryType.SEMANTIC,
            tags=["tag1"],
            weight=0.8,
            source="test",
            created_at=datetime.now(),
            accessed_at=datetime.now(),
            access_count=1,
            status=MemoryStatus.ACTIVE,
            metadata={},
        )
        memory_b = Memory(
            id="mem-b",
            content="Content B",
            summary="Summary B",
            memory_type=MemoryType.SEMANTIC,
            tags=["tag2"],
            weight=0.6,
            source="test",
            created_at=datetime.now(),
            accessed_at=datetime.now(),
            access_count=0,
            status=MemoryStatus.ACTIVE,
            metadata={},
        )
        store.list.return_value = [memory_a, memory_b]
        store._vector._embed.return_value = [[1.0, 0.0], [1.0, 0.0]]

        consolidator = Consolidator(store=store, threshold=0.85)
        result = consolidator.consolidate(dry_run=True)

        assert result.pairs_found == 1
        assert result.memories_consolidated == 0  # Dry run, no actual merge
        assert len(result.pairs) == 1
        # No actual updates should be made
        store._meta.update_memory.assert_not_called()

    def test_consolidate_with_type_filter(self):
        """Test consolidation with type filter."""
        from iris_cli.memory.consolidate import Consolidator

        store = MagicMock()
        memory_a = Memory(
            id="mem-a",
            content="Content A",
            summary="Summary A",
            memory_type=MemoryType.SEMANTIC,
            tags=["tag1"],
            weight=0.8,
            source="test",
            created_at=datetime.now(),
            accessed_at=datetime.now(),
            access_count=1,
            status=MemoryStatus.ACTIVE,
            metadata={},
        )
        memory_b = Memory(
            id="mem-b",
            content="Content B",
            summary="Summary B",
            memory_type=MemoryType.PROCEDURAL,
            tags=["tag2"],
            weight=0.6,
            source="test",
            created_at=datetime.now(),
            accessed_at=datetime.now(),
            access_count=0,
            status=MemoryStatus.ACTIVE,
            metadata={},
        )
        store.list.return_value = [memory_a, memory_b]
        store._vector._embed.return_value = [[1.0, 0.0], [1.0, 0.0]]

        consolidator = Consolidator(store=store, threshold=0.85)
        result = consolidator.consolidate(
            dry_run=True,
            memory_type=MemoryType.SEMANTIC,
        )

        # Only semantic type should be searched
        assert store.list.call_args[1]["memory_type"] == MemoryType.SEMANTIC

    def test_consolidate_threshold_override(self):
        """Test threshold override during consolidation."""
        from iris_cli.memory.consolidate import Consolidator

        store = MagicMock()
        store.list.return_value = []
        consolidator = Consolidator(store=store, threshold=0.85)

        consolidator.consolidate(dry_run=True, threshold=0.95)
        assert consolidator.threshold == 0.95
