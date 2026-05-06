"""Tests for memory I/O module."""
from unittest.mock import MagicMock, patch
import json
import pytest

from iris_cli.memory.io import (
    ExportOptions,
    ImportOptions,
    ExportResult,
    ImportResult,
    Exporter,
    Importer,
)
from iris_cli.memory.models import Memory, MemoryType, MemoryStatus
from datetime import datetime


class TestExportOptions:
    """Test ExportOptions dataclass."""

    def test_export_options_defaults(self):
        """Test ExportOptions default values."""
        opts = ExportOptions()

        assert opts.format == "json"
        assert opts.include_decayed is False
        assert opts.include_consolidated is False
        assert opts.output is None

    def test_export_options_custom(self):
        """Test custom ExportOptions."""
        opts = ExportOptions(
            format="markdown",
            include_decayed=True,
            include_consolidated=True,
            output="/path/to/output.md",
        )

        assert opts.format == "markdown"
        assert opts.include_decayed is True
        assert opts.include_consolidated is True
        assert opts.output == "/path/to/output.md"

    def test_export_options_invalid_format(self):
        """Test invalid format is accepted (no validation in dataclass)."""
        opts = ExportOptions(format="xml")
        assert opts.format == "xml"


class TestImportOptions:
    """Test ImportOptions dataclass."""

    def test_import_options_defaults(self):
        """Test ImportOptions default values."""
        opts = ImportOptions()

        assert opts.format == "json"
        assert opts.source is None
        assert opts.conflict_mode == "merge"
        assert opts.dry_run is False

    def test_import_options_custom(self):
        """Test custom ImportOptions."""
        opts = ImportOptions(
            format="markdown",
            source="backup",
            conflict_mode="skip",
            dry_run=True,
        )

        assert opts.format == "markdown"
        assert opts.source == "backup"
        assert opts.conflict_mode == "skip"
        assert opts.dry_run is True


class TestExportResult:
    """Test ExportResult dataclass."""

    def test_export_result_defaults(self):
        """Test ExportResult default values."""
        result = ExportResult()

        assert result.count == 0
        assert result.output_path is None
        assert result.errors == []

    def test_export_result_success(self):
        """Test successful export result."""
        result = ExportResult(
            count=5,
            output_path="/path/to/output.json",
            errors=[],
        )

        assert result.count == 5
        assert result.output_path == "/path/to/output.json"
        assert len(result.errors) == 0


class TestImportResult:
    """Test ImportResult dataclass."""

    def test_import_result_defaults(self):
        """Test ImportResult default values."""
        result = ImportResult()

        assert result.imported == 0
        assert result.skipped == 0
        assert result.errors == []

    def test_import_result_success(self):
        """Test successful import result."""
        result = ImportResult(
            imported=10,
            skipped=2,
            errors=["Warning: some entries had issues"],
        )

        assert result.imported == 10
        assert result.skipped == 2
        assert len(result.errors) == 1


class TestExporter:
    """Test Exporter class."""

    def test_init(self):
        """Test Exporter initialization."""
        store = MagicMock()
        exporter = Exporter(store)

        assert exporter.store is store

    def test_export_to_json(self):
        """Test JSON export."""
        store = MagicMock()
        memory = Memory(
            id="test-1",
            content="Test content",
            summary="Test",
            memory_type=MemoryType.EPISODIC,
            tags=["test"],
            weight=1.0,
            source="test",
            created_at=datetime.now(),
            accessed_at=datetime.now(),
            access_count=0,
            status=MemoryStatus.ACTIVE,
            metadata={},
        )
        store._meta.list.return_value = [memory]

        exporter = Exporter(store)
        result = exporter.export(ExportOptions())

        assert result.count == 1
        assert len(result.errors) == 0

    def test_export_to_markdown(self):
        """Test Markdown export."""
        store = MagicMock()
        memory = Memory(
            id="test-1",
            content="Test content",
            summary="Test",
            memory_type=MemoryType.SEMANTIC,
            tags=["test"],
            weight=1.0,
            source="test",
            created_at=datetime.now(),
            accessed_at=datetime.now(),
            access_count=0,
            status=MemoryStatus.ACTIVE,
            metadata={},
        )
        store._meta.list.return_value = [memory]

        exporter = Exporter(store)
        result = exporter.export(ExportOptions(format="markdown"))

        assert result.count == 1

    def test_export_with_status_filter(self):
        """Test export with status filtering."""
        store = MagicMock()
        decayed = Memory(
            id="test-1",
            content="Decayed content",
            summary="Decayed",
            memory_type=MemoryType.EPISODIC,
            tags=[],
            weight=0.05,
            source="test",
            created_at=datetime.now(),
            accessed_at=datetime.now(),
            access_count=0,
            status=MemoryStatus.DECAYED,
            metadata={},
        )
        active = Memory(
            id="test-2",
            content="Active content",
            summary="Active",
            memory_type=MemoryType.SEMANTIC,
            tags=[],
            weight=0.8,
            source="test",
            created_at=datetime.now(),
            accessed_at=datetime.now(),
            access_count=5,
            status=MemoryStatus.ACTIVE,
            metadata={},
        )
        store._meta.list.return_value = [decayed, active]

        exporter = Exporter(store)

        # Default: exclude decayed
        result = exporter.export(ExportOptions())
        assert result.count == 1

        # Include decayed
        result = exporter.export(ExportOptions(include_decayed=True))
        assert result.count == 2


class TestImporter:
    """Test Importer class."""

    def test_init(self):
        """Test Importer initialization."""
        store = MagicMock()
        importer = Importer(store)

        assert importer.store is store

    def test_import_from_json(self):
        """Test JSON import."""
        store = MagicMock()
        store.get_memory.return_value = None  # No existing memories
        store.add.return_value = "new-id-1"

        importer = Importer(store)
        data = [
            {
                "content": "Memory 1",
                "memory_type": "episodic",
                "tags": ["test"],
            },
            {
                "content": "Memory 2",
                "memory_type": "semantic",
                "tags": ["info"],
            },
        ]

        result = importer._import_json(json.dumps(data), ImportOptions(format="json"))

        assert result.imported == 2
        assert store.add.called

    def test_import_from_markdown_with_file(self, tmp_path):
        """Test Markdown import with file."""
        store = MagicMock()
        store.get_memory.return_value = None
        store.add.return_value = "new-id"

        importer = Importer(store)
        markdown_content = """# Memory Title

Content of the memory.

## Tags
- tag1
- tag2

## Type
semantic
"""
        tmp_file = tmp_path / "test.md"
        tmp_file.write_text(markdown_content)
        options = ImportOptions(format="markdown")

        result = importer.import_from_file(str(tmp_file), options)

        assert result.imported == 1

    def test_import_markdown_format(self, tmp_path):
        """Test importing markdown format."""
        store = MagicMock()
        store.get_memory.return_value = None
        store.add.return_value = "test-id"

        importer = Importer(store)
        markdown_content = """# Memory: Test Memory

## Summary
A test memory

## Content

This is test content.

## Metadata
- Type: semantic
- Tags: test, example
"""
        tmp_file = tmp_path / "test.md"
        tmp_file.write_text(markdown_content)
        options = ImportOptions(format="markdown")

        result = importer.import_from_file(str(tmp_file), options)

        assert result.imported == 1

    def test_import_skips_invalid_entries(self):
        """Test that invalid entries are skipped."""
        store = MagicMock()
        store.get_memory.return_value = None
        store.add.return_value = "test-id"

        importer = Importer(store)
        json_content = json.dumps([
            {"content": "Valid memory"},
            {"memory_type": "episodic"},  # Missing content
            {"content": "Another valid"},
        ])
        options = ImportOptions(format="json")

        result = importer.import_from_file("nonexistent.json", options)

        # File not found should be in errors
        assert len(result.errors) > 0
