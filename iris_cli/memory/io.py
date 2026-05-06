"""记忆导入导出模块."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rich.console import Console

from iris_cli.memory.models import Memory, MemoryStatus, MemoryType

if TYPE_CHECKING:
    from iris_cli.memory.store import MemoryStore

console = Console()


@dataclass
class ExportOptions:
    """导出选项."""

    format: str = "json"  # json | markdown
    include_decayed: bool = False
    include_consolidated: bool = False
    output: str | None = None


@dataclass
class ImportOptions:
    """导入选项."""

    format: str = "json"  # json | markdown
    source: str | None = None
    merge_existing: bool = True  # 如果 ID 存在则合并


@dataclass
class ExportResult:
    """导出结果."""

    count: int = 0
    output_path: str | None = None
    errors: list[str] = field(default_factory=list)


@dataclass
class ImportResult:
    """导入结果."""

    imported: int = 0
    skipped: int = 0
    errors: list[str] = field(default_factory=list)


class Exporter:
    """记忆导出器."""

    def __init__(self, store: MemoryStore) -> None:
        """初始化导出器."""
        self.store = store

    def export(self, options: ExportOptions) -> ExportResult:
        """导出记忆.

        Args:
            options: 导出选项

        Returns:
            导出结果
        """
        result = ExportResult()

        # 获取所有记忆
        memories = self.store._meta.list(status=None, limit=10000)

        # 过滤状态
        filtered = []
        for m in memories:
            if m.status == MemoryStatus.CONSOLIDATED and not options.include_consolidated:
                continue
            if m.status == MemoryStatus.DECAYED and not options.include_decayed:
                continue
            filtered.append(m)

        result.count = len(filtered)

        if options.format == "json":
            content = self._export_json(filtered)
        elif options.format == "markdown":
            content = self._export_markdown(filtered)
        else:
            result.errors.append(f"Unsupported format: {options.format}")
            return result

        # 输出
        if options.output:
            Path(options.output).write_text(content, encoding="utf-8")
            result.output_path = options.output
            console.print(f"[green]Exported {result.count} memories to {options.output}[/green]")
        else:
            print(content)

        return result

    def _export_json(self, memories: list[Memory]) -> str:
        """导出为 JSON 格式."""
        data = []
        for m in memories:
            data.append({
                "id": m.id,
                "content": m.content,
                "summary": m.summary,
                "memory_type": m.memory_type.value,
                "tags": m.tags,
                "weight": m.weight,
                "source": m.source,
                "created_at": m.created_at.isoformat(),
                "accessed_at": m.accessed_at.isoformat(),
                "access_count": m.access_count,
                "status": m.status.value,
                "metadata": m.metadata or {},
            })

        return json.dumps(data, ensure_ascii=False, indent=2)

    def _export_markdown(self, memories: list[Memory]) -> str:
        """导出为 Markdown 格式."""
        lines = [
            "# Iris Memory Export",
            "",
            f"Exported at: {datetime.now().isoformat()}",
            f"Total memories: {len(memories)}",
            "",
        ]

        for m in memories:
            lines.append(f"## {m.summary or m.content[:50]}...")
            lines.append("")
            lines.append(f"- **ID**: `{m.id}`")
            lines.append(f"- **Type**: {m.memory_type.value}")
            lines.append(f"- **Status**: {m.status.value}")
            lines.append(f"- **Weight**: {m.weight:.3f}")
            lines.append(f"- **Tags**: {', '.join(m.tags) if m.tags else 'None'}")
            lines.append(f"- **Created**: {m.created_at.strftime('%Y-%m-%d %H:%M:%S')}")
            lines.append(f"- **Accessed**: {m.accessed_at.strftime('%Y-%m-%d %H:%M:%S')}")
            lines.append(f"- **Access Count**: {m.access_count}")
            lines.append("")
            lines.append("### Content")
            lines.append("")
            lines.append(m.content)
            lines.append("")
            lines.append("---")
            lines.append("")

        return "\n".join(lines)


class Importer:
    """记忆导入器."""

    def __init__(self, store: MemoryStore) -> None:
        """初始化导入器."""
        self.store = store

    def import_from_file(self, file_path: str, options: ImportOptions) -> ImportResult:
        """从文件导入记忆.

        Args:
            file_path: 文件路径
            options: 导入选项

        Returns:
            导入结果
        """
        result = ImportResult()

        path = Path(file_path)
        if not path.exists():
            result.errors.append(f"File not found: {file_path}")
            return result

        try:
            content = path.read_text(encoding="utf-8")
        except Exception as e:
            result.errors.append(f"Failed to read file: {e}")
            return result

        if options.format == "json":
            return self._import_json(content, options)
        elif options.format == "markdown":
            return self._import_markdown(content, options)
        else:
            result.errors.append(f"Unsupported format: {options.format}")
            return result

    def _import_json(self, content: str, options: ImportOptions) -> ImportResult:
        """从 JSON 导入."""
        result = ImportResult()

        try:
            data = json.loads(content)
        except json.JSONDecodeError as e:
            result.errors.append(f"Invalid JSON: {e}")
            return result

        if not isinstance(data, list):
            data = [data]

        for item in data:
            try:
                # 解析记忆
                memory = self._parse_memory(item)
                if not memory:
                    result.skipped += 1
                    continue

                # 检查是否已存在
                existing = self.store.get_memory(memory.id)
                if existing:
                    if options.merge_existing:
                        # 合并元数据
                        existing.metadata = existing.metadata or {}
                        existing.metadata["imported_versions"] = (
                            existing.metadata.get("imported_versions", [])
                            + [memory.content]
                        )
                        self.store._meta.update(memory.id, metadata=existing.metadata)
                        result.imported += 1
                    else:
                        result.skipped += 1
                else:
                    # 添加新记忆
                    self.store.add(memory)
                    result.imported += 1

            except Exception as e:
                result.errors.append(f"Failed to import: {e}")

        console.print(
            f"[green]Imported {result.imported} memories, "
            f"skipped {result.skipped}[/green]"
        )
        return result

    def _import_markdown(self, content: str, options: ImportOptions) -> ImportResult:
        """从 Markdown 导入."""
        result = ImportResult()
        
        # 解析 Markdown 格式的记忆
        # 支持格式: # Memory: Title\n## Summary\n...\n## Content\n...
        memory_blocks = self._split_markdown_memories(content)
        
        for block in memory_blocks:
            try:
                memory = self._parse_markdown_memory(block)
                if not memory:
                    result.skipped += 1
                    continue
                
                # 检查是否已存在
                existing = self.store.get_memory(memory.id)
                if existing:
                    if options.merge_existing:
                        existing.metadata = existing.metadata or {}
                        existing.metadata["imported_versions"] = (
                            existing.metadata.get("imported_versions", [])
                            + [memory.content]
                        )
                        self.store._meta.update(memory.id, metadata=existing.metadata)
                        result.imported += 1
                    else:
                        result.skipped += 1
                else:
                    self.store.add(memory)
                    result.imported += 1
                    
            except Exception as e:
                result.errors.append(f"Failed to import markdown block: {e}")
        
        console.print(
            f"[green]Imported {result.imported} memories, "
            f"skipped {result.skipped}[/green]"
        )
        return result
    
    def _split_markdown_memories(self, content: str) -> list[str]:
        """将 Markdown 内容分割成多个记忆块."""
        # 按一级标题分割（支持 # Memory、# Memory: 或 # Memory Title 格式）
        blocks = []
        current_block = []
        in_block = False
        
        for line in content.split("\n"):
            stripped = line.lstrip()
            if stripped.startswith("# "):
                # 检查是否以 "Memory" 开头
                after_hash = stripped[1:].strip()
                if after_hash.startswith("Memory"):
                    if current_block:
                        blocks.append("\n".join(current_block))
                    current_block = [line]
                    in_block = True
                    continue
            if in_block:
                current_block.append(line)
        
        if current_block:
            blocks.append("\n".join(current_block))
        
        return blocks

    def _parse_markdown_metadata_section(self, content: str, memory_type: MemoryType, tags: list[str]) -> None:
        """解析 Markdown 元数据部分."""
        for meta_line in content.strip().split("\n"):
            meta_line = meta_line.strip()
            if meta_line.startswith("Type:") or meta_line.startswith("- Type:"):
                type_str = meta_line.replace("Type:", "").replace("- Type:", "").strip().lower()
                if type_str in ["episodic", "semantic", "procedural"]:
                    memory_type.value = type_str
            elif meta_line.startswith("Tags:") or meta_line.startswith("- Tags:"):
                tags_str = meta_line.replace("Tags:", "").replace("- Tags:", "").strip()
                if tags_str:
                    tags.extend([t.strip() for t in tags_str.split(",")])

    def _parse_markdown_memory(self, block: str) -> Memory | None:
        """解析单个 Markdown 记忆块."""
        try:
            lines = block.strip().split("\n")
            if not lines:
                return None
            
            # 解析标题（支持 # Memory、# Memory: 或 # Memory Title 格式）
            title_line = lines[0]
            stripped = title_line.lstrip()
            if not (stripped.startswith("# ") and stripped[1:].lstrip().startswith("Memory")):
                return None
            
            # 移除 # Memory: 或 # Memory 并提取标题
            if ':' in title_line:
                title = title_line.split(':', 1)[1].strip()
            else:
                title = title_line.split(' ', 1)[1].strip() if ' ' in title_line else ""
            
            # 解析各部分
            content_parts = []
            summary = ""
            memory_type = MemoryType.EPISODIC
            tags = []
            
            current_section = None
            current_section_content = []
            
            for line in lines[1:]:
                line = line.rstrip()
                
                if line.startswith("## Summary") or line.startswith("##summary"):
                    if current_section and current_section_content:
                        if current_section == "content":
                            content_parts.append("\n".join(current_section_content))
                        elif current_section == "metadata":
                            self._parse_metadata_section("\n".join(current_section_content), memory_type, tags)
                    current_section = "summary"
                    current_section_content = []
                elif line.startswith("## Content") or line.startswith("##content"):
                    if current_section and current_section_content:
                        if current_section == "summary":
                            summary = "\n".join(current_section_content).strip()
                        elif current_section == "metadata":
                            self._parse_metadata_section("\n".join(current_section_content), memory_type, tags)
                    current_section = "content"
                    current_section_content = []
                elif line.startswith("## Metadata") or line.startswith("##metadata"):
                    if current_section and current_section_content:
                        if current_section == "summary":
                            summary = "\n".join(current_section_content).strip()
                        elif current_section == "content":
                            content_parts.append("\n".join(current_section_content))
                    current_section = "metadata"
                    current_section_content = []
                elif line.startswith("## Type") or line.startswith("##type"):
                    continue  # 已在 metadata 中处理
                elif line.startswith("## Tags") or line.startswith("##tags"):
                    continue  # 已在 metadata 中处理
                elif current_section:
                    current_section_content.append(line)
            
            # 处理最后一个 section
            if current_section == "summary":
                summary = "\n".join(current_section_content).strip()
            elif current_section == "content":
                content_parts.append("\n".join(current_section_content))
            elif current_section == "metadata":
                self._parse_markdown_metadata_section("\n".join(current_section_content), memory_type, tags)
            
            content = "\n\n".join(content_parts).strip()
            if not content:
                content = title
            
            now = datetime.now()
            return Memory(
                id=str(uuid.uuid4()),
                content=content,
                summary=summary or title,
                memory_type=memory_type,
                tags=tags,
                weight=1.0,
                source="import",
                metadata={},
                created_at=now,
                accessed_at=now,
                access_count=0,
                status=MemoryStatus.ACTIVE,
            )
            
        except Exception:
            return None
    
    def _parse_memory(self, data: dict[str, Any]) -> Memory | None:
        """解析记忆数据."""
        try:
            # 解析枚举
            memory_type = MemoryType(data.get("memory_type", "episodic"))
            status = MemoryStatus(data.get("status", "active"))

            # 解析时间
            created_at = data.get("created_at")
            if isinstance(created_at, str):
                created_at = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
            elif not created_at:
                created_at = datetime.now()

            accessed_at = data.get("accessed_at")
            if isinstance(accessed_at, str):
                accessed_at = datetime.fromisoformat(accessed_at.replace("Z", "+00:00"))
            elif not accessed_at:
                accessed_at = datetime.now()

            return Memory(
                id=data.get("id") or str(uuid.uuid4()),
                content=data["content"],
                summary=data.get("summary", ""),
                memory_type=memory_type,
                tags=data.get("tags", []),
                weight=data.get("weight", 1.0),
                source=data.get("source", "import"),
                metadata=data.get("metadata") or {},
                created_at=created_at,
                accessed_at=accessed_at,
                access_count=data.get("access_count", 0),
                status=status,
            )
        except KeyError as e:
            console.print(f"[red]Missing required field: {e}[/red]")
            return None
