"""记忆管理 CLI 命令.

提供 memory 子命令组，支持衰减、检索等高级功能。
"""

from datetime import datetime
from typing import Optional
import uuid

import typer
from rich.console import Console
from rich.table import Table

from iris_cli.cli.utils import (
    console,
    print_success,
    print_error,
    print_info,
    print_memory_table,
    print_memory_detail,
    print_search_results,
    print_stats,
    confirm,
    print_decay_stats,
    print_consolidation_pairs,
    print_import_export_result,
)
from iris_cli.config import ensure_initialized
from iris_cli.memory.models import Memory, MemoryType, MemoryStatus
from iris_cli.memory.store import VectorStore, MetaStore
from iris_cli.memory.decay import DecayEngine, MemoryNotFoundError
from iris_cli.memory.search import SearchEngine, SearchMode, SearchOptions
from iris_cli.memory.consolidate import ConsolidationEngine
from iris_cli.memory.io import MemoryExporter, MemoryImporter

memory_app = typer.Typer(help="记忆管理命令")


def get_stores() -> tuple[VectorStore, MetaStore]:
    """获取存储实例.

    Returns:
        (VectorStore, MetaStore) 元组
    """
    ensure_initialized()
    return VectorStore(), MetaStore()


@memory_app.command("add")
def add_memory(
    content: str = typer.Argument(..., help="记忆内容"),
    memory_type: str = typer.Option(
        "episodic",
        "--type",
        "-t",
        help="记忆类型: episodic/semantic/procedural",
    ),
    tags: Optional[str] = typer.Option(
        None,
        "--tags",
        help="标签，逗号分隔",
    ),
    source: str = typer.Option(
        "manual",
        "--source",
        "-s",
        help="来源标识",
    ),
    importance: int = typer.Option(
        5,
        "--importance",
        "-i",
        help="重要程度 1-10，影响初始权重",
    ),
) -> None:
    """添加新记忆.

    示例:
        iris memory add "用户喜欢蓝色的配色方案" --type semantic --tags design,preference
        iris memory add "今天完成了代码审查" --type episodic
    """
    try:
        # 验证记忆类型
        try:
            mem_type = MemoryType(memory_type)
        except ValueError:
            print_error(f"无效的记忆类型: {memory_type}")
            print_info("有效类型: episodic, semantic, procedural")
            raise typer.Exit(1)

        # 解析标签
        tag_list = []
        if tags:
            tag_list = [t.strip() for t in tags.split(",") if t.strip()]

        # 计算初始权重（重要程度映射到 0.5-1.0）
        weight = 0.5 + (importance / 10) * 0.5

        # 生成摘要（取前100字）
        summary = content[:100] + "..." if len(content) > 100 else content

        # 创建记忆
        now = datetime.now()
        memory = Memory(
            id=str(uuid.uuid4()),
            content=content,
            summary=summary,
            memory_type=mem_type,
            tags=tag_list,
            weight=weight,
            source=source,
            created_at=now,
            accessed_at=now,
            access_count=0,
            status=MemoryStatus.ACTIVE,
            metadata={},
        )

        # 保存到存储
        vector_store, meta_store = get_stores()

        try:
            vector_store.add(memory)
            meta_store.add(memory)
            print_success(f"记忆已添加: {memory.id}")
            console.print(f"[dim]摘要: {summary}[/dim]")
        finally:
            vector_store.close()
            meta_store.close()

    except Exception as e:
        print_error(f"添加记忆失败: {e}")
        raise typer.Exit(1)


@memory_app.command("get")
def get_memory(
    memory_id: str = typer.Argument(..., help="记忆 ID"),
) -> None:
    """获取记忆详情.

    示例:
        iris memory get 550e8400-e29b-41d4-a716-446655440000
    """
    try:
        _, meta_store = get_stores()

        try:
            memory = meta_store.get(memory_id)

            if memory:
                print_memory_detail(memory)
            else:
                print_error(f"记忆不存在: {memory_id}")

        finally:
            meta_store.close()

    except Exception as e:
        print_error(f"获取记忆失败: {e}")
        raise typer.Exit(1)


@memory_app.command("list")
def list_memories(
    memory_type: Optional[str] = typer.Option(
        None,
        "--type",
        "-t",
        help="记忆类型过滤",
    ),
    status: Optional[str] = typer.Option(
        None,
        "--status",
        "-s",
        help="状态过滤: active/archived/decayed",
    ),
    limit: int = typer.Option(
        20,
        "--limit",
        "-l",
        help="返回数量",
    ),
    offset: int = typer.Option(
        0,
        "--offset",
        "-o",
        help="偏移量",
    ),
) -> None:
    """列出记忆.

    示例:
        iris memory list
        iris memory list --type semantic --status active --limit 10
    """
    try:
        _, meta_store = get_stores()

        try:
            # 验证类型和状态
            if memory_type:
                try:
                    MemoryType(memory_type)
                except ValueError:
                    print_error(f"无效的记忆类型: {memory_type}")
                    raise typer.Exit(1)

            if status:
                try:
                    MemoryStatus(status)
                except ValueError:
                    print_error(f"无效的状态: {status}")
                    print_info("有效状态: active, archived, decayed")
                    raise typer.Exit(1)

            memories = meta_store.list(
                memory_type=memory_type,
                status=status,
                limit=limit,
                offset=offset,
            )

            print_memory_table(memories, title=f"记忆列表 (共 {len(memories)} 条)")

        finally:
            meta_store.close()

    except Exception as e:
        print_error(f"列出记忆失败: {e}")
        raise typer.Exit(1)


@memory_app.command("search")
def search_memories(
    query: str = typer.Argument(..., help="搜索关键词"),
    mode: SearchMode = typer.Option(
        SearchMode.HYBRID,
        "--mode",
        "-m",
        help="搜索模式: hybrid/semantic/keyword",
    ),
    memory_type: Optional[str] = typer.Option(
        None,
        "--type",
        "-t",
        help="记忆类型过滤",
    ),
    tags: Optional[str] = typer.Option(
        None,
        "--tags",
        help="标签过滤，逗号分隔",
    ),
    from_date: Optional[str] = typer.Option(
        None,
        "--from",
        help="起始日期 (YYYY-MM-DD)",
    ),
    to_date: Optional[str] = typer.Option(
        None,
        "--to",
        help="结束日期 (YYYY-MM-DD)",
    ),
    limit: int = typer.Option(
        5,
        "--limit",
        "-l",
        help="返回数量",
    ),
    no_reinforce: bool = typer.Option(
        False,
        "--no-reinforce",
        help="禁用自动强化",
    ),
) -> None:
    """搜索记忆.

    示例:
        iris memory search "Python异步编程"
        iris memory search "配色" --mode semantic
        iris memory search "设计" --type semantic --tags design
        iris memory search "调研" --from 2026-04-01 --to 2026-05-01
    """
    try:
        # 解析日期
        from_dt = None
        to_dt = None

        if from_date:
            try:
                from_dt = datetime.strptime(from_date, "%Y-%m-%d")
            except ValueError:
                print_error(f"无效的日期格式: {from_date}")
                print_info("请使用 YYYY-MM-DD 格式")
                raise typer.Exit(1)

        if to_date:
            try:
                to_dt = datetime.strptime(to_date, "%Y-%m-%d")
                to_dt = to_dt.replace(hour=23, minute=59, second=59)
            except ValueError:
                print_error(f"无效的日期格式: {to_date}")
                print_info("请使用 YYYY-MM-DD 格式")
                raise typer.Exit(1)

        # 解析标签
        tag_list = None
        if tags:
            tag_list = [t.strip() for t in tags.split(",") if t.strip()]

        # 解析记忆类型
        mem_type = None
        if memory_type:
            try:
                mem_type = MemoryType(memory_type)
            except ValueError:
                print_error(f"无效的记忆类型: {memory_type}")
                raise typer.Exit(1)

        # 执行搜索
        ensure_initialized()
        search_engine = SearchEngine()

        options = SearchOptions(
            mode=mode,
            memory_type=mem_type,
            tags=tag_list,
            from_date=from_dt,
            to_date=to_dt,
            limit=limit,
            auto_reinforce=not no_reinforce,
        )

        results = search_engine.search(query, options)

        if not results:
            print_info(f"未找到与 '{query}' 相关的记忆")
            return

        # 格式化输出
        formatted_results = [
            {
                "memory": r.memory,
                "score": r.score,
                "source": r.matched_by or mode.value,
            }
            for r in results
        ]

        print_search_results(formatted_results, query)

        if not no_reinforce:
            print_info(f"[dim]已自动强化 {len(results)} 条记忆[/dim]")

    except Exception as e:
        print_error(f"搜索失败: {e}")
        raise typer.Exit(1)


@memory_app.command("delete")
def delete_memory(
    memory_id: str = typer.Argument(..., help="记忆 ID"),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="跳过确认",
    ),
) -> None:
    """删除记忆.

    示例:
        iris memory delete 550e8400-e29b-41d4-a716-446655440000
        iris memory delete 550e8400 --force
    """
    try:
        vector_store, meta_store = get_stores()

        try:
            # 检查是否存在
            memory = meta_store.get(memory_id)
            if not memory:
                print_error(f"记忆不存在: {memory_id}")
                raise typer.Exit(1)

            # 确认删除
            if not force:
                print_info(f"即将删除记忆: {memory_id}")
                if not confirm("确认删除?"):
                    print_info("已取消")
                    return

            # 删除
            vector_store.delete(memory_id)
            meta_store.delete(memory_id)

            print_success(f"记忆已删除: {memory_id}")

        finally:
            vector_store.close()
            meta_store.close()

    except Exception as e:
        print_error(f"删除记忆失败: {e}")
        raise typer.Exit(1)


@memory_app.command("decay")
def decay_memories(
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="预览模式，不实际执行",
    ),
    stats_only: bool = typer.Option(
        False,
        "--stats",
        help="仅显示统计信息",
    ),
    cleanup: bool = typer.Option(
        False,
        "--cleanup",
        help="清理已衰减的记忆",
    ),
) -> None:
    """执行记忆衰减.

    基于艾宾浩斯遗忘曲线衰减记忆权重。

    示例:
        iris memory decay
        iris memory decay --stats
        iris memory decay --cleanup
        iris memory decay --dry-run
    """
    try:
        ensure_initialized()
        decay_engine = DecayEngine()

        # 显示统计
        stats = decay_engine.get_stats()
        _print_decay_stats(stats)

        if stats_only:
            return

        if dry_run:
            print_info("预览模式，仅显示预计变化")
        else:
            print_info("执行衰减...")

        # 执行衰减
        decay_stats = decay_engine.decay_all(dry_run=dry_run)

        if dry_run:
            print_info("\n[yellow]预览结果 (dry-run 模式):[/yellow]")
            _print_decay_stats(decay_stats)
        else:
            print_success("衰减执行完成")

        # 清理
        if cleanup and not dry_run:
            print_info("\n清理已衰减的记忆...")
            cleaned = decay_engine.cleanup_decayed(dry_run=False)
            if cleaned:
                print_success(f"已清理 {len(cleaned)} 条已衰减记忆")
            else:
                print_info("没有需要清理的记忆")

    except Exception as e:
        print_error(f"衰减失败: {e}")
        raise typer.Exit(1)


def _print_decay_stats(stats) -> None:
    """打印衰减统计信息。"""
    table = Table(title="记忆状态统计", show_header=True, header_style="bold magenta")
    table.add_column("指标", style="cyan")
    table.add_column("数量/值", style="green")

    table.add_row("总记忆数", str(stats.total_memories))
    table.add_row("活跃", str(stats.active_count))
    table.add_row("已归档", str(stats.archived_count))
    table.add_row("已衰减", str(stats.decayed_count))
    table.add_row("平均权重", f"{stats.avg_weight:.3f}")
    table.add_row("最小权重", f"{stats.min_weight:.3f}")
    table.add_row("最大权重", f"{stats.max_weight:.3f}")

    if stats.memories_to_archive:
        table.add_row("待归档", str(len(stats.memories_to_archive)))
    if stats.memories_to_decay:
        table.add_row("待衰减", str(len(stats.memories_to_decay)))

    console.print(table)


@memory_app.command("reinforce")
def reinforce_memory(
    memory_id: str = typer.Argument(..., help="记忆 ID"),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="跳过确认",
    ),
) -> None:
    """强化指定记忆.

    强化效果:
    - weight += reinforce_boost（上限1.0）
    - access_count += 1
    - 重置衰减起点

    示例:
        iris memory reinforce 550e8400-e29b-41d4-a716-446655440000
    """
    try:
        ensure_initialized()
        decay_engine = DecayEngine()

        try:
            result = decay_engine.reinforce(memory_id)

            table = Table(title="强化结果", show_header=False)
            table.add_column("字段", style="cyan")
            table.add_column("旧值", style="yellow")
            table.add_column("新值", style="green")

            table.add_row("权重", f"{result.old_weight:.3f}", f"{result.new_weight:.3f}")
            table.add_row("访问次数", str(result.old_access_count), str(result.new_access_count))
            table.add_row("状态", result.old_status.value, result.new_status.value)

            console.print(table)
            print_success(f"记忆已强化: {memory_id}")

        except MemoryNotFoundError:
            print_error(f"记忆不存在: {memory_id}")
            raise typer.Exit(1)

    except Exception as e:
        print_error(f"强化失败: {e}")
        raise typer.Exit(1)


@memory_app.command("stats")
def memory_stats() -> None:
    """显示统计信息.

    示例:
        iris memory stats
    """
    try:
        _, meta_store = get_stores()

        try:
            total = meta_store.count()

            if total == 0:
                print_info("暂无记忆")
                return

            # 按类型统计
            by_type = {}
            for mem_type in MemoryType:
                count = meta_store.count(memory_type=mem_type.value)
                if count > 0:
                    by_type[mem_type.value] = count

            # 按状态统计
            by_status = {}
            for status in MemoryStatus:
                count = meta_store.count(status=status.value)
                if count > 0:
                    by_status[status.value] = count

            # 计算平均权重和权重分布
            memories = meta_store.list(limit=1000)
            if memories:
                avg_weight = sum(m.weight for m in memories) / len(memories)

                # 权重分布
                weight_distribution = {
                    "高 (0.7-1.0)": sum(1 for m in memories if m.weight >= 0.7),
                    "中 (0.4-0.7)": sum(1 for m in memories if 0.4 <= m.weight < 0.7),
                    "低 (0.1-0.4)": sum(1 for m in memories if 0.1 <= m.weight < 0.4),
                    "衰减 (<0.1)": sum(1 for m in memories if m.weight < 0.1),
                }

                # 热门标签统计
                tag_counts: dict[str, int] = {}
                for m in memories:
                    for tag in m.tags:
                        tag_counts[tag] = tag_counts.get(tag, 0) + 1

                top_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)
            else:
                avg_weight = 0
                weight_distribution = None
                top_tags = None

            print_stats(total, by_type, by_status, avg_weight, weight_distribution, top_tags)

        finally:
            meta_store.close()

    except Exception as e:
        print_error(f"获取统计失败: {e}")
        raise typer.Exit(1)


# ============== 记忆整合命令 ==============

@memory_app.command("consolidate")
def consolidate_cmd(
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="预览可整合的记忆对，不执行整合",
    ),
    memory_type: str = typer.Option(
        None,
        "--type",
        "-t",
        help="只整合指定类型的记忆 (episodic/semantic/procedural)",
    ),
    threshold: float = typer.Option(
        None,
        "--threshold",
        help="自定义相似度阈值 (0.0-1.0)",
    ),
) -> None:
    """
    整合相似记忆。

    扫描所有 active 状态的记忆，合并语义相似度超过阈值的记忆对。
    默认阈值: 0.85
    """
    try:
        from iris_cli.memory import ConsolidationEngine
        from iris_cli.memory.models import MemoryType

        engine = ConsolidationEngine()

        # 验证类型
        mem_type = None
        if memory_type:
            try:
                mem_type = MemoryType(memory_type)
            except ValueError:
                print_error(f"无效的记忆类型: {memory_type}")
                raise typer.Exit(1)

        # 验证阈值
        sim_threshold = threshold
        if sim_threshold is not None and not (0 <= sim_threshold <= 1):
            print_error("阈值必须在 0.0 到 1.0 之间")
            raise typer.Exit(1)

        console.print("\n[bold cyan]🔄 记忆整合[/bold cyan]\n")

        if dry_run:
            console.print("[yellow]⚠️  Dry-run 模式：只显示预览，不执行整合[/yellow]\n")

        # 执行或预览整合
        result = engine.consolidate(
            dry_run=dry_run,
            memory_type=mem_type,
            threshold=sim_threshold,
        )

        if result.merged_count == 0 and result.skipped_count == 0:
            console.print("\n[green]✓ 没有找到需要整合的记忆对[/green]\n")
        else:
            console.print(f"\n[green]✓ 整合完成:[/green]")
            console.print(f"  - 合并: {result.merged_count} 对")
            console.print(f"  - 跳过: {result.skipped_count} 对")
            console.print(f"  - 总耗时: {result.duration:.2f}s\n")

    except Exception as e:
        print_error(f"整合失败: {e}")
        raise typer.Exit(1)


# ============== 导入导出命令 ==============

@memory_app.command("export")
def export_cmd(
    output: str = typer.Option(
        "memories.json",
        "--output",
        "-o",
        help="输出文件路径",
    ),
    export_format: str = typer.Option(
        "json",
        "--format",
        "-f",
        help="导出格式 (json/markdown)",
    ),
    memory_type: str = typer.Option(
        None,
        "--type",
        "-t",
        help="只导出指定类型的记忆",
    ),
    status: str = typer.Option(
        None,
        "--status",
        "-s",
        help="只导出指定状态的记忆",
    ),
) -> None:
    """
    导出记忆到文件。

    支持 JSON 和 Markdown 格式。
    """
    try:
        from iris_cli.memory import MemoryExporter
        from iris_cli.memory.models import MemoryType, MemoryStatus

        exporter = MemoryExporter()

        # 验证格式
        valid_formats = ["json", "markdown"]
        if export_format not in valid_formats:
            print_error(f"不支持的格式: {export_format}")
            console.print(f"支持的格式: {', '.join(valid_formats)}")
            raise typer.Exit(1)

        # 验证类型
        mem_type = None
        if memory_type:
            try:
                mem_type = MemoryType(memory_type)
            except ValueError:
                print_error(f"无效的记忆类型: {memory_type}")
                raise typer.Exit(1)

        # 验证状态
        mem_status = None
        if status:
            try:
                mem_status = MemoryStatus(status)
            except ValueError:
                print_error(f"无效的记忆状态: {status}")
                raise typer.Exit(1)

        console.print(f"\n[bold cyan]📤 导出记忆[/bold cyan]\n")
        console.print(f"格式: [yellow]{export_format}[/yellow]")
        console.print(f"输出: [yellow]{output}[/yellow]")

        # 执行导出
        result = exporter.export(
            output_path=output,
            export_format=export_format,
            memory_type=mem_type,
            status=mem_status,
        )

        console.print(f"\n[green]✓ 导出完成[/green]")
        console.print(f"  - 导出记录: {result['exported_count']}")
        console.print(f"  - 输出文件: {result['output_path']}\n")

    except Exception as e:
        print_error(f"导出失败: {e}")
        raise typer.Exit(1)


@memory_app.command("import")
def import_cmd(
    input_path: str = typer.Argument(..., help="导入文件路径"),
    import_format: str = typer.Option(
        "auto",
        "--format",
        "-f",
        help="导入格式 (auto/json/markdown)",
    ),
) -> None:
    """
    从文件导入记忆。

    支持 JSON 和 Markdown 格式，自动检测格式。
    """
    try:
        from iris_cli.memory import MemoryImporter

        importer = MemoryImporter()

        console.print(f"\n[bold cyan]📥 导入记忆[/bold cyan]\n")
        console.print(f"文件: [yellow]{input_path}[/yellow]")

        # 执行导入
        result = importer.import_memories(
            input_path=input_path,
            import_format=import_format,
        )

        console.print(f"\n[green]✓ 导入完成[/green]")
        console.print(f"  - 成功: {result['success_count']}")
        console.print(f"  - 跳过: {result['skipped_count']}")
        console.print(f"  - 失败: {result['failed_count']}\n")

        if result['failed_count'] > 0 and result.get('errors'):
            console.print("[yellow]错误详情:[/yellow]")
            for error in result['errors'][:5]:
                console.print(f"  - {error}")
            if len(result['errors']) > 5:
                console.print(f"  ... 还有 {len(result['errors']) - 5} 个错误")

    except FileNotFoundError:
        print_error(f"文件不存在: {input_path}")
        raise typer.Exit(1)
    except Exception as e:
        print_error(f"导入失败: {e}")
        raise typer.Exit(1)
