"""CLI 工具函数.

提供 CLI 输出格式化和通用工具。
"""

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from typing import Optional

console = Console()


def print_success(message: str) -> None:
    """打印成功消息.

    Args:
        message: 消息内容
    """
    console.print(f"[green]✓[/green] {message}")


def print_error(message: str) -> None:
    """打印错误消息.

    Args:
        message: 错误内容
    """
    console.print(f"[red]✗[/red] {message}", style="red")


def print_info(message: str) -> None:
    """打印信息消息.

    Args:
        message: 消息内容
    """
    console.print(f"[blue]ℹ[/blue] {message}")


def print_warning(message: str) -> None:
    """打印警告消息.

    Args:
        message: 警告内容
    """
    console.print(f"[yellow]⚠[/yellow] {message}")


def print_memory_table(memories: list, title: str = "记忆列表") -> None:
    """打印记忆表格.

    Args:
        memories: 记忆列表
        title: 表格标题
    """
    if not memories:
        console.print("[dim]暂无记忆[/dim]")
        return

    table = Table(title=title, show_header=True, header_style="bold magenta")
    table.add_column("ID", style="cyan", width=8)
    table.add_column("类型", width=10)
    table.add_column("摘要", width=40)
    table.add_column("权重", width=8, justify="right")
    table.add_column("状态", width=8)
    table.add_column("访问次数", width=10, justify="right")

    for memory in memories:
        table.add_row(
            memory.id[:8],
            memory.memory_type.value,
            memory.summary[:38] + ".." if len(memory.summary) > 38 else memory.summary,
            f"{memory.weight:.2f}",
            memory.status.value,
            str(memory.access_count),
        )

    console.print(table)


def print_memory_detail(memory) -> None:
    """打印记忆详情.

    Args:
        memory: Memory 对象
    """
    from rich.table import Table

    table = Table(title=f"记忆详情 - {memory.id[:8]}", show_header=False)
    table.add_column("字段", style="cyan")
    table.add_column("内容")

    table.add_row("ID", memory.id)
    table.add_row("类型", memory.memory_type.value)
    table.add_row("摘要", memory.summary)
    table.add_row("权重", f"{memory.weight:.2f}")
    table.add_row("状态", memory.status.value)
    table.add_row("来源", memory.source)
    table.add_row("标签", ", ".join(memory.tags) if memory.tags else "-")
    table.add_row("访问次数", str(memory.access_count))
    table.add_row("创建时间", memory.created_at.strftime("%Y-%m-%d %H:%M:%S"))
    table.add_row("最近访问", memory.accessed_at.strftime("%Y-%m-%d %H:%M:%S"))
    table.add_row("内容", memory.content)

    console.print(table)


def print_search_results(results: list, query: str) -> None:
    """打印搜索结果.

    Args:
        results: 搜索结果列表
        query: 查询文本
    """
    if not results:
        console.print(f"[dim]未找到与 '{query}' 相关的记忆[/dim]")
        return

    console.print(f"\n[bold]搜索结果 (关键词: '{query}')[/bold]\n")

    for i, result in enumerate(results, 1):
        memory = result.get("memory") or result.get("obj")
        score = result.get("score", result.get("similarity", 0))

        panel = Panel(
            f"[bold cyan]{memory.summary}[/bold cyan]\n\n"
            f"{memory.content[:200]}{'...' if len(memory.content) > 200 else ''}\n\n"
            f"[dim]类型: {memory.memory_type.value} | "
            f"权重: {memory.weight:.2f} | "
            f"相似度: {score:.2f}[/dim]",
            title=f"#{i} - {memory.id[:8]}",
            border_style="blue",
        )
        console.print(panel)


def print_stats(
    total: int,
    by_type: dict[str, int],
    by_status: dict[str, int],
    avg_weight: float,
    weight_distribution: dict[str, int] | None = None,
    top_tags: list[tuple[str, int]] | None = None,
    recent_activity: dict | None = None,
    storage_size: dict | None = None,
) -> None:
    """打印统计信息.

    Args:
        total: 总数
        by_type: 按类型统计
        by_status: 按状态统计
        avg_weight: 平均权重
        weight_distribution: 权重分布
        top_tags: 热门标签
        recent_activity: 近期活动统计（7天内新增/衰减/访问）
        storage_size: 存储大小信息
    """
    from rich.table import Table
    from rich.panel import Panel

    # 总体统计
    table = Table(title="[bold]记忆统计[/bold]", show_header=False)
    table.add_column("指标", style="cyan")
    table.add_column("数值", justify="right")

    table.add_row("总记忆数", f"[bold]{total}[/bold]")
    table.add_row("平均权重", f"{avg_weight:.3f}")

    console.print(table)

    # 存储大小
    if storage_size:
        size_table = Table(title="存储信息", show_header=False)
        size_table.add_column("指标", style="cyan")
        size_table.add_column("大小", justify="right")
        size_table.add_row("SQLite数据库", storage_size.get("sqlite", "N/A"))
        size_table.add_row("向量数据库", storage_size.get("chroma", "N/A"))
        size_table.add_row("总计", f"[bold]{storage_size.get('total', 'N/A')}[/bold]")
        console.print(size_table)

    # 近期活动
    if recent_activity:
        activity_table = Table(title="近期活动 (7天内)", show_header=False)
        activity_table.add_column("指标", style="cyan")
        activity_table.add_column("数量", justify="right")
        activity_table.add_row("新增记忆", str(recent_activity.get("new_count", 0)))
        activity_table.add_row("衰减记忆", str(recent_activity.get("decayed_count", 0)))
        activity_table.add_row("被访问", str(recent_activity.get("accessed_count", 0)))
        if recent_activity.get("last_accessed"):
            activity_table.add_row("最近访问", recent_activity.get("last_accessed"))
        console.print(activity_table)

    # 类型分布
    type_table = Table(title="按类型分布", show_header=True)
    type_table.add_column("类型", style="cyan")
    type_table.add_column("数量", justify="right")
    type_table.add_column("占比", justify="right")
    type_table.add_column("分布", style="green")

    for mem_type, count in by_type.items():
        pct = (count / total * 100) if total > 0 else 0
        bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
        type_table.add_row(mem_type, str(count), f"{pct:.1f}%", bar)

    console.print(type_table)

    # 状态分布
    status_table = Table(title="按状态分布", show_header=True)
    status_table.add_column("状态", style="cyan")
    status_table.add_column("数量", justify="right")
    status_table.add_column("分布", style="green")

    for status, count in by_status.items():
        pct = (count / total * 100) if total > 0 else 0
        bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
        status_table.add_row(status, str(count), bar)

    console.print(status_table)

    # 权重分布
    if weight_distribution:
        weight_table = Table(title="权重分布", show_header=True)
        weight_table.add_column("区间", style="cyan")
        weight_table.add_column("数量", justify="right")
        weight_table.add_column("分布", style="green")

        for range_label, count in weight_distribution.items():
            pct = (count / total * 100) if total > 0 else 0
            bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
            weight_table.add_row(range_label, str(count), bar)

        console.print(weight_table)

    # 热门标签
    if top_tags:
        tag_table = Table(title="热门标签 (Top 10)", show_header=True)
        tag_table.add_column("排名", justify="right", style="dim")
        tag_table.add_column("标签", style="cyan")
        tag_table.add_column("使用次数", justify="right")

        for i, (tag, count) in enumerate(top_tags[:10], 1):
            tag_table.add_row(str(i), tag, str(count))

        console.print(tag_table)


def print_decay_stats(
    decayed_count: int,
    archived_count: int,
    preserved_count: int,
    duration: float,
) -> None:
    """打印衰减统计.

    Args:
        decayed_count: 被衰减的记忆数量
        archived_count: 被归档的记忆数量
        preserved_count: 被保留的记忆数量
        duration: 执行耗时
    """
    table = Table(title="衰减统计", show_header=False)
    table.add_column("指标", style="cyan")
    table.add_column("数量", justify="right")

    table.add_row("衰减记忆", str(decayed_count))
    table.add_row("归档记忆", str(archived_count))
    table.add_row("保留记忆", str(preserved_count))
    table.add_row("执行耗时", f"{duration:.2f}s")

    console.print(table)


def print_consolidation_pairs(pairs: list) -> None:
    """打印整合候选对.

    Args:
        pairs: 整合候选对列表
    """
    if not pairs:
        console.print("[yellow]没有找到可整合的记忆对[/yellow]")
        return

    table = Table(title=f"整合候选对 ({len(pairs)} 对)", show_header=True)
    table.add_column("记忆1", style="cyan", no_wrap=False)
    table.add_column("记忆2", style="cyan", no_wrap=False)
    table.add_column("相似度", justify="right", style="green")

    for pair in pairs[:10]:  # 最多显示10对
        mem1 = pair.memory1
        mem2 = pair.memory2
        sim = pair.similarity
        table.add_row(
            mem1.content[:30] + "..." if len(mem1.content) > 30 else mem1.content,
            mem2.content[:30] + "..." if len(mem2.content) > 30 else mem2.content,
            f"{sim:.3f}",
        )

    console.print(table)

    if len(pairs) > 10:
        console.print(f"[dim]... 还有 {len(pairs) - 10} 对未显示[/dim]")


def print_import_export_result(
    operation: str,
    count: int,
    details: str = "",
) -> None:
    """打印导入导出结果.

    Args:
        operation: 操作类型 (import/export)
        count: 处理数量
        details: 详细信息
    """
    icon = "📥" if operation == "import" else "📤"
    verb = "导入" if operation == "import" else "导出"

    console.print(f"\n{icon} [bold]{verb}完成[/bold]")
    console.print(f"  - {verb}记录: [green]{count}[/green]")

    if details:
        console.print(f"  - 详情: {details}")

    console.print()

    if details:
        console.print(f"  - 详情: {details}")

    console.print()


def confirm(prompt: str, default: bool = False) -> bool:
    """确认提示.

    Args:
        prompt: 提示文本
        default: 默认值

    Returns:
        用户选择
    """
    suffix = " [Y/n]" if default else " [y/N]"
    response = console.input(f"{prompt}{suffix}: ").strip().lower()

    if not response:
        return default

    return response in ("y", "yes")
