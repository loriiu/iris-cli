"""CLI 主入口.

定义根命令和全局选项。
"""

from pathlib import Path

import typer

from iris_cli.cli.memory import memory_app
from iris_cli.cli.utils import console, print_success, print_info
from iris_cli.config import get_config, ensure_initialized, init_config

app = typer.Typer(
    name="iris",
    help="iris-cli - Agent 记忆系统",
    add_completion=False,
)

# 注册子命令组
app.add_typer(memory_app, name="memory")


@app.callback()
def main_callback(
    ctx: typer.Context,
    config_path: Path = typer.Option(
        None,
        "--config",
        "-c",
        help="配置文件路径",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="详细输出",
    ),
) -> None:
    """iris-cli - Agent 记忆系统 CLI 工具."""
    # 设置配置路径
    if config_path:
        # 临时覆盖配置路径
        pass

    if verbose:
        import logging
        logging.basicConfig(level=logging.DEBUG)


@app.command("init")
def init_command() -> None:
    """初始化 iris-cli 配置.

    示例:
        iris init
    """
    try:
        config = init_config()
        print_success("iris-cli 初始化完成")
        console.print(f"[dim]数据目录: {config.data_dir}[/dim]")
        console.print(f"[dim]配置文件: ~/.iris/config.toml[/dim]")
    except Exception as e:
        console.print(f"[red]初始化失败: {e}[/red]")
        raise typer.Exit(1)


@app.command("doctor")
def doctor_command() -> None:
    """检查环境配置.

    示例:
        iris doctor
    """
    from iris_cli.memory.embedder import Embedder

    console.print("\n[bold]iris-cli 环境检查[/bold]\n")

    # 检查配置
    try:
        config = get_config()
        print_success(f"配置: {config.config_file}")
    except Exception as e:
        print_info(f"配置: 未初始化 ({e})")

    # 检查目录
    try:
        config = get_config()
        data_dir = Path(config.data_dir).expanduser()
        if data_dir.exists():
            print_success(f"数据目录: {data_dir}")
        else:
            print_info(f"数据目录: {data_dir} (不存在)")
    except Exception:
        print_info("数据目录: 未检查")

    # 检查嵌入模型
    try:
        embedder = Embedder.get_instance()
        dim = embedder.dimension
        print_success(f"嵌入模型: {embedder.model_name} (维度: {dim})")
    except Exception as e:
        print_info(f"嵌入模型: 未加载 ({e})")

    # 检查依赖
    console.print("\n[bold]依赖检查[/bold]")
    deps = [
        ("typer", "typer"),
        ("rich", "rich"),
        ("chromadb", "chromadb"),
        ("sentence-transformers", "sentence_transformers"),
        ("pydantic", "pydantic"),
    ]

    for name, import_name in deps:
        try:
            __import__(import_name)
            print_success(f"{name}: 已安装")
        except ImportError:
            print_info(f"{name}: 未安装")

    console.print()


def main() -> None:
    """入口函数."""
    app()


if __name__ == "__main__":
    main()
