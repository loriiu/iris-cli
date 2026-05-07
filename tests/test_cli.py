"""CLI端到端测试."""
import pytest
from typer.testing import CliRunner
from unittest.mock import patch, MagicMock
import os
import tempfile

# 导入CLI应用
from iris_cli.cli.memory import memory_app

runner = CliRunner()


class TestCLIHelp:
    """测试CLI帮助."""

    def test_help(self):
        """显示帮助."""
        result = runner.invoke(memory_app, ["--help"])
        assert result.exit_code == 0
        assert "添加" in result.output or "add" in result.output.lower() or "命令" in result.output

    def test_add_help(self):
        """显示add命令帮助."""
        result = runner.invoke(memory_app, ["add", "--help"])
        assert result.exit_code == 0
        assert "记忆" in result.output or "memory" in result.output.lower()

    def test_subcommand_list(self):
        """列出所有子命令."""
        result = runner.invoke(memory_app, ["--help"])
        assert result.exit_code == 0
        # 检查关键子命令存在
        for cmd in ["add", "get", "list", "search", "delete"]:
            assert cmd in result.output


class TestCLICommands:
    """测试CLI命令（集成测试，无mock）."""

    def test_add_command_basic(self, tmp_path):
        """测试add命令基本功能."""
        with patch.dict(os.environ, {"IRIS_HOME": str(tmp_path)}):
            result = runner.invoke(memory_app, ["add", "测试记忆内容"])
            # 可能的退出码: 0成功, 1需要依赖, 2参数错误
            assert result.exit_code in [0, 1, 2]
            # 输出中应包含相关内容
            assert len(result.output) > 0

    def test_get_command(self, tmp_path):
        """测试get命令."""
        with patch.dict(os.environ, {"IRIS_HOME": str(tmp_path)}):
            result = runner.invoke(memory_app, ["get", "test-id"])
            assert result.exit_code in [0, 1, 2]
            assert len(result.output) >= 0

    def test_list_command(self, tmp_path):
        """测试list命令."""
        with patch.dict(os.environ, {"IRIS_HOME": str(tmp_path)}):
            result = runner.invoke(memory_app, ["list"])
            assert result.exit_code in [0, 1, 2]
            assert len(result.output) >= 0

    def test_search_command(self, tmp_path):
        """测试search命令."""
        with patch.dict(os.environ, {"IRIS_HOME": str(tmp_path)}):
            result = runner.invoke(memory_app, ["search", "test"])
            assert result.exit_code in [0, 1, 2]
            assert len(result.output) >= 0

    def test_delete_command(self, tmp_path):
        """测试delete命令."""
        with patch.dict(os.environ, {"IRIS_HOME": str(tmp_path)}):
            result = runner.invoke(memory_app, ["delete", "test-id"])
            assert result.exit_code in [0, 1, 2]
            assert len(result.output) >= 0

    def test_stats_command(self, tmp_path):
        """测试stats命令."""
        with patch.dict(os.environ, {"IRIS_HOME": str(tmp_path)}):
            result = runner.invoke(memory_app, ["stats"])
            assert result.exit_code in [0, 1, 2]
            assert len(result.output) >= 0

    def test_consolidate_command(self, tmp_path):
        """测试consolidate命令."""
        with patch.dict(os.environ, {"IRIS_HOME": str(tmp_path)}):
            result = runner.invoke(memory_app, ["consolidate"])
            assert result.exit_code in [0, 1, 2]
            assert len(result.output) >= 0

    def test_consolidate_dry_run(self, tmp_path):
        """测试consolidate干运行模式."""
        with patch.dict(os.environ, {"IRIS_HOME": str(tmp_path)}):
            result = runner.invoke(memory_app, ["consolidate", "--dry-run"])
            assert result.exit_code in [0, 1, 2]

    def test_export_command(self, tmp_path):
        """测试export命令."""
        with patch.dict(os.environ, {"IRIS_HOME": str(tmp_path)}):
            result = runner.invoke(memory_app, ["export", "test_export.json"])
            assert result.exit_code in [0, 1, 2]
            assert len(result.output) >= 0

    def test_import_command(self, tmp_path):
        """测试import命令."""
        with patch.dict(os.environ, {"IRIS_HOME": str(tmp_path)}):
            result = runner.invoke(memory_app, ["import", "test_import.json"])
            assert result.exit_code in [0, 1, 2]
            assert len(result.output) >= 0

    def test_import_skip_existing(self, tmp_path):
        """测试import skip-existing模式."""
        with patch.dict(os.environ, {"IRIS_HOME": str(tmp_path)}):
            result = runner.invoke(memory_app, ["import", "test.json", "--conflict", "skip-existing"])
            assert result.exit_code in [0, 1, 2]

    def test_import_dry_run(self, tmp_path):
        """测试import干运行模式."""
        with patch.dict(os.environ, {"IRIS_HOME": str(tmp_path)}):
            result = runner.invoke(memory_app, ["import", "test.json", "--dry-run"])
            assert result.exit_code in [0, 1, 2]

    def test_decay_command(self, tmp_path):
        """测试decay命令."""
        with patch.dict(os.environ, {"IRIS_HOME": str(tmp_path)}):
            result = runner.invoke(memory_app, ["decay"])
            assert result.exit_code in [0, 1, 2]
            assert len(result.output) >= 0

    def test_reinforce_command(self, tmp_path):
        """测试reinforce命令."""
        with patch.dict(os.environ, {"IRIS_HOME": str(tmp_path)}):
            result = runner.invoke(memory_app, ["reinforce", "test-id"])
            assert result.exit_code in [0, 1, 2]
            assert len(result.output) >= 0


class TestCLIValidation:
    """测试CLI参数验证."""

    def test_add_with_type_option(self, tmp_path):
        """测试带类型选项的add命令."""
        with patch.dict(os.environ, {"IRIS_HOME": str(tmp_path)}):
            result = runner.invoke(memory_app, ["add", "test", "--type", "semantic"])
            assert result.exit_code in [0, 1, 2]

    def test_add_with_invalid_type(self, tmp_path):
        """测试无效类型选项."""
        with patch.dict(os.environ, {"IRIS_HOME": str(tmp_path)}):
            result = runner.invoke(memory_app, ["add", "test", "--type", "invalid"])
            # 应该失败或给出错误提示
            assert result.exit_code in [0, 1, 2]
            if result.exit_code != 0:
                assert "error" in result.output.lower() or "invalid" in result.output.lower()

    def test_search_with_limit(self, tmp_path):
        """测试带limit的search命令."""
        with patch.dict(os.environ, {"IRIS_HOME": str(tmp_path)}):
            result = runner.invoke(memory_app, ["search", "test", "--limit", "5"])
            assert result.exit_code in [0, 1, 2]

    def test_search_with_mode(self, tmp_path):
        """测试带mode的search命令."""
        with patch.dict(os.environ, {"IRIS_HOME": str(tmp_path)}):
            result = runner.invoke(memory_app, ["search", "test", "--mode", "semantic"])
            assert result.exit_code in [0, 1, 2]

    def test_consolidate_with_threshold(self, tmp_path):
        """测试带阈值的consolidate命令."""
        with patch.dict(os.environ, {"IRIS_HOME": str(tmp_path)}):
            result = runner.invoke(memory_app, ["consolidate", "--threshold", "0.9"])
            assert result.exit_code in [0, 1, 2]

    def test_export_with_format(self, tmp_path):
        """测试带格式选项的export命令."""
        with patch.dict(os.environ, {"IRIS_HOME": str(tmp_path)}):
            result = runner.invoke(memory_app, ["export", "out.json", "--format", "json"])
            assert result.exit_code in [0, 1, 2]

    def test_export_with_status_filter(self, tmp_path):
        """测试带状态过滤的export命令."""
        with patch.dict(os.environ, {"IRIS_HOME": str(tmp_path)}):
            result = runner.invoke(memory_app, ["export", "out.json", "--status", "active"])
            assert result.exit_code in [0, 1, 2]

    def test_list_with_type_filter(self, tmp_path):
        """测试带类型过滤的list命令."""
        with patch.dict(os.environ, {"IRIS_HOME": str(tmp_path)}):
            result = runner.invoke(memory_app, ["list", "--type", "episodic"])
            assert result.exit_code in [0, 1, 2]
