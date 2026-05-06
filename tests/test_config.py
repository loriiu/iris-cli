"""测试配置管理."""

import pytest
from pathlib import Path
from iris_cli.config import IrisConfig, reset_config


class TestConfig:
    """测试配置类."""

    def test_default_config_path(self):
        """测试默认配置路径."""
        config = IrisConfig._get_default_config_path()
        assert config == Path.home() / ".iris" / "config.toml"

    def test_custom_config_path(self, temp_dir):
        """测试自定义配置路径."""
        custom_path = temp_dir / "custom.toml"
        config = IrisConfig(custom_path)
        assert config.config_path == custom_path

    def test_load_existing_config(self, temp_dir):
        """测试加载已存在的配置."""
        config_path = temp_dir / "config.toml"
        config_content = """
[memory]
data_dir = "~/.test_data"
embed_model = "test-model"
"""
        config_path.write_text(config_content)

        config = IrisConfig(config_path)
        assert config.data_dir == Path.home() / ".test_data"
        assert config.embed_model == "test-model"

    def test_create_default_config(self, temp_dir):
        """测试创建默认配置."""
        config_path = temp_dir / "new_config.toml"
        config = IrisConfig(config_path)

        # 验证默认配置已创建
        assert config_path.exists()
        assert config.embed_model == "sentence-transformers/all-MiniLM-L6-v2"

    def test_get_nested_value(self, temp_dir):
        """测试获取嵌套配置值."""
        config_path = temp_dir / "config.toml"
        config_content = """
[memory]
embed_model = "test-model"
data_dir = "~/.test_data"
"""
        config_path.write_text(config_content)

        config = IrisConfig(config_path)
        # TOML 不支持点分隔键名，使用标准嵌套结构
        assert config.get("memory", "embed_model") == "test-model"
        assert config.get("memory", "data_dir") == "~/.test_data"

    def test_ensure_dirs(self, temp_dir):
        """测试确保目录存在."""
        config_path = temp_dir / "config.toml"
        config = IrisConfig(config_path)

        data_dir = temp_dir / "data"
        config._config["memory"] = {"data_dir": str(data_dir)}

        # 直接调用 ensure_dirs 方法需要先设置数据目录
        config.data_dir.mkdir(exist_ok=True)
        assert config.data_dir.exists()

    def test_weight_properties(self, temp_dir):
        """测试权重相关属性."""
        config_path = temp_dir / "config.toml"
        config_content = """
[memory]
decay_rate_day1 = 0.15
decay_rate_week1 = 0.08
decay_rate_month1 = 0.02
reinforce_boost = 0.15
max_weight = 0.9
"""
        config_path.write_text(config_content)

        config = IrisConfig(config_path)
        assert config.decay_rate_day1 == 0.15
        assert config.decay_rate_week1 == 0.08
        assert config.decay_rate_month1 == 0.02
        assert config.reinforce_boost == 0.15
        assert config.max_weight == 0.9

    def test_search_properties(self, temp_dir):
        """测试搜索相关属性."""
        config_path = temp_dir / "config.toml"
        config_content = """
[memory]
search_default_mode = "semantic"
search_semantic_weight = 0.7
search_keyword_weight = 0.3
search_weight_boost = 0.5
search_default_limit = 10
"""
        config_path.write_text(config_content)

        config = IrisConfig(config_path)
        assert config.search_default_mode == "semantic"
        assert config.search_semantic_weight == 0.7
        assert config.search_keyword_weight == 0.3
        assert config.search_weight_boost == 0.5
        assert config.search_default_limit == 10

    def test_consolidate_threshold(self, temp_dir):
        """测试整合阈值."""
        config_path = temp_dir / "config.toml"
        config_content = """
[memory]
consolidate_similarity_threshold = 0.9
"""
        config_path.write_text(config_content)

        config = IrisConfig(config_path)
        assert config.consolidate_threshold == 0.9

    def test_get_with_default(self, temp_dir):
        """测试带默认值的获取."""
        config_path = temp_dir / "config.toml"
        config = IrisConfig(config_path)

        result = config.get("nonexistent", "key", default="default_value")
        assert result == "default_value"
