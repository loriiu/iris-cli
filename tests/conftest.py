"""pytest 配置和 fixtures."""

import pytest
import tempfile
import shutil
from pathlib import Path


@pytest.fixture
def temp_dir():
    """创建临时目录用于测试."""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def mock_config(temp_dir, monkeypatch):
    """模拟配置."""
    from iris_cli.config import reset_config

    config_path = temp_dir / "config.toml"
    config_content = """
[memory]
data_dir = "{data_dir}"
embed_model = "all-MiniLM-L6-v2"
decay_rate_day1 = 0.10
decay_rate_week1 = 0.05
decay_rate_month1 = 0.01
decay_cleanup_threshold = 0.10
decay_archive_threshold = 0.30
reinforce_boost = 0.10
max_weight = 1.0
search_default_mode = "hybrid"
search_semantic_weight = 0.6
search_keyword_weight = 0.4
search_weight_boost = 0.3
search_default_limit = 5
consolidate_similarity_threshold = 0.85
""".format(data_dir=str(temp_dir / "data"))

    config_path.write_text(config_content)
    reset_config(config_path)

    yield config_path

    # 清理
    from iris_cli.config import reset_config as reset
    reset()
