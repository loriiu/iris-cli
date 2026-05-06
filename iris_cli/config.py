"""配置管理模块.

管理 Iris CLI 的配置，使用 TOML 格式。
配置路径: ~/.iris/config.toml
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any

import tomllib


class IrisConfig:
    """Iris 配置类."""

    DEFAULT_CONFIG = """
[memory]
# 存储路径
data_dir = "~/.iris/data"

# 衰减参数
decay_rate_day1 = 0.10
decay_rate_week1 = 0.05
decay_rate_month1 = 0.01
decay_cleanup_threshold = 0.10
decay_archive_threshold = 0.30

# 强化参数
reinforce_boost = 0.10
max_weight = 1.0

# 检索参数
search_default_mode = "hybrid"
search_semantic_weight = 0.6
search_keyword_weight = 0.4
search_weight_boost = 0.3
search_default_limit = 5

# 整合参数
consolidate_similarity_threshold = 0.85

# 嵌入模型
embed_model = "sentence-transformers/all-MiniLM-L6-v2"
"""

    def __init__(self, config_path: Optional[Path] = None):
        """初始化配置.

        Args:
            config_path: 配置文件路径
        """
        if config_path is None:
            config_path = self._get_default_config_path()

        self.config_path = Path(config_path).expanduser()
        self._config: Dict[str, Any] = {}
        self._load()

    @staticmethod
    def _get_default_config_path() -> Path:
        """获取默认配置路径."""
        return Path.home() / ".iris" / "config.toml"

    def _load(self) -> None:
        """加载配置."""
        if self.config_path.exists():
            with open(self.config_path, "rb") as f:
                self._config = tomllib.load(f)
        else:
            self._create_default_config()
            self._load()

    def _create_default_config(self) -> None:
        """创建默认配置."""
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, "w") as f:
            f.write(self.DEFAULT_CONFIG.strip())

    def get(self, *keys: str, default: Any = None) -> Any:
        """获取配置值.

        Args:
            *keys: 嵌套键路径，如 "memory", "data_dir"
            default: 默认值

        Returns:
            配置值
        """
        value = self._config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value

    @property
    def data_dir(self) -> Path:
        """获取数据目录."""
        return Path(self.get("memory", "data_dir", default="~/.iris/data")).expanduser()

    @property
    def db_path(self) -> Path:
        """获取数据库路径."""
        return self.data_dir / "iris.db"

    @property
    def chroma_path(self) -> Path:
        """获取 ChromaDB 路径."""
        return self.data_dir / "chroma"

    @property
    def embed_model(self) -> str:
        """获取嵌入模型."""
        return self.get("memory", "embed_model", default="sentence-transformers/all-MiniLM-L6-v2")

    @property
    def decay_rate_day1(self) -> float:
        """获取第一天衰减率."""
        return self.get("memory", "decay_rate_day1", default=0.10)

    @property
    def decay_rate_week1(self) -> float:
        """获取第一周衰减率."""
        return self.get("memory", "decay_rate_week1", default=0.05)

    @property
    def decay_rate_month1(self) -> float:
        """获取第一个月衰减率."""
        return self.get("memory", "decay_rate_month1", default=0.01)

    @property
    def decay_cleanup_threshold(self) -> float:
        """获取清理阈值."""
        return self.get("memory", "decay_cleanup_threshold", default=0.10)

    @property
    def decay_archive_threshold(self) -> float:
        """获取归档阈值."""
        return self.get("memory", "decay_archive_threshold", default=0.30)

    @property
    def reinforce_boost(self) -> float:
        """获取强化提升值."""
        return self.get("memory", "reinforce_boost", default=0.10)

    @property
    def max_weight(self) -> float:
        """获取最大权重."""
        return self.get("memory", "max_weight", default=1.0)

    @property
    def search_default_mode(self) -> str:
        """获取默认搜索模式."""
        return self.get("memory", "search_default_mode", default="hybrid")

    @property
    def search_semantic_weight(self) -> float:
        """获取语义搜索权重."""
        return self.get("memory", "search_semantic_weight", default=0.6)

    @property
    def search_keyword_weight(self) -> float:
        """获取关键词搜索权重."""
        return self.get("memory", "search_keyword_weight", default=0.4)

    @property
    def search_weight_boost(self) -> float:
        """获取权重加成."""
        return self.get("memory", "search_weight_boost", default=0.3)

    @property
    def search_default_limit(self) -> int:
        """获取默认搜索限制."""
        return self.get("memory", "search_default_limit", default=5)

    @property
    def consolidate_threshold(self) -> float:
        """获取整合相似度阈值."""
        return self.get("memory", "consolidate_similarity_threshold", default=0.85)

    def ensure_dirs(self) -> None:
        """确保数据目录存在."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.chroma_path.mkdir(parents=True, exist_ok=True)


# 全局配置实例
_config: Optional[IrisConfig] = None


def get_config() -> IrisConfig:
    """获取全局配置实例."""
    global _config
    if _config is None:
        _config = IrisConfig()
    return _config


def reset_config(config_path: Optional[Path] = None) -> None:
    """重置配置实例.

    Args:
        config_path: 新的配置路径
    """
    global _config
    _config = IrisConfig(config_path)


def init_config() -> IrisConfig:
    """初始化配置.

    如果配置文件不存在，则创建默认配置。

    Returns:
        配置实例
    """
    config = get_config()
    config.ensure_dirs()
    return config


def ensure_initialized() -> None:
    """确保系统已初始化.

    如果未初始化，则创建默认配置和目录。
    """
    init_config()
