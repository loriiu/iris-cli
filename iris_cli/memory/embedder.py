"""嵌入生成器模块.

使用 sentence-transformers 生成文本向量嵌入。
提供本地模型和简单 hash fallback。
当 sentence-transformers 未安装时，自动降级为 hash 嵌入。
"""

import hashlib
import struct
from typing import Optional

from iris_cli.config import get_config


class Embedder:
    """文本嵌入生成器.

    使用本地 sentence-transformers 模型生成文本向量。
    如果模型不可用，使用基于 hash 的简单嵌入。
    """

    _instance: Optional["Embedder"] = None

    def __init__(self, model_name: Optional[str] = None):
        """初始化嵌入生成器.

        Args:
            model_name: 模型名称，默认使用配置中的模型
        """
        self.model_name = model_name or get_config().embed_model
        self._initialized = False
        self._model = None
        self._dimension = 384  # 默认维度 (MiniLM-L6-v2)

    def _ensure_initialized(self) -> None:
        """确保模型已加载."""
        if self._initialized:
            return

        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, device="cpu")
            self._dimension = self._model.get_sentence_embedding_dimension()
            self._initialized = True
        except ImportError:
            # sentence-transformers 未安装，使用 hash fallback
            self._initialized = True
            self._model = None
        except Exception:
            # 模型加载失败，使用 fallback
            self._initialized = True
            self._model = None

    def _hash_embedding(self, text: str) -> list[float]:
        """生成基于 hash 的简单嵌入向量.

        用于模型不可用时的 fallback。

        Args:
            text: 文本内容

        Returns:
            固定维度的嵌入向量
        """
        # 使用 MD5 hash 生成确定性随机种子
        hash_bytes = hashlib.md5(text.encode()).digest()
        seed = struct.unpack('<Q', hash_bytes[:8])[0]

        # 使用种子生成伪随机向量
        import random
        random.seed(seed)

        embedding = [random.uniform(-1, 1) for _ in range(self._dimension)]

        # L2 归一化
        norm = sum(x * x for x in embedding) ** 0.5
        if norm > 0:
            embedding = [x / norm for x in embedding]

        return embedding

    def embed(self, texts: str | list[str]) -> list[list[float]]:
        """生成文本嵌入.

        Args:
            texts: 单个文本或文本列表

        Returns:
            嵌入向量列表，每个向量为浮点数列表
        """
        self._ensure_initialized()

        if isinstance(texts, str):
            texts = [texts]

        if self._model is not None:
            embeddings = self._model.encode(texts, convert_to_numpy=True)
            return embeddings.tolist()
        else:
            # 使用 fallback
            return [self._hash_embedding(text) for text in texts]

    def embed_single(self, text: str) -> list[float]:
        """生成单个文本的嵌入.

        Args:
            text: 文本内容

        Returns:
            嵌入向量
        """
        return self.embed(texts=[text])[0]

    @property
    def dimension(self) -> int:
        """获取嵌入维度."""
        self._ensure_initialized()
        return self._dimension

    def __call__(self, text: str) -> list[float]:
        """便捷调用方式.

        Args:
            text: 文本内容

        Returns:
            嵌入向量
        """
        return self.embed_single(text)

    @classmethod
    def get_instance(cls, model_name: Optional[str] = None) -> "Embedder":
        """获取单例实例.

        Args:
            model_name: 模型名称

        Returns:
            Embedder 实例
        """
        if cls._instance is None:
            cls._instance = cls(model_name)
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """重置单例实例（用于测试或切换模型）."""
        cls._instance = None
