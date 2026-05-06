"""测试嵌入生成器."""

import pytest
from iris_cli.memory.embedder import Embedder


class TestEmbedder:
    """测试嵌入生成器."""

    def test_singleton_instance(self):
        """测试单例模式."""
        Embedder.reset()
        instance1 = Embedder.get_instance()
        instance2 = Embedder.get_instance()
        assert instance1 is instance2

    def test_reset_instance(self):
        """测试重置单例."""
        Embedder.reset()
        instance1 = Embedder.get_instance()
        Embedder.reset()
        instance2 = Embedder.get_instance()
        assert instance1 is not instance2

    def test_embed_single_text(self):
        """测试单个文本嵌入."""
        Embedder.reset()
        embedder = Embedder.get_instance()
        embedding = embedder.embed_single("这是一个测试文本")

        assert isinstance(embedding, list)
        assert len(embedding) == embedder.dimension
        assert all(-1 <= x <= 1 for x in embedding)

    def test_embed_multiple_texts(self):
        """测试多个文本嵌入."""
        Embedder.reset()
        embedder = Embedder.get_instance()
        texts = ["文本一", "文本二", "文本三"]
        embeddings = embedder.embed(texts)

        assert len(embeddings) == 3
        assert all(len(e) == embedder.dimension for e in embeddings)

    def test_embed_consistency(self):
        """测试嵌入一致性（相同文本产生相同向量）."""
        Embedder.reset()
        embedder = Embedder.get_instance()
        text = "一致性测试"

        emb1 = embedder.embed_single(text)
        emb2 = embedder.embed_single(text)

        assert emb1 == emb2

    def test_embed_different_texts(self):
        """测试不同文本产生不同向量."""
        Embedder.reset()
        embedder = Embedder.get_instance()

        emb1 = embedder.embed_single("文本A")
        emb2 = embedder.embed_single("文本B")

        assert emb1 != emb2

    def test_callable_interface(self):
        """测试可调用接口."""
        Embedder.reset()
        embedder = Embedder.get_instance()

        result = embedder("测试文本")
        assert isinstance(result, list)
        assert len(result) == embedder.dimension

    def test_dimension_property(self):
        """测试维度属性."""
        Embedder.reset()
        embedder = Embedder.get_instance()

        dim = embedder.dimension
        assert isinstance(dim, int)
        assert dim > 0

    def test_hash_embedding_normalized(self):
        """测试 hash 嵌入已归一化."""
        Embedder.reset()
        embedder = Embedder.get_instance()

        embedding = embedder.embed_single("归一化测试")
        norm = sum(x * x for x in embedding) ** 0.5

        # 归一化向量的 L2 范数应该接近 1
        assert abs(norm - 1.0) < 0.0001

    def test_custom_model_name(self):
        """测试自定义模型名称."""
        Embedder.reset()
        embedder = Embedder("test-model")
        assert embedder.model_name == "test-model"

    def test_embedding_list_input(self):
        """测试列表输入."""
        Embedder.reset()
        embedder = Embedder.get_instance()

        result = embedder(["单个元素列表"])
        assert len(result) == 1
        assert len(result[0]) == embedder.dimension

    def test_embedding_empty_string(self):
        """测试空字符串输入."""
        Embedder.reset()
        embedder = Embedder.get_instance()

        embedding = embedder.embed_single("")
        assert isinstance(embedding, list)
        assert len(embedding) == embedder.dimension

    def test_embedding_unicode(self):
        """测试 Unicode 输入."""
        Embedder.reset()
        embedder = Embedder.get_instance()

        embedding = embedder.embed_single("中文测试 🎉")
        assert isinstance(embedding, list)
        assert len(embedding) == embedder.dimension
