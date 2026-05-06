"""检索引擎 - 支持语义、关键词和混合搜索。

三种检索策略:
    - 语义搜索: ChromaDB向量相似度
    - 关键词搜索: SQLite FTS5
    - 混合搜索: 语义×0.6 + 关键词×0.4 + 权重×0.3

检索时自动强化命中的记忆。
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional

from iris_cli.config import get_config
from iris_cli.memory.models import Memory, MemoryType
from iris_cli.memory.store import MemoryStore


class SearchMode(str, Enum):
    """搜索模式枚举。"""
    SEMANTIC = "semantic"
    KEYWORD = "keyword"
    HYBRID = "hybrid"


@dataclass
class SearchResult:
    """搜索结果项。"""
    memory: Memory
    score: float
    semantic_score: Optional[float] = None
    keyword_score: Optional[float] = None
    matched_by: Optional[str] = None


@dataclass
class SearchOptions:
    """搜索选项。"""
    mode: SearchMode = SearchMode.HYBRID
    memory_type: Optional[MemoryType] = None
    tags: Optional[list[str]] = None
    from_date: Optional[datetime] = None
    to_date: Optional[datetime] = None
    limit: int = 5
    auto_reinforce: bool = True


class SearchEngine:
    """检索引擎，支持语义、关键词和混合搜索。"""

    def __init__(self, store: Optional[MemoryStore] = None) -> None:
        """初始化检索引擎。

        Args:
            store: 记忆存储实例，如果为None则使用全局实例
        """
        self._store = store

    @property
    def store(self) -> MemoryStore:
        """获取存储实例。"""
        if self._store is None:
            self._store = MemoryStore()
        return self._store

    def search(
        self,
        query: str,
        options: Optional[SearchOptions] = None
    ) -> list[SearchResult]:
        """执行搜索。

        Args:
            query: 搜索查询
            options: 搜索选项

        Returns:
            搜索结果列表，按分数降序排列
        """
        if options is None:
            options = SearchOptions()

        config = get_config()
        if options.limit <= 0:
            options.limit = config.get("memory.search_default_limit", 5)

        if options.mode == SearchMode.SEMANTIC:
            return self._semantic_search(query, options)
        elif options.mode == SearchMode.KEYWORD:
            return self._keyword_search(query, options)
        else:
            return self._hybrid_search(query, options)

    def _semantic_search(
        self,
        query: str,
        options: SearchOptions
    ) -> list[SearchResult]:
        """语义搜索。

        Args:
            query: 搜索查询
            options: 搜索选项

        Returns:
            搜索结果列表
        """
        results = self.store.semantic_search(query, top_k=options.limit * 2)

        search_results: list[SearchResult] = []
        seen_ids: set[str] = set()

        for result in results:
            memory_id = result.memory_id
            if memory_id in seen_ids:
                continue
            seen_ids.add(memory_id)

            memory = self.store.get_memory(memory_id)
            if memory is None:
                continue

            if not self._matches_filters(memory, options):
                continue

            search_results.append(SearchResult(
                memory=memory,
                score=result.distance,
                semantic_score=result.distance,
                matched_by="semantic"
            ))

            if options.auto_reinforce:
                self._reinforce_memory(memory)

        search_results.sort(key=lambda x: x.score, reverse=True)
        return search_results[:options.limit]

    def _keyword_search(
        self,
        query: str,
        options: SearchOptions
    ) -> list[SearchResult]:
        """关键词搜索。

        使用SQLite FTS5进行全文搜索。

        Args:
            query: 搜索查询
            options: 搜索选项

        Returns:
            搜索结果列表
        """
        memories = self.store.keyword_search(query, limit=options.limit * 2)

        search_results: list[SearchResult] = []
        seen_ids: set[str] = set()

        for memory in memories:
            if memory.id in seen_ids:
                continue
            seen_ids.add(memory.id)

            if not self._matches_filters(memory, options):
                continue

            keyword_score = self._calculate_keyword_score(memory.content, query)
            search_results.append(SearchResult(
                memory=memory,
                score=keyword_score,
                keyword_score=keyword_score,
                matched_by="keyword"
            ))

            if options.auto_reinforce:
                self._reinforce_memory(memory)

        search_results.sort(key=lambda x: x.score, reverse=True)
        return search_results[:options.limit]

    def _hybrid_search(
        self,
        query: str,
        options: SearchOptions
    ) -> list[SearchResult]:
        """混合搜索。

        评分公式:
        final_score = semantic_score * 0.6 + keyword_score * 0.4 + weight * 0.3

        Args:
            query: 搜索查询
            options: 搜索选项

        Returns:
            搜索结果列表
        """
        config = get_config()
        semantic_weight = config.get("memory.search_semantic_weight", 0.6)
        keyword_weight = config.get("memory.search_keyword_weight", 0.4)
        weight_boost = config.get("memory.search_weight_boost", 0.3)

        semantic_results = self.store.semantic_search(query, top_k=options.limit * 3)
        keyword_memories = self.store.keyword_search(query, limit=options.limit * 3)

        keyword_scores: dict[str, float] = {}
        for memory in keyword_memories:
            keyword_scores[memory.id] = self._calculate_keyword_score(
                memory.content, query
            )

        seen_ids: set[str] = set()
        combined_results: list[SearchResult] = []

        for result in semantic_results:
            memory_id = result.memory_id
            if memory_id in seen_ids:
                continue

            memory = self.store.get_memory(memory_id)
            if memory is None:
                continue

            if not self._matches_filters(memory, options):
                continue

            seen_ids.add(memory_id)

            semantic_score = result.distance
            keyword_score = keyword_scores.get(memory_id, 0.0)

            if keyword_score > 0:
                final_score = (
                    semantic_score * semantic_weight +
                    keyword_score * keyword_weight +
                    memory.weight * weight_boost
                )
            else:
                final_score = (
                    semantic_score * (semantic_weight + keyword_weight) +
                    memory.weight * weight_boost
                )

            matched_by = "hybrid"
            if semantic_score > 0 and keyword_score == 0:
                matched_by = "semantic"
            elif keyword_score > 0 and semantic_score == 0:
                matched_by = "keyword"

            combined_results.append(SearchResult(
                memory=memory,
                score=final_score,
                semantic_score=semantic_score,
                keyword_score=keyword_score,
                matched_by=matched_by
            ))

        combined_results.sort(key=lambda x: x.score, reverse=True)
        results = combined_results[:options.limit]

        if options.auto_reinforce:
            for result in results:
                self._reinforce_memory(result.memory)

        return results

    def _matches_filters(
        self,
        memory: Memory,
        options: SearchOptions
    ) -> bool:
        """检查记忆是否匹配过滤条件。

        Args:
            memory: 记忆对象
            options: 搜索选项

        Returns:
            是否匹配
        """
        if options.memory_type is not None:
            if memory.memory_type != options.memory_type:
                return False

        if options.tags:
            if not any(tag in memory.tags for tag in options.tags):
                return False

        if options.from_date is not None:
            if memory.created_at < options.from_date:
                return False

        if options.to_date is not None:
            if memory.created_at > options.to_date:
                return False

        return True

    def _calculate_keyword_score(self, content: str, query: str) -> float:
        """计算关键词匹配分数。

        基于查询词在内容中的出现次数。

        Args:
            content: 内容
            query: 查询

        Returns:
            匹配分数 [0.0, 1.0]
        """
        content_lower = content.lower()
        query_lower = query.lower()

        query_words = query_lower.split()
        matches = sum(1 for word in query_words if word in content_lower)

        if not query_words:
            return 0.0

        return min(1.0, matches / len(query_words))

    def _reinforce_memory(self, memory: Memory) -> None:
        """强化记忆。

        Args:
            memory: 记忆对象
        """
        config = get_config()
        reinforce_boost = config.get("memory.reinforce_boost", 0.1)
        max_weight = config.get("memory.max_weight", 1.0)

        memory.weight = min(max_weight, memory.weight + reinforce_boost)
        memory.access_count += 1
        memory.accessed_at = datetime.now()

        archive_threshold = config.get("memory.decay_archive_threshold", 0.3)
        if memory.weight >= archive_threshold:
            from iris_cli.memory.models import MemoryStatus
            memory.status = MemoryStatus.ACTIVE

        self.store.update_memory(memory)
