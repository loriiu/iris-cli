"""记忆整合模块 - 合并相似记忆

当多条记忆的语义相似度 > 阈值时，自动合并为一条。
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
import time
import uuid

from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn

from iris_cli.memory.models import Memory, MemoryStatus, MemoryType
from iris_cli.memory.store import MemoryStore


@dataclass
class ConsolidationPair:
    """可整合的记忆对"""
    memory_a: Memory
    memory_b: Memory
    similarity: float
    suggested_action: str = "merge_a_into_b"  # merge_a_into_b | merge_b_into_a


@dataclass
class ConsolidationResult:
    """整合结果"""
    pairs_found: int = 0
    memories_consolidated: int = 0
    memories_preserved: int = 0
    pairs: list[ConsolidationPair] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    duration: float = 0.0  # 耗时（秒）


class Consolidator:
    """记忆整合器

    当多条记忆的语义相似度超过阈值时，自动合并为一条。
    """

    def __init__(
        self,
        store: Optional[MemoryStore] = None,
        threshold: float = 0.85,
        embedder=None,
    ):
        """初始化整合器

        Args:
            store: MemoryStore 实例
            threshold: 相似度阈值，默认 0.85
            embedder: 嵌入生成器（用于计算相似度）
        """
        self._store = store or MemoryStore()
        self._threshold = threshold
        self._embedder = embedder

    @property
    def threshold(self) -> float:
        """获取当前阈值"""
        return self._threshold

    @threshold.setter
    def threshold(self, value: float) -> None:
        """设置阈值"""
        if not 0.0 <= value <= 1.0:
            raise ValueError("Threshold must be between 0.0 and 1.0")
        self._threshold = value

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """计算两条文本的语义相似度

        Args:
            text1: 文本1
            text2: 文本2

        Returns:
            相似度分数 [0.0, 1.0]
        """
        if self._embedder:
            embedding1 = self._embedder.embed([text1])[0]
            embedding2 = self._embedder.embed([text2])[0]
        else:
            embedding1 = self._store.embed([text1])[0]
            embedding2 = self._store.embed([text2])[0]

        # 余弦相似度
        dot_product = sum(a * b for a, b in zip(embedding1, embedding2))
        norm1 = sum(a * a for a in embedding1) ** 0.5
        norm2 = sum(b * b for b in embedding2) ** 0.5

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def find_consolidation_pairs(
        self,
        memory_type: Optional[MemoryType] = None,
    ) -> list[ConsolidationPair]:
        """查找可整合的记忆对

        Args:
            memory_type: 可选，只查找指定类型的记忆

        Returns:
            可整合的记忆对列表

        优化策略:
        - n >= 100: 使用 ChromaDB 语义搜索批量筛选候选对 (top_k=5)，减少 O(n²) 计算
        - n < 100: 直接 O(n²) 两两比较（精确计算）
        """
        # 获取所有 active 状态的记忆
        if memory_type:
            memories = self._store.list(
                status=MemoryStatus.ACTIVE,
                memory_type=memory_type,
                limit=10000,
            )
        else:
            memories = self._store.list(status=MemoryStatus.ACTIVE, limit=10000)

        pairs: list[ConsolidationPair] = []
        n = len(memories)

        if n < 100:
            # 小数据量：直接 O(n²) 两两比较
            pairs = self._find_pairs_brute_force(memories)
        else:
            # 大数据量：使用 ChromaDB 语义搜索优化
            pairs = self._find_pairs_optimized(memories)

        return pairs

    def _find_pairs_brute_force(
        self,
        memories: list[Memory],
    ) -> list[ConsolidationPair]:
        """O(n²) 暴力查找 - 适用于小数据量"""
        pairs: list[ConsolidationPair] = []
        n = len(memories)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
        ) as progress:
            task = progress.add_task(
                "[cyan]暴力扫描记忆对...",
                total=n * (n - 1) // 2,
            )

            for i in range(n):
                for j in range(i + 1, n):
                    mem_a = memories[i]
                    mem_b = memories[j]

                    similarity = self._calculate_similarity(
                        mem_a.content,
                        mem_b.content,
                    )

                    progress.advance(task)

                    if similarity >= self._threshold:
                        action = "merge_b_into_a" if mem_a.weight >= mem_b.weight else "merge_a_into_b"
                        pairs.append(ConsolidationPair(
                            memory_a=mem_a,
                            memory_b=mem_b,
                            similarity=similarity,
                            suggested_action=action,
                        ))

        return pairs

    def _find_pairs_optimized(
        self,
        memories: list[Memory],
    ) -> list[ConsolidationPair]:
        """使用 ChromaDB 语义搜索优化 - 适用于大数据量 (n >= 100)

        策略：对每个记忆，用 ChromaDB 搜索 top_k=5 个候选，然后只对候选对做精确比较
        这样可以把 O(n²) 降低到约 O(n * top_k)
        """
        pairs: list[ConsolidationPair] = []
        memory_map = {m.id: m for m in memories}
        checked_pairs: set[tuple[str, str]] = set()

        # 预热 ChromaDB（如果需要）
        self._warm_up_chroma()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
        ) as progress:
            task = progress.add_task(
                f"[cyan]语义搜索候选对 (n={len(memories)})...",
                total=len(memories),
            )

            for mem in memories:
                # 用 ChromaDB 搜索相似记忆
                candidates = self._store.search_vector(mem.content, top_k=6)

                for candidate_id, chroma_score in candidates:
                    # 跳过自己
                    if candidate_id == mem.id:
                        continue

                    # 跳过已检查的对
                    pair_key = tuple(sorted([mem.id, candidate_id]))
                    if pair_key in checked_pairs:
                        continue

                    checked_pairs.add(pair_key)

                    # 获取候选记忆
                    candidate = memory_map.get(candidate_id)
                    if not candidate:
                        continue

                    # 跳过不同类型的记忆
                    if mem.memory_type != candidate.memory_type:
                        continue

                    # 精确计算相似度
                    similarity = self._calculate_similarity(
                        mem.content,
                        candidate.content,
                    )

                    if similarity >= self._threshold:
                        action = "merge_b_into_a" if mem.weight >= candidate.weight else "merge_a_into_b"
                        pairs.append(ConsolidationPair(
                            memory_a=mem,
                            memory_b=candidate,
                            similarity=similarity,
                            suggested_action=action,
                        ))

                progress.advance(task)

        return pairs

    def _warm_up_chroma(self) -> None:
        """预热 ChromaDB 连接"""
        try:
            self._store.search_vector("warm up", top_k=1)
        except Exception:
            pass

    def _merge_memories(
        self,
        main_memory: Memory,
        to_merge: Memory,
    ) -> None:
        """执行记忆合并

        Args:
            main_memory: 主记忆（保留）
            to_merge: 要合并的记忆
        """
        # 1. 更新主记忆
        now = datetime.now().isoformat()

        # 合并内容到 metadata
        merged_content = main_memory.metadata.get("merged_content", [])
        merged_content.append({
            "content": to_merge.content,
            "original_id": to_merge.id,
            "merged_at": now,
        })
        main_memory.metadata["merged_content"] = merged_content

        # 更新权重
        main_memory.weight = min(
            max(main_memory.weight, to_merge.weight) + 0.05,
            1.0,
        )

        # 更新元数据
        self._store.update_memory(main_memory)

        # 2. 标记被合并记忆
        to_merge.status = MemoryStatus.CONSOLIDATED
        to_merge.metadata["consolidated_into"] = main_memory.id
        to_merge.metadata["consolidated_at"] = now

        self._store.update_memory(to_merge)

        # 3. 从向量存储中删除
        self._store.delete(to_merge.id)

    def consolidate(
        self,
        dry_run: bool = False,
        memory_type: Optional[MemoryType] = None,
        threshold: Optional[float] = None,
    ) -> ConsolidationResult:
        """执行记忆整合

        Args:
            dry_run: 是否只预览不执行
            memory_type: 可选，只整合指定类型
            threshold: 可选，自定义阈值

        Returns:
            整合结果
        """
        if threshold is not None:
            self.threshold = threshold

        result = ConsolidationResult()
        start_time = time.time()

        # 查找可整合的记忆对
        pairs = self.find_consolidation_pairs(memory_type=memory_type)
        result.pairs_found = len(pairs)
        result.pairs = pairs

        if dry_run:
            # 只预览，不执行
            result.memories_preserved = len(pairs) * 2
            return result

        # 执行合并
        processed_ids: set[str] = set()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
        ) as progress:
            task = progress.add_task(
                "[cyan]整合记忆...",
                total=len(pairs),
            )

            for pair in pairs:
                try:
                    # 检查是否已经被处理过
                    if pair.memory_a.id in processed_ids or pair.memory_b.id in processed_ids:
                        progress.advance(task)
                        continue

                    # 确定主记忆和要合并的记忆
                    if pair.suggested_action == "merge_b_into_a":
                        main = pair.memory_a
                        to_merge = pair.memory_b
                    else:
                        main = pair.memory_b
                        to_merge = pair.memory_a

                    # 执行合并
                    self._merge_memories(main, to_merge)

                    processed_ids.add(pair.memory_a.id)
                    processed_ids.add(pair.memory_b.id)
                    result.memories_consolidated += 1

                except Exception as e:
                    result.errors.append(f"合并失败 {pair.memory_a.id}/{pair.memory_b.id}: {str(e)}")

                progress.advance(task)

        result.memories_preserved = result.pairs_found - result.memories_consolidated + (
            len(set(self._store.list(limit=10000))) - len(processed_ids)
        )

        result.duration = time.time() - start_time
        return result


# CLI 兼容别名
ConsolidationEngine = Consolidator
