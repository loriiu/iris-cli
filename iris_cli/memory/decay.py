"""衰减引擎 - 基于艾宾浩斯遗忘曲线管理记忆权重。

公式:
    weight(t) = weight_0 × R(t)
    R(t) = e^(-t/S)
    S = S_0 × (access_count + 1)^0.5

衰减策略:
    - Day 1: -10%
    - Day 2-7: -5%/天
    - Day 8-30: -1%/天
    - weight < 0.1 → decayed
    - weight < 0.3 → archived
"""

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

from iris_cli.config import get_config
from iris_cli.memory.models import Memory, MemoryStatus
from iris_cli.memory.store import MemoryStore


@dataclass
class DecayStats:
    """衰减统计信息。"""
    total_memories: int = 0
    active_count: int = 0
    archived_count: int = 0
    decayed_count: int = 0
    avg_weight: float = 0.0
    min_weight: float = 1.0
    max_weight: float = 0.0
    memories_to_archive: list[str] = field(default_factory=list)
    memories_to_decay: list[str] = field(default_factory=list)


@dataclass
class ReinforceResult:
    """强化操作结果。"""
    memory_id: str
    old_weight: float
    new_weight: float
    old_access_count: int
    new_access_count: int
    old_status: MemoryStatus
    new_status: MemoryStatus


class DecayEngine:
    """艾宾浩斯遗忘曲线衰减引擎。"""

    def __init__(self, store: Optional[MemoryStore] = None) -> None:
        """初始化衰减引擎。

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

    def calculate_decay(
        self,
        memory: Memory,
        current_time: Optional[datetime] = None
    ) -> float:
        """计算给定时间点的记忆权重。

        使用艾宾浩斯遗忘曲线公式:
        weight(t) = weight_0 × e^(-t/S)
        S = S_0 × (access_count + 1)^0.5

        Args:
            memory: 记忆对象
            current_time: 当前时间，默认为now()

        Returns:
            计算后的权重值 [0.0, 1.0]
        """
        if current_time is None:
            current_time = datetime.now()

        config = get_config()
        time_diff = current_time - memory.accessed_at
        days_elapsed = time_diff.total_seconds() / (24 * 3600)

        if days_elapsed < 1.0:
            return memory.weight

        S_0 = config.get("memory", "decay_stability_base", default=7.0)
        S = S_0 * math.sqrt(memory.access_count + 1)

        retention = math.exp(-days_elapsed / S)
        new_weight = memory.weight * retention

        return max(0.0, min(1.0, new_weight))

    def decay_all(self, dry_run: bool = False) -> DecayStats:
        """对所有活跃记忆执行衰减。

        Args:
            dry_run: 如果为True，只计算不保存

        Returns:
            衰减统计信息
        """
        config = get_config()
        archive_threshold = config.get("memory", "decay_archive_threshold", default=0.3)
        cleanup_threshold = config.get("memory", "decay_cleanup_threshold", default=0.1)

        memories = self.store.list_memories(status=MemoryStatus.ACTIVE)
        stats = DecayStats(total_memories=len(memories))

        to_archive: list[str] = []
        to_decay: list[str] = []

        weights: list[float] = []

        for memory in memories:
            new_weight = self.calculate_decay(memory)
            weights.append(new_weight)

            stats.min_weight = min(stats.min_weight, new_weight)
            stats.max_weight = max(stats.max_weight, new_weight)

            if new_weight < cleanup_threshold:
                to_decay.append(memory.id)
                stats.memories_to_decay.append(memory.id)
            elif new_weight < archive_threshold:
                to_archive.append(memory.id)
                stats.memories_to_archive.append(memory.id)
            else:
                stats.active_count += 1

            if not dry_run:
                memory.weight = new_weight

                if new_weight < cleanup_threshold:
                    memory.status = MemoryStatus.DECAYED
                elif new_weight < archive_threshold:
                    memory.status = MemoryStatus.ARCHIVED

                self.store.update_memory(memory)

        stats.archived_count = len(to_archive)
        stats.decayed_count = len(to_decay)

        if weights:
            stats.avg_weight = sum(weights) / len(weights)

        all_memories = self.store.list_memories()
        for m in all_memories:
            if m.status == MemoryStatus.ARCHIVED:
                stats.archived_count += 1
            elif m.status == MemoryStatus.DECAYED:
                stats.decayed_count += 1

        return stats

    def get_stats(self) -> DecayStats:
        """获取衰减统计信息。

        Returns:
            当前衰减统计信息
        """
        stats = DecayStats()

        all_memories = self.store.list_memories()
        stats.total_memories = len(all_memories)

        weights: list[float] = []

        for memory in all_memories:
            weights.append(memory.weight)
            stats.min_weight = min(stats.min_weight, memory.weight)
            stats.max_weight = max(stats.max_weight, memory.weight)

            if memory.status == MemoryStatus.ACTIVE:
                stats.active_count += 1
            elif memory.status == MemoryStatus.ARCHIVED:
                stats.archived_count += 1
            elif memory.status == MemoryStatus.DECAYED:
                stats.decayed_count += 1

        if weights:
            stats.avg_weight = sum(weights) / len(weights)

        return stats

    def reinforce(self, memory_id: str) -> ReinforceResult:
        """强化指定记忆。

        强化效果:
        - weight += reinforce_boost（上限1.0）
        - access_count += 1
        - 重置衰减起点：accessed_at = now()

        Args:
            memory_id: 记忆ID

        Returns:
            强化结果

        Raises:
            MemoryNotFoundError: 如果记忆不存在
        """
        memory = self.store.get_memory(memory_id)
        if memory is None:
            raise MemoryNotFoundError(f"记忆不存在: {memory_id}")

        config = get_config()
        reinforce_boost = config.get("memory", "reinforce_boost", default=0.1)
        max_weight = config.get("memory", "max_weight", default=1.0)

        old_weight = memory.weight
        old_access_count = memory.access_count
        old_status = memory.status

        memory.weight = min(max_weight, memory.weight + reinforce_boost)
        memory.access_count += 1
        memory.accessed_at = datetime.now()

        archive_threshold = config.get("memory", "decay_archive_threshold", default=0.3)
        if memory.weight >= archive_threshold:
            memory.status = MemoryStatus.ACTIVE

        self.store.update_memory(memory)

        return ReinforceResult(
            memory_id=memory_id,
            old_weight=old_weight,
            new_weight=memory.weight,
            old_access_count=old_access_count,
            new_access_count=memory.access_count,
            old_status=old_status,
            new_status=memory.status
        )

    def cleanup_decayed(self, dry_run: bool = False) -> list[str]:
        """清理已衰减的记忆。

        Args:
            dry_run: 如果为True，只返回不删除

        Returns:
            被清理的记忆ID列表
        """
        decayed_memories = self.store.list_memories(status=MemoryStatus.DECAYED)
        decayed_ids = [m.id for m in decayed_memories]

        if not dry_run:
            for memory_id in decayed_ids:
                self.store.delete_memory(memory_id)

        return decayed_ids


class MemoryNotFoundError(Exception):
    """记忆不存在错误。"""
    pass
