"""存储层模块.

实现 ChromaDB 向量存储和 SQLite 元数据存储。
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Any, List, Dict

import chromadb
from chromadb.config import Settings

from iris_cli.config import get_config
from iris_cli.memory.models import Memory, SearchResult, SearchMode
from iris_cli.memory.embedder import Embedder


@dataclass
class VectorSearchResult:
    """向量搜索结果."""
    memory_id: str
    distance: float


class VectorStore:
    """ChromaDB 向量存储封装.

    管理记忆的向量嵌入和语义搜索。
    """

    COLLECTION_NAME = "iris_memories"

    def __init__(self, persist_directory: Optional[Path] = None):
        """初始化向量存储.

        Args:
            persist_directory: 持久化目录
        """
        if persist_directory is None:
            persist_directory = get_config().chroma_path

        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        self._client: Optional[chromadb.PersistentClient] = None
        self._collection = None
        self._embedder = Embedder.get_instance()

    def _ensure_connected(self) -> None:
        """确保连接已建立."""
        if self._client is None:
            self._client = chromadb.PersistentClient(
                path=str(self.persist_directory),
                settings=Settings(anonymized_telemetry=False),
            )

            # 获取或创建集合，使用自定义嵌入函数
            try:
                self._collection = self._client.get_collection(name=self.COLLECTION_NAME)
            except Exception:
                # 集合不存在，创建新集合
                self._collection = self._client.create_collection(
                    name=self.COLLECTION_NAME,
                    metadata={"hnsw:space": "cosine"},
                )

    def embed(self, texts: str | list[str]) -> list[list[float]]:
        """获取文本的向量嵌入.

        Args:
            texts: 单个文本或文本列表

        Returns:
            向量嵌入列表
        """
        return self._embedder.embed(texts)

    def add(self, memory: Memory) -> None:
        """添加记忆向量.

        Args:
            memory: Memory 对象
        """
        self._ensure_connected()

        embedding = self._embedder.embed_single(memory.content)

        metadata = {
            "memory_type": memory.memory_type.value,
            "tags": json.dumps(memory.tags),
            "weight": memory.weight,
            "source": memory.source,
            "created_at": memory.created_at.isoformat(),
        }

        self._collection.add(
            ids=[memory.id],
            embeddings=[embedding],
            documents=[memory.content],
            metadatas=[metadata],
        )

    def search(
        self,
        query: str,
        top_k: int = 5,
        filter_tags: Optional[list[str]] = None,
        filter_type: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """语义搜索.

        Args:
            query: 查询文本
            top_k: 返回数量
            filter_tags: 标签过滤
            filter_type: 类型过滤

        Returns:
            搜索结果列表
        """
        self._ensure_connected()

        # 构建过滤条件
        where_filter = None
        if filter_type or filter_tags:
            where_filter = {}
            if filter_type:
                where_filter["memory_type"] = filter_type

        # 执行搜索
        results = self._collection.query(
            query_texts=[query],
            n_results=top_k,
            where=where_filter,
            include=["metadatas", "distances", "documents"],
        )

        search_results = []
        if results["ids"] and len(results["ids"]) > 0:
            for i, memory_id in enumerate(results["ids"][0]):
                distance = results["distances"][0][i]
                metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                document = results["documents"][0][i] if results["documents"] else ""

                search_results.append({
                    "id": memory_id,
                    "distance": distance,
                    "similarity": 1.0 - distance,
                    "metadata": metadata,
                    "content": document,
                })

        return search_results

    def update(self, memory: Memory) -> None:
        """更新记忆向量.

        Args:
            memory: Memory 对象
        """
        self._ensure_connected()

        embedding = self._embedder.embed_single(memory.content)

        metadata = {
            "memory_type": memory.memory_type.value,
            "tags": json.dumps(memory.tags),
            "weight": memory.weight,
            "source": memory.source,
            "created_at": memory.created_at.isoformat(),
        }

        # ChromaDB 的 update 需要先删除再添加
        try:
            self._collection.delete(ids=[memory.id])
        except Exception:
            pass

        self._collection.add(
            ids=[memory.id],
            embeddings=[embedding],
            documents=[memory.content],
            metadatas=[metadata],
        )

    def delete(self, memory_id: str) -> None:
        """删除记忆向量.

        Args:
            memory_id: 记忆 ID
        """
        self._ensure_connected()

        try:
            self._collection.delete(ids=[memory_id])
        except Exception:
            pass

    def get_by_id(self, memory_id: str) -> Optional[dict[str, Any]]:
        """根据 ID 获取向量.

        Args:
            memory_id: 记忆 ID

        Returns:
            向量数据字典
        """
        self._ensure_connected()

        try:
            result = self._collection.get(
                ids=[memory_id],
                include=["metadatas", "documents"],
            )

            if result["ids"] and memory_id in result["ids"]:
                idx = result["ids"].index(memory_id)
                return {
                    "id": memory_id,
                    "content": result["documents"][idx] if result["documents"] else "",
                    "metadata": result["metadatas"][idx] if result["metadatas"] else {},
                }
        except Exception:
            pass

        return None

    def close(self) -> None:
        """关闭连接."""
        self._client = None
        self._collection = None

    def __enter__(self) -> "VectorStore":
        """上下文管理器入口."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """上下文管理器出口."""
        self.close()


class MemoryStore:
    """统一存储接口.

    组合 VectorStore 和 MetaStore，提供统一的记忆存储和检索接口。
    """

    def __init__(
        self,
        persist_directory: Optional[Path] = None,
        chroma_path: Optional[Path] = None,
        db_path: Optional[Path] = None,
    ):
        """初始化存储.

        Args:
            persist_directory: 统一持久化目录
            chroma_path: ChromaDB 路径（优先于 persist_directory）
            db_path: SQLite 数据库路径（优先于 persist_directory）
        """
        if persist_directory is None:
            config = get_config()
            persist_directory = config.data_dir
            if chroma_path is None:
                chroma_path = config.chroma_path
            if db_path is None:
                db_path = config.db_path

        self._vector = VectorStore(persist_directory=chroma_path)
        self._meta = MetaStore(db_path=db_path)

    @classmethod
    def for_testing(cls, temp_dir: Path) -> "MemoryStore":
        """为测试创建临时存储.

        Args:
            temp_dir: 临时目录

        Returns:
            MemoryStore 实例
        """
        return cls(
            chroma_path=temp_dir / "chroma",
            db_path=temp_dir / "iris.db",
        )

    # ==================== 向量存储代理 ====================

    def embed(self, texts: str | list[str]) -> list[list[float]]:
        """获取文本的向量嵌入.

        Args:
            texts: 单个文本或文本列表

        Returns:
            向量嵌入列表
        """
        return self._vector.embed(texts)

    def search_vector(self, query: str, top_k: int = 5) -> list[tuple[str, float]]:
        """向量语义搜索.

        Args:
            query: 查询文本
            top_k: 返回数量

        Returns:
            [(memory_id, score), ...]
        """
        return self._vector.search(query, top_k)

    def semantic_search(self, query: str, top_k: int = 5) -> list["VectorSearchResult"]:
        """语义搜索（返回 VectorSearchResult 对象）.

        Args:
            query: 查询文本
            top_k: 返回数量

        Returns:
            VectorSearchResult 列表
        """
        results = self._vector.search(query, top_k)
        return [
            VectorSearchResult(memory_id=mid, distance=score)
            for mid, score in results
        ]

    # ==================== 元数据存储代理 ====================

    def get_memory(self, memory_id: str) -> Optional[Memory]:
        """获取记忆.

        Args:
            memory_id: 记忆 ID

        Returns:
            Memory 对象或 None
        """
        return self._meta.get(memory_id)

    def list_memories(
        self,
        memory_type: Optional[str] = None,
        status: Optional[str] = None,
        tags: Optional[list[str]] = None,
        limit: int = 20,
        offset: int = 0,
    ) -> list[Memory]:
        """列出记忆.

        Args:
            memory_type: 类型过滤
            status: 状态过滤
            tags: 标签过滤
            limit: 数量限制
            offset: 偏移量

        Returns:
            Memory 列表
        """
        return self._meta.list(
            memory_type=memory_type,
            status=status,
            tags=tags,
            limit=limit,
            offset=offset,
        )

    def search_fts(
        self,
        query: str,
        limit: int = 5,
    ) -> list[tuple[str, float, str]]:
        """全文搜索.

        Args:
            query: 查询文本
            limit: 数量限制

        Returns:
            [(memory_id, score, snippet), ...]
        """
        results = self._meta.search_keyword(query, limit)
        return [(r["memory"].id, r["score"], r["memory"].content) for r in results]

    def update_memory(self, memory: Memory) -> bool:
        """更新记忆.

        Args:
            memory: Memory 对象

        Returns:
            是否更新成功
        """
        return self._meta.update(memory)

    def delete_memory(self, memory_id: str) -> bool:
        """删除记忆.

        Args:
            memory_id: 记忆 ID

        Returns:
            是否删除成功
        """
        return self._meta.delete(memory_id)

    def count(self, memory_type: Optional[str] = None, status: Optional[str] = None) -> int:
        """统计数量.

        Args:
            memory_type: 类型过滤
            status: 状态过滤

        Returns:
            数量
        """
        return self._meta.count(memory_type=memory_type, status=status)

    def close(self) -> None:
        """关闭存储."""
        self._meta.close()


class MetaStore:
    """SQLite 元数据存储.

    管理记忆的结构化数据和全文搜索索引。
    """

    def __init__(self, db_path: Optional[Path] = None):
        """初始化元数据存储.

        Args:
            db_path: 数据库路径
        """
        if db_path is None:
            db_path = get_config().db_path

        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._conn: Optional[sqlite3.Connection] = None
        self._init_schema()

    def _init_schema(self) -> None:
        """初始化数据库 schema."""
        conn = self._get_connection()

        cursor = conn.cursor()

        # 创建主表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                summary TEXT,
                memory_type TEXT NOT NULL,
                tags TEXT DEFAULT '[]',
                weight REAL DEFAULT 1.0,
                source TEXT DEFAULT 'manual',
                created_at TEXT NOT NULL,
                accessed_at TEXT NOT NULL,
                access_count INTEGER DEFAULT 0,
                status TEXT DEFAULT 'active',
                metadata TEXT DEFAULT '{}'
            )
        """)

        # 创建索引
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(memory_type)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_memories_status ON memories(status)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_memories_weight ON memories(weight)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_memories_created ON memories(created_at)
        """)

        # 创建全文搜索虚拟表
        cursor.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
                id,
                content,
                summary,
                tags,
                content='memories',
                content_rowid='rowid'
            )
        """)

        conn.commit()

    def _get_connection(self) -> sqlite3.Connection:
        """获取数据库连接."""
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.db_path))
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def add(self, memory: Memory) -> None:
        """添加记忆.

        Args:
            memory: Memory 对象
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        data = memory.to_db_dict()

        cursor.execute("""
            INSERT INTO memories (
                id, content, summary, memory_type, tags, weight,
                source, created_at, accessed_at, access_count, status, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            data["id"],
            data["content"],
            data["summary"],
            data["memory_type"],
            data["tags"],
            data["weight"],
            data["source"],
            data["created_at"],
            data["accessed_at"],
            data["access_count"],
            data["status"],
            data["metadata"],
        ))

        # 添加到全文搜索索引
        cursor.execute("""
            INSERT INTO memories_fts (id, content, summary, tags)
            VALUES (?, ?, ?, ?)
        """, (memory.id, memory.content, memory.summary, data["tags"]))

        conn.commit()

    def get(self, memory_id: str) -> Optional[Memory]:
        """获取记忆.

        Args:
            memory_id: 记忆 ID

        Returns:
            Memory 对象
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM memories WHERE id = ?", (memory_id,))
        row = cursor.fetchone()

        if row:
            return Memory.from_db_row(dict(row))

        return None

    def update(self, memory: Memory) -> bool:
        """更新记忆.

        Args:
            memory: Memory 对象

        Returns:
            是否更新成功
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        data = memory.to_db_dict()

        cursor.execute("""
            UPDATE memories SET
                content = ?,
                summary = ?,
                memory_type = ?,
                tags = ?,
                weight = ?,
                source = ?,
                accessed_at = ?,
                access_count = ?,
                status = ?,
                metadata = ?
            WHERE id = ?
        """, (
            data["content"],
            data["summary"],
            data["memory_type"],
            data["tags"],
            data["weight"],
            data["source"],
            data["accessed_at"],
            data["access_count"],
            data["status"],
            data["metadata"],
            memory.id,
        ))

        # 更新全文搜索索引
        cursor.execute("""
            DELETE FROM memories_fts WHERE id = ?
        """, (memory.id,))

        cursor.execute("""
            INSERT INTO memories_fts (id, content, summary, tags)
            VALUES (?, ?, ?, ?)
        """, (memory.id, memory.content, memory.summary, data["tags"]))

        conn.commit()

        return cursor.rowcount > 0

    def delete(self, memory_id: str) -> bool:
        """删除记忆.

        Args:
            memory_id: 记忆 ID

        Returns:
            是否删除成功
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("DELETE FROM memories WHERE id = ?", (memory_id,))

        # 从全文搜索索引中删除
        cursor.execute("DELETE FROM memories_fts WHERE id = ?", (memory_id,))

        conn.commit()

        return cursor.rowcount > 0

    def list(
        self,
        memory_type: Optional[str] = None,
        status: Optional[str] = None,
        tags: Optional[list[str]] = None,
        limit: int = 20,
        offset: int = 0,
    ) -> list[Memory]:
        """列出记忆.

        Args:
            memory_type: 记忆类型过滤
            status: 状态过滤
            tags: 标签过滤
            limit: 返回数量限制
            offset: 偏移量

        Returns:
            Memory 对象列表
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        conditions = []
        params: list[Any] = []

        if memory_type:
            conditions.append("memory_type = ?")
            params.append(memory_type)

        if status:
            conditions.append("status = ?")
            params.append(status)

        if tags:
            tag_conditions = " OR ".join(["tags LIKE ?"] * len(tags))
            conditions.append(f"({tag_conditions})")
            params.extend([f'%"{tag}"%' for tag in tags])

        where_clause = " AND ".join(conditions) if conditions else "1=1"

        sql = f"""
            SELECT * FROM memories
            WHERE {where_clause}
            ORDER BY created_at DESC
            LIMIT ? OFFSET ?
        """
        params.extend([limit, offset])

        cursor.execute(sql, params)
        rows = cursor.fetchall()

        return [Memory.from_db_row(dict(row)) for row in rows]

    def search_keyword(
        self,
        query: str,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """关键词搜索（使用 FTS5）。

        Args:
            query: 搜索关键词
            limit: 返回数量

        Returns:
            搜索结果列表
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        # FTS5 搜索
        cursor.execute("""
            SELECT m.*, memories_fts.rank
            FROM memories_fts
            JOIN memories m ON memories_fts.id = m.id
            WHERE memories_fts MATCH ?
            ORDER BY rank
            LIMIT ?
        """, (query, limit))

        rows = cursor.fetchall()

        results = []
        for row in rows:
            results.append({
                "memory": Memory.from_db_row(dict(row)),
                "score": 1.0 / (row["rank"] + 1),  # 将排名转换为相似度分数
            })

        return results

    def count(self, memory_type: Optional[str] = None, status: Optional[str] = None) -> int:
        """统计记忆数量.

        Args:
            memory_type: 记忆类型过滤
            status: 状态过滤

        Returns:
            记忆数量
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        conditions = []
        params: list[Any] = []

        if memory_type:
            conditions.append("memory_type = ?")
            params.append(memory_type)

        if status:
            conditions.append("status = ?")
            params.append(status)

        where_clause = " AND ".join(conditions) if conditions else "1=1"

        cursor.execute(f"SELECT COUNT(*) FROM memories WHERE {where_clause}", params)
        return cursor.fetchone()[0]

    def close(self) -> None:
        """关闭连接."""
        if self._conn:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> "MetaStore":
        """上下文管理器入口."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """上下文管理器出口."""
        self.close()
