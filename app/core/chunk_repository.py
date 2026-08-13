from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Sequence

from KBzhy.app.core.document_models import KnowledgeChunk


@dataclass(frozen=True)
class ContextFamily:
    parent: KnowledgeChunk | None
    children: tuple[KnowledgeChunk, ...]


class ChunkRepository:
    _columns = (
        "chunk_id",
        "task_id",
        "doc_id",
        "document_version",
        "parent_chunk_id",
        "chunk_type",
        "content",
        "retrieval_text",
        "content_hash",
        "section_path_json",
        "page_start",
        "page_end",
        "position",
        "token_count",
        "index_version",
        "status",
        "metadata_json",
        "created_at",
        "updated_at",
    )

    def __init__(self, store_or_connection):
        self._conn = getattr(store_or_connection, "_conn", store_or_connection)

    def replace_staging(
        self,
        task_id: str,
        document_id: str,
        version: int,
        chunks: Sequence[KnowledgeChunk],
    ) -> None:
        now = datetime.now()
        values = []
        for chunk in chunks:
            if chunk.document_id != document_id or chunk.document_version != version:
                raise ValueError("chunk document identity does not match staging target")
            values.append(self._serialize(task_id, chunk, "staging", now))
        try:
            cur = self._conn.cursor()
            cur.execute(
                "DELETE FROM document_chunks WHERE task_id=%s AND status='staging'",
                (task_id,),
            )
            if values:
                placeholders = ", ".join(["%s"] * len(self._columns))
                cur.executemany(
                    f"INSERT INTO document_chunks ({', '.join(self._columns)}) VALUES ({placeholders})",
                    values,
                )
            cur.close()
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

    def activate_version(self, document_id: str, version: int) -> None:
        try:
            cur = self._conn.cursor()
            cur.execute(
                """
                SELECT COUNT(*) AS child_count, MAX(index_version) AS index_version
                FROM document_chunks
                WHERE doc_id=%s AND document_version=%s AND status='staging' AND chunk_type='child'
                """,
                (document_id, version),
            )
            row = cur.fetchone() or {}
            if int(row.get("child_count") or 0) < 1:
                raise ValueError("version must contain at least one staging active child")
            index_version = int(row["index_version"])
            cur.execute(
                """
                UPDATE document_chunks SET status='inactive', updated_at=NOW(3)
                WHERE doc_id=%s AND status='active'
                """,
                (document_id,),
            )
            cur.execute(
                """
                UPDATE document_chunks SET status='active', updated_at=NOW(3)
                WHERE doc_id=%s AND document_version=%s AND status='staging'
                """,
                (document_id, version),
            )
            cur.execute(
                """
                UPDATE documents SET current_version=%s, active_index_version=%s, updated_at=NOW(3)
                WHERE doc_id=%s
                """,
                (version, index_version, document_id),
            )
            cur.close()
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

    def discard_task(self, task_id: str) -> None:
        try:
            cur = self._conn.cursor()
            cur.execute(
                "DELETE FROM document_chunks WHERE task_id=%s AND status='staging'",
                (task_id,),
            )
            cur.close()
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

    def list_by_task(self, task_id: str) -> list[KnowledgeChunk]:
        return self._query_chunks(
            "SELECT * FROM document_chunks WHERE task_id=%s ORDER BY position ASC",
            (task_id,),
        )

    def list_active_children(self, document_id: str | None = None) -> list[KnowledgeChunk]:
        if document_id is None:
            return self._query_chunks(
                "SELECT * FROM document_chunks WHERE status='active' AND chunk_type='child' ORDER BY doc_id, position"
            )
        return self._query_chunks(
            "SELECT * FROM document_chunks WHERE status='active' AND chunk_type='child' AND doc_id=%s ORDER BY position",
            (document_id,),
        )

    def get_context_family(self, chunk_id: str, neighbor_window: int = 1) -> ContextFamily:
        if neighbor_window < 0:
            raise ValueError("neighbor_window must be non-negative")
        row = self._query_one(
            "SELECT * FROM document_chunks WHERE chunk_id=%s AND status='active' AND chunk_type='child'",
            (chunk_id,),
        )
        if row is None:
            return ContextFamily(parent=None, children=())
        child = self._from_row(row)
        children = self._query_chunks(
            """
            SELECT * FROM document_chunks
            WHERE parent_chunk_id=%s AND status='active' AND chunk_type='child'
              AND position BETWEEN %s AND %s
            ORDER BY position
            """,
            (child.parent_chunk_id, child.position - neighbor_window, child.position + neighbor_window),
        )
        parent_row = self._query_one(
            "SELECT * FROM document_chunks WHERE chunk_id=%s AND status='active' AND chunk_type='parent'",
            (child.parent_chunk_id,),
        )
        return ContextFamily(
            parent=self._from_row(parent_row) if parent_row else None,
            children=tuple(children),
        )

    def get_active_versions(self, document_ids: Iterable[str]) -> dict[str, int]:
        ids = tuple(dict.fromkeys(document_ids))
        if not ids:
            return {}
        placeholders = ", ".join(["%s"] * len(ids))
        cur = self._conn.cursor()
        cur.execute(
            f"""
            SELECT doc_id, document_version FROM document_chunks
            WHERE status='active' AND chunk_type='child' AND doc_id IN ({placeholders})
            GROUP BY doc_id, document_version
            """,
            ids,
        )
        rows = cur.fetchall()
        cur.close()
        return {row["doc_id"]: int(row["document_version"]) for row in rows}

    @classmethod
    def _serialize(cls, task_id: str, chunk: KnowledgeChunk, status: str, now: datetime) -> tuple:
        return (
            chunk.chunk_id,
            task_id,
            chunk.document_id,
            chunk.document_version,
            chunk.parent_chunk_id,
            chunk.chunk_type,
            chunk.content,
            chunk.retrieval_text,
            chunk.content_hash,
            json.dumps(chunk.section_path, ensure_ascii=False),
            chunk.page_start,
            chunk.page_end,
            chunk.position,
            chunk.token_count,
            chunk.index_version,
            status,
            json.dumps(chunk.metadata, ensure_ascii=False),
            now,
            now,
        )

    @staticmethod
    def _from_row(row: dict) -> KnowledgeChunk:
        section_path = row.get("section_path_json") or "[]"
        metadata = row.get("metadata_json") or "{}"
        return KnowledgeChunk(
            chunk_id=row["chunk_id"],
            document_id=row["doc_id"],
            document_version=int(row["document_version"]),
            parent_chunk_id=row.get("parent_chunk_id"),
            chunk_type=row["chunk_type"],
            content=row["content"],
            retrieval_text=row["retrieval_text"],
            content_hash=row["content_hash"],
            section_path=tuple(json.loads(section_path) if isinstance(section_path, str) else section_path),
            page_start=row.get("page_start"),
            page_end=row.get("page_end"),
            position=int(row["position"]),
            token_count=int(row["token_count"]),
            index_version=int(row.get("index_version") or 1),
            metadata=json.loads(metadata) if isinstance(metadata, str) else metadata,
        )

    def _query_one(self, sql: str, params: tuple = ()) -> dict | None:
        cur = self._conn.cursor()
        cur.execute(sql, params)
        row = cur.fetchone()
        cur.close()
        return row

    def _query_chunks(self, sql: str, params: tuple = ()) -> list[KnowledgeChunk]:
        cur = self._conn.cursor()
        cur.execute(sql, params)
        rows = cur.fetchall()
        cur.close()
        return [self._from_row(row) for row in rows]
