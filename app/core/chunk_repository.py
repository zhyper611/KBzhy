from __future__ import annotations

import json
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Iterable, Iterator, Sequence

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

    def __init__(self, store_or_connection=None, *, connection_factory: Callable | None = None):
        if connection_factory is not None:
            self._connection_factory = connection_factory
            self._owns_connections = True
        elif hasattr(store_or_connection, "create_connection"):
            self._connection_factory = store_or_connection.create_connection
            self._owns_connections = True
        elif store_or_connection is not None:
            self._connection_factory = lambda: store_or_connection
            self._owns_connections = False
        else:
            raise ValueError("store, connection, or connection_factory is required")

    @contextmanager
    def _connection(self) -> Iterator:
        connection = self._connection_factory()
        try:
            yield connection
        finally:
            if self._owns_connections:
                connection.close()

    def replace_staging(
        self,
        task_id: str,
        document_id: str,
        version: int,
        chunks: Sequence[KnowledgeChunk],
        parsed_artifact_path: str | None = None,
    ) -> None:
        now = datetime.now()
        values = []
        for chunk in chunks:
            if chunk.document_id != document_id or chunk.document_version != version:
                raise ValueError("chunk document identity does not match staging target")
            values.append(self._serialize(task_id, chunk, "staging", now))
        with self._connection() as connection:
            cursor = connection.cursor()
            try:
                cursor.execute(
                    "SELECT doc_id, task_id, status FROM documents WHERE doc_id=%s FOR UPDATE",
                    (document_id,),
                )
                document = cursor.fetchone()
                if (
                    not document
                    or document.get("task_id") != task_id
                    or document.get("status") == "deleting"
                ):
                    raise ValueError("indexing task no longer owns document")
                cursor.execute(
                    "SELECT * FROM document_index_tasks WHERE task_id=%s FOR UPDATE",
                    (task_id,),
                )
                task = cursor.fetchone()
                if (
                    not task
                    or task.get("doc_id") != document_id
                    or int(task.get("document_version") or 0) != version
                    or task.get("status") not in {"queued", "parsing", "chunking", "indexing"}
                ):
                    raise ValueError("indexing task identity or status does not match staging target")
                cursor.execute(
                    "SELECT * FROM document_versions "
                    "WHERE doc_id=%s AND version=%s AND status='staging' FOR UPDATE",
                    (document_id, version),
                )
                if cursor.fetchone() is None:
                    raise ValueError("staging document version does not exist")
                if parsed_artifact_path is not None:
                    cursor.execute(
                        "UPDATE document_versions SET parsed_artifact_path=%s "
                        "WHERE doc_id=%s AND version=%s AND status='staging'",
                        (parsed_artifact_path, document_id, version),
                    )
                    if cursor.rowcount != 1:
                        raise RuntimeError("staging document version changed during artifact update")
                cursor.execute(
                    "DELETE FROM document_chunks WHERE task_id=%s AND status='staging'",
                    (task_id,),
                )
                if values:
                    placeholders = ", ".join(["%s"] * len(self._columns))
                    cursor.executemany(
                        f"INSERT INTO document_chunks ({', '.join(self._columns)}) VALUES ({placeholders})",
                        values,
                    )
                connection.commit()
            except Exception:
                connection.rollback()
                raise
            finally:
                cursor.close()

    def activate_version(self, document_id: str, version: int, task_id: str) -> None:
        with self._connection() as connection:
            cursor = connection.cursor()
            try:
                cursor.execute(
                    "SELECT doc_id, task_id, status FROM documents WHERE doc_id=%s FOR UPDATE",
                    (document_id,),
                )
                document = cursor.fetchone()
                cursor.execute(
                    "SELECT * FROM document_index_tasks WHERE task_id=%s FOR UPDATE",
                    (task_id,),
                )
                task = cursor.fetchone()
                if (
                    not task
                    or task.get("doc_id") != document_id
                    or int(task.get("document_version") or 0) != version
                    or task.get("status") != "indexing"
                ):
                    raise ValueError("indexing task identity or status does not match activation target")
                if document is None:
                    raise ValueError("document does not exist")
                if document.get("task_id") != task_id or document.get("status") == "deleting":
                    raise ValueError("indexing task no longer owns document")
                cursor.execute(
                    """
                    SELECT * FROM document_versions
                    WHERE doc_id=%s AND version=%s AND status='staging'
                    FOR UPDATE
                    """,
                    (document_id, version),
                )
                version_row = cursor.fetchone()
                if version_row is None:
                    raise ValueError("staging document version does not exist")
                cursor.execute(
                    """
                    SELECT * FROM document_chunks
                    WHERE doc_id=%s AND document_version=%s AND task_id=%s AND status='staging'
                    FOR UPDATE
                    """,
                    (document_id, version, task_id),
                )
                rows = cursor.fetchall()
                index_version = self._validate_staging_batch(rows)
                child_count = sum(row["chunk_type"] == "child" for row in rows)
                if int(task.get("index_version") or 0) != index_version:
                    raise ValueError("indexing task index_version does not match staging batch")
                expected_count = len(rows)
                cursor.execute(
                    """
                    UPDATE document_chunks SET status='inactive', updated_at=NOW(3)
                    WHERE doc_id=%s AND status='active'
                    """,
                    (document_id,),
                )
                cursor.execute(
                    """
                    UPDATE document_chunks SET status='active', updated_at=NOW(3)
                    WHERE doc_id=%s AND document_version=%s AND task_id=%s AND status='staging'
                    """,
                    (document_id, version, task_id),
                )
                if cursor.rowcount != expected_count:
                    raise RuntimeError("activated chunk row count changed during transaction")
                cursor.execute(
                    """
                    UPDATE document_versions SET status='inactive'
                    WHERE doc_id=%s AND status='active'
                    """,
                    (document_id,),
                )
                cursor.execute(
                    """
                    UPDATE document_versions SET status='active'
                    WHERE doc_id=%s AND version=%s AND status='staging'
                    """,
                    (document_id, version),
                )
                if cursor.rowcount != 1:
                    raise RuntimeError("document version status changed during transaction")
                cursor.execute(
                    """
                    UPDATE documents
                    SET content_hash=%s, filename=%s, file_type=%s, storage_path=%s,
                        parser_version=%s, parsed_artifact_path=%s,
                        current_version=%s, active_index_version=%s,
                        status='ready', chunk_count=%s,
                        error_message=NULL, updated_at=NOW(3)
                    WHERE doc_id=%s AND task_id=%s AND status<>'deleting'
                    """,
                    (
                        version_row.get("content_hash"), version_row.get("filename"),
                        version_row.get("file_type"), version_row.get("storage_path"),
                        version_row.get("parser_version"), version_row.get("parsed_artifact_path"),
                        version, index_version, child_count, document_id, task_id,
                    ),
                )
                if cursor.rowcount != 1:
                    raise RuntimeError("document ownership changed during activation")
                cursor.execute(
                    """
                    UPDATE document_index_tasks
                    SET status='ready', error_message=NULL, updated_at=NOW(3)
                    WHERE task_id=%s AND doc_id=%s AND document_version=%s AND status='indexing'
                    """,
                    (task_id, document_id, version),
                )
                if cursor.rowcount != 1:
                    raise RuntimeError("indexing task changed during activation")
                connection.commit()
            except Exception:
                connection.rollback()
                raise
            finally:
                cursor.close()

    def replace_reindex_staging(
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
                raise ValueError("chunk document identity does not match reindex target")
            values.append(self._serialize(task_id, chunk, "staging", now))
        if not values:
            raise ValueError("reindex staging requires chunks")
        with self._connection() as connection:
            cursor = connection.cursor()
            try:
                cursor.execute(
                    "SELECT current_version, status FROM documents WHERE doc_id=%s FOR UPDATE",
                    (document_id,),
                )
                document = cursor.fetchone()
                if (
                    not document
                    or int(document.get("current_version") or 0) != version
                    or document.get("status") == "deleting"
                ):
                    raise ValueError("document version changed during reindex")
                cursor.execute(
                    "SELECT * FROM document_index_tasks WHERE task_id=%s FOR UPDATE",
                    (task_id,),
                )
                task = cursor.fetchone()
                if (
                    not task
                    or task.get("doc_id") != document_id
                    or int(task.get("document_version") or 0) != version
                    or task.get("status") != "reindexing"
                ):
                    raise ValueError("reindex task identity or status does not match target")
                cursor.execute(
                    "SELECT version FROM document_versions "
                    "WHERE doc_id=%s AND version=%s AND status='active' FOR UPDATE",
                    (document_id, version),
                )
                if cursor.fetchone() is None:
                    raise ValueError("active document version does not exist")
                cursor.execute(
                    "DELETE FROM document_chunks WHERE task_id=%s AND status='staging'",
                    (task_id,),
                )
                placeholders = ", ".join(["%s"] * len(self._columns))
                cursor.executemany(
                    f"INSERT INTO document_chunks ({', '.join(self._columns)}) VALUES ({placeholders})",
                    values,
                )
                connection.commit()
            except Exception:
                connection.rollback()
                raise
            finally:
                cursor.close()

    @staticmethod
    def _validate_staging_batch(rows: Sequence[dict]) -> int:
        children = [row for row in rows if row["chunk_type"] == "child"]
        if not children:
            raise ValueError("version must contain at least one staging active child")
        parent_ids = {row["chunk_id"] for row in rows if row["chunk_type"] == "parent"}
        if not parent_ids:
            raise ValueError("version must contain at least one staging parent")
        if any(row.get("parent_chunk_id") not in parent_ids for row in children):
            raise ValueError("every child parent must belong to the same staging batch")
        index_versions = {int(row["index_version"]) for row in rows}
        if len(index_versions) != 1:
            raise ValueError("staging batch must use one index_version")
        return index_versions.pop()

    def discard_task(self, task_id: str) -> None:
        with self._connection() as connection:
            cursor = connection.cursor()
            try:
                cursor.execute(
                    "DELETE FROM document_chunks WHERE task_id=%s AND status='staging'",
                    (task_id,),
                )
                connection.commit()
            except Exception:
                connection.rollback()
                raise
            finally:
                cursor.close()

    def list_by_task(self, task_id: str) -> list[KnowledgeChunk]:
        with self._connection() as connection:
            return self._query_chunks(
                connection,
                """
                SELECT * FROM document_chunks WHERE task_id=%s
                ORDER BY position ASC, CASE WHEN chunk_type='parent' THEN 0 ELSE 1 END, row_id
                """,
                (task_id,),
            )

    def list_active_children(self, document_id: str | None = None) -> list[KnowledgeChunk]:
        with self._connection() as connection:
            if document_id is None:
                return self._query_chunks(
                    connection,
                    "SELECT * FROM document_chunks WHERE status='active' AND chunk_type='child' ORDER BY doc_id, position",
                )
            return self._query_chunks(
                connection,
                "SELECT * FROM document_chunks WHERE doc_id=%s AND status='active' AND chunk_type='child' ORDER BY position",
                (document_id,),
            )

    def get_context_family(self, chunk_id: str, neighbor_window: int = 1) -> ContextFamily:
        if neighbor_window < 0:
            raise ValueError("neighbor_window must be non-negative")
        with self._connection() as connection:
            row = self._query_one(
                connection,
                "SELECT * FROM document_chunks WHERE chunk_id=%s AND status='active' AND chunk_type='child'",
                (chunk_id,),
            )
            if row is None:
                return ContextFamily(parent=None, children=())
            child = self._from_row(row)
            batch_identity = (
                row["task_id"],
                child.document_id,
                child.document_version,
            )
            children = self._query_chunks(
                connection,
                """
                SELECT * FROM document_chunks
                WHERE task_id=%s AND doc_id=%s AND document_version=%s
                  AND parent_chunk_id=%s AND status='active' AND chunk_type='child'
                  AND position BETWEEN %s AND %s
                ORDER BY position
                """,
                batch_identity
                + (child.parent_chunk_id, child.position - neighbor_window, child.position + neighbor_window),
            )
            parent_row = self._query_one(
                connection,
                """
                SELECT * FROM document_chunks
                WHERE task_id=%s AND doc_id=%s AND document_version=%s
                  AND chunk_id=%s AND status='active' AND chunk_type='parent'
                """,
                batch_identity + (child.parent_chunk_id,),
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
        with self._connection() as connection:
            cursor = connection.cursor()
            try:
                cursor.execute(
                    f"""
                    SELECT doc_id, document_version
                    FROM document_chunks
                    WHERE status='active' AND chunk_type='child'
                      AND doc_id IN ({placeholders})
                    GROUP BY doc_id, document_version
                    """,
                    ids,
                )
                rows = cursor.fetchall()
            finally:
                cursor.close()
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

    @staticmethod
    def _query_one(connection, sql: str, params: tuple = ()) -> dict | None:
        cursor = connection.cursor()
        try:
            cursor.execute(sql, params)
            return cursor.fetchone()
        finally:
            cursor.close()

    @classmethod
    def _query_chunks(cls, connection, sql: str, params: tuple = ()) -> list[KnowledgeChunk]:
        cursor = connection.cursor()
        try:
            cursor.execute(sql, params)
            rows = cursor.fetchall()
        finally:
            cursor.close()
        return [cls._from_row(row) for row in rows]
