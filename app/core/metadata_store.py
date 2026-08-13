from __future__ import annotations

import logging
import json
import os
import uuid
from datetime import datetime
from typing import Any

from KBzhy.config import (
    DATA_DIR,
    MYSQL_HOST,
    MYSQL_PORT,
    MYSQL_USER,
    MYSQL_PASSWORD,
    MYSQL_DATABASE,
)

logger = logging.getLogger(__name__)


_COLUMN_MIGRATIONS = {
    ("knowledge_bases", "active_collection_name"): "VARCHAR(255) NULL",
    ("documents", "content_hash"): "VARCHAR(64) NULL",
    ("documents", "current_version"): "INT NOT NULL DEFAULT 1",
    ("documents", "parser_version"): "VARCHAR(64) NULL",
    ("documents", "active_index_version"): "INT NOT NULL DEFAULT 1",
    ("documents", "parsed_artifact_path"): "TEXT NULL",
    ("document_index_tasks", "document_version"): "INT NOT NULL DEFAULT 1",
    ("document_index_tasks", "index_version"): "INT NOT NULL DEFAULT 1",
    ("document_index_tasks", "attempt_count"): "INT NOT NULL DEFAULT 0",
    ("document_versions", "filename"): "VARCHAR(255) NULL",
    ("document_versions", "file_type"): "VARCHAR(32) NULL",
    ("document_versions", "storage_path"): "TEXT NULL",
}

_CHUNK_INDEXES = {
    "uq_chunks_task_chunk": ("UNIQUE KEY", ("task_id", "chunk_id")),
    "idx_chunks_doc_active_children": ("INDEX", ("doc_id", "status", "chunk_type", "position")),
    "idx_chunks_active_children": ("INDEX", ("status", "chunk_type", "doc_id", "position")),
}

_CHUNK_FOREIGN_KEYS = {
    "fk_document_chunks_version": (
        ("doc_id", "document_version"),
        "document_versions",
        ("doc_id", "version"),
        "CASCADE",
    ),
}


class MetadataStoreUnavailable(RuntimeError):
    pass


def now_iso() -> str:
    return datetime.now().isoformat()


class MySQLMetadataStore:
    def __init__(self):
        try:
            import pymysql
            from pymysql.cursors import DictCursor

            self._pymysql = pymysql
            self._connect_kwargs = {
                "host": MYSQL_HOST,
                "port": MYSQL_PORT,
                "user": MYSQL_USER,
                "password": MYSQL_PASSWORD,
                "database": MYSQL_DATABASE,
                "charset": "utf8mb4",
                "autocommit": False,
                "cursorclass": DictCursor,
                "connect_timeout": 5,
            }
            self._conn = self._connect()
            self._ensure_schema()
            self._migrate_legacy_json_if_present()
        except Exception as exc:
            raise MetadataStoreUnavailable(f"MySQL metadata store unavailable: {exc}") from exc

    def _connect(self):
        return self._pymysql.connect(**self._connect_kwargs)

    def create_connection(self):
        return self._connect()

    def _reconnect(self):
        try:
            self._conn.close()
        except Exception:
            pass
        self._conn = self._connect()

    def _rollback_quietly(self):
        try:
            self._conn.rollback()
        except Exception as exc:
            logger.warning("MySQL rollback skipped after connection error: %s", exc)

    def _execute(self, sql: str, params: tuple = ()):
        try:
            cur = self._conn.cursor()
            cur.execute(sql, params)
            return cur
        except Exception:
            try:
                try:
                    self._conn.ping(reconnect=True)
                except Exception:
                    self._reconnect()
                cur = self._conn.cursor()
                cur.execute(sql, params)
                return cur
            except Exception:
                self._rollback_quietly()
                raise

    def _ensure_schema(self):
        statements = [
            """
            CREATE TABLE IF NOT EXISTS knowledge_bases (
                kb_id VARCHAR(64) PRIMARY KEY,
                name VARCHAR(100) NOT NULL,
                description VARCHAR(500) NOT NULL DEFAULT '',
                created_at DATETIME(3) NOT NULL
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """,
            """
            CREATE TABLE IF NOT EXISTS documents (
                doc_id VARCHAR(128) PRIMARY KEY,
                kb_id VARCHAR(64) NOT NULL,
                filename VARCHAR(255) NOT NULL,
                file_type VARCHAR(32) NOT NULL DEFAULT '',
                status VARCHAR(32) NOT NULL,
                chunk_count INT NOT NULL DEFAULT 0,
                task_id VARCHAR(128) DEFAULT NULL,
                storage_path TEXT DEFAULT NULL,
                error_message TEXT DEFAULT NULL,
                created_at DATETIME(3) NOT NULL,
                updated_at DATETIME(3) NOT NULL,
                INDEX idx_documents_kb (kb_id, updated_at),
                INDEX idx_documents_task (task_id),
                CONSTRAINT fk_documents_kb FOREIGN KEY (kb_id)
                    REFERENCES knowledge_bases(kb_id) ON DELETE CASCADE
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """,
            """
            CREATE TABLE IF NOT EXISTS document_index_tasks (
                task_id VARCHAR(128) PRIMARY KEY,
                doc_id VARCHAR(128) NOT NULL,
                kb_id VARCHAR(64) NOT NULL,
                status VARCHAR(32) NOT NULL,
                error_message TEXT DEFAULT NULL,
                created_at DATETIME(3) NOT NULL,
                updated_at DATETIME(3) NOT NULL,
                INDEX idx_tasks_status (status, updated_at),
                INDEX idx_tasks_doc (doc_id),
                CONSTRAINT fk_tasks_doc FOREIGN KEY (doc_id)
                    REFERENCES documents(doc_id) ON DELETE CASCADE
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """,
            """
            CREATE TABLE IF NOT EXISTS document_versions (
                version_id VARCHAR(128) PRIMARY KEY,
                doc_id VARCHAR(128) NOT NULL,
                version INT NOT NULL,
                content_hash VARCHAR(64) NULL,
                filename VARCHAR(255) NULL,
                file_type VARCHAR(32) NULL,
                storage_path TEXT NULL,
                parser_version VARCHAR(64) NULL,
                parsed_artifact_path TEXT NULL,
                status VARCHAR(32) NOT NULL,
                created_at DATETIME(3) NOT NULL,
                UNIQUE KEY uq_document_versions_doc_version (doc_id, version),
                INDEX idx_document_versions_doc_status (doc_id, status),
                CONSTRAINT fk_document_versions_doc FOREIGN KEY (doc_id)
                    REFERENCES documents(doc_id) ON DELETE CASCADE
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """,
            """
            CREATE TABLE IF NOT EXISTS document_chunks (
                row_id BIGINT AUTO_INCREMENT PRIMARY KEY,
                chunk_id VARCHAR(64) NOT NULL,
                task_id VARCHAR(128) NOT NULL,
                doc_id VARCHAR(128) NOT NULL,
                document_version INT NOT NULL,
                parent_chunk_id VARCHAR(64) NULL,
                chunk_type VARCHAR(16) NOT NULL,
                content LONGTEXT NOT NULL,
                retrieval_text LONGTEXT NOT NULL,
                content_hash VARCHAR(64) NOT NULL,
                section_path_json LONGTEXT NOT NULL,
                page_start INT NULL,
                page_end INT NULL,
                position INT NOT NULL,
                token_count INT NOT NULL,
                index_version INT NOT NULL DEFAULT 1,
                status VARCHAR(32) NOT NULL,
                metadata_json LONGTEXT NOT NULL,
                created_at DATETIME(3) NOT NULL,
                updated_at DATETIME(3) NOT NULL,
                UNIQUE KEY uq_chunks_task_chunk (task_id, chunk_id),
                INDEX idx_chunks_doc_version_position (doc_id, document_version, position),
                INDEX idx_chunks_doc_active_children (doc_id, status, chunk_type, position),
                INDEX idx_chunks_active_children (status, chunk_type, doc_id, position),
                INDEX idx_chunks_parent (parent_chunk_id),
                INDEX idx_chunks_status_index (status, index_version),
                INDEX idx_chunks_task (task_id),
                CONSTRAINT fk_document_chunks_version FOREIGN KEY (doc_id, document_version)
                    REFERENCES document_versions(doc_id, version) ON DELETE CASCADE
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """,
        ]
        for sql in statements:
            cur = self._conn.cursor()
            try:
                cur.execute(sql)
            finally:
                cur.close()
        for (table, column), ddl in _COLUMN_MIGRATIONS.items():
            self._ensure_column(table, column, ddl)
        self._ensure_document_chunks_shape()
        self._conn.commit()

    def _ensure_document_chunks_shape(self):
        row_id = self._get_column("document_chunks", "row_id")
        primary_columns = self._get_index_columns("document_chunks", "PRIMARY")
        if not row_id:
            if primary_columns == ("chunk_id",):
                self._execute_chunk_ddl(
                    "ALTER TABLE document_chunks DROP PRIMARY KEY, "
                    "ADD COLUMN row_id BIGINT NOT NULL AUTO_INCREMENT PRIMARY KEY FIRST",
                    {1060, 1068},
                )
            elif not primary_columns:
                self._execute_chunk_ddl(
                    "ALTER TABLE document_chunks "
                    "ADD COLUMN row_id BIGINT NOT NULL AUTO_INCREMENT PRIMARY KEY FIRST",
                    {1060, 1068},
                )
            else:
                raise RuntimeError("unsupported document_chunks primary key shape")
            if not self._document_chunks_primary_matches():
                raise RuntimeError("document_chunks primary key migration did not produce auto-increment row_id")
        elif not self._document_chunks_primary_matches():
            raise RuntimeError("row_id exists but is not the document_chunks primary key")

        chunk_id = self._get_column("document_chunks", "chunk_id")
        if chunk_id and chunk_id.get("IS_NULLABLE") != "NO":
            self._execute_chunk_ddl(
                "ALTER TABLE document_chunks MODIFY COLUMN chunk_id VARCHAR(64) NOT NULL",
                set(),
            )

        for name, (kind, columns) in _CHUNK_INDEXES.items():
            existing = self._get_index_columns("document_chunks", name)
            expected_unique = kind == "UNIQUE KEY"
            if existing == columns and self._index_is_unique("document_chunks", name) == expected_unique:
                continue
            if existing:
                legacy_global_index = (
                    name == "idx_chunks_active_children"
                    and existing == ("status", "chunk_type", "position")
                )
                if not legacy_global_index:
                    raise RuntimeError(f"unexpected document_chunks index shape: {name}")
                self._execute_chunk_ddl(
                    "ALTER TABLE document_chunks DROP INDEX idx_chunks_active_children",
                    set(),
                )
            column_sql = ", ".join(columns)
            self._execute_chunk_ddl(
                f"ALTER TABLE document_chunks ADD {kind} {name} ({column_sql})",
                {1061},
            )
            if (
                self._get_index_columns("document_chunks", name) != columns
                or self._index_is_unique("document_chunks", name) != expected_unique
            ):
                raise RuntimeError(f"document_chunks index shape did not match after migration: {name}")

        for name, (columns, referenced_table, referenced_columns, delete_rule) in _CHUNK_FOREIGN_KEYS.items():
            existing = self._get_foreign_key("document_chunks", name)
            expected = (columns, referenced_table, referenced_columns, delete_rule)
            if existing == expected:
                continue
            if existing:
                raise RuntimeError(f"unexpected document_chunks foreign key shape: {name}")
            self._backfill_missing_document_versions()
            column_sql = ", ".join(columns)
            referenced_sql = ", ".join(referenced_columns)
            self._execute_chunk_ddl(
                f"ALTER TABLE document_chunks ADD CONSTRAINT {name} "
                f"FOREIGN KEY ({column_sql}) REFERENCES {referenced_table} ({referenced_sql}) ON DELETE {delete_rule}",
                {1826},
            )
            if self._get_foreign_key("document_chunks", name) != expected:
                raise RuntimeError(f"document_chunks foreign key shape did not match after migration: {name}")

    def _document_chunks_primary_matches(self) -> bool:
        row_id = self._get_column("document_chunks", "row_id")
        return bool(
            row_id
            and "auto_increment" in (row_id.get("EXTRA") or "").lower()
            and self._get_index_columns("document_chunks", "PRIMARY") == ("row_id",)
        )

    def _backfill_missing_document_versions(self):
        cur = self._conn.cursor()
        try:
            cur.execute(
                """
                INSERT IGNORE INTO document_versions
                    (version_id, doc_id, version, content_hash, parser_version,
                     parsed_artifact_path, status, created_at)
                SELECT
                    SHA2(CONCAT('legacy:', dc.doc_id, ':', dc.document_version), 256),
                    dc.doc_id,
                    dc.document_version,
                    CASE WHEN d.current_version=dc.document_version THEN d.content_hash ELSE NULL END,
                    CASE WHEN d.current_version=dc.document_version THEN d.parser_version ELSE NULL END,
                    CASE WHEN d.current_version=dc.document_version THEN d.parsed_artifact_path ELSE NULL END,
                    CASE WHEN d.current_version=dc.document_version THEN 'active' ELSE 'inactive' END,
                    COALESCE(d.created_at, NOW(3))
                FROM (
                    SELECT DISTINCT doc_id, document_version FROM document_chunks
                ) dc
                INNER JOIN documents d ON d.doc_id=dc.doc_id
                LEFT JOIN document_versions dv
                    ON dv.doc_id=dc.doc_id AND dv.version=dc.document_version
                WHERE dv.doc_id IS NULL
                """
            )
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise
        finally:
            cur.close()

        cur = self._conn.cursor()
        try:
            cur.execute(
                """
                SELECT COUNT(*) AS missing_count
                FROM (
                    SELECT DISTINCT doc_id, document_version FROM document_chunks
                ) dc
                LEFT JOIN document_versions dv
                    ON dv.doc_id=dc.doc_id AND dv.version=dc.document_version
                WHERE dv.doc_id IS NULL
                """
            )
            missing_count = int((cur.fetchone() or {}).get("missing_count") or 0)
        finally:
            cur.close()
        if missing_count:
            raise RuntimeError("document version backfill left chunk versions unresolved")

    def _get_column(self, table: str, column: str) -> dict | None:
        if table != "document_chunks" or column not in {"row_id", "chunk_id"}:
            raise ValueError("column shape lookup is not allowed")
        cur = self._conn.cursor()
        try:
            cur.execute(
                """
                SELECT IS_NULLABLE, EXTRA FROM INFORMATION_SCHEMA.COLUMNS
                WHERE TABLE_SCHEMA=DATABASE() AND TABLE_NAME=%s AND COLUMN_NAME=%s
                """,
                (table, column),
            )
            return cur.fetchone()
        finally:
            cur.close()

    def _get_index_columns(self, table: str, name: str) -> tuple[str, ...]:
        if table != "document_chunks" or name not in {"PRIMARY", *_CHUNK_INDEXES}:
            raise ValueError("index shape lookup is not allowed")
        cur = self._conn.cursor()
        try:
            cur.execute(
                """
                SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.STATISTICS
                WHERE TABLE_SCHEMA=DATABASE() AND TABLE_NAME=%s AND INDEX_NAME=%s
                ORDER BY SEQ_IN_INDEX
                """,
                (table, name),
            )
            return tuple(row["COLUMN_NAME"] for row in cur.fetchall())
        finally:
            cur.close()

    def _index_is_unique(self, table: str, name: str) -> bool:
        if table != "document_chunks" or name not in _CHUNK_INDEXES:
            raise ValueError("index uniqueness lookup is not allowed")
        cur = self._conn.cursor()
        try:
            cur.execute(
                """
                SELECT NON_UNIQUE FROM INFORMATION_SCHEMA.STATISTICS
                WHERE TABLE_SCHEMA=DATABASE() AND TABLE_NAME=%s AND INDEX_NAME=%s
                LIMIT 1
                """,
                (table, name),
            )
            row = cur.fetchone()
            return bool(row is not None and int(row["NON_UNIQUE"]) == 0)
        finally:
            cur.close()

    def _get_foreign_key(self, table: str, name: str):
        if table != "document_chunks" or name not in _CHUNK_FOREIGN_KEYS:
            raise ValueError("foreign key shape lookup is not allowed")
        cur = self._conn.cursor()
        try:
            cur.execute(
                """
                SELECT CONSTRAINT_TYPE FROM INFORMATION_SCHEMA.TABLE_CONSTRAINTS
                WHERE TABLE_SCHEMA=DATABASE() AND TABLE_NAME=%s AND CONSTRAINT_NAME=%s
                """,
                (table, name),
            )
            constraint = cur.fetchone()
            if not constraint:
                return None
            if constraint["CONSTRAINT_TYPE"] != "FOREIGN KEY":
                raise RuntimeError(f"unexpected document_chunks constraint type: {name}")
            cur.execute(
                """
                SELECT COLUMN_NAME, REFERENCED_TABLE_NAME, REFERENCED_COLUMN_NAME
                FROM INFORMATION_SCHEMA.KEY_COLUMN_USAGE
                WHERE TABLE_SCHEMA=DATABASE() AND TABLE_NAME=%s AND CONSTRAINT_NAME=%s
                  AND REFERENCED_TABLE_NAME IS NOT NULL
                ORDER BY ORDINAL_POSITION
                """,
                (table, name),
            )
            rows = cur.fetchall()
            cur.execute(
                """
                SELECT DELETE_RULE FROM INFORMATION_SCHEMA.REFERENTIAL_CONSTRAINTS
                WHERE CONSTRAINT_SCHEMA=DATABASE() AND TABLE_NAME=%s AND CONSTRAINT_NAME=%s
                """,
                (table, name),
            )
            referential = cur.fetchone()
        finally:
            cur.close()
        if not rows:
            return None
        return (
            tuple(row["COLUMN_NAME"] for row in rows),
            rows[0]["REFERENCED_TABLE_NAME"],
            tuple(row["REFERENCED_COLUMN_NAME"] for row in rows),
            (referential or {}).get("DELETE_RULE"),
        )

    def _execute_chunk_ddl(self, sql: str, ignored_errnos: set[int]):
        allowed = {
            "ALTER TABLE document_chunks DROP PRIMARY KEY, ADD COLUMN row_id BIGINT NOT NULL AUTO_INCREMENT PRIMARY KEY FIRST",
            "ALTER TABLE document_chunks ADD COLUMN row_id BIGINT NOT NULL AUTO_INCREMENT PRIMARY KEY FIRST",
            "ALTER TABLE document_chunks MODIFY COLUMN chunk_id VARCHAR(64) NOT NULL",
            "ALTER TABLE document_chunks DROP INDEX idx_chunks_active_children",
            *(
                f"ALTER TABLE document_chunks ADD {kind} {name} ({', '.join(columns)})"
                for name, (kind, columns) in _CHUNK_INDEXES.items()
            ),
            *(
                f"ALTER TABLE document_chunks ADD CONSTRAINT {name} FOREIGN KEY ({', '.join(columns)}) "
                f"REFERENCES {referenced_table} ({', '.join(referenced_columns)}) ON DELETE {delete_rule}"
                for name, (columns, referenced_table, referenced_columns, delete_rule) in _CHUNK_FOREIGN_KEYS.items()
            ),
        }
        if sql not in allowed:
            raise ValueError("chunk schema migration is not allowed")
        cur = self._conn.cursor()
        try:
            cur.execute(sql)
        except Exception as exc:
            if not exc.args or exc.args[0] not in ignored_errnos:
                raise
        finally:
            cur.close()

    def _ensure_column(self, table: str, column: str, ddl: str):
        if _COLUMN_MIGRATIONS.get((table, column)) != ddl:
            raise ValueError("column migration is not allowed")
        cur = self._conn.cursor()
        try:
            cur.execute(
                """
                SELECT 1 AS present
                FROM INFORMATION_SCHEMA.COLUMNS
                WHERE TABLE_SCHEMA=DATABASE() AND TABLE_NAME=%s AND COLUMN_NAME=%s
                LIMIT 1
                """,
                (table, column),
            )
            present = cur.fetchone()
        finally:
            cur.close()
        if present:
            return
        cur = self._conn.cursor()
        duplicate_race = False
        try:
            cur.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")
        except Exception as exc:
            if not exc.args or exc.args[0] != 1060:
                raise
            duplicate_race = True
        finally:
            cur.close()
        if duplicate_race:
            cur = self._conn.cursor()
            try:
                cur.execute(
                    """
                    SELECT 1 AS present FROM INFORMATION_SCHEMA.COLUMNS
                    WHERE TABLE_SCHEMA=DATABASE() AND TABLE_NAME=%s AND COLUMN_NAME=%s
                    LIMIT 1
                    """,
                    (table, column),
                )
                present = cur.fetchone()
            finally:
                cur.close()
            if not present:
                raise RuntimeError(f"duplicate column race did not produce expected column: {table}.{column}")

    @staticmethod
    def _dt(value: Any) -> str:
        if hasattr(value, "isoformat"):
            return value.isoformat()
        return str(value or "")

    @staticmethod
    def _mysql_dt(value: Any) -> datetime:
        if isinstance(value, datetime):
            return value
        if isinstance(value, str) and value:
            try:
                return datetime.fromisoformat(value)
            except ValueError:
                try:
                    return datetime.strptime(value.replace("T", " "), "%Y-%m-%d %H:%M:%S")
                except ValueError:
                    return datetime.now()
        return datetime.now()

    @classmethod
    def _kb_from_row(cls, row: dict | None) -> dict | None:
        if not row:
            return None
        return {
            "kb_id": row["kb_id"],
            "name": row["name"],
            "description": row.get("description") or "",
            "created_at": cls._dt(row["created_at"]),
        }

    @classmethod
    def _doc_from_row(cls, row: dict | None) -> dict | None:
        if not row:
            return None
        return {
            "id": row["doc_id"],
            "filename": row["filename"],
            "file_type": row.get("file_type") or "",
            "kb_id": row["kb_id"],
            "status": row["status"],
            "chunk_count": int(row.get("chunk_count") or 0),
            "task_id": row.get("task_id"),
            "storage_path": row.get("storage_path"),
            "error_message": row.get("error_message"),
            "content_hash": row.get("content_hash"),
            "current_version": int(row.get("current_version") or 0),
            "active_index_version": int(row.get("active_index_version") or 0),
            "created_at": cls._dt(row["created_at"]),
            "updated_at": cls._dt(row["updated_at"]),
        }

    @classmethod
    def _task_from_row(cls, row: dict | None) -> dict | None:
        if not row:
            return None
        return {
            "task_id": row["task_id"],
            "doc_id": row["doc_id"],
            "kb_id": row["kb_id"],
            "status": row["status"],
            "error_message": row.get("error_message"),
            "document_version": int(row.get("document_version") or 1),
            "index_version": int(row.get("index_version") or 1),
            "created_at": cls._dt(row["created_at"]),
            "updated_at": cls._dt(row["updated_at"]),
        }

    def create_knowledge_base(self, kb_id: str, name: str, description: str, created_at: str):
        cur = self._execute(
            "INSERT INTO knowledge_bases (kb_id, name, description, created_at) VALUES (%s, %s, %s, %s)",
            (kb_id, name, description, self._mysql_dt(created_at)),
        )
        cur.close()
        self._conn.commit()

    def knowledge_base_exists(self, kb_id: str) -> bool:
        cur = self._execute("SELECT kb_id FROM knowledge_bases WHERE kb_id=%s", (kb_id,))
        row = cur.fetchone()
        cur.close()
        return row is not None

    def get_kb(self, kb_id: str) -> dict | None:
        cur = self._execute("SELECT * FROM knowledge_bases WHERE kb_id=%s", (kb_id,))
        row = cur.fetchone()
        cur.close()
        return self._kb_from_row(row)

    def list_knowledge_bases(self) -> list[dict]:
        cur = self._execute(
            """
            SELECT kb.kb_id, kb.name, kb.description, kb.created_at,
                   COALESCE(SUM(CASE WHEN d.status='ready' THEN 1 ELSE 0 END), 0) AS doc_count
            FROM knowledge_bases kb
            LEFT JOIN documents d ON d.kb_id = kb.kb_id
            GROUP BY kb.kb_id, kb.name, kb.description, kb.created_at
            ORDER BY kb.created_at DESC
            """
        )
        rows = cur.fetchall()
        cur.close()
        result = []
        for row in rows:
            item = self._kb_from_row(row)
            item["doc_count"] = int(row.get("doc_count") or 0)
            result.append(item)
        return result

    def delete_knowledge_base(self, kb_id: str):
        cur = self._execute("DELETE FROM knowledge_bases WHERE kb_id=%s", (kb_id,))
        cur.close()
        self._conn.commit()

    def create_document(self, data: dict):
        cur = self._execute(
            """
            INSERT INTO documents
                (doc_id, kb_id, filename, file_type, status, chunk_count, task_id, storage_path, error_message, created_at, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                data["id"],
                data["kb_id"],
                data["filename"],
                data.get("file_type") or "",
                data["status"],
                data.get("chunk_count", 0),
                data.get("task_id"),
                data.get("storage_path"),
                data.get("error_message"),
                self._mysql_dt(data["created_at"]),
                self._mysql_dt(data["updated_at"]),
            ),
        )
        cur.close()
        self._conn.commit()

    def create_document_with_task(self, document: dict, task: dict):
        cur = self._conn.cursor()
        try:
            cur.execute(
                """
                INSERT INTO documents
                    (doc_id, kb_id, filename, file_type, status, chunk_count, task_id,
                     storage_path, error_message, content_hash, current_version,
                     active_index_version, created_at, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    document["id"],
                    document["kb_id"],
                    document["filename"],
                    document.get("file_type") or "",
                    document["status"],
                    document.get("chunk_count", 0),
                    document.get("task_id"),
                    document.get("storage_path"),
                    document.get("error_message"),
                    None,
                    0,
                    0,
                    self._mysql_dt(document["created_at"]),
                    self._mysql_dt(document["updated_at"]),
                ),
            )
            cur.execute(
                """
                INSERT INTO document_versions
                    (version_id, doc_id, version, content_hash, filename, file_type,
                     storage_path, parser_version, parsed_artifact_path, status, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    uuid.uuid4().hex,
                    document["id"],
                    1,
                    document["content_hash"],
                    document["filename"],
                    document.get("file_type") or "",
                    document.get("storage_path"),
                    None,
                    None,
                    "staging",
                    self._mysql_dt(document["created_at"]),
                ),
            )
            cur.execute(
                """
                INSERT INTO document_index_tasks
                    (task_id, doc_id, kb_id, status, error_message, created_at, updated_at,
                     document_version, index_version, attempt_count)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    task["task_id"],
                    task["doc_id"],
                    task["kb_id"],
                    task["status"],
                    task.get("error_message"),
                    self._mysql_dt(task["created_at"]),
                    self._mysql_dt(task["updated_at"]),
                    1,
                    task.get("index_version", 1),
                    task.get("attempt_count", 0),
                ),
            )
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise
        finally:
            cur.close()

    def find_document_by_hash(
        self,
        kb_id: str,
        content_hash: str,
        exclude_document_id: str | None = None,
    ) -> dict | None:
        exclusion_sql = " AND d.doc_id<>%s" if exclude_document_id else ""
        params = [kb_id]
        if exclude_document_id:
            params.append(exclude_document_id)
        params.extend((content_hash, content_hash))
        cur = self._execute(
            f"""
            SELECT d.*
            FROM documents d
            WHERE d.kb_id=%s
              {exclusion_sql}
              AND d.status<>'deleting'
              AND (
                  d.content_hash=%s
                  OR EXISTS (
                      SELECT 1 FROM document_versions dv
                      WHERE dv.doc_id=d.doc_id
                        AND dv.content_hash=%s
                        AND dv.status='staging'
                  )
              )
            LIMIT 1
            """,
            tuple(params),
        )
        row = cur.fetchone()
        cur.close()
        return self._doc_from_row(row)

    def create_document_version_and_task(
        self,
        document_id: str,
        kb_id: str,
        content_hash: str,
        storage_path: str,
        filename: str,
        file_type: str,
        task_id: str,
        now: str,
    ) -> int:
        cur = self._conn.cursor()
        try:
            cur.execute(
                "SELECT doc_id, current_version, content_hash FROM documents "
                "WHERE doc_id=%s AND kb_id=%s FOR UPDATE",
                (document_id, kb_id),
            )
            if not cur.fetchone():
                raise ValueError("document does not exist")
            cur.execute(
                "SELECT COALESCE(MAX(version), 0) AS max_version "
                "FROM document_versions WHERE doc_id=%s",
                (document_id,),
            )
            new_version = int((cur.fetchone() or {}).get("max_version") or 0) + 1
            cur.execute(
                """
                INSERT INTO document_versions
                    (version_id, doc_id, version, content_hash, filename, file_type,
                     storage_path, parser_version, parsed_artifact_path, status, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    uuid.uuid4().hex, document_id, new_version, content_hash,
                    filename, file_type, storage_path, None, None, "staging",
                    self._mysql_dt(now),
                ),
            )
            cur.execute(
                """
                UPDATE documents
                SET filename=%s, file_type=%s, status=%s, chunk_count=%s,
                    task_id=%s, storage_path=%s, error_message=%s, updated_at=%s
                WHERE doc_id=%s AND kb_id=%s
                """,
                (
                    filename, file_type, "queued", 0, task_id, storage_path,
                    None, self._mysql_dt(now), document_id, kb_id,
                ),
            )
            cur.execute(
                """
                INSERT INTO document_index_tasks
                    (task_id, doc_id, kb_id, status, error_message, created_at, updated_at,
                     document_version, index_version, attempt_count)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    task_id, document_id, kb_id, "queued", None,
                    self._mysql_dt(now), self._mysql_dt(now), new_version, 1, 0,
                ),
            )
            self._conn.commit()
            return new_version
        except Exception:
            self._conn.rollback()
            raise
        finally:
            cur.close()

    def update_document(self, doc_id: str, **changes):
        if not changes:
            return
        changes.setdefault("updated_at", now_iso())
        columns = []
        params = []
        for key, value in changes.items():
            columns.append(f"{key}=%s")
            params.append(self._mysql_dt(value) if key.endswith("_at") else value)
        params.append(doc_id)
        cur = self._execute(f"UPDATE documents SET {', '.join(columns)} WHERE doc_id=%s", tuple(params))
        cur.close()
        self._conn.commit()

    def get_document(self, kb_id: str, doc_id: str) -> dict | None:
        cur = self._execute("SELECT * FROM documents WHERE kb_id=%s AND doc_id=%s", (kb_id, doc_id))
        row = cur.fetchone()
        cur.close()
        return self._doc_from_row(row)

    def list_documents(self, kb_id: str, page: int, page_size: int, status: str | None = None) -> tuple[int, list[dict]]:
        params: list[Any] = [kb_id]
        where = "WHERE kb_id=%s"
        if status:
            where += " AND status=%s"
            params.append(status)
        cur = self._execute(f"SELECT COUNT(*) AS total FROM documents {where}", tuple(params))
        total = int((cur.fetchone() or {}).get("total") or 0)
        cur.close()
        offset = (page - 1) * page_size
        cur = self._execute(
            f"SELECT * FROM documents {where} ORDER BY updated_at DESC LIMIT %s OFFSET %s",
            tuple(params + [page_size, offset]),
        )
        rows = cur.fetchall()
        cur.close()
        return total, [self._doc_from_row(row) for row in rows]

    def delete_document_record(self, doc_id: str):
        cur = self._execute("DELETE FROM documents WHERE doc_id=%s", (doc_id,))
        cur.close()
        self._conn.commit()

    def create_task(self, data: dict):
        cur = self._execute(
            """
            INSERT INTO document_index_tasks
                (task_id, doc_id, kb_id, status, error_message, created_at, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """,
            (
                data["task_id"],
                data["doc_id"],
                data["kb_id"],
                data["status"],
                data.get("error_message"),
                self._mysql_dt(data["created_at"]),
                self._mysql_dt(data["updated_at"]),
            ),
        )
        cur.close()
        self._conn.commit()

    def update_task(self, task_id: str, **changes):
        if not changes:
            return
        changes.setdefault("updated_at", now_iso())
        columns = []
        params = []
        for key, value in changes.items():
            columns.append(f"{key}=%s")
            params.append(self._mysql_dt(value) if key.endswith("_at") else value)
        params.append(task_id)
        cur = self._execute(f"UPDATE document_index_tasks SET {', '.join(columns)} WHERE task_id=%s", tuple(params))
        cur.close()
        self._conn.commit()

    def claim_task(self, task_id: str) -> bool:
        cur = self._execute(
            """
            UPDATE document_index_tasks
            SET status=%s, updated_at=%s
            WHERE task_id=%s AND status=%s
            """,
            ("parsing", datetime.now(), task_id, "queued"),
        )
        claimed = cur.rowcount == 1
        cur.close()
        self._conn.commit()
        return claimed

    def get_task(self, task_id: str) -> dict | None:
        cur = self._execute("SELECT * FROM document_index_tasks WHERE task_id=%s", (task_id,))
        row = cur.fetchone()
        cur.close()
        return self._task_from_row(row)

    def list_recoverable_tasks(self) -> list[dict]:
        cur = self._execute(
            """
            SELECT * FROM document_index_tasks
            WHERE status IN ('queued', 'parsing', 'chunking', 'indexing')
            ORDER BY created_at ASC
            """
        )
        rows = cur.fetchall()
        cur.close()
        return [self._task_from_row(row) for row in rows]

    def _migrate_legacy_json_if_present(self):
        kb_meta_path = os.path.join(DATA_DIR, "kb_meta.json")
        doc_registry_path = os.path.join(DATA_DIR, "doc_registry.json")
        if not os.path.exists(kb_meta_path) and not os.path.exists(doc_registry_path):
            return

        try:
            kb_meta = self._load_json(kb_meta_path, {})
            doc_registry = self._load_json(doc_registry_path, {})
        except Exception as exc:
            logger.warning("Legacy JSON metadata migration skipped: %s", exc)
            return

        try:
            cur = self._conn.cursor()
            for kb_id, meta in kb_meta.items():
                cur.execute(
                    """
                    INSERT IGNORE INTO knowledge_bases (kb_id, name, description, created_at)
                    VALUES (%s, %s, %s, %s)
                    """,
                    (
                        kb_id,
                        meta.get("name") or f"知识库-{kb_id}",
                        meta.get("description") or "",
                        self._mysql_dt(meta.get("created_at")),
                    ),
                )

            for kb_id, docs in doc_registry.items():
                if kb_id not in kb_meta:
                    cur.execute(
                        """
                        INSERT IGNORE INTO knowledge_bases (kb_id, name, description, created_at)
                        VALUES (%s, %s, %s, %s)
                        """,
                        (kb_id, f"知识库-{kb_id}", "", datetime.now()),
                    )
                for doc_id, doc in docs.items():
                    filename = doc.get("filename") or doc_id
                    created_at = doc.get("created_at") or datetime.now().isoformat()
                    updated_at = doc.get("updated_at") or created_at
                    cur.execute(
                        """
                        INSERT IGNORE INTO documents
                            (doc_id, kb_id, filename, file_type, status, chunk_count, task_id, storage_path, error_message, created_at, updated_at)
                        VALUES (%s, %s, %s, %s, %s, %s, NULL, NULL, NULL, %s, %s)
                        """,
                        (
                            doc_id,
                            kb_id,
                            filename,
                            os.path.splitext(filename)[1],
                            doc.get("status") or "ready",
                            int(doc.get("chunk_count") or 0),
                            self._mysql_dt(created_at),
                            self._mysql_dt(updated_at),
                        ),
                    )
            self._conn.commit()
            self._archive_legacy_json(kb_meta_path)
            self._archive_legacy_json(doc_registry_path)
            logger.info("Legacy JSON metadata migration completed")
        except Exception as exc:
            self._conn.rollback()
            logger.warning("Legacy JSON metadata migration failed: %s", exc)

    @staticmethod
    def _archive_legacy_json(path: str):
        if not os.path.exists(path):
            return
        os.replace(path, f"{path}.migrated")

    @staticmethod
    def _load_json(path: str, default: dict) -> dict:
        if not os.path.exists(path):
            return default
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)


_metadata_store: MySQLMetadataStore | None = None


def get_metadata_store() -> MySQLMetadataStore:
    global _metadata_store
    if _metadata_store is None:
        _metadata_store = MySQLMetadataStore()
    return _metadata_store
