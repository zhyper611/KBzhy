from __future__ import annotations

import logging
import json
import os
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
                INDEX idx_chunks_active_children (status, chunk_type, position),
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
        self._conn.commit()

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
        try:
            cur.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")
        except Exception as exc:
            if not exc.args or exc.args[0] != 1060:
                raise
        finally:
            cur.close()

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
        try:
            cur = self._conn.cursor()
            cur.execute(
                """
                INSERT INTO documents
                    (doc_id, kb_id, filename, file_type, status, chunk_count, task_id, storage_path, error_message, created_at, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
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
                    self._mysql_dt(document["created_at"]),
                    self._mysql_dt(document["updated_at"]),
                ),
            )
            cur.execute(
                """
                INSERT INTO document_index_tasks
                    (task_id, doc_id, kb_id, status, error_message, created_at, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    task["task_id"],
                    task["doc_id"],
                    task["kb_id"],
                    task["status"],
                    task.get("error_message"),
                    self._mysql_dt(task["created_at"]),
                    self._mysql_dt(task["updated_at"]),
                ),
            )
            cur.close()
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

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
