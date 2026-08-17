from __future__ import annotations

import hashlib
import threading
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta

import pytest

from KBzhy.app.core import metadata_store
from KBzhy.app.core.metadata_store import MySQLMetadataStore


class SnapshotDatabase:
    def __init__(self):
        self.tasks = {}

    def commit_task(self, task_id):
        self.tasks[task_id] = {
            "task_id": task_id,
            "doc_id": "doc1",
            "kb_id": "kb1",
            "status": "queued",
            "error_message": None,
            "document_version": 1,
            "index_version": 1,
            "recovery_owner": None,
            "recovery_lease_until": None,
            "created_at": datetime(2026, 8, 17, 15, 0),
            "updated_at": datetime(2026, 8, 17, 15, 0),
        }


class SnapshotCursor:
    def __init__(self, connection):
        self.connection = connection
        self.row = None

    def execute(self, sql, params=()):
        if "FROM document_index_tasks WHERE task_id=%s" not in " ".join(sql.split()):
            raise AssertionError(f"unexpected SQL: {sql}")
        tasks = self.connection.visible_tasks()
        self.row = deepcopy(tasks.get(params[0]))

    def fetchone(self):
        return self.row

    def close(self):
        pass


class SnapshotConnection:
    def __init__(self, database, autocommit):
        self.database = database
        self.autocommit_enabled = autocommit
        self.snapshot = None

    def cursor(self):
        return SnapshotCursor(self)

    def visible_tasks(self):
        if self.autocommit_enabled:
            return self.database.tasks
        if self.snapshot is None:
            self.snapshot = deepcopy(self.database.tasks)
        return self.snapshot

    def close(self):
        pass


class SnapshotPyMySQL:
    def __init__(self, database):
        self.database = database
        self.connections = []

    def connect(self, **kwargs):
        connection = SnapshotConnection(self.database, kwargs["autocommit"])
        self.connections.append(connection)
        return connection


def _snapshot_store(monkeypatch):
    import pymysql

    database = SnapshotDatabase()
    fake_pymysql = SnapshotPyMySQL(database)
    monkeypatch.setattr(pymysql, "connect", fake_pymysql.connect)
    monkeypatch.setattr(MySQLMetadataStore, "_ensure_schema", lambda self: None)
    monkeypatch.setattr(MySQLMetadataStore, "_migrate_legacy_json_if_present", lambda self: None)
    return MySQLMetadataStore(), database, fake_pymysql


def test_runtime_connection_sees_external_commit_in_same_thread(monkeypatch):
    store, database, fake_pymysql = _snapshot_store(monkeypatch)

    assert store.get_task("task1") is None
    database.commit_task("task1")

    assert store.get_task("task1")["status"] == "queued"
    assert fake_pymysql.connections[-1].autocommit_enabled is True


def test_worker_thread_uses_an_isolated_runtime_connection(monkeypatch):
    store, database, fake_pymysql = _snapshot_store(monkeypatch)
    assert store.get_task("task1") is None
    database.commit_task("task1")
    worker_results = []

    thread = threading.Thread(target=lambda: worker_results.append(store.get_task("task1")))
    thread.start()
    thread.join()

    assert worker_results[0]["status"] == "queued"
    assert len(fake_pymysql.connections) == 3
    assert fake_pymysql.connections[-1].autocommit_enabled is True
    transaction_connection = store.create_connection()
    assert transaction_connection.autocommit_enabled is False


class FakeCursor:
    def __init__(self, rows=None):
        self.rows = rows or []
        self.closed = False
        self.executed = []

    def execute(self, sql, params=()):
        self.executed.append((sql, params))

    def fetchall(self):
        return list(self.rows)

    def fetchone(self):
        return self.rows[0] if self.rows else None

    def close(self):
        self.closed = True


class BrokenConnection:
    def cursor(self):
        raise ValueError("read of closed file")

    def ping(self, reconnect=True):
        raise AttributeError("'NoneType' object has no attribute 'getsockopt'")

    def rollback(self):
        raise RuntimeError("rollback should not be called on broken connection")

    def close(self):
        pass


class FakePyMySQL:
    class cursors:
        class DictCursor:
            pass

    def __init__(self):
        self.cursor = FakeCursor(rows=[{"kb_id": "kb1"}])
        self.connect_calls = 0

    def connect(self, **kwargs):
        self.connect_calls += 1
        return HealthyConnection(self.cursor)


class HealthyConnection:
    def __init__(self, cursor):
        self.cursor_obj = cursor

    def cursor(self):
        return self.cursor_obj

    def ping(self, reconnect=True):
        return None


class SchemaConnection:
    def __init__(self, existing_columns=(), alter_error=None, statement_error=None, missing_indexes=()):
        self.existing_columns = set(existing_columns)
        self.missing_indexes = set(missing_indexes)
        self.alter_error = alter_error
        self.statement_error = statement_error
        self.executed = []
        self.cursors = []
        self.commits = 0

    def cursor(self):
        connection = self

        class Cursor:
            def execute(self, sql, params=()):
                normalized = " ".join(sql.split())
                connection.executed.append((normalized, params))
                self._row = None
                self._rows = []
                if normalized.startswith("CREATE TABLE") and connection.statement_error:
                    raise connection.statement_error
                if "INFORMATION_SCHEMA.COLUMNS" in normalized:
                    if tuple(params) == ("document_chunks", "row_id"):
                        self._row = {"IS_NULLABLE": "NO", "EXTRA": "auto_increment"}
                    elif tuple(params) == ("document_chunks", "chunk_id"):
                        self._row = {"IS_NULLABLE": "NO", "EXTRA": ""}
                    else:
                        self._row = {"present": 1} if tuple(params) in connection.existing_columns else None
                elif "INFORMATION_SCHEMA.STATISTICS" in normalized:
                    indexes = {
                        "PRIMARY": ("row_id",),
                        "uq_chunks_task_chunk": ("task_id", "chunk_id"),
                        "idx_chunks_doc_active_children": ("doc_id", "status", "chunk_type", "position"),
                        "idx_chunks_active_children": ("status", "chunk_type", "doc_id", "position"),
                        "idx_documents_kb_hash": ("kb_id", "content_hash", "current_version", "status"),
                        "idx_document_versions_hash_status": ("content_hash", "status", "doc_id", "version"),
                        "idx_tasks_doc_version_status": ("doc_id", "document_version", "status", "task_id"),
                    }
                    if tuple(params) in connection.missing_indexes:
                        indexes[params[1]] = ()
                    unique = params[1] in {"PRIMARY", "uq_chunks_task_chunk"}
                    self._rows = [
                        {"COLUMN_NAME": name, "NON_UNIQUE": 0 if unique else 1}
                        for name in indexes.get(params[1], ())
                    ]
                    self._row = self._rows[0] if self._rows else None
                elif "INFORMATION_SCHEMA.TABLE_CONSTRAINTS" in normalized:
                    self._row = {"CONSTRAINT_TYPE": "FOREIGN KEY"}
                elif "INFORMATION_SCHEMA.REFERENTIAL_CONSTRAINTS" in normalized:
                    self._row = {"DELETE_RULE": "CASCADE"}
                elif "INFORMATION_SCHEMA.KEY_COLUMN_USAGE" in normalized:
                    self._rows = [
                        {"COLUMN_NAME": "doc_id", "REFERENCED_TABLE_NAME": "document_versions", "REFERENCED_COLUMN_NAME": "doc_id"},
                        {"COLUMN_NAME": "document_version", "REFERENCED_TABLE_NAME": "document_versions", "REFERENCED_COLUMN_NAME": "version"},
                    ]
                elif "COUNT(*) AS missing_count" in normalized:
                    self._row = {"missing_count": 0}
                elif normalized.startswith("ALTER TABLE") and connection.alter_error:
                    if connection.alter_error.args[0] == 1060:
                        parts = normalized.split()
                        connection.existing_columns.add((parts[2], parts[5]))
                    raise connection.alter_error
                elif normalized.startswith("ALTER TABLE") and " ADD INDEX " in normalized:
                    parts = normalized.split()
                    connection.missing_indexes.discard((parts[2], parts[5]))

            def fetchone(self):
                return self._row

            def fetchall(self):
                return getattr(self, "_rows", [self._row] if self._row else [])

            def close(self):
                self.closed = True

        cursor = Cursor()
        cursor.closed = False
        self.cursors.append(cursor)
        return cursor

    def commit(self):
        self.commits += 1


def test_execute_reconnects_with_new_connection_when_stale_connection_ping_fails():
    store = object.__new__(MySQLMetadataStore)
    store._pymysql = FakePyMySQL()
    store._conn = BrokenConnection()
    store._connect_kwargs = {"host": "localhost"}

    cur = store._execute("SELECT kb_id FROM knowledge_bases")

    assert cur.fetchall() == [{"kb_id": "kb1"}]
    assert store._pymysql.connect_calls == 1


def test_ensure_schema_creates_version_and_chunk_tables_and_new_columns():
    store = object.__new__(MySQLMetadataStore)
    store._conn = SchemaConnection()

    store._ensure_schema()

    sql = "\n".join(statement for statement, _ in store._conn.executed)
    assert "CREATE TABLE IF NOT EXISTS document_versions" in sql
    assert "CREATE TABLE IF NOT EXISTS document_chunks" in sql
    assert "row_id BIGINT AUTO_INCREMENT PRIMARY KEY" in sql
    assert "chunk_id VARCHAR(64) NOT NULL" in sql
    assert "UNIQUE KEY uq_chunks_task_chunk (task_id, chunk_id)" in sql
    assert "FOREIGN KEY (doc_id, document_version) REFERENCES document_versions(doc_id, version) ON DELETE CASCADE" in sql
    assert "INDEX idx_chunks_doc_active_children (doc_id, status, chunk_type, position)" in sql
    assert "INDEX idx_chunks_active_children (status, chunk_type, doc_id, position)" in sql
    assert "INDEX idx_documents_kb_hash (kb_id, content_hash, current_version, status)" in sql
    assert "INDEX idx_document_versions_hash_status (content_hash, status, doc_id, version)" in sql
    assert "INDEX idx_tasks_doc_version_status (doc_id, document_version, status, task_id)" in sql
    assert "ADD COLUMN active_collection_name VARCHAR(255) NULL" in sql
    assert "ADD COLUMN content_hash VARCHAR(64) NULL" in sql
    assert "ADD COLUMN current_version INT NOT NULL DEFAULT 0" in sql
    assert "ADD COLUMN parser_version VARCHAR(64) NULL" in sql
    assert "ADD COLUMN active_index_version INT NOT NULL DEFAULT 1" in sql
    assert "ADD COLUMN parsed_artifact_path TEXT NULL" in sql
    assert "ADD COLUMN document_version INT NOT NULL DEFAULT 1" in sql
    assert "ADD COLUMN index_version INT NOT NULL DEFAULT 1" in sql
    assert "ADD COLUMN attempt_count INT NOT NULL DEFAULT 0" in sql
    assert "ADD COLUMN recovery_owner VARCHAR(128) NULL" in sql
    assert "ADD COLUMN recovery_lease_until DATETIME(3) NULL" in sql
    assert "ADD COLUMN filename VARCHAR(255) NULL" in sql
    assert "ADD COLUMN file_type VARCHAR(32) NULL" in sql
    assert "ADD COLUMN storage_path TEXT NULL" in sql


def test_ensure_column_does_not_alter_existing_column():
    store = object.__new__(MySQLMetadataStore)
    store._conn = SchemaConnection({("documents", "content_hash")})

    added = store._ensure_column("documents", "content_hash", "VARCHAR(64) NULL")

    assert added is False
    assert len(store._conn.executed) == 1
    assert "INFORMATION_SCHEMA.COLUMNS" in store._conn.executed[0][0]


def test_ensure_column_alters_missing_column_once():
    store = object.__new__(MySQLMetadataStore)
    store._conn = SchemaConnection()

    added = store._ensure_column("documents", "content_hash", "VARCHAR(64) NULL")

    assert added is True
    alter_statements = [sql for sql, _ in store._conn.executed if sql.startswith("ALTER TABLE")]
    assert alter_statements == ["ALTER TABLE documents ADD COLUMN content_hash VARCHAR(64) NULL"]


def test_ensure_column_ignores_duplicate_column_race():
    store = object.__new__(MySQLMetadataStore)
    store._conn = SchemaConnection(alter_error=RuntimeError(1060, "Duplicate column"))

    added = store._ensure_column("documents", "content_hash", "VARCHAR(64) NULL")

    assert added is False


def test_ensure_schema_only_classifies_legacy_states_when_current_version_is_added():
    missing = object.__new__(MySQLMetadataStore)
    missing._conn = SchemaConnection()
    missing._ensure_schema()

    existing = object.__new__(MySQLMetadataStore)
    existing._conn = SchemaConnection({("documents", "current_version")})
    existing._ensure_schema()

    missing_sql = "\n".join(sql for sql, _ in missing._conn.executed)
    existing_sql = "\n".join(sql for sql, _ in existing._conn.executed)
    assert "UPDATE documents SET current_version=1" in missing_sql
    assert "legacy-staging:" in missing_sql
    assert "status IN ('queued','parsing','chunking','indexing')" in missing_sql
    assert "UPDATE documents SET current_version=1" not in existing_sql
    assert "legacy-staging:" not in existing_sql


def test_ensure_column_does_not_hide_other_alter_errors():
    store = object.__new__(MySQLMetadataStore)
    store._conn = SchemaConnection(alter_error=RuntimeError(1061, "Duplicate key"))

    with pytest.raises(RuntimeError) as exc_info:
        store._ensure_column("documents", "content_hash", "VARCHAR(64) NULL")

    assert exc_info.value.args[0] == 1061


def test_ensure_query_index_adds_missing_legacy_index():
    store = object.__new__(MySQLMetadataStore)
    store._conn = SchemaConnection(missing_indexes={("document_versions", "idx_document_versions_hash_status")})

    store._ensure_query_index(
        "document_versions",
        "idx_document_versions_hash_status",
        ("content_hash", "status", "doc_id", "version"),
    )

    alter_sql = [sql for sql, _ in store._conn.executed if sql.startswith("ALTER TABLE")]
    assert alter_sql == [
        "ALTER TABLE document_versions ADD INDEX idx_document_versions_hash_status "
        "(content_hash, status, doc_id, version)"
    ]


def test_create_connection_uses_saved_connection_configuration():
    store = object.__new__(MySQLMetadataStore)
    store._connect_kwargs = {"host": "db", "database": "kb"}
    store._pymysql = FakePyMySQL()

    connection = store.create_connection()

    assert isinstance(connection, HealthyConnection)
    assert store._pymysql.connect_calls == 1


def test_ensure_schema_closes_cursor_when_statement_fails():
    store = object.__new__(MySQLMetadataStore)
    store._conn = SchemaConnection(statement_error=RuntimeError("ddl failed"))

    with pytest.raises(RuntimeError, match="ddl failed"):
        store._ensure_schema()

    assert all(cursor.closed for cursor in store._conn.cursors)


class ChunkShapeConnection:
    def __init__(self, *, current=False):
        self.columns = {
            "chunk_id": {"IS_NULLABLE": "NO", "EXTRA": ""},
            "task_id": {"IS_NULLABLE": "NO", "EXTRA": ""},
            "doc_id": {"IS_NULLABLE": "NO", "EXTRA": ""},
            "document_version": {"IS_NULLABLE": "NO", "EXTRA": ""},
        }
        self.primary = ("chunk_id",)
        self.indexes = {
            "idx_chunks_active_children": ("status", "chunk_type", "position"),
        }
        self.index_unique = {"idx_chunks_active_children": False}
        self.foreign_keys = {}
        if current:
            self.columns["row_id"] = {"IS_NULLABLE": "NO", "EXTRA": "auto_increment"}
            self.primary = ("row_id",)
            self.indexes = {
                "uq_chunks_task_chunk": ("task_id", "chunk_id"),
                "idx_chunks_doc_active_children": ("doc_id", "status", "chunk_type", "position"),
                "idx_chunks_active_children": ("status", "chunk_type", "doc_id", "position"),
            }
            self.index_unique = {
                "uq_chunks_task_chunk": True,
                "idx_chunks_doc_active_children": False,
                "idx_chunks_active_children": False,
            }
            self.foreign_keys["fk_document_chunks_version"] = (
                ("doc_id", "document_version"),
                "document_versions",
                ("doc_id", "version"),
                "CASCADE",
            )
        self.executed = []
        self.alter_errors = {}
        self.alter_races = {}
        self.chunk_versions = {("doc-legacy", 7)} if not current else set()
        self.documents = {
            "doc-legacy": {
                "current_version": 7,
                "content_hash": "current-hash",
                "parser_version": "parser-v2",
                "parsed_artifact_path": "/parsed/current.json",
            }
        }
        self.document_versions = set()
        self.version_rows = []
        self.commits = 0
        self.events = []
        self.ignore_insert = False
        self.concurrent_duplicate_1062 = False

    def cursor(self):
        return ChunkShapeCursor(self)

    def commit(self):
        self.commits += 1
        self.events.append("commit")

    def rollback(self):
        pass


class ChunkShapeCursor:
    def __init__(self, connection):
        self.connection = connection
        self.rows = []

    def execute(self, sql, params=()):
        normalized = " ".join(sql.split())
        self.connection.executed.append((normalized, params))
        if normalized.startswith("ALTER TABLE"):
            self.connection.events.append(normalized)
        if "INFORMATION_SCHEMA.COLUMNS" in normalized:
            column = self.connection.columns.get(params[1])
            self.rows = [column] if column else []
        elif "INFORMATION_SCHEMA.STATISTICS" in normalized:
            if params[1] == "PRIMARY":
                columns = self.connection.primary
            else:
                columns = self.connection.indexes.get(params[1], ())
            unique = params[1] == "PRIMARY" or self.connection.index_unique.get(params[1], False)
            self.rows = [{"COLUMN_NAME": name, "NON_UNIQUE": 0 if unique else 1} for name in columns]
        elif "INFORMATION_SCHEMA.TABLE_CONSTRAINTS" in normalized:
            self.rows = (
                [{"CONSTRAINT_TYPE": "FOREIGN KEY"}]
                if params[1] in self.connection.foreign_keys
                else []
            )
        elif "INFORMATION_SCHEMA.REFERENTIAL_CONSTRAINTS" in normalized:
            foreign_key = self.connection.foreign_keys.get(params[1])
            delete_rule = foreign_key[3] if foreign_key and len(foreign_key) > 3 else "CASCADE"
            self.rows = [{"DELETE_RULE": delete_rule}] if foreign_key else []
        elif "INFORMATION_SCHEMA.KEY_COLUMN_USAGE" in normalized:
            foreign_key = self.connection.foreign_keys.get(params[1])
            if foreign_key:
                local, referenced_table, remote = foreign_key[:3]
                self.rows = [
                    {
                        "COLUMN_NAME": local_name,
                        "REFERENCED_TABLE_NAME": referenced_table,
                        "REFERENCED_COLUMN_NAME": remote_name,
                    }
                    for local_name, remote_name in zip(local, remote)
                ]
            else:
                self.rows = []
        elif normalized.startswith("INSERT IGNORE INTO document_versions"):
            if self.connection.concurrent_duplicate_1062:
                self.connection.document_versions.update(self.connection.chunk_versions)
                self.rows = []
                return
            if self.connection.ignore_insert:
                self.rows = []
                return
            for doc_id, version in sorted(self.connection.chunk_versions - self.connection.document_versions):
                version_id = hashlib.sha256(f"legacy:{doc_id}:{version}".encode()).hexdigest()
                document = self.connection.documents[doc_id]
                is_current = document["current_version"] == version
                self.connection.document_versions.add((doc_id, version))
                self.connection.version_rows.append(
                    {
                        "version_id": version_id,
                        "doc_id": doc_id,
                        "version": version,
                        "content_hash": document["content_hash"] if is_current else None,
                        "parser_version": document["parser_version"] if is_current else None,
                        "parsed_artifact_path": document["parsed_artifact_path"] if is_current else None,
                        "status": "active" if is_current else "inactive",
                    }
                )
            self.rows = []
        elif "COUNT(*) AS missing_count" in normalized:
            missing = self.connection.chunk_versions - self.connection.document_versions
            self.rows = [{"missing_count": len(missing)}]
        elif normalized.startswith("ALTER TABLE document_chunks"):
            for marker, (error, apply_race) in self.connection.alter_races.items():
                if marker in normalized:
                    apply_race(self.connection)
                    raise error
            for marker, error in self.connection.alter_errors.items():
                if marker in normalized:
                    raise error
            self._apply_alter(normalized)
        else:
            self.rows = []

    def _apply_alter(self, sql):
        connection = self.connection
        if "ADD COLUMN row_id" in sql:
            connection.columns["row_id"] = {"IS_NULLABLE": "NO", "EXTRA": "auto_increment"}
            connection.primary = ("row_id",)
        if "MODIFY COLUMN chunk_id VARCHAR(64) NOT NULL" in sql:
            connection.columns["chunk_id"]["IS_NULLABLE"] = "NO"
        if "ADD UNIQUE KEY uq_chunks_task_chunk" in sql:
            connection.indexes["uq_chunks_task_chunk"] = ("task_id", "chunk_id")
            connection.index_unique["uq_chunks_task_chunk"] = True
        if "ADD INDEX idx_chunks_doc_active_children" in sql:
            connection.indexes["idx_chunks_doc_active_children"] = ("doc_id", "status", "chunk_type", "position")
        if "ADD INDEX idx_chunks_active_children" in sql:
            connection.indexes["idx_chunks_active_children"] = ("status", "chunk_type", "doc_id", "position")
        if "DROP INDEX idx_chunks_active_children" in sql:
            connection.indexes.pop("idx_chunks_active_children", None)
            connection.index_unique.pop("idx_chunks_active_children", None)
        if "ADD CONSTRAINT fk_document_chunks_version" in sql:
            connection.foreign_keys["fk_document_chunks_version"] = (
                ("doc_id", "document_version"),
                "document_versions",
                ("doc_id", "version"),
                "CASCADE",
            )

    def fetchone(self):
        return self.rows[0] if self.rows else None

    def fetchall(self):
        return list(self.rows)

    def close(self):
        pass


def test_document_chunks_shape_migrates_legacy_primary_key_and_objects():
    store = object.__new__(MySQLMetadataStore)
    store._conn = ChunkShapeConnection()

    store._ensure_document_chunks_shape()

    assert store._conn.primary == ("row_id",)
    assert store._conn.columns["row_id"]["EXTRA"] == "auto_increment"
    assert store._conn.indexes["uq_chunks_task_chunk"] == ("task_id", "chunk_id")
    assert store._conn.indexes["idx_chunks_doc_active_children"] == ("doc_id", "status", "chunk_type", "position")
    assert store._conn.indexes["idx_chunks_active_children"] == ("status", "chunk_type", "doc_id", "position")
    assert store._conn.foreign_keys["fk_document_chunks_version"] == (
        ("doc_id", "document_version"),
        "document_versions",
        ("doc_id", "version"),
        "CASCADE",
    )
    assert store._conn.document_versions == {("doc-legacy", 7)}
    assert len(store._conn.version_rows[0]["version_id"]) == 64
    fk_event = next(i for i, event in enumerate(store._conn.events) if "ADD CONSTRAINT fk_document_chunks_version" in event)
    assert "commit" in store._conn.events[:fk_event]
    alter_sql = [sql for sql, _ in store._conn.executed if sql.startswith("ALTER TABLE")]
    assert any("DROP PRIMARY KEY, ADD COLUMN row_id BIGINT NOT NULL AUTO_INCREMENT PRIMARY KEY FIRST" in sql for sql in alter_sql)
    store._conn.executed.clear()

    store._ensure_document_chunks_shape()

    assert not [sql for sql, _ in store._conn.executed if sql.startswith("ALTER TABLE")]


def test_document_chunks_shape_is_noop_when_repeated():
    store = object.__new__(MySQLMetadataStore)
    store._conn = ChunkShapeConnection(current=True)

    store._ensure_document_chunks_shape()
    store._ensure_document_chunks_shape()

    assert not [sql for sql, _ in store._conn.executed if sql.startswith("ALTER TABLE")]


def test_chunk_shape_ddl_ignores_only_configured_duplicate_errno():
    store = object.__new__(MySQLMetadataStore)
    store._conn = SchemaConnection(alter_error=RuntimeError(1061, "Duplicate key"))

    store._execute_chunk_ddl(
        "ALTER TABLE document_chunks ADD INDEX idx_chunks_doc_active_children "
        "(doc_id, status, chunk_type, position)",
        {1061},
    )


def test_chunk_shape_ddl_propagates_nonduplicate_errno():
    store = object.__new__(MySQLMetadataStore)
    store._conn = SchemaConnection(alter_error=RuntimeError(1091, "Cannot drop"))

    with pytest.raises(RuntimeError) as exc_info:
        store._execute_chunk_ddl(
            "ALTER TABLE document_chunks DROP INDEX idx_chunks_active_children",
            set(),
        )

    assert exc_info.value.args[0] == 1091


def test_chunk_shape_migration_does_not_hide_failed_legacy_index_drop():
    store = object.__new__(MySQLMetadataStore)
    store._conn = ChunkShapeConnection(current=True)
    store._conn.indexes["idx_chunks_active_children"] = ("status", "chunk_type", "position")
    store._conn.alter_errors["DROP INDEX idx_chunks_active_children"] = RuntimeError(1091, "Cannot drop")

    with pytest.raises(RuntimeError) as exc_info:
        store._ensure_document_chunks_shape()

    assert exc_info.value.args[0] == 1091


def test_document_version_backfill_is_idempotent():
    store = object.__new__(MySQLMetadataStore)
    store._conn = ChunkShapeConnection()

    store._backfill_missing_document_versions()
    first_rows = list(store._conn.version_rows)
    store._backfill_missing_document_versions()

    assert store._conn.version_rows == first_rows
    assert store._conn.commits == 2


def test_document_version_backfill_only_copies_metadata_to_current_version():
    store = object.__new__(MySQLMetadataStore)
    store._conn = ChunkShapeConnection(current=True)
    store._conn.chunk_versions = {("doc-legacy", 1), ("doc-legacy", 2)}
    store._conn.documents["doc-legacy"]["current_version"] = 2

    store._backfill_missing_document_versions()

    by_version = {row["version"]: row for row in store._conn.version_rows}
    assert by_version[1]["status"] == "inactive"
    assert by_version[1]["content_hash"] is None
    assert by_version[1]["parser_version"] is None
    assert by_version[1]["parsed_artifact_path"] is None
    assert by_version[2]["status"] == "active"
    assert by_version[2]["content_hash"] == "current-hash"
    assert by_version[2]["parser_version"] == "parser-v2"
    assert by_version[2]["parsed_artifact_path"] == "/parsed/current.json"
    insert_sql = next(sql for sql, _ in store._conn.executed if sql.startswith("INSERT"))
    assert insert_sql.startswith("INSERT IGNORE INTO document_versions")
    assert "CASE WHEN d.current_version=dc.document_version THEN d.content_hash ELSE NULL END" in insert_sql


def test_document_version_backfill_rejects_unresolved_insert_ignore_rows():
    store = object.__new__(MySQLMetadataStore)
    store._conn = ChunkShapeConnection()
    store._conn.ignore_insert = True

    with pytest.raises(RuntimeError, match="unresolved"):
        store._backfill_missing_document_versions()


def test_document_version_backfill_accepts_concurrent_1062_unique_race_after_recheck():
    store = object.__new__(MySQLMetadataStore)
    store._conn = ChunkShapeConnection()
    store._conn.concurrent_duplicate_1062 = True

    store._backfill_missing_document_versions()

    assert store._conn.document_versions == {("doc-legacy", 7)}
    assert store._conn.commits == 1


def test_duplicate_index_errno_rechecks_and_accepts_matching_race_shape():
    store = object.__new__(MySQLMetadataStore)
    store._conn = ChunkShapeConnection(current=True)
    store._conn.indexes.pop("idx_chunks_doc_active_children")
    store._conn.alter_races["ADD INDEX idx_chunks_doc_active_children"] = (
        RuntimeError(1061, "Duplicate key"),
        lambda connection: connection.indexes.__setitem__(
            "idx_chunks_doc_active_children", ("doc_id", "status", "chunk_type", "position")
        ),
    )

    store._ensure_document_chunks_shape()


def test_duplicate_index_errno_rechecks_and_rejects_wrong_race_shape():
    store = object.__new__(MySQLMetadataStore)
    store._conn = ChunkShapeConnection(current=True)
    store._conn.indexes.pop("idx_chunks_doc_active_children")
    store._conn.alter_races["ADD INDEX idx_chunks_doc_active_children"] = (
        RuntimeError(1061, "Duplicate key"),
        lambda connection: connection.indexes.__setitem__(
            "idx_chunks_doc_active_children", ("doc_id", "status")
        ),
    )

    with pytest.raises(RuntimeError, match="index shape"):
        store._ensure_document_chunks_shape()


def test_duplicate_primary_errno_rechecks_and_rejects_non_auto_increment_row_id():
    store = object.__new__(MySQLMetadataStore)
    store._conn = ChunkShapeConnection()

    def apply_wrong_primary(connection):
        connection.columns["row_id"] = {"IS_NULLABLE": "NO", "EXTRA": ""}
        connection.primary = ("row_id",)

    store._conn.alter_races["DROP PRIMARY KEY"] = (
        RuntimeError(1060, "Duplicate column"),
        apply_wrong_primary,
    )

    with pytest.raises(RuntimeError, match="primary key"):
        store._ensure_document_chunks_shape()


def test_duplicate_primary_errno_rechecks_and_accepts_matching_race_shape():
    store = object.__new__(MySQLMetadataStore)
    store._conn = ChunkShapeConnection()

    def apply_expected_primary(connection):
        connection.columns["row_id"] = {"IS_NULLABLE": "NO", "EXTRA": "auto_increment"}
        connection.primary = ("row_id",)

    store._conn.alter_races["DROP PRIMARY KEY"] = (
        RuntimeError(1060, "Duplicate column"),
        apply_expected_primary,
    )

    store._ensure_document_chunks_shape()


def test_duplicate_unique_errno_rechecks_and_rejects_reversed_columns():
    store = object.__new__(MySQLMetadataStore)
    store._conn = ChunkShapeConnection(current=True)
    store._conn.indexes.pop("uq_chunks_task_chunk")
    store._conn.alter_races["ADD UNIQUE KEY uq_chunks_task_chunk"] = (
        RuntimeError(1061, "Duplicate key"),
        lambda connection: connection.indexes.__setitem__(
            "uq_chunks_task_chunk", ("chunk_id", "task_id")
        ),
    )

    with pytest.raises(RuntimeError, match="index shape"):
        store._ensure_document_chunks_shape()


def test_duplicate_unique_errno_rechecks_and_accepts_matching_columns():
    store = object.__new__(MySQLMetadataStore)
    store._conn = ChunkShapeConnection(current=True)
    store._conn.indexes.pop("uq_chunks_task_chunk")
    store._conn.alter_races["ADD UNIQUE KEY uq_chunks_task_chunk"] = (
        RuntimeError(1061, "Duplicate key"),
        lambda connection: (
            connection.indexes.__setitem__("uq_chunks_task_chunk", ("task_id", "chunk_id")),
            connection.index_unique.__setitem__("uq_chunks_task_chunk", True),
        ),
    )

    store._ensure_document_chunks_shape()


def test_existing_nonunique_task_chunk_index_is_rejected():
    store = object.__new__(MySQLMetadataStore)
    store._conn = ChunkShapeConnection(current=True)
    store._conn.index_unique["uq_chunks_task_chunk"] = False

    with pytest.raises(RuntimeError, match="index shape"):
        store._ensure_document_chunks_shape()


def test_duplicate_foreign_key_errno_rechecks_and_rejects_wrong_reference():
    store = object.__new__(MySQLMetadataStore)
    store._conn = ChunkShapeConnection(current=True)
    store._conn.foreign_keys.clear()
    store._conn.alter_races["ADD CONSTRAINT fk_document_chunks_version"] = (
        RuntimeError(1826, "Duplicate foreign key"),
        lambda connection: connection.foreign_keys.__setitem__(
            "fk_document_chunks_version",
            (("doc_id",), "documents", ("doc_id",)),
        ),
    )

    with pytest.raises(RuntimeError, match="foreign key shape"):
        store._ensure_document_chunks_shape()


def test_duplicate_foreign_key_errno_rechecks_and_accepts_matching_reference():
    store = object.__new__(MySQLMetadataStore)
    store._conn = ChunkShapeConnection(current=True)
    store._conn.foreign_keys.clear()
    store._conn.alter_races["ADD CONSTRAINT fk_document_chunks_version"] = (
        RuntimeError(1826, "Duplicate foreign key"),
        lambda connection: connection.foreign_keys.__setitem__(
            "fk_document_chunks_version",
            (("doc_id", "document_version"), "document_versions", ("doc_id", "version")),
        ),
    )

    store._ensure_document_chunks_shape()


def test_foreign_key_with_restrict_delete_rule_is_rejected():
    store = object.__new__(MySQLMetadataStore)
    store._conn = ChunkShapeConnection(current=True)
    store._conn.foreign_keys["fk_document_chunks_version"] = (
        ("doc_id", "document_version"),
        "document_versions",
        ("doc_id", "version"),
        "RESTRICT",
    )

    with pytest.raises(RuntimeError, match="foreign key shape"):
        store._ensure_document_chunks_shape()


class DocumentTransactionConnection:
    def __init__(self, *, max_version=1, fail_on=None, hash_rows=None):
        self.max_version = max_version
        self.fail_on = fail_on
        self.executed = []
        self.commits = 0
        self.rollbacks = 0
        self.closed = False
        self.hash_rows = list(hash_rows or [])

    def cursor(self):
        connection = self

        class Cursor:
            def execute(self, sql, params=()):
                normalized = " ".join(sql.split())
                connection.executed.append((normalized, params))
                if connection.fail_on and connection.fail_on in normalized:
                    raise RuntimeError("write failed")
                if normalized.startswith("SELECT kb_id FROM knowledge_bases"):
                    self.row = {"kb_id": "kb1"}
                elif "AS hash_match_type" in normalized:
                    self.row = connection.hash_rows.pop(0) if connection.hash_rows else None
                elif normalized.startswith("SELECT doc_id"):
                    self.row = {"doc_id": "doc1", "current_version": 1, "content_hash": "active-hash"}
                elif normalized.startswith("SELECT COALESCE(MAX(version)"):
                    self.row = {"max_version": connection.max_version}
                else:
                    self.row = None

            def fetchone(self):
                return self.row

            def close(self):
                pass

        return Cursor()

    def commit(self):
        self.commits += 1

    def rollback(self):
        self.rollbacks += 1

    def close(self):
        self.closed = True


def _store_with_connection_factory(*, max_version=1, fail_on=None, hash_rows=None):
    store = object.__new__(MySQLMetadataStore)
    connections = []

    def create_connection():
        connection = DocumentTransactionConnection(
            max_version=max_version, fail_on=fail_on, hash_rows=hash_rows
        )
        connections.append(connection)
        return connection

    store.create_connection = create_connection
    store._conn = type("ForbiddenSharedConnection", (), {
        "cursor": lambda self: (_ for _ in ()).throw(AssertionError("shared connection must not be used")),
    })()
    return store, connections


def test_create_document_version_and_task_locks_document_and_uses_max_version():
    store, connections = _store_with_connection_factory(max_version=4)

    version = store.create_document_version_and_task(
        "doc1", "kb1", "new-hash", "/uploads/kb1/doc1/task2/new.txt",
        "new.txt", ".txt", "task2", "2026-08-13T10:00:00",
    )

    assert version == 5
    assert len(connections) == 1
    connection = connections[0]
    sql = [statement for statement, _ in connection.executed]
    assert any(statement.endswith("FOR UPDATE") for statement in sql)
    assert any("SELECT COALESCE(MAX(version), 0) AS max_version" in statement for statement in sql)
    version_insert = next((statement, params) for statement, params in connection.executed if statement.startswith("INSERT INTO document_versions"))
    assert version_insert[1][2] == 5
    assert version_insert[1][3] == "new-hash"
    task_insert = next((statement, params) for statement, params in connection.executed if statement.startswith("INSERT INTO document_index_tasks"))
    assert task_insert[1][-3:-1] == (5, 1)
    document_update = next(statement for statement in sql if statement.startswith("UPDATE documents"))
    assert "current_version" not in document_update
    assert "content_hash" not in document_update
    assert "chunk_count" not in document_update
    assert connection.commits == 1
    assert connection.rollbacks == 0
    assert connection.closed is True


def test_legacy_current_version_one_without_version_rows_advances_to_v2():
    store, _ = _store_with_connection_factory(max_version=0)

    version = store.create_document_version_and_task(
        "doc1", "kb1", "new-hash", "/uploads/new.txt",
        "new.txt", ".txt", "task2", "2026-08-13T10:00:00",
    )

    assert version == 2


def test_create_version_treats_current_pending_hash_as_unchanged():
    store, connections = _store_with_connection_factory(hash_rows=[
        None,
        {
            "doc_id": "doc1", "kb_id": "kb1", "hash_match_type": "pending",
            "hash_task_id": "task1", "filename": "pending.txt", "status": "queued",
            "created_at": "2026-08-13T10:00:00", "updated_at": "2026-08-13T10:00:00",
        },
    ])

    with pytest.raises(metadata_store.DocumentContentUnchanged):
        store.create_document_version_and_task(
            "doc1", "kb1", "same-hash", "/uploads/new.txt",
            "new.txt", ".txt", "task2", "2026-08-13T10:00:00",
        )

    hash_queries = [item for item in connections[0].executed if "AS hash_match_type" in item[0]]
    assert len(hash_queries) == 2
    assert "d.doc_id<>%s" in hash_queries[0][0]
    assert "d.doc_id<>%s" not in hash_queries[1][0]
    assert connections[0].rollbacks == 1


class IndexCompletionConnection:
    def __init__(self):
        self.task = {
            "task_id": "task2", "doc_id": "doc1", "kb_id": "kb1", "status": "indexing",
            "document_version": 2, "index_version": 4,
        }
        self.document = {
            "doc_id": "doc1", "kb_id": "kb1", "task_id": "task2", "status": "indexing",
            "current_version": 1,
        }
        self.version = {
            "doc_id": "doc1", "version": 2, "status": "staging", "filename": "new.txt",
            "file_type": ".txt", "storage_path": "/uploads/task2/new.txt",
            "content_hash": "new-hash", "parser_version": None, "parsed_artifact_path": None,
        }
        self.executed = []
        self.commits = 0
        self.rollbacks = 0
        self.closed = False

    def cursor(self):
        connection = self

        class Cursor:
            rowcount = 0
            row = None

            def execute(self, sql, params=()):
                normalized = " ".join(sql.split())
                connection.executed.append((normalized, params))
                self.rowcount = 0
                if normalized.startswith("SELECT doc_id, kb_id FROM document_index_tasks"):
                    self.row = {"doc_id": connection.task["doc_id"], "kb_id": connection.task["kb_id"]}
                elif normalized.startswith("SELECT * FROM document_index_tasks"):
                    self.row = dict(connection.task)
                elif normalized.startswith("SELECT * FROM documents"):
                    self.row = dict(connection.document)
                elif normalized.startswith("SELECT * FROM document_versions"):
                    self.row = dict(connection.version) if connection.version["status"] == "staging" else None
                elif normalized.startswith("UPDATE document_versions SET status='inactive'"):
                    self.row = None
                elif normalized.startswith("UPDATE document_versions SET status='active'"):
                    connection.version["status"] = "active"
                    self.rowcount = 1
                elif normalized.startswith("UPDATE documents"):
                    connection.document.update(
                        filename=params[0], file_type=params[1], storage_path=params[2],
                        content_hash=params[3], current_version=params[6],
                        active_index_version=params[7], status="ready", chunk_count=params[8],
                    )
                    self.rowcount = 1
                elif normalized.startswith("UPDATE document_index_tasks"):
                    connection.task["status"] = "ready"
                    self.rowcount = 1

            def fetchone(self):
                return self.row

            def close(self):
                pass

        return Cursor()

    def commit(self):
        self.commits += 1

    def rollback(self):
        self.rollbacks += 1

    def close(self):
        self.closed = True


def test_complete_indexing_task_activates_metadata_and_task_in_one_transaction():
    connection = IndexCompletionConnection()
    store = object.__new__(MySQLMetadataStore)
    store.create_connection = lambda: connection

    store.complete_indexing_task("task2", 9)

    assert connection.document["filename"] == "new.txt"
    assert connection.document["current_version"] == 2
    assert connection.document["active_index_version"] == 4
    assert connection.document["chunk_count"] == 9
    assert connection.version["status"] == "active"
    assert connection.task["status"] == "ready"
    assert connection.commits == 1
    assert connection.rollbacks == 0
    assert connection.closed is True
    sql = [statement for statement, _ in connection.executed]
    prefetch = next(i for i, statement in enumerate(sql) if statement.startswith("SELECT doc_id, kb_id FROM document_index_tasks"))
    document_lock = next(i for i, statement in enumerate(sql) if statement.startswith("SELECT * FROM documents") and statement.endswith("FOR UPDATE"))
    task_lock = next(i for i, statement in enumerate(sql) if statement.startswith("SELECT * FROM document_index_tasks") and statement.endswith("FOR UPDATE"))
    version_lock = next(i for i, statement in enumerate(sql) if statement.startswith("SELECT * FROM document_versions") and statement.endswith("FOR UPDATE"))
    assert prefetch < document_lock < task_lock < version_lock
    assert any("task_id=%s AND status<>'deleting'" in statement for statement in sql)
    assert any("document_version=%s AND status='indexing'" in statement for statement in sql)


def test_create_document_version_and_task_rolls_back_all_writes():
    store, connections = _store_with_connection_factory(fail_on="INSERT INTO document_index_tasks")

    with pytest.raises(RuntimeError, match="write failed"):
        store.create_document_version_and_task(
            "doc1", "kb1", "new-hash", "/uploads/new.txt",
            "new.txt", ".txt", "task2", "2026-08-13T10:00:00",
        )

    assert connections[0].commits == 0
    assert connections[0].rollbacks == 1
    assert connections[0].closed is True


def test_get_document_version_queries_exact_version_and_returns_dict():
    store = object.__new__(MySQLMetadataStore)
    cursor = FakeCursor(rows=[{"doc_id": "doc1", "version": 2, "filename": "new.txt"}])
    captured = {}

    def execute(sql, params=()):
        captured["sql"] = " ".join(sql.split())
        captured["params"] = params
        return cursor

    store._execute = execute

    assert store.get_document_version("doc1", 2)["filename"] == "new.txt"
    assert captured["params"] == ("doc1", 2)
    assert "WHERE doc_id=%s AND version=%s" in captured["sql"]


def test_update_document_version_status_targets_exact_staging_version():
    store = object.__new__(MySQLMetadataStore)
    cursor = FakeCursor()
    store._conn = type("Connection", (), {"commit": lambda self: None})()
    captured = {}

    def execute(sql, params=()):
        captured["sql"] = " ".join(sql.split())
        captured["params"] = params
        return cursor

    store._execute = execute

    store.update_document_version_status("doc1", 2, "failed")

    assert captured["params"] == ("failed", "doc1", 2)
    assert "doc_id=%s AND version=%s AND status='staging'" in captured["sql"]


def test_create_document_with_task_creates_v1_staging_and_versioned_task():
    store, connections = _store_with_connection_factory(max_version=0)
    document = {
        "id": "doc-new", "kb_id": "kb1", "filename": "guide.txt", "file_type": ".txt",
        "status": "queued", "chunk_count": 0, "task_id": "task1", "storage_path": "/uploads/v1/guide.txt",
        "content_hash": "hash-v1", "created_at": "2026-08-13T10:00:00", "updated_at": "2026-08-13T10:00:00",
    }
    task = {
        "task_id": "task1", "doc_id": "doc-new", "kb_id": "kb1", "status": "queued",
        "created_at": "2026-08-13T10:00:00", "updated_at": "2026-08-13T10:00:00",
    }

    store.create_document_with_task(document, task)

    assert len(connections) == 1
    connection = connections[0]
    statements = connection.executed
    document_insert = next((sql, params) for sql, params in statements if sql.startswith("INSERT INTO documents"))
    assert document_insert[1][9:12] == (None, 0, 0)
    version_insert = next((sql, params) for sql, params in statements if sql.startswith("INSERT INTO document_versions"))
    assert version_insert[1][1:5] == ("doc-new", 1, "hash-v1", "guide.txt")
    assert version_insert[1][-2] == "staging"
    task_insert = next((sql, params) for sql, params in statements if sql.startswith("INSERT INTO document_index_tasks"))
    assert task_insert[1][-3:-1] == (1, 1)
    assert connection.commits == 1
    assert connection.closed is True


def test_create_document_with_task_rolls_back_and_closes_independent_connection():
    store, connections = _store_with_connection_factory(fail_on="INSERT INTO document_versions")
    document = {
        "id": "doc-new", "kb_id": "kb1", "filename": "guide.txt", "file_type": ".txt",
        "status": "queued", "task_id": "task1", "storage_path": "/uploads/v1/guide.txt",
        "content_hash": "hash-v1", "created_at": "2026-08-13T10:00:00", "updated_at": "2026-08-13T10:00:00",
    }
    task = {
        "task_id": "task1", "doc_id": "doc-new", "kb_id": "kb1", "status": "queued",
        "created_at": "2026-08-13T10:00:00", "updated_at": "2026-08-13T10:00:00",
    }

    with pytest.raises(RuntimeError, match="write failed"):
        store.create_document_with_task(document, task)

    assert len(connections) == 1
    assert connections[0].commits == 0
    assert connections[0].rollbacks == 1
    assert connections[0].closed is True


class ConcurrentVersionDatabase:
    def __init__(self):
        self.document_lock = threading.Lock()
        self.attempt_lock = threading.Lock()
        self.second_lock_attempt = threading.Event()
        self.lock_attempts = 0
        self.versions = [1]
        self.connections = []

    def connect(self):
        connection = ConcurrentVersionConnection(self)
        self.connections.append(connection)
        return connection


class ConcurrentVersionConnection:
    def __init__(self, database):
        self.database = database
        self.locked = False
        self.closed = False
        self.row = None
        self.pending_version = None

    def cursor(self):
        return self

    def execute(self, sql, params=()):
        normalized = " ".join(sql.split())
        if normalized.startswith("SELECT kb_id FROM knowledge_bases"):
            self.row = {"kb_id": "kb1"}
        elif "AS hash_match_type" in normalized:
            self.row = None
        elif normalized.startswith("SELECT doc_id"):
            with self.database.attempt_lock:
                self.database.lock_attempts += 1
                first_attempt = self.database.lock_attempts == 1
                if self.database.lock_attempts == 2:
                    self.database.second_lock_attempt.set()
            self.database.document_lock.acquire()
            self.locked = True
            if first_attempt:
                assert self.database.second_lock_attempt.wait(timeout=2)
            self.row = {"doc_id": "doc1", "current_version": 1, "content_hash": "active"}
        elif normalized.startswith("SELECT COALESCE(MAX(version)"):
            self.row = {"max_version": max(self.database.versions)}
        elif normalized.startswith("INSERT INTO document_versions"):
            self.pending_version = params[2]
            self.row = None
        else:
            self.row = None

    def fetchone(self):
        return self.row

    def commit(self):
        if self.pending_version is not None:
            self.database.versions.append(self.pending_version)
            self.pending_version = None
        self._release_lock()

    def rollback(self):
        self.pending_version = None
        self._release_lock()

    def _release_lock(self):
        if self.locked:
            self.locked = False
            self.database.document_lock.release()

    def close(self):
        self.closed = True


def test_concurrent_document_updates_use_independent_locked_transactions_for_continuous_versions():
    database = ConcurrentVersionDatabase()
    store = object.__new__(MySQLMetadataStore)
    store.create_connection = database.connect

    def create_version(task_number):
        return store.create_document_version_and_task(
            "doc1", "kb1", f"hash-{task_number}", f"/uploads/task-{task_number}/new.txt",
            "new.txt", ".txt", f"task-{task_number}", "2026-08-13T10:00:00",
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        versions = list(executor.map(create_version, (1, 2)))

    assert sorted(versions) == [2, 3]
    assert sorted(database.versions) == [1, 2, 3]
    assert len(database.connections) == 2
    assert all(connection.closed for connection in database.connections)


def test_find_document_by_hash_only_reserves_active_or_recoverable_staging_hashes():
    store = object.__new__(MySQLMetadataStore)
    cursor = FakeCursor(rows=[{
        "doc_id": "doc1", "kb_id": "kb1", "filename": "guide.txt", "file_type": ".txt",
        "status": "ready", "chunk_count": 2, "task_id": "task1", "storage_path": "/guide.txt",
        "content_hash": "same-hash", "current_version": 1, "active_index_version": 1,
        "created_at": "2026-08-13T10:00:00", "updated_at": "2026-08-13T10:00:00",
    }])
    captured = {}

    def execute(sql, params=()):
        captured["sql"] = " ".join(sql.split())
        captured["params"] = params
        return cursor

    store._execute = execute

    result = store.find_document_by_hash("kb1", "same-hash")

    assert result["id"] == "doc1"
    assert "d.status<>'deleting'" in captured["sql"]
    assert "dv.status='staging'" in captured["sql"]
    assert "dit.status IN ('queued', 'parsing', 'chunking', 'indexing')" in captured["sql"]
    assert "dit.status IN ('queued', 'parsing', 'chunking', 'indexing', 'failed', 'stale')" not in captured["sql"]
    assert "ORDER BY" in captured["sql"]


def test_find_document_by_hash_returns_match_metadata():
    store = object.__new__(MySQLMetadataStore)
    cursor = FakeCursor(rows=[{
        "doc_id": "doc1", "kb_id": "kb1", "filename": "guide.txt", "file_type": ".txt",
        "status": "ready", "chunk_count": 2, "task_id": "active-task", "storage_path": "/guide.txt",
        "content_hash": "same-hash", "current_version": 1, "active_index_version": 1,
        "hash_match_type": "pending", "hash_task_id": "pending-task",
        "created_at": "2026-08-13T10:00:00", "updated_at": "2026-08-13T10:00:00",
    }])
    store._execute = lambda sql, params=(): cursor

    result = store.find_document_by_hash("kb1", "same-hash")

    assert result["match_type"] == "pending"
    assert result["match_task_id"] == "pending-task"


class ConcurrentHashDatabase:
    def __init__(self):
        self.kb_lock = threading.Lock()
        self.attempt_lock = threading.Lock()
        self.second_lock_attempt = threading.Event()
        self.lock_attempts = 0
        self.documents = []
        self.connections = []

    def connect(self):
        connection = ConcurrentHashConnection(self)
        self.connections.append(connection)
        return connection


class ConcurrentHashConnection:
    def __init__(self, database):
        self.database = database
        self.locked = False
        self.pending_document = None
        self.row = None

    def cursor(self):
        return self

    def execute(self, sql, params=()):
        normalized = " ".join(sql.split())
        if normalized.startswith("SELECT kb_id FROM knowledge_bases") and normalized.endswith("FOR UPDATE"):
            with self.database.attempt_lock:
                self.database.lock_attempts += 1
                first_attempt = self.database.lock_attempts == 1
                if self.database.lock_attempts == 2:
                    self.database.second_lock_attempt.set()
            self.database.kb_lock.acquire()
            self.locked = True
            if first_attempt:
                assert self.database.second_lock_attempt.wait(timeout=2)
            self.row = {"kb_id": params[0]}
        elif "AS hash_match_type" in normalized:
            content_hash = next(value for value in params if value == "same-hash")
            duplicate = next((item for item in self.database.documents if item["content_hash"] == content_hash), None)
            self.row = duplicate
        elif normalized.startswith("INSERT INTO documents"):
            self.pending_document = {
                "doc_id": params[0], "kb_id": params[1], "filename": params[2], "file_type": params[3],
                "status": params[4], "chunk_count": params[5], "task_id": params[6], "storage_path": params[7],
                "error_message": params[8], "content_hash": params[9], "current_version": params[10],
                "active_index_version": params[11], "created_at": params[12], "updated_at": params[13],
                "hash_match_type": "pending", "hash_task_id": params[6],
            }
            self.row = None
        elif normalized.startswith("INSERT INTO document_versions"):
            self.pending_document["content_hash"] = params[3]
            self.row = None
        else:
            self.row = None

    def fetchone(self):
        return self.row

    def commit(self):
        if self.pending_document:
            self.database.documents.append(self.pending_document)
        self._unlock()

    def rollback(self):
        self._unlock()

    def _unlock(self):
        if self.locked:
            self.locked = False
            self.database.kb_lock.release()

    def close(self):
        pass


def test_concurrent_same_hash_uploads_are_serialized_and_one_is_rejected():
    database = ConcurrentHashDatabase()
    store = object.__new__(MySQLMetadataStore)
    store.create_connection = database.connect

    def create(document_number):
        now = "2026-08-13T10:00:00"
        document = {
            "id": f"doc-{document_number}", "kb_id": "kb1", "filename": "guide.txt", "file_type": ".txt",
            "status": "queued", "chunk_count": 0, "task_id": f"task-{document_number}",
            "storage_path": f"/uploads/doc-{document_number}/guide.txt", "content_hash": "same-hash",
            "created_at": now, "updated_at": now,
        }
        task = {
            "task_id": f"task-{document_number}", "doc_id": f"doc-{document_number}", "kb_id": "kb1",
            "status": "queued", "created_at": now, "updated_at": now,
        }
        try:
            store.create_document_with_task(document, task)
            return "created"
        except metadata_store.DuplicateDocumentError as exc:
            return exc.document_id

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(create, (1, 2)))

    assert sorted(results) == ["created", "doc-1"]
    assert len(database.documents) == 1


def test_create_version_locks_knowledge_base_before_document_and_rechecks_hash():
    store, connections = _store_with_connection_factory(max_version=1)

    store.create_document_version_and_task(
        "doc1", "kb1", "new-hash", "/uploads/new.txt",
        "new.txt", ".txt", "task2", "2026-08-13T10:00:00",
    )

    statements = [sql for sql, _ in connections[0].executed]
    kb_lock = next(i for i, sql in enumerate(statements) if sql.startswith("SELECT kb_id FROM knowledge_bases"))
    hash_recheck = next(i for i, sql in enumerate(statements) if "AS hash_match_type" in sql)
    document_lock = next(i for i, sql in enumerate(statements) if sql.startswith("SELECT doc_id"))
    assert kb_lock < hash_recheck < document_lock


def test_active_document_version_backfill_seeds_legacy_document_without_chunks():
    store = object.__new__(MySQLMetadataStore)
    cursor = FakeCursor()
    store._conn = type("Connection", (), {
        "cursor": lambda self: cursor,
        "commit": lambda self: None,
        "rollback": lambda self: None,
    })()

    store._backfill_active_document_versions()

    sql, _ = cursor.executed[0]
    normalized = " ".join(sql.split())
    assert normalized.startswith("INSERT IGNORE INTO document_versions")
    assert "FROM documents d" in normalized
    assert "d.current_version > 0" in normalized
    assert "d.filename" in normalized
    assert "'active'" in normalized


class ActiveVersionBackfillConnection:
    def __init__(self):
        self.documents = {"doc1": {
            "doc_id": "doc1", "current_version": 1, "content_hash": "hash-v1",
            "filename": "guide.pdf", "file_type": ".pdf", "storage_path": "/uploads/guide.pdf",
            "parser_version": "parser-v1", "parsed_artifact_path": "/parsed/guide.json",
            "created_at": "2026-08-13T10:00:00",
        }}
        self.versions = {("doc1", 1): {
            "doc_id": "doc1", "version": 1, "content_hash": None, "filename": None,
            "file_type": None, "storage_path": None, "parser_version": None,
            "parsed_artifact_path": None, "status": "active",
        }}

    def cursor(self):
        connection = self

        class Cursor:
            def execute(self, sql, params=()):
                normalized = " ".join(sql.split())
                if normalized.startswith("INSERT IGNORE INTO document_versions"):
                    return
                if normalized.startswith("UPDATE document_versions dv INNER JOIN documents d"):
                    for key, version in connection.versions.items():
                        document = connection.documents[key[0]]
                        if key[1] != document["current_version"]:
                            continue
                        for field in (
                            "content_hash", "filename", "file_type", "storage_path",
                            "parser_version", "parsed_artifact_path",
                        ):
                            if version[field] is None:
                                version[field] = document[field]

            def close(self):
                pass

        return Cursor()

    def commit(self):
        pass

    def rollback(self):
        pass


def test_active_version_backfill_repairs_null_metadata_in_existing_legacy_row():
    store = object.__new__(MySQLMetadataStore)
    store._conn = ActiveVersionBackfillConnection()

    store._backfill_active_document_versions()

    version = store._conn.versions[("doc1", 1)]
    assert version["content_hash"] == "hash-v1"
    assert version["filename"] == "guide.pdf"
    assert version["file_type"] == ".pdf"
    assert version["storage_path"] == "/uploads/guide.pdf"
    assert version["parser_version"] == "parser-v1"
    assert version["parsed_artifact_path"] == "/parsed/guide.json"


class FinishTaskConnection:
    def __init__(self, task_status):
        self.task = {"task_id": "task2", "doc_id": "doc1", "kb_id": "kb1", "status": task_status, "document_version": 2}
        self.document = {
            "doc_id": "doc1", "kb_id": "kb1", "task_id": "task2",
            "status": "ready" if task_status == "ready" else "indexing",
            "current_version": 2 if task_status == "ready" else 1,
        }
        self.version = {"doc_id": "doc1", "version": 2, "status": "active" if task_status == "ready" else "staging"}
        self.executed = []

    def cursor(self):
        connection = self

        class Cursor:
            rowcount = 0

            def execute(self, sql, params=()):
                normalized = " ".join(sql.split())
                connection.executed.append((normalized, params))
                self.rowcount = 0
                if normalized.startswith("SELECT doc_id, kb_id FROM document_index_tasks"):
                    self.row = {"doc_id": connection.task["doc_id"], "kb_id": connection.task["kb_id"]}
                elif normalized.startswith("SELECT * FROM document_index_tasks"):
                    self.row = dict(connection.task)
                elif normalized.startswith("SELECT * FROM documents"):
                    self.row = dict(connection.document)
                elif normalized.startswith("UPDATE document_index_tasks"):
                    connection.task.update(status=params[0], error_message=params[1])
                    self.rowcount = 1
                elif normalized.startswith("UPDATE document_versions"):
                    if connection.version["status"] == "staging":
                        connection.version["status"] = params[0]
                        self.rowcount = 1
                elif normalized.startswith("UPDATE documents"):
                    connection.document.update(status=params[0], error_message=params[1])
                    self.rowcount = 1

            def fetchone(self):
                return self.row

            def close(self):
                pass

        return Cursor()

    def commit(self):
        pass

    def rollback(self):
        pass

    def close(self):
        pass


class CompletionCheckConnection:
    def __init__(self, row=None, execute_error=None):
        self.row = row
        self.execute_error = execute_error
        self.executed = []
        self.commits = 0
        self.rollbacks = 0
        self.closed = False
        self.cursor_closed = False

    def cursor(self):
        connection = self

        class Cursor:
            def execute(self, sql, params=()):
                connection.executed.append((" ".join(sql.split()), params))
                if connection.execute_error:
                    raise connection.execute_error

            def fetchone(self):
                return connection.row

            def close(self):
                connection.cursor_closed = True

        return Cursor()

    def commit(self):
        self.commits += 1

    def rollback(self):
        self.rollbacks += 1

    def close(self):
        self.closed = True


def test_completion_check_uses_fresh_connection_and_exact_authoritative_state():
    shared_connection = CompletionCheckConnection()
    fresh_connection = CompletionCheckConnection(row={"committed": 1})
    store = object.__new__(MySQLMetadataStore)
    store._conn = shared_connection
    store.create_connection = lambda: fresh_connection

    assert store.is_indexing_completion_committed("task2") is True

    assert shared_connection.executed == []
    sql, params = fresh_connection.executed[0]
    assert "FROM document_index_tasks dit" in sql
    assert "JOIN documents d" in sql
    assert "JOIN document_versions dv" in sql
    assert "dit.status='ready'" in sql
    assert "d.status='ready'" in sql
    assert "d.task_id=dit.task_id" in sql
    assert "d.current_version=dit.document_version" in sql
    assert "dv.status='active'" in sql
    assert params == ("task2",)
    assert fresh_connection.commits == 1
    assert fresh_connection.cursor_closed is True
    assert fresh_connection.closed is True


def test_completion_check_failure_rolls_back_closes_and_propagates():
    fresh_connection = CompletionCheckConnection(execute_error=RuntimeError("read failed"))
    store = object.__new__(MySQLMetadataStore)
    store.create_connection = lambda: fresh_connection

    with pytest.raises(RuntimeError, match="read failed"):
        store.is_indexing_completion_committed("task2")

    assert fresh_connection.commits == 0
    assert fresh_connection.rollbacks == 1
    assert fresh_connection.cursor_closed is True
    assert fresh_connection.closed is True


@pytest.mark.parametrize("terminal_status", ["ready", "failed", "stale"])
def test_finish_indexing_task_does_not_overwrite_terminal_task(terminal_status):
    connection = FinishTaskConnection(terminal_status)
    store = object.__new__(MySQLMetadataStore)
    store.create_connection = lambda: connection

    changed = store.finish_indexing_task("task2", "failed", "late failure")

    assert changed is False
    assert connection.task["status"] == terminal_status
    assert not any(sql.startswith("UPDATE documents") for sql, _ in connection.executed)


def test_finish_indexing_task_transitions_recoverable_task_and_locks_in_order():
    connection = FinishTaskConnection("indexing")
    store = object.__new__(MySQLMetadataStore)
    store.create_connection = lambda: connection

    changed = store.finish_indexing_task("task2", "failed", "embedding failed")

    assert changed is True
    assert connection.task["status"] == "failed"
    assert connection.version["status"] == "failed"
    statements = [sql for sql, _ in connection.executed]
    document_lock = next(i for i, sql in enumerate(statements) if sql.startswith("SELECT * FROM documents"))
    task_lock = next(i for i, sql in enumerate(statements) if sql.startswith("SELECT * FROM document_index_tasks"))
    version_lock = next(i for i, sql in enumerate(statements) if sql.startswith("SELECT * FROM document_versions"))
    assert document_lock < task_lock < version_lock


class LifecycleLockOrderConnection:
    def __init__(self, task_status="parsing", locked_doc_id="doc1"):
        self.prefetch = {"doc_id": "doc1", "kb_id": "kb1"}
        self.task = {
            "task_id": "task1", "doc_id": locked_doc_id, "kb_id": "kb1",
            "status": task_status, "document_version": 2,
        }
        self.document = {
            "doc_id": "doc1", "kb_id": "kb1", "task_id": "task1",
            "status": task_status, "current_version": 1,
        }
        self.version = {"doc_id": "doc1", "version": 2, "status": "staging"}
        self.executed = []

    def cursor(self):
        connection = self

        class Cursor:
            rowcount = 0

            def execute(self, sql, params=()):
                normalized = " ".join(sql.split())
                connection.executed.append((normalized, params))
                self.rowcount = 0
                if normalized.startswith("SELECT doc_id, kb_id FROM document_index_tasks"):
                    self.row = dict(connection.prefetch)
                elif normalized.startswith("SELECT * FROM documents") or normalized.startswith("SELECT doc_id, task_id, status FROM documents"):
                    self.row = dict(connection.document)
                elif normalized.startswith("SELECT * FROM document_index_tasks"):
                    self.row = dict(connection.task)
                elif normalized.startswith("SELECT version FROM document_versions"):
                    self.row = dict(connection.version)
                elif normalized.startswith("UPDATE document_index_tasks SET status='queued'"):
                    connection.task["status"] = "queued"
                    self.rowcount = 1
                elif normalized.startswith("UPDATE document_index_tasks SET status='indexing'"):
                    connection.task["status"] = "indexing"
                    self.rowcount = 1
                elif normalized.startswith("UPDATE document_index_tasks SET status='chunking'"):
                    connection.task["status"] = "chunking"
                    self.rowcount = 1
                elif normalized.startswith("UPDATE documents"):
                    self.rowcount = 1

            def fetchone(self):
                return self.row

            def close(self):
                pass

        return Cursor()

    def commit(self):
        pass

    def rollback(self):
        pass

    def close(self):
        pass


class RecoveryLeaseConnection:
    def __init__(
        self, recovery_owner=None, recovery_lease_until=None,
        updated_at=datetime(2026, 8, 17, 10, 0, 0),
    ):
        self.prefetch = {"doc_id": "doc1", "kb_id": "kb1"}
        self.task = {
            "task_id": "task1", "doc_id": "doc1", "kb_id": "kb1",
            "status": "indexing", "document_version": 2,
            "recovery_owner": recovery_owner,
            "recovery_lease_until": recovery_lease_until,
            "updated_at": updated_at,
        }
        self.document = {
            "doc_id": "doc1", "kb_id": "kb1", "task_id": "task1",
            "status": "indexing", "current_version": 1,
        }
        self.version = {"doc_id": "doc1", "version": 2, "status": "staging"}
        self.executed = []

    def cursor(self):
        connection = self

        class Cursor:
            rowcount = 0

            def execute(self, sql, params=()):
                normalized = " ".join(sql.split())
                connection.executed.append((normalized, params))
                self.rowcount = 0
                self.row = None
                if normalized.startswith("SELECT doc_id, kb_id FROM document_index_tasks"):
                    self.row = dict(connection.prefetch)
                elif normalized.startswith("SELECT * FROM documents"):
                    self.row = dict(connection.document)
                elif normalized.startswith("SELECT * FROM document_index_tasks"):
                    self.row = dict(connection.task)
                elif normalized.startswith("SELECT version FROM document_versions"):
                    self.row = dict(connection.version)
                elif normalized.startswith("UPDATE document_index_tasks SET recovery_owner"):
                    connection.task.update(
                        recovery_owner=params[0], recovery_lease_until=params[1],
                        updated_at=params[2],
                    )
                    self.rowcount = 1
                elif normalized.startswith("UPDATE document_index_tasks SET status='queued'"):
                    connection.task["status"] = "queued"
                    self.rowcount = 1
                elif normalized.startswith("UPDATE documents SET status='queued'"):
                    connection.document["status"] = "queued"
                    self.rowcount = 1

            def fetchone(self):
                return self.row

            def close(self):
                pass

        return Cursor()

    def commit(self):
        pass

    def rollback(self):
        pass

    def close(self):
        pass


def test_recovery_lease_only_allows_one_owner_and_locks_in_order():
    connection = RecoveryLeaseConnection()
    store = object.__new__(MySQLMetadataStore)
    store.create_connection = lambda: connection
    now = datetime(2026, 8, 17, 10, 0, 0)

    assert store.claim_task_recovery(
        "task1", "worker-a", now, now + timedelta(minutes=5)
    ) is True
    assert store.claim_task_recovery(
        "task1", "worker-b", now, now + timedelta(minutes=5)
    ) is False

    statements = [sql for sql, _ in connection.executed]
    document_lock = next(i for i, sql in enumerate(statements) if "FROM documents" in sql)
    task_lock = next(
        i for i, sql in enumerate(statements)
        if sql.startswith("SELECT * FROM document_index_tasks")
    )
    version_lock = next(
        i for i, sql in enumerate(statements)
        if sql.startswith("SELECT version FROM document_versions")
    )
    assert document_lock < task_lock < version_lock


def test_expired_recovery_lease_can_be_reclaimed():
    now = datetime(2026, 8, 17, 10, 0, 0)
    connection = RecoveryLeaseConnection(
        recovery_owner="worker-a",
        recovery_lease_until=now - timedelta(seconds=1),
    )
    store = object.__new__(MySQLMetadataStore)
    store.create_connection = lambda: connection

    assert store.claim_task_recovery(
        "task1", "worker-b", now, now + timedelta(minutes=5)
    ) is True
    assert connection.task["recovery_owner"] == "worker-b"


def test_recovery_claim_rejects_stale_listing_snapshot():
    current = datetime(2026, 8, 17, 10, 1, 0)
    connection = RecoveryLeaseConnection(updated_at=current)
    store = object.__new__(MySQLMetadataStore)
    store.create_connection = lambda: connection

    assert store.claim_task_recovery(
        "task1",
        "worker-b",
        current,
        current + timedelta(minutes=5),
        current - timedelta(minutes=1),
    ) is False
    assert connection.task["recovery_owner"] is None


def test_complete_task_recovery_requires_matching_owner():
    connection = RecoveryLeaseConnection(
        recovery_owner="worker-a",
        recovery_lease_until=datetime(2026, 8, 17, 10, 5, 0),
    )
    store = object.__new__(MySQLMetadataStore)
    store.create_connection = lambda: connection

    assert store.complete_task_recovery("task1", "worker-b") is False
    assert connection.task["status"] == "indexing"
    assert store.complete_task_recovery("task1", "worker-a") is True
    assert connection.task["status"] == "queued"
    assert connection.document["status"] == "queued"
    assert connection.task["recovery_owner"] == "worker-a"
    assert connection.task["recovery_lease_until"] == datetime(2026, 8, 17, 10, 5, 0)


def test_normal_claim_accepts_matching_recovery_owner_and_clears_lease():
    executed = []

    class Cursor:
        rowcount = 1

        def close(self):
            pass

    class Connection:
        def commit(self):
            pass

    store = object.__new__(MySQLMetadataStore)
    store._conn = Connection()

    def execute(sql, params):
        executed.append((" ".join(sql.split()), params))
        return Cursor()

    store._execute = execute

    assert store.claim_task("task1", recovery_owner="worker-a") is True
    sql, params = executed[0]
    assert "recovery_owner=NULL" in sql
    assert "recovery_lease_until=NULL" in sql
    assert "OR recovery_owner=%s" in sql
    assert params[-1] == "worker-a"


@pytest.mark.parametrize(
    ("method_name", "task_status", "args"),
    [
        ("requeue_indexing_task", "indexing", ("task1",)),
        ("set_indexing_phase", "parsing", ("task1", "chunking")),
        ("set_indexing_phase", "parsing", ("task1", "indexing")),
    ],
)
def test_worker_lifecycle_transactions_lock_document_before_task_and_version(
    method_name, task_status, args
):
    connection = LifecycleLockOrderConnection(task_status=task_status)
    store = object.__new__(MySQLMetadataStore)
    store.create_connection = lambda: connection

    assert getattr(store, method_name)(*args) is True

    statements = [sql for sql, _ in connection.executed]
    prefetch = next(i for i, sql in enumerate(statements) if sql.startswith("SELECT doc_id, kb_id FROM document_index_tasks"))
    document_lock = next(i for i, sql in enumerate(statements) if "FROM documents" in sql and sql.endswith("FOR UPDATE"))
    task_lock = next(i for i, sql in enumerate(statements) if sql.startswith("SELECT * FROM document_index_tasks") and sql.endswith("FOR UPDATE"))
    assert prefetch < document_lock < task_lock
    if method_name == "requeue_indexing_task":
        version_lock = next(i for i, sql in enumerate(statements) if sql.startswith("SELECT version FROM document_versions"))
        assert task_lock < version_lock


def test_task_owner_prefetch_is_not_used_as_authoritative_state():
    connection = LifecycleLockOrderConnection(task_status="indexing", locked_doc_id="doc2")
    store = object.__new__(MySQLMetadataStore)
    store.create_connection = lambda: connection

    changed = store.requeue_indexing_task("task1")

    assert changed is False
    assert not any(sql.startswith("UPDATE ") for sql, _ in connection.executed)


def test_duplicate_foreign_key_errno_rechecks_delete_rule():
    store = object.__new__(MySQLMetadataStore)
    store._conn = ChunkShapeConnection(current=True)
    store._conn.foreign_keys.clear()
    store._conn.alter_races["ADD CONSTRAINT fk_document_chunks_version"] = (
        RuntimeError(1826, "Duplicate foreign key"),
        lambda connection: connection.foreign_keys.__setitem__(
            "fk_document_chunks_version",
            (
                ("doc_id", "document_version"),
                "document_versions",
                ("doc_id", "version"),
                "RESTRICT",
            ),
        ),
    )

    with pytest.raises(RuntimeError, match="foreign key shape"):
        store._ensure_document_chunks_shape()
