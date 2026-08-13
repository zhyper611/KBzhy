from __future__ import annotations

import pytest

from KBzhy.app.core.metadata_store import MySQLMetadataStore


class FakeCursor:
    def __init__(self, rows=None):
        self.rows = rows or []
        self.closed = False
        self.executed = []

    def execute(self, sql, params=()):
        self.executed.append((sql, params))

    def fetchall(self):
        return list(self.rows)

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
    def __init__(self, existing_columns=(), alter_error=None, statement_error=None):
        self.existing_columns = set(existing_columns)
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
                if normalized.startswith("CREATE TABLE") and connection.statement_error:
                    raise connection.statement_error
                if "INFORMATION_SCHEMA.COLUMNS" in normalized:
                    self._row = {"present": 1} if tuple(params) in connection.existing_columns else None
                elif normalized.startswith("ALTER TABLE") and connection.alter_error:
                    raise connection.alter_error

            def fetchone(self):
                return self._row

            def fetchall(self):
                return [self._row] if self._row else []

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
    assert "ADD COLUMN active_collection_name VARCHAR(255) NULL" in sql
    assert "ADD COLUMN content_hash VARCHAR(64) NULL" in sql
    assert "ADD COLUMN current_version INT NOT NULL DEFAULT 1" in sql
    assert "ADD COLUMN parser_version VARCHAR(64) NULL" in sql
    assert "ADD COLUMN active_index_version INT NOT NULL DEFAULT 1" in sql
    assert "ADD COLUMN parsed_artifact_path TEXT NULL" in sql
    assert "ADD COLUMN document_version INT NOT NULL DEFAULT 1" in sql
    assert "ADD COLUMN index_version INT NOT NULL DEFAULT 1" in sql
    assert "ADD COLUMN attempt_count INT NOT NULL DEFAULT 0" in sql


def test_ensure_column_does_not_alter_existing_column():
    store = object.__new__(MySQLMetadataStore)
    store._conn = SchemaConnection({("documents", "content_hash")})

    store._ensure_column("documents", "content_hash", "VARCHAR(64) NULL")

    assert len(store._conn.executed) == 1
    assert "INFORMATION_SCHEMA.COLUMNS" in store._conn.executed[0][0]


def test_ensure_column_alters_missing_column_once():
    store = object.__new__(MySQLMetadataStore)
    store._conn = SchemaConnection()

    store._ensure_column("documents", "content_hash", "VARCHAR(64) NULL")

    alter_statements = [sql for sql, _ in store._conn.executed if sql.startswith("ALTER TABLE")]
    assert alter_statements == ["ALTER TABLE documents ADD COLUMN content_hash VARCHAR(64) NULL"]


def test_ensure_column_ignores_duplicate_column_race():
    store = object.__new__(MySQLMetadataStore)
    store._conn = SchemaConnection(alter_error=RuntimeError(1060, "Duplicate column"))

    store._ensure_column("documents", "content_hash", "VARCHAR(64) NULL")


def test_ensure_column_does_not_hide_other_alter_errors():
    store = object.__new__(MySQLMetadataStore)
    store._conn = SchemaConnection(alter_error=RuntimeError(1061, "Duplicate key"))

    with pytest.raises(RuntimeError) as exc_info:
        store._ensure_column("documents", "content_hash", "VARCHAR(64) NULL")

    assert exc_info.value.args[0] == 1061


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
        self.foreign_keys = {}
        if current:
            self.columns["row_id"] = {"IS_NULLABLE": "NO", "EXTRA": "auto_increment"}
            self.primary = ("row_id",)
            self.indexes = {
                "uq_chunks_task_chunk": ("task_id", "chunk_id"),
                "idx_chunks_doc_active_children": ("doc_id", "status", "chunk_type", "position"),
                "idx_chunks_active_children": ("status", "chunk_type", "doc_id", "position"),
            }
            self.foreign_keys["fk_document_chunks_version"] = (
                ("doc_id", "document_version"),
                "document_versions",
                ("doc_id", "version"),
            )
        self.executed = []
        self.alter_errors = {}

    def cursor(self):
        return ChunkShapeCursor(self)


class ChunkShapeCursor:
    def __init__(self, connection):
        self.connection = connection
        self.rows = []

    def execute(self, sql, params=()):
        normalized = " ".join(sql.split())
        self.connection.executed.append((normalized, params))
        if "INFORMATION_SCHEMA.COLUMNS" in normalized:
            column = self.connection.columns.get(params[1])
            self.rows = [column] if column else []
        elif "INFORMATION_SCHEMA.STATISTICS" in normalized:
            if params[1] == "PRIMARY":
                columns = self.connection.primary
            else:
                columns = self.connection.indexes.get(params[1], ())
            self.rows = [{"COLUMN_NAME": name} for name in columns]
        elif "INFORMATION_SCHEMA.TABLE_CONSTRAINTS" in normalized:
            self.rows = (
                [{"CONSTRAINT_TYPE": "FOREIGN KEY"}]
                if params[1] in self.connection.foreign_keys
                else []
            )
        elif "INFORMATION_SCHEMA.KEY_COLUMN_USAGE" in normalized:
            foreign_key = self.connection.foreign_keys.get(params[1])
            if foreign_key:
                local, referenced_table, remote = foreign_key
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
        elif normalized.startswith("ALTER TABLE document_chunks"):
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
        if "ADD INDEX idx_chunks_doc_active_children" in sql:
            connection.indexes["idx_chunks_doc_active_children"] = ("doc_id", "status", "chunk_type", "position")
        if "ADD INDEX idx_chunks_active_children" in sql:
            connection.indexes["idx_chunks_active_children"] = ("status", "chunk_type", "doc_id", "position")
        if "DROP INDEX idx_chunks_active_children" in sql:
            connection.indexes.pop("idx_chunks_active_children", None)
        if "ADD CONSTRAINT fk_document_chunks_version" in sql:
            connection.foreign_keys["fk_document_chunks_version"] = (
                ("doc_id", "document_version"),
                "document_versions",
                ("doc_id", "version"),
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
    )
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
