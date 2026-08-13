from __future__ import annotations

import hashlib

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
                    }
                    unique = params[1] in {"PRIMARY", "uq_chunks_task_chunk"}
                    self._rows = [
                        {"COLUMN_NAME": name, "NON_UNIQUE": 0 if unique else 1}
                        for name in indexes.get(params[1], ())
                    ]
                    self._row = self._rows[0] if self._rows else None
                elif "INFORMATION_SCHEMA.TABLE_CONSTRAINTS" in normalized:
                    self._row = {"CONSTRAINT_TYPE": "FOREIGN KEY"}
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
            )
        self.executed = []
        self.alter_errors = {}
        self.alter_races = {}
        self.chunk_versions = {("doc-legacy", 7)} if not current else set()
        self.document_versions = set()
        self.version_rows = []
        self.commits = 0
        self.events = []

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
        elif normalized.startswith("INSERT INTO document_versions"):
            for doc_id, version in sorted(self.connection.chunk_versions - self.connection.document_versions):
                version_id = hashlib.sha256(f"legacy:{doc_id}:{version}".encode()).hexdigest()
                self.connection.document_versions.add((doc_id, version))
                self.connection.version_rows.append(
                    {"version_id": version_id, "doc_id": doc_id, "version": version}
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
