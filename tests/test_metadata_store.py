from __future__ import annotations

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
    def __init__(self, existing_columns=()):
        self.existing_columns = set(existing_columns)
        self.executed = []
        self.commits = 0

    def cursor(self):
        connection = self

        class Cursor:
            def execute(self, sql, params=()):
                normalized = " ".join(sql.split())
                connection.executed.append((normalized, params))
                self._row = None
                if "INFORMATION_SCHEMA.COLUMNS" in normalized:
                    self._row = {"present": 1} if tuple(params) in connection.existing_columns else None

            def fetchone(self):
                return self._row

            def close(self):
                pass

        return Cursor()

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
