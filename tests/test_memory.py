from __future__ import annotations

import sys
from types import SimpleNamespace

from KBzhy.app.core import memory


class StaleConnection:
    def __init__(self):
        self.closed = False

    def ping(self, reconnect=True):
        raise RuntimeError("lost connection")

    def close(self):
        self.closed = True


class HealthyConnection:
    def __init__(self):
        self.pings = []

    def ping(self, reconnect=True):
        self.pings.append(reconnect)


def test_shared_mysql_replaces_stale_cached_connection(monkeypatch):
    stale = StaleConnection()
    healthy = HealthyConnection()
    connect_calls = []
    fake_pymysql = SimpleNamespace(
        connect=lambda **kwargs: connect_calls.append(kwargs) or healthy
    )
    monkeypatch.setitem(sys.modules, "pymysql", fake_pymysql)
    monkeypatch.setattr(memory, "_shared_mysql", stale)
    monkeypatch.setattr(memory, "_shared_mysql_checked", True)

    connection = memory._get_shared_mysql()

    assert connection is healthy
    assert stale.closed is True
    assert len(connect_calls) == 1


def test_memory_operation_fetches_current_shared_connection(monkeypatch):
    rows = [("user", "hello", None, "2026-08-17 10:00:00")]

    class Cursor:
        def execute(self, sql, params):
            pass

        def fetchall(self):
            return rows

        def close(self):
            pass

    class Connection:
        def cursor(self):
            return Cursor()

    current = Connection()
    monkeypatch.setattr(memory, "_get_shared_mysql", lambda: current)
    instance = object.__new__(memory.ConversationMemory)
    instance.session_id = "session-1"
    instance._mysql_conn = StaleConnection()
    instance._file_read_log = lambda limit: []

    history = instance.get_history()

    assert history == [
        {"role": "user", "content": "hello", "created_at": "2026-08-17 10:00:00"}
    ]
