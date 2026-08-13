from __future__ import annotations

from copy import deepcopy

import pytest

from KBzhy.app.core.chunk_repository import ChunkRepository
from KBzhy.app.core.document_models import KnowledgeChunk, content_hash


def make_parent(chunk_id="parent-1", version=1, position=0, index_version=3):
    return KnowledgeChunk(
        chunk_id=chunk_id,
        document_id="doc-1",
        document_version=version,
        parent_chunk_id=None,
        chunk_type="parent",
        content="父内容",
        retrieval_text="章节\n\n父内容",
        content_hash=content_hash("父内容"),
        section_path=("章节",),
        page_start=1,
        page_end=2,
        position=position,
        token_count=10,
        index_version=index_version,
        metadata={"source": "测试"},
    )


def make_child(chunk_id, position, version=1, parent_id="parent-1", index_version=3):
    content = f"子内容 {position}"
    return KnowledgeChunk(
        chunk_id=chunk_id,
        document_id="doc-1",
        document_version=version,
        parent_chunk_id=parent_id,
        chunk_type="child",
        content=content,
        retrieval_text=f"章节\n\n{content}",
        content_hash=content_hash(content),
        section_path=("章节", "小节"),
        page_start=position + 1,
        page_end=position + 1,
        position=position,
        token_count=5,
        index_version=index_version,
        metadata={"position": position, "中文": "值"},
    )


class MemoryConnection:
    def __init__(self):
        self.rows = []
        self.documents = {"doc-1": {"current_version": 1, "active_index_version": 1}}
        self.commits = 0
        self.rollbacks = 0
        self.fail_insert = False
        self.sql = []

    def cursor(self):
        return MemoryCursor(self)

    def commit(self):
        self.commits += 1

    def rollback(self):
        self.rollbacks += 1


class MemoryCursor:
    def __init__(self, connection):
        self.connection = connection
        self.results = []
        self.rowcount = 0

    def execute(self, sql, params=()):
        normalized = " ".join(sql.split())
        self.connection.sql.append((normalized, params))
        if normalized.startswith("DELETE FROM document_chunks WHERE task_id=%s AND status='staging'"):
            task_id = params[0]
            before = len(self.connection.rows)
            self.connection.rows = [r for r in self.connection.rows if not (r["task_id"] == task_id and r["status"] == "staging")]
            self.rowcount = before - len(self.connection.rows)
        elif normalized.startswith("SELECT"):
            self.results = self._select(normalized, params)
        elif normalized.startswith("UPDATE document_chunks SET status='inactive'"):
            doc_id, version = params
            for row in self.connection.rows:
                if row["doc_id"] == doc_id and row["status"] == "active" and row["document_version"] != version:
                    row["status"] = "inactive"
        elif normalized.startswith("UPDATE document_chunks SET status='active'"):
            doc_id, version = params
            for row in self.connection.rows:
                if row["doc_id"] == doc_id and row["document_version"] == version and row["status"] == "staging":
                    row["status"] = "active"
        elif normalized.startswith("UPDATE documents SET current_version"):
            version, index_version, doc_id = params
            self.connection.documents[doc_id] = {"current_version": version, "active_index_version": index_version}
        elif normalized.startswith("DELETE FROM document_chunks WHERE task_id=%s"):
            task_id = params[0]
            self.connection.rows = [r for r in self.connection.rows if not (r["task_id"] == task_id and r["status"] == "staging")]

    def executemany(self, sql, values):
        if self.connection.fail_insert:
            raise RuntimeError("insert failed")
        columns = [part.strip() for part in sql.split("(", 1)[1].split(")", 1)[0].split(",")]
        for value in values:
            self.connection.rows.append(dict(zip(columns, value)))

    def _select(self, sql, params):
        rows = self.connection.rows
        if "COUNT(*) AS child_count" in sql:
            doc_id, version = params
            matching = [r for r in rows if r["doc_id"] == doc_id and r["document_version"] == version and r["status"] == "staging"]
            if "chunk_type='child'" in sql:
                matching = [r for r in matching if r["chunk_type"] == "child"]
            return [{"child_count": len(matching), "index_version": max((r["index_version"] for r in matching), default=None)}]
        if "WHERE task_id=%s" in sql:
            rows = [r for r in rows if r["task_id"] == params[0]]
        elif "WHERE chunk_id=%s AND status='active' AND chunk_type='child'" in sql:
            rows = [r for r in rows if r["chunk_id"] == params[0] and r["status"] == "active" and r["chunk_type"] == "child"]
        elif "parent_chunk_id=%s" in sql:
            parent_id, lower, upper = params
            rows = [r for r in rows if r["parent_chunk_id"] == parent_id and r["status"] == "active" and lower <= r["position"] <= upper]
        elif "chunk_id=%s AND status='active' AND chunk_type='parent'" in sql:
            rows = [r for r in rows if r["chunk_id"] == params[0] and r["status"] == "active" and r["chunk_type"] == "parent"]
        elif "status='active' AND chunk_type='child'" in sql:
            rows = [r for r in rows if r["status"] == "active" and r["chunk_type"] == "child"]
            if params:
                rows = [r for r in rows if r["doc_id"] == params[0]]
        elif "SELECT doc_id, document_version" in sql:
            ids = set(params)
            rows = [r for r in rows if r["doc_id"] in ids and r["status"] == "active" and r["chunk_type"] == "child"]
            unique = {}
            for row in rows:
                unique[row["doc_id"]] = {"doc_id": row["doc_id"], "document_version": row["document_version"]}
            return list(unique.values())
        return sorted((deepcopy(r) for r in rows), key=lambda r: r.get("position", 0))

    def fetchone(self):
        return self.results[0] if self.results else None

    def fetchall(self):
        return self.results

    def close(self):
        pass


def test_replace_staging_roundtrips_chunks_and_is_idempotent():
    connection = MemoryConnection()
    repo = ChunkRepository(connection)
    chunks = [make_parent(), make_child("child-0", 0), make_child("child-1", 1)]

    repo.replace_staging("task-1", "doc-1", 1, chunks)
    repo.replace_staging("task-1", "doc-1", 1, chunks)

    assert repo.list_by_task("task-1") == chunks
    assert len(connection.rows) == 3
    assert connection.commits == 2


def test_replace_staging_rolls_back_when_insert_fails():
    connection = MemoryConnection()
    connection.fail_insert = True

    with pytest.raises(RuntimeError, match="insert failed"):
        ChunkRepository(connection).replace_staging("task-1", "doc-1", 1, [make_parent()])

    assert connection.rollbacks == 1
    assert connection.commits == 0


def test_activate_version_switches_status_and_updates_document_versions():
    connection = MemoryConnection()
    repo = ChunkRepository(connection)
    old_chunks = [make_parent(version=1), make_child("old-child", 0, version=1)]
    repo.replace_staging("old-task", "doc-1", 1, old_chunks)
    for row in connection.rows:
        row["status"] = "active"
    new_chunks = [make_parent("parent-2", version=2, index_version=5), make_child("new-child", 0, version=2, parent_id="parent-2", index_version=5)]
    repo.replace_staging("new-task", "doc-1", 2, new_chunks)

    repo.activate_version("doc-1", 2)

    assert {r["status"] for r in connection.rows if r["document_version"] == 1} == {"inactive"}
    assert {r["status"] for r in connection.rows if r["document_version"] == 2} == {"active"}
    assert connection.documents["doc-1"] == {"current_version": 2, "active_index_version": 5}


@pytest.mark.parametrize("chunks", [[], [make_parent(version=2)]])
def test_activate_version_rejects_missing_or_childless_version(chunks):
    connection = MemoryConnection()
    repo = ChunkRepository(connection)
    if chunks:
        repo.replace_staging("task", "doc-1", 2, chunks)

    with pytest.raises(ValueError, match="active child"):
        repo.activate_version("doc-1", 2)

    assert connection.rollbacks == 1


def test_discard_task_only_deletes_staging_rows():
    connection = MemoryConnection()
    repo = ChunkRepository(connection)
    repo.replace_staging("task-1", "doc-1", 1, [make_parent(), make_child("active", 0)])
    connection.rows[1]["status"] = "active"

    repo.discard_task("task-1")

    assert [row["chunk_id"] for row in connection.rows] == ["active"]


def test_list_active_children_filters_document_and_roundtrips_json():
    connection = MemoryConnection()
    repo = ChunkRepository(connection)
    repo.replace_staging("task", "doc-1", 1, [make_parent(), make_child("child", 0)])
    for row in connection.rows:
        row["status"] = "active"

    assert repo.list_active_children("doc-1") == [make_child("child", 0)]


def test_get_context_family_returns_parent_and_position_neighbors():
    connection = MemoryConnection()
    repo = ChunkRepository(connection)
    chunks = [make_parent()] + [make_child(f"child-{i}", i) for i in range(4)]
    repo.replace_staging("task", "doc-1", 1, chunks)
    for row in connection.rows:
        row["status"] = "active"

    family = repo.get_context_family("child-2", neighbor_window=1)

    assert family.parent == make_parent()
    assert family.children == (make_child("child-1", 1), make_child("child-2", 2), make_child("child-3", 3))


def test_get_context_family_rejects_negative_window():
    with pytest.raises(ValueError, match="neighbor_window"):
        ChunkRepository(MemoryConnection()).get_context_family("child", -1)


def test_get_active_versions_returns_without_sql_for_empty_input():
    connection = MemoryConnection()

    assert ChunkRepository(connection).get_active_versions([]) == {}
    assert connection.sql == []


def test_get_active_versions_returns_versions_for_requested_documents():
    connection = MemoryConnection()
    repo = ChunkRepository(connection)
    repo.replace_staging("task", "doc-1", 1, [make_parent(), make_child("child", 0)])
    for row in connection.rows:
        row["status"] = "active"

    assert repo.get_active_versions(["doc-1", "missing", "doc-1"]) == {"doc-1": 1}
