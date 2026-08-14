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
        self.documents = {"doc-1": {
            "current_version": 1,
            "active_index_version": 1,
            "content_hash": "old-hash",
            "filename": "old.txt",
            "file_type": ".txt",
            "storage_path": "/uploads/kb/doc-1/v1/old.txt",
            "parser_version": "parser-v1",
            "parsed_artifact_path": "/parsed/v1.json",
            "status": "ready",
        }}
        self.tasks = {}
        self.document_versions = {
            ("doc-1", 1): {
                "doc_id": "doc-1", "version": 1, "content_hash": "old-hash",
                "filename": "old.txt", "file_type": ".txt",
                "storage_path": "/uploads/kb/doc-1/v1/old.txt",
                "parser_version": "parser-v1", "parsed_artifact_path": "/parsed/v1.json",
                "status": "active",
            },
            ("doc-1", 2): {
                "doc_id": "doc-1", "version": 2, "content_hash": "new-hash",
                "filename": "new.pdf", "file_type": ".pdf",
                "storage_path": "/uploads/kb/doc-1/task-2/new.pdf",
                "parser_version": "parser-v2", "parsed_artifact_path": "/parsed/v2.json",
                "status": "staging",
            },
        }
        self.commits = 0
        self.rollbacks = 0
        self.fail_insert = False
        self.force_activation_rowcount = None
        self.force_document_rowcount = None
        self.closed = False
        self.cursors = []
        self.sql = []

    def cursor(self):
        cursor = MemoryCursor(self)
        self.cursors.append(cursor)
        return cursor

    def commit(self):
        self.commits += 1

    def rollback(self):
        self.rollbacks += 1

    def close(self):
        self.closed = True


class MemoryCursor:
    def __init__(self, connection):
        self.connection = connection
        self.results = []
        self.rowcount = 0
        self.closed = False

    def execute(self, sql, params=()):
        normalized = " ".join(sql.split())
        self.rowcount = 0
        self.connection.sql.append((normalized, params))
        if normalized.startswith("DELETE FROM document_chunks WHERE task_id=%s AND status='staging'"):
            task_id = params[0]
            before = len(self.connection.rows)
            self.connection.rows = [r for r in self.connection.rows if not (r["task_id"] == task_id and r["status"] == "staging")]
            self.rowcount = before - len(self.connection.rows)
        elif normalized.startswith("SELECT"):
            self.results = self._select(normalized, params)
        elif normalized.startswith("UPDATE document_chunks SET status='inactive'"):
            doc_id = params[0]
            for row in self.connection.rows:
                version_matches = len(params) == 1 or row["document_version"] != params[1]
                if row["doc_id"] == doc_id and row["status"] == "active" and version_matches:
                    row["status"] = "inactive"
                    self.rowcount += 1
        elif normalized.startswith("UPDATE document_chunks SET status='active'"):
            doc_id, version, task_id = params
            for row in self.connection.rows:
                if row["doc_id"] == doc_id and row["document_version"] == version and row["task_id"] == task_id and row["status"] == "staging":
                    row["status"] = "active"
                    self.rowcount += 1
            if self.connection.force_activation_rowcount is not None:
                self.rowcount = self.connection.force_activation_rowcount
        elif normalized.startswith("UPDATE document_versions SET status='inactive'"):
            doc_id = params[0]
            for row in self.connection.document_versions.values():
                if row["doc_id"] == doc_id and row["status"] == "active":
                    row["status"] = "inactive"
                    self.rowcount += 1
        elif normalized.startswith("UPDATE document_versions SET status='active'"):
            doc_id, version = params
            row = self.connection.document_versions.get((doc_id, version))
            if row and row["status"] == "staging":
                row["status"] = "active"
                self.rowcount = 1
        elif normalized.startswith("UPDATE documents"):
            version, index_version, doc_id, task_id = params[-4:]
            if doc_id in self.connection.documents:
                version_row = self.connection.document_versions[(doc_id, version)]
                self.connection.documents[doc_id].update(
                    current_version=version,
                    active_index_version=index_version,
                    content_hash=version_row["content_hash"],
                    filename=version_row["filename"],
                    file_type=version_row["file_type"],
                    storage_path=version_row["storage_path"],
                    parser_version=version_row["parser_version"],
                    parsed_artifact_path=version_row["parsed_artifact_path"],
                    status="ready",
                )
                self.rowcount = 1
            if self.connection.force_document_rowcount is not None:
                self.rowcount = self.connection.force_document_rowcount
        elif normalized.startswith("UPDATE document_index_tasks"):
            task_id, doc_id, version = params
            task = self.connection.tasks.get(task_id)
            if (
                task and task["doc_id"] == doc_id and task["document_version"] == version
                and task["status"] == "indexing"
            ):
                task["status"] = "ready"
                self.rowcount = 1
        elif normalized.startswith("DELETE FROM document_chunks WHERE task_id=%s"):
            task_id = params[0]
            self.connection.rows = [r for r in self.connection.rows if not (r["task_id"] == task_id and r["status"] == "staging")]

    def executemany(self, sql, values):
        if self.connection.fail_insert:
            raise RuntimeError("insert failed")
        columns = [part.strip() for part in sql.split("(", 1)[1].split(")", 1)[0].split(",")]
        for value in values:
            candidate = dict(zip(columns, value))
            if any(row["task_id"] == candidate["task_id"] and row["chunk_id"] == candidate["chunk_id"] for row in self.connection.rows):
                raise RuntimeError("duplicate task chunk")
            candidate["row_id"] = len(self.connection.rows) + 1
            self.connection.rows.append(candidate)

    def _select(self, sql, params):
        rows = self.connection.rows
        if "FROM documents" in sql and "FOR UPDATE" in sql:
            doc_id = params[0]
            if doc_id not in self.connection.documents:
                return []
            document = self.connection.documents[doc_id]
            return [{
                "doc_id": doc_id,
                "task_id": document.get("task_id"),
                "status": document.get("status"),
            }]
        if "FROM document_index_tasks" in sql and "FOR UPDATE" in sql:
            task = self.connection.tasks.get(params[0])
            return [deepcopy(task)] if task else []
        if "FROM document_versions" in sql and "FOR UPDATE" in sql:
            row = self.connection.document_versions.get(tuple(params))
            if row and "status='staging'" in sql and row.get("status") != "staging":
                return []
            return [deepcopy(row)] if row else []
        if "FROM documents" in sql and "current_version" in sql:
            ids = set(params)
            return [{"doc_id": doc_id, "current_version": values["current_version"]} for doc_id, values in self.connection.documents.items() if doc_id in ids]
        if "status='staging'" in sql and "task_id=%s" in sql and normalized_selects_chunks(sql):
            doc_id, version, task_id = params
            rows = [r for r in rows if r["doc_id"] == doc_id and r["document_version"] == version and r["task_id"] == task_id and r["status"] == "staging"]
            return sorted((deepcopy(r) for r in rows), key=lambda r: r.get("position", 0))
        if "WHERE task_id=%s" in sql and "parent_chunk_id=%s" not in sql and "chunk_id=%s" not in sql:
            rows = [r for r in rows if r["task_id"] == params[0]]
        elif "WHERE chunk_id=%s AND status='active' AND chunk_type='child'" in sql:
            rows = [r for r in rows if r["chunk_id"] == params[0] and r["status"] == "active" and r["chunk_type"] == "child"]
        elif "parent_chunk_id=%s" in sql:
            task_id, doc_id, version, parent_id, lower, upper = params
            rows = [r for r in rows if r["task_id"] == task_id and r["doc_id"] == doc_id and r["document_version"] == version and r["parent_chunk_id"] == parent_id and r["status"] == "active" and r["chunk_type"] == "child" and lower <= r["position"] <= upper]
        elif "chunk_id=%s AND status='active' AND chunk_type='parent'" in sql:
            task_id, doc_id, version, chunk_id = params
            rows = [r for r in rows if r["task_id"] == task_id and r["doc_id"] == doc_id and r["document_version"] == version and r["chunk_id"] == chunk_id and r["status"] == "active" and r["chunk_type"] == "parent"]
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
        if "CASE WHEN chunk_type='parent' THEN 0 ELSE 1 END" in sql:
            return sorted(
                (deepcopy(r) for r in rows),
                key=lambda r: (r.get("position", 0), 0 if r["chunk_type"] == "parent" else 1, r["row_id"]),
            )
        return sorted((deepcopy(r) for r in rows), key=lambda r: r.get("position", 0))

    def fetchone(self):
        return self.results[0] if self.results else None

    def fetchall(self):
        return self.results

    def close(self):
        self.closed = True


def normalized_selects_chunks(sql):
    return sql.startswith("SELECT") and "FROM document_chunks" in sql


def prepare_staging(connection, task_id, document_id, version, chunks, *, task_status="indexing"):
    connection.documents[document_id]["task_id"] = task_id
    connection.documents[document_id]["status"] = task_status
    index_version = chunks[0].index_version if chunks else 1
    connection.tasks[task_id] = {
        "task_id": task_id, "doc_id": document_id, "document_version": version,
        "index_version": index_version, "status": task_status,
    }
    version_row = connection.document_versions.setdefault(
        (document_id, version),
        {
            "doc_id": document_id, "version": version, "content_hash": f"hash-v{version}",
            "filename": f"v{version}.txt", "file_type": ".txt",
            "storage_path": f"/uploads/kb/{document_id}/v{version}.txt",
            "parser_version": "parser-v1", "parsed_artifact_path": f"/parsed/v{version}.json",
        },
    )
    version_row["status"] = "staging"


def test_replace_staging_locks_parent_rows_before_chunks():
    connection = MemoryConnection()
    chunks = [make_parent(version=2), make_child("child", 0, version=2)]
    prepare_staging(connection, "task-2", "doc-1", 2, chunks)

    ChunkRepository(connection).replace_staging("task-2", "doc-1", 2, chunks)

    statements = [sql for sql, _ in connection.sql]
    document_lock = next(i for i, sql in enumerate(statements) if "FROM documents" in sql and sql.endswith("FOR UPDATE"))
    task_lock = next(i for i, sql in enumerate(statements) if "FROM document_index_tasks" in sql and sql.endswith("FOR UPDATE"))
    version_lock = next(i for i, sql in enumerate(statements) if "FROM document_versions" in sql and sql.endswith("FOR UPDATE"))
    chunk_delete = next(i for i, sql in enumerate(statements) if sql.startswith("DELETE FROM document_chunks"))
    assert document_lock < task_lock < version_lock < chunk_delete


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda connection: connection.documents["doc-1"].update(task_id="new-task"), "owns document"),
        (lambda connection: connection.documents["doc-1"].update(status="deleting"), "owns document"),
        (lambda connection: connection.tasks["task-2"].update(doc_id="other-doc"), "identity or status"),
        (lambda connection: connection.tasks["task-2"].update(document_version=1), "identity or status"),
        (lambda connection: connection.tasks["task-2"].update(status="ready"), "identity or status"),
        (lambda connection: connection.document_versions[("doc-1", 2)].update(status="active"), "staging document version"),
    ],
)
def test_replace_staging_rejects_invalid_ownership_task_or_version(mutation, message):
    connection = MemoryConnection()
    chunks = [make_parent(version=2), make_child("child", 0, version=2)]
    prepare_staging(connection, "task-2", "doc-1", 2, chunks)
    mutation(connection)

    with pytest.raises(ValueError, match=message):
        ChunkRepository(connection).replace_staging("task-2", "doc-1", 2, chunks)

    assert connection.rows == []
    assert connection.commits == 0
    assert connection.rollbacks == 1


def test_replace_staging_roundtrips_chunks_and_is_idempotent():
    connection = MemoryConnection()
    repo = ChunkRepository(connection)
    chunks = [make_parent(), make_child("child-0", 0), make_child("child-1", 1)]

    prepare_staging(connection, "task-1", "doc-1", 1, chunks)
    repo.replace_staging("task-1", "doc-1", 1, chunks)
    repo.replace_staging("task-1", "doc-1", 1, chunks)

    assert repo.list_by_task("task-1") == chunks
    assert len(connection.rows) == 3
    assert connection.commits == 2


def test_list_by_task_has_stable_parent_first_order_for_equal_positions():
    connection = MemoryConnection()
    repo = ChunkRepository(connection)
    child = make_child("child-0", 0)
    parent = make_parent(position=0)
    prepare_staging(connection, "task-1", "doc-1", 1, [child, parent])
    repo.replace_staging("task-1", "doc-1", 1, [child, parent])

    assert repo.list_by_task("task-1") == [parent, child]
    assert "CASE WHEN chunk_type='parent' THEN 0 ELSE 1 END, row_id" in connection.sql[-1][0]


def test_replace_staging_rolls_back_when_insert_fails():
    connection = MemoryConnection()
    connection.fail_insert = True
    chunks = [make_parent()]
    prepare_staging(connection, "task-1", "doc-1", 1, chunks)

    with pytest.raises(RuntimeError, match="insert failed"):
        ChunkRepository(connection).replace_staging("task-1", "doc-1", 1, chunks)

    assert connection.rollbacks == 1
    assert connection.commits == 0


def test_activate_version_switches_status_and_updates_document_versions():
    connection = MemoryConnection()
    repo = ChunkRepository(connection)
    old_chunks = [make_parent(version=1), make_child("old-child", 0, version=1)]
    prepare_staging(connection, "old-task", "doc-1", 1, old_chunks)
    repo.replace_staging("old-task", "doc-1", 1, old_chunks)
    for row in connection.rows:
        row["status"] = "active"
    connection.document_versions[("doc-1", 1)]["status"] = "active"
    new_chunks = [make_parent("parent-2", version=2, index_version=5), make_child("new-child", 0, version=2, parent_id="parent-2", index_version=5)]
    prepare_staging(connection, "new-task", "doc-1", 2, new_chunks)
    repo.replace_staging("new-task", "doc-1", 2, new_chunks)

    repo.activate_version("doc-1", 2, "new-task")

    assert {r["status"] for r in connection.rows if r["document_version"] == 1} == {"inactive"}
    assert {r["status"] for r in connection.rows if r["document_version"] == 2} == {"active"}
    assert connection.documents["doc-1"] == {
        "current_version": 2, "active_index_version": 5, "content_hash": "new-hash",
        "filename": "new.pdf", "file_type": ".pdf",
        "storage_path": "/uploads/kb/doc-1/task-2/new.pdf",
        "parser_version": "parser-v2", "parsed_artifact_path": "/parsed/v2.json",
        "status": "ready", "task_id": "new-task",
    }
    assert connection.document_versions[("doc-1", 1)]["status"] == "inactive"
    assert connection.document_versions[("doc-1", 2)]["status"] == "active"


def test_activate_version_locks_document_before_task_and_version():
    connection = MemoryConnection()
    chunks = [make_parent(version=2), make_child("child", 1, version=2)]
    prepare_staging(connection, "task-2", "doc-1", 2, chunks)
    repo = ChunkRepository(connection)
    repo.replace_staging("task-2", "doc-1", 2, chunks)

    repo.activate_version("doc-1", 2, "task-2")

    statements = [sql for sql, _ in connection.sql]
    task_lock = next(i for i, sql in enumerate(statements) if sql.startswith("SELECT * FROM document_index_tasks"))
    document_lock = next(i for i, sql in enumerate(statements) if sql.startswith("SELECT doc_id, task_id, status FROM documents"))
    version_lock = next(i for i, sql in enumerate(statements) if sql.startswith("SELECT * FROM document_versions"))
    assert document_lock < task_lock < version_lock


def test_activate_version_deactivates_old_active_batch_of_same_version():
    connection = MemoryConnection()
    repo = ChunkRepository(connection)
    old_chunks = [
        make_parent("old-parent", version=2, index_version=4),
        make_child("old-child", 0, version=2, parent_id="old-parent", index_version=4),
    ]
    prepare_staging(connection, "old-task", "doc-1", 2, old_chunks)
    repo.replace_staging("old-task", "doc-1", 2, old_chunks)
    for row in connection.rows:
        row["status"] = "active"
    new_chunks = [
        make_parent("new-parent", version=2, index_version=5),
        make_child("new-child", 0, version=2, parent_id="new-parent", index_version=5),
    ]
    prepare_staging(connection, "new-task", "doc-1", 2, new_chunks)
    repo.replace_staging("new-task", "doc-1", 2, new_chunks)

    repo.activate_version("doc-1", 2, "new-task")

    assert {row["status"] for row in connection.rows if row["task_id"] == "old-task"} == {"inactive"}
    assert {row["status"] for row in connection.rows if row["task_id"] == "new-task"} == {"active"}


@pytest.mark.parametrize("chunks", [[], [make_parent(version=2)]])
def test_activate_version_rejects_missing_or_childless_version(chunks):
    connection = MemoryConnection()
    repo = ChunkRepository(connection)
    prepare_staging(connection, "task", "doc-1", 2, chunks)
    if chunks:
        repo.replace_staging("task", "doc-1", 2, chunks)

    with pytest.raises(ValueError, match="active child"):
        repo.activate_version("doc-1", 2, "task")

    assert connection.rollbacks == 1


def test_discard_task_only_deletes_staging_rows():
    connection = MemoryConnection()
    repo = ChunkRepository(connection)
    chunks = [make_parent(), make_child("active", 0)]
    prepare_staging(connection, "task-1", "doc-1", 1, chunks)
    repo.replace_staging("task-1", "doc-1", 1, chunks)
    connection.rows[1]["status"] = "active"

    repo.discard_task("task-1")

    assert [row["chunk_id"] for row in connection.rows] == ["active"]


def test_list_active_children_filters_document_and_roundtrips_json():
    connection = MemoryConnection()
    repo = ChunkRepository(connection)
    chunks = [make_parent(), make_child("child", 0)]
    prepare_staging(connection, "task", "doc-1", 1, chunks)
    repo.replace_staging("task", "doc-1", 1, chunks)
    for row in connection.rows:
        row["status"] = "active"

    assert repo.list_active_children("doc-1") == [make_child("child", 0)]


def test_get_context_family_returns_parent_and_position_neighbors():
    connection = MemoryConnection()
    repo = ChunkRepository(connection)
    chunks = [make_parent()] + [make_child(f"child-{i}", i) for i in range(4)]
    prepare_staging(connection, "task", "doc-1", 1, chunks)
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
    chunks = [make_parent(version=2), make_child("child", 0, version=2)]
    prepare_staging(connection, "task", "doc-1", 2, chunks)
    repo.replace_staging(
        "task",
        "doc-1",
        2,
        chunks,
    )
    for row in connection.rows:
        row["status"] = "active"

    assert repo.get_active_versions(["doc-1", "missing", "doc-1"]) == {"doc-1": 1}


def test_same_chunk_id_can_stage_in_different_tasks():
    connection = MemoryConnection()
    repo = ChunkRepository(connection)
    chunks = [make_parent(), make_child("stable-child", 0)]

    prepare_staging(connection, "task-1", "doc-1", 1, chunks)
    repo.replace_staging("task-1", "doc-1", 1, chunks)
    prepare_staging(connection, "task-2", "doc-1", 1, chunks)
    repo.replace_staging("task-2", "doc-1", 1, chunks)

    assert len(connection.rows) == 4
    assert {(row["task_id"], row["chunk_id"]) for row in connection.rows} == {
        ("task-1", "parent-1"),
        ("task-1", "stable-child"),
        ("task-2", "parent-1"),
        ("task-2", "stable-child"),
    }


def test_activate_version_only_activates_selected_staging_task():
    connection = MemoryConnection()
    repo = ChunkRepository(connection)
    chunks = [make_parent(version=2), make_child("stable-child", 0, version=2)]
    prepare_staging(connection, "task-a", "doc-1", 2, chunks)
    repo.replace_staging("task-a", "doc-1", 2, chunks)
    prepare_staging(connection, "task-b", "doc-1", 2, chunks)
    repo.replace_staging("task-b", "doc-1", 2, chunks)

    repo.activate_version("doc-1", 2, "task-b")

    assert {row["status"] for row in connection.rows if row["task_id"] == "task-a"} == {"staging"}
    assert {row["status"] for row in connection.rows if row["task_id"] == "task-b"} == {"active"}


def test_activate_version_rejects_child_with_parent_from_another_batch():
    connection = MemoryConnection()
    repo = ChunkRepository(connection)
    parent_chunks = [make_parent("outside-parent", version=2)]
    prepare_staging(connection, "parent-task", "doc-1", 2, parent_chunks)
    repo.replace_staging("parent-task", "doc-1", 2, parent_chunks)
    child_chunks = [make_parent("inside-parent", version=2), make_child("child", 0, version=2, parent_id="outside-parent")]
    prepare_staging(connection, "child-task", "doc-1", 2, child_chunks)
    repo.replace_staging(
        "child-task",
        "doc-1",
        2,
        child_chunks,
    )

    with pytest.raises(ValueError, match="parent"):
        repo.activate_version("doc-1", 2, "child-task")

    assert connection.rollbacks == 1


def test_activate_version_rejects_batch_without_parent():
    connection = MemoryConnection()
    repo = ChunkRepository(connection)
    chunks = [make_child("child", 0, version=2, parent_id="missing")]
    prepare_staging(connection, "task", "doc-1", 2, chunks)
    repo.replace_staging("task", "doc-1", 2, chunks)

    with pytest.raises(ValueError, match="parent"):
        repo.activate_version("doc-1", 2, "task")

    assert connection.rollbacks == 1
    assert all(cursor.closed for cursor in connection.cursors)


def test_activate_version_rejects_mixed_index_versions():
    connection = MemoryConnection()
    repo = ChunkRepository(connection)
    chunks = [make_parent(version=2, index_version=4), make_child("child", 0, version=2, index_version=5)]
    prepare_staging(connection, "task", "doc-1", 2, chunks)
    repo.replace_staging(
        "task",
        "doc-1",
        2,
        chunks,
    )

    with pytest.raises(ValueError, match="index_version"):
        repo.activate_version("doc-1", 2, "task")

    assert connection.rollbacks == 1


def test_activate_version_rejects_old_task_that_no_longer_owns_document():
    connection = MemoryConnection()
    repo = ChunkRepository(connection)
    chunks = [make_parent(version=2), make_child("child", 0, version=2)]
    prepare_staging(connection, "old-task", "doc-1", 2, chunks)
    repo.replace_staging("old-task", "doc-1", 2, chunks)
    connection.documents["doc-1"]["task_id"] = "new-task"

    with pytest.raises(ValueError, match="owns document"):
        repo.activate_version("doc-1", 2, "old-task")

    assert connection.tasks["old-task"]["status"] == "indexing"
    assert connection.document_versions[("doc-1", 2)]["status"] == "staging"
    assert connection.rollbacks == 1


def test_activate_version_rolls_back_on_activation_rowcount_mismatch():
    connection = MemoryConnection()
    repo = ChunkRepository(connection)
    chunks = [make_parent(version=2), make_child("child", 0, version=2)]
    prepare_staging(connection, "task", "doc-1", 2, chunks)
    repo.replace_staging("task", "doc-1", 2, chunks)
    active_document = deepcopy(connection.documents["doc-1"])
    connection.force_activation_rowcount = 0

    with pytest.raises(RuntimeError, match="row count"):
        repo.activate_version("doc-1", 2, "task")

    assert connection.rollbacks == 1
    assert connection.documents["doc-1"] == active_document


def test_activate_version_rejects_zero_changed_rows_for_document_update():
    connection = MemoryConnection()
    connection.documents["doc-1"] = {"current_version": 2, "active_index_version": 3}
    connection.force_document_rowcount = 0
    repo = ChunkRepository(connection)
    chunks = [make_parent(version=2), make_child("child", 0, version=2)]
    prepare_staging(connection, "task", "doc-1", 2, chunks)
    repo.replace_staging("task", "doc-1", 2, chunks)

    with pytest.raises(RuntimeError, match="ownership changed"):
        repo.activate_version("doc-1", 2, "task")

    assert connection.commits == 1
    assert connection.rollbacks == 1


def test_repository_created_from_store_uses_owned_connection_and_closes_it():
    owned_connection = MemoryConnection()

    class Store:
        _conn = object()

        def create_connection(self):
            return owned_connection

    assert ChunkRepository(Store()).list_active_children() == []
    assert owned_connection.closed is True


def test_owned_connection_and_cursor_close_when_operation_fails():
    owned_connection = MemoryConnection()
    owned_connection.fail_insert = True
    chunks = [make_parent()]
    prepare_staging(owned_connection, "task", "doc-1", 1, chunks)

    with pytest.raises(RuntimeError, match="insert failed"):
        ChunkRepository(connection_factory=lambda: owned_connection).replace_staging(
            "task", "doc-1", 1, chunks
        )

    assert owned_connection.closed is True
    assert all(cursor.closed for cursor in owned_connection.cursors)


def test_borrowed_connection_is_not_closed():
    connection = MemoryConnection()

    ChunkRepository(connection).list_active_children()

    assert connection.closed is False


def test_context_family_uses_parent_from_same_active_task():
    connection = MemoryConnection()
    repo = ChunkRepository(connection)
    old_chunks = [make_parent("parent-1"), make_child("old-child", 0)]
    prepare_staging(connection, "old-task", "doc-1", 1, old_chunks)
    repo.replace_staging("old-task", "doc-1", 1, old_chunks)
    active_chunks = [make_parent("parent-1"), make_child("active-child", 0)]
    prepare_staging(connection, "active-task", "doc-1", 1, active_chunks)
    repo.replace_staging("active-task", "doc-1", 1, active_chunks)
    for row in connection.rows:
        row["status"] = "active" if row["task_id"] == "active-task" else "inactive"

    family = repo.get_context_family("active-child")

    assert family.parent == make_parent("parent-1")
