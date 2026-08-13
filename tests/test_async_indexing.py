from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path

from KBzhy.app.api import documents
from KBzhy.app.core.indexing_worker import IndexingWorker
from KBzhy.app.core.metadata_store import MySQLMetadataStore


class FakeUploadFile:
    def __init__(self, filename: str, content: bytes):
        self.filename = filename
        self._content = content

    async def read(self):
        return self._content


class InMemoryStore:
    def __init__(self):
        self.kbs = {
            "kb1": {
                "kb_id": "kb1",
                "name": "KB",
                "description": "",
                "created_at": "2026-06-12T10:00:00",
            }
        }
        self.documents = {}
        self.tasks = {}

    def knowledge_base_exists(self, kb_id):
        return kb_id in self.kbs

    def create_document(self, data):
        self.documents[data["id"]] = dict(data)

    def create_task(self, data):
        self.tasks[data["task_id"]] = dict(data)

    def create_document_with_task(self, document, task):
        self.create_document(document)
        self.create_task(task)

    def update_document(self, doc_id, **changes):
        self.documents[doc_id].update(changes)

    def update_task(self, task_id, **changes):
        self.tasks[task_id].update(changes)

    def claim_task(self, task_id):
        task = self.tasks.get(task_id)
        if not task or task.get("status") != "queued":
            return False
        task["status"] = "parsing"
        return True

    def get_task(self, task_id):
        task = self.tasks.get(task_id)
        return dict(task) if task else None

    def get_document(self, kb_id, doc_id):
        doc = self.documents.get(doc_id)
        if doc and doc["kb_id"] == kb_id:
            return dict(doc)
        return None


def test_upload_document_queues_indexing_without_calling_engine(monkeypatch, tmp_path):
    store = InMemoryStore()
    queued = []

    class QueueOnlyWorker:
        def enqueue(self, task_id):
            queued.append(task_id)

    monkeypatch.setattr(documents, "get_metadata_store", lambda: store)
    monkeypatch.setattr(documents, "get_indexing_worker", lambda: QueueOnlyWorker())
    monkeypatch.setattr(documents, "UPLOAD_STORAGE_DIR", str(tmp_path))
    monkeypatch.setattr(documents, "get_engine", lambda: (_ for _ in ()).throw(AssertionError("engine should not run in request")))

    response = asyncio.run(documents.upload_document("kb1", FakeUploadFile("guide.txt", b"hello")))

    assert response.status.value == "queued"
    assert response.chunk_count == 0
    assert response.task_id in queued
    saved_path = Path(store.documents[response.id]["storage_path"])
    assert saved_path.exists()
    assert saved_path.read_bytes() == b"hello"


def test_indexing_worker_rolls_back_vectors_and_marks_failed(tmp_path):
    file_path = tmp_path / "bad.txt"
    file_path.write_text("bad content", encoding="utf-8")
    store = InMemoryStore()
    store.documents["doc1"] = {
        "id": "doc1",
        "kb_id": "kb1",
        "filename": "bad.txt",
        "file_type": ".txt",
        "status": "queued",
        "chunk_count": 0,
        "task_id": "task1",
        "storage_path": str(file_path),
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
    }
    store.tasks["task1"] = {"task_id": "task1", "doc_id": "doc1", "kb_id": "kb1", "status": "queued"}
    removed = []

    class FailingEngine:
        def index_document(self, file_path_arg, kb_id, display_name=None, doc_id=None, task_id=None):
            assert file_path_arg == str(file_path)
            assert kb_id == "kb1"
            assert display_name == "bad.txt"
            assert doc_id == "doc1"
            assert task_id == "task1"
            raise RuntimeError("embedding failed")

        def remove_document(self, filename, kb_id, doc_id=None, task_id=None):
            removed.append((filename, kb_id, doc_id, task_id))

    worker = IndexingWorker(store=store, engine_factory=lambda: FailingEngine(), autostart=False)
    worker.process_task("task1")

    assert store.documents["doc1"]["status"] == "failed"
    assert "embedding failed" in store.documents["doc1"]["error_message"]
    assert store.tasks["task1"]["status"] == "failed"
    assert removed == [("bad.txt", "kb1", "doc1", "task1")]


def test_indexing_worker_skips_stale_task_when_document_has_newer_task(tmp_path):
    old_path = tmp_path / "old.txt"
    old_path.write_text("old content", encoding="utf-8")
    store = InMemoryStore()
    store.documents["doc1"] = {
        "id": "doc1",
        "kb_id": "kb1",
        "filename": "new.txt",
        "file_type": ".txt",
        "status": "queued",
        "chunk_count": 0,
        "task_id": "task2",
        "storage_path": str(old_path),
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
    }
    store.tasks["task1"] = {"task_id": "task1", "doc_id": "doc1", "kb_id": "kb1", "status": "queued"}
    called = False

    class Engine:
        def index_document(self, *args, **kwargs):
            nonlocal called
            called = True
            return 1

    worker = IndexingWorker(store=store, engine_factory=lambda: Engine(), autostart=False)
    worker.process_task("task1")

    assert called is False
    assert store.tasks["task1"]["status"] == "stale"
    assert store.documents["doc1"]["status"] == "queued"
    assert store.documents["doc1"]["task_id"] == "task2"


def test_indexing_worker_skips_task_that_was_already_claimed(tmp_path):
    file_path = tmp_path / "doc.txt"
    file_path.write_text("content", encoding="utf-8")
    store = InMemoryStore()
    store.documents["doc1"] = {
        "id": "doc1",
        "kb_id": "kb1",
        "filename": "doc.txt",
        "file_type": ".txt",
        "status": "queued",
        "chunk_count": 0,
        "task_id": "task1",
        "storage_path": str(file_path),
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
    }
    store.tasks["task1"] = {"task_id": "task1", "doc_id": "doc1", "kb_id": "kb1", "status": "parsing"}
    called = False

    class Engine:
        def index_document(self, *args, **kwargs):
            nonlocal called
            called = True
            return 1

    worker = IndexingWorker(store=store, engine_factory=lambda: Engine(), autostart=False)
    worker.process_task("task1")

    assert called is False
    assert store.documents["doc1"]["status"] == "queued"


def test_legacy_json_metadata_migration_imports_kbs_and_documents(monkeypatch, tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "kb_meta.json").write_text(
        '{"kb1":{"name":"Legacy KB","description":"old","created_at":"2026-06-12T10:00:00"}}',
        encoding="utf-8",
    )
    (data_dir / "doc_registry.json").write_text(
        '{"kb1":{"doc1":{"filename":"guide.pdf","status":"ready","chunk_count":3,"created_at":"2026-06-12T10:01:00","updated_at":"2026-06-12T10:02:00"}}}',
        encoding="utf-8",
    )

    executed = []

    class FakeCursor:
        def execute(self, sql, params=None):
            executed.append((sql, params))

        def close(self):
            pass

    class FakeConn:
        def cursor(self):
            return FakeCursor()

        def commit(self):
            executed.append(("COMMIT", None))

        def rollback(self):
            executed.append(("ROLLBACK", None))

    monkeypatch.setattr("KBzhy.app.core.metadata_store.DATA_DIR", str(data_dir))
    store = MySQLMetadataStore.__new__(MySQLMetadataStore)
    store._conn = FakeConn()

    store._migrate_legacy_json_if_present()

    params = [item[1] for item in executed if item[1]]
    assert ("kb1", "Legacy KB", "old", MySQLMetadataStore._mysql_dt("2026-06-12T10:00:00")) in params
    assert any(p and p[0] == "doc1" and p[1] == "kb1" and p[2] == "guide.pdf" and p[5] == 3 for p in params)
    assert ("COMMIT", None) in executed
    assert not (data_dir / "kb_meta.json").exists()
    assert not (data_dir / "doc_registry.json").exists()
    assert (data_dir / "kb_meta.json.migrated").exists()
    assert (data_dir / "doc_registry.json.migrated").exists()
