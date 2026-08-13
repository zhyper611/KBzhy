from __future__ import annotations

import asyncio
import hashlib
from datetime import datetime
from pathlib import Path

import pytest

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
        self.versions = {}

    def knowledge_base_exists(self, kb_id):
        return kb_id in self.kbs

    def create_document(self, data):
        self.documents[data["id"]] = dict(data)

    def create_task(self, data):
        self.tasks[data["task_id"]] = dict(data)

    def create_document_with_task(self, document, task):
        self.create_document(document)
        self.create_task(task)
        self.documents[document["id"]].update(content_hash=None, current_version=0, active_index_version=0)
        self.versions[(document["id"], 1)] = {
            "content_hash": document["content_hash"],
            "storage_path": document["storage_path"],
            "status": "staging",
        }
        self.tasks[task["task_id"]].update(document_version=1, index_version=1)

    def find_document_by_hash(self, kb_id, content_hash, exclude_document_id=None):
        for document in self.documents.values():
            if document["kb_id"] != kb_id or document.get("status") == "deleting":
                continue
            if document["id"] == exclude_document_id:
                continue
            if document.get("content_hash") == content_hash:
                return dict(document)
            if any(
                version["content_hash"] == content_hash and version["status"] == "staging"
                for (version_doc_id, _), version in self.versions.items()
                if version_doc_id == document["id"]
            ):
                return dict(document)
        return None

    def create_document_version_and_task(
        self, document_id, kb_id, content_hash, storage_path, filename, file_type, task_id, now
    ):
        document = self.documents[document_id]
        version = max((v for doc_id, v in self.versions if doc_id == document_id), default=0) + 1
        self.versions[(document_id, version)] = {
            "content_hash": content_hash,
            "storage_path": storage_path,
            "status": "staging",
        }
        document.update(
            filename=filename,
            file_type=file_type,
            status="queued",
            chunk_count=0,
            task_id=task_id,
            storage_path=storage_path,
            error_message=None,
            updated_at=now,
        )
        self.tasks[task_id] = {
            "task_id": task_id,
            "doc_id": document_id,
            "kb_id": kb_id,
            "status": "queued",
            "document_version": version,
            "index_version": 1,
        }
        return version

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
    assert store.documents[response.id]["content_hash"] is None
    assert store.documents[response.id]["current_version"] == 0
    assert store.versions[(response.id, 1)]["status"] == "staging"
    assert store.versions[(response.id, 1)]["content_hash"] == hashlib.sha256(b"hello").hexdigest()
    assert store.tasks[response.task_id]["document_version"] == 1
    assert store.tasks[response.task_id]["index_version"] == 1


def test_upload_duplicate_returns_structured_409_without_writing_or_enqueueing(monkeypatch, tmp_path):
    store = InMemoryStore()
    content = b"same bytes"
    store.documents["existing"] = {
        "id": "existing",
        "kb_id": "kb1",
        "filename": "existing.txt",
        "status": "ready",
        "content_hash": hashlib.sha256(content).hexdigest(),
    }
    queued = []

    class Worker:
        def enqueue(self, task_id):
            queued.append(task_id)

    monkeypatch.setattr(documents, "get_metadata_store", lambda: store)
    monkeypatch.setattr(documents, "get_indexing_worker", lambda: Worker())
    monkeypatch.setattr(documents, "UPLOAD_STORAGE_DIR", str(tmp_path))

    with pytest.raises(documents.HTTPException) as exc_info:
        asyncio.run(documents.upload_document("kb1", FakeUploadFile("copy.txt", content)))

    assert exc_info.value.status_code == 409
    assert exc_info.value.detail["document_id"] == "existing"
    assert exc_info.value.detail["message"]
    assert queued == []
    assert list(tmp_path.rglob("*")) == []


def test_upload_enqueue_failure_keeps_queued_task_and_file(monkeypatch, tmp_path):
    store = InMemoryStore()

    class FailingWorker:
        def enqueue(self, task_id):
            raise RuntimeError("queue unavailable")

    monkeypatch.setattr(documents, "get_metadata_store", lambda: store)
    monkeypatch.setattr(documents, "get_indexing_worker", lambda: FailingWorker())
    monkeypatch.setattr(documents, "UPLOAD_STORAGE_DIR", str(tmp_path))

    response = asyncio.run(documents.upload_document("kb1", FakeUploadFile("guide.txt", b"content")))

    assert response.status.value == "queued"
    assert store.tasks[response.task_id]["status"] == "queued"
    assert Path(store.documents[response.id]["storage_path"]).read_bytes() == b"content"


def test_upload_transaction_failure_removes_new_document_directory(monkeypatch, tmp_path):
    class FailingStore(InMemoryStore):
        def create_document_with_task(self, document, task):
            raise RuntimeError("database unavailable")

    store = FailingStore()
    monkeypatch.setattr(documents, "get_metadata_store", lambda: store)
    monkeypatch.setattr(documents, "UPLOAD_STORAGE_DIR", str(tmp_path))

    with pytest.raises(documents.HTTPException) as exc_info:
        asyncio.run(documents.upload_document("kb1", FakeUploadFile("guide.txt", b"content")))

    assert exc_info.value.status_code == 500
    assert list((tmp_path / "kb1").iterdir()) == []
    assert store.documents == {}
    assert store.tasks == {}


def test_update_unchanged_document_is_noop(monkeypatch, tmp_path):
    store = InMemoryStore()
    content = b"unchanged"
    old_path = tmp_path / "kb1" / "doc1" / "v1" / "old.txt"
    old_path.parent.mkdir(parents=True)
    old_path.write_bytes(content)
    store.documents["doc1"] = {
        "id": "doc1", "kb_id": "kb1", "filename": "old.txt", "file_type": ".txt",
        "status": "ready", "chunk_count": 3, "task_id": "old-task", "storage_path": str(old_path),
        "content_hash": hashlib.sha256(content).hexdigest(), "current_version": 1,
    }
    queued = []
    monkeypatch.setattr(documents, "get_metadata_store", lambda: store)
    monkeypatch.setattr(documents, "get_indexing_worker", lambda: type("Worker", (), {"enqueue": lambda _, task: queued.append(task)})())
    monkeypatch.setattr(documents, "UPLOAD_STORAGE_DIR", str(tmp_path))
    monkeypatch.setattr(documents, "get_engine", lambda: (_ for _ in ()).throw(AssertionError("engine should not run")))

    response = asyncio.run(documents.update_document("kb1", "doc1", FakeUploadFile("renamed.txt", content)))

    assert response.message == "文档内容未变化，无需重新索引"
    assert response.filename == "old.txt"
    assert response.status.value == "ready"
    assert queued == []
    assert old_path.exists()
    assert len(store.tasks) == 0


def test_update_rejects_hash_owned_by_another_document(monkeypatch, tmp_path):
    store = InMemoryStore()
    content = b"duplicate"
    store.documents["doc1"] = {"id": "doc1", "kb_id": "kb1", "filename": "one.txt", "status": "ready"}
    store.documents["doc2"] = {
        "id": "doc2", "kb_id": "kb1", "filename": "two.txt", "status": "ready",
        "content_hash": hashlib.sha256(content).hexdigest(),
    }
    monkeypatch.setattr(documents, "get_metadata_store", lambda: store)
    monkeypatch.setattr(documents, "UPLOAD_STORAGE_DIR", str(tmp_path))

    with pytest.raises(documents.HTTPException) as exc_info:
        asyncio.run(documents.update_document("kb1", "doc1", FakeUploadFile("one.txt", content)))

    assert exc_info.value.status_code == 409
    assert exc_info.value.detail["document_id"] == "doc2"
    assert list(tmp_path.rglob("*")) == []


def test_update_prioritizes_other_duplicate_over_self_noop(monkeypatch, tmp_path):
    store = InMemoryStore()
    content_hash = hashlib.sha256(b"same").hexdigest()
    store.documents["doc1"] = {
        "id": "doc1", "kb_id": "kb1", "filename": "one.txt", "file_type": ".txt",
        "status": "ready", "content_hash": content_hash,
    }
    store.documents["doc2"] = {
        "id": "doc2", "kb_id": "kb1", "filename": "two.txt", "status": "ready",
        "content_hash": content_hash,
    }
    monkeypatch.setattr(documents, "get_metadata_store", lambda: store)
    monkeypatch.setattr(documents, "UPLOAD_STORAGE_DIR", str(tmp_path))

    with pytest.raises(documents.HTTPException) as exc_info:
        asyncio.run(documents.update_document("kb1", "doc1", FakeUploadFile("one.txt", b"same")))

    assert exc_info.value.status_code == 409
    assert exc_info.value.detail["document_id"] == "doc2"


def test_changed_update_preserves_active_file_and_index_and_creates_staging_version(monkeypatch, tmp_path):
    store = InMemoryStore()
    old_path = tmp_path / "kb1" / "doc1" / "v1" / "old.txt"
    old_path.parent.mkdir(parents=True)
    old_path.write_bytes(b"old")
    store.documents["doc1"] = {
        "id": "doc1", "kb_id": "kb1", "filename": "old.txt", "file_type": ".txt",
        "status": "ready", "chunk_count": 2, "task_id": "task1", "storage_path": str(old_path),
        "content_hash": hashlib.sha256(b"old").hexdigest(), "current_version": 1,
    }
    store.versions[("doc1", 1)] = {"content_hash": hashlib.sha256(b"old").hexdigest(), "storage_path": str(old_path), "status": "active"}
    queued = []
    monkeypatch.setattr(documents, "get_metadata_store", lambda: store)
    monkeypatch.setattr(documents, "get_indexing_worker", lambda: type("Worker", (), {"enqueue": lambda _, task: queued.append(task)})())
    monkeypatch.setattr(documents, "UPLOAD_STORAGE_DIR", str(tmp_path))
    monkeypatch.setattr(documents, "get_engine", lambda: (_ for _ in ()).throw(AssertionError("old index must remain")))

    response = asyncio.run(documents.update_document("kb1", "doc1", FakeUploadFile("new.txt", b"new")))

    assert old_path.read_bytes() == b"old"
    assert store.documents["doc1"]["current_version"] == 1
    assert store.documents["doc1"]["content_hash"] == hashlib.sha256(b"old").hexdigest()
    staging = store.versions[("doc1", 2)]
    assert Path(staging["storage_path"]).read_bytes() == b"new"
    assert response.task_id in staging["storage_path"]
    assert store.tasks[response.task_id]["document_version"] == 2
    assert queued == [response.task_id]


def test_update_enqueue_failure_keeps_staging_task_and_files(monkeypatch, tmp_path):
    store = InMemoryStore()
    old_path = tmp_path / "kb1" / "doc1" / "v1" / "old.txt"
    old_path.parent.mkdir(parents=True)
    old_path.write_bytes(b"old")
    store.documents["doc1"] = {
        "id": "doc1", "kb_id": "kb1", "filename": "old.txt", "file_type": ".txt",
        "status": "ready", "chunk_count": 2, "storage_path": str(old_path),
        "content_hash": hashlib.sha256(b"old").hexdigest(), "current_version": 1,
    }
    store.versions[("doc1", 1)] = {
        "content_hash": hashlib.sha256(b"old").hexdigest(), "storage_path": str(old_path), "status": "active",
    }

    class FailingWorker:
        def enqueue(self, task_id):
            raise RuntimeError("queue unavailable")

    monkeypatch.setattr(documents, "get_metadata_store", lambda: store)
    monkeypatch.setattr(documents, "get_indexing_worker", lambda: FailingWorker())
    monkeypatch.setattr(documents, "UPLOAD_STORAGE_DIR", str(tmp_path))

    response = asyncio.run(documents.update_document("kb1", "doc1", FakeUploadFile("new.txt", b"new")))

    assert response.status.value == "queued"
    assert store.tasks[response.task_id]["status"] == "queued"
    assert old_path.read_bytes() == b"old"
    assert Path(store.versions[("doc1", 2)]["storage_path"]).read_bytes() == b"new"


def test_update_transaction_failure_removes_only_new_task_directory(monkeypatch, tmp_path):
    class FailingStore(InMemoryStore):
        def create_document_version_and_task(self, *args, **kwargs):
            raise RuntimeError("database unavailable")

    store = FailingStore()
    old_path = tmp_path / "kb1" / "doc1" / "v1" / "old.txt"
    old_path.parent.mkdir(parents=True)
    old_path.write_bytes(b"old")
    store.documents["doc1"] = {
        "id": "doc1", "kb_id": "kb1", "filename": "old.txt", "file_type": ".txt",
        "status": "ready", "chunk_count": 2, "storage_path": str(old_path),
        "content_hash": hashlib.sha256(b"old").hexdigest(), "current_version": 1,
    }
    monkeypatch.setattr(documents, "get_metadata_store", lambda: store)
    monkeypatch.setattr(documents, "UPLOAD_STORAGE_DIR", str(tmp_path))

    with pytest.raises(documents.HTTPException) as exc_info:
        asyncio.run(documents.update_document("kb1", "doc1", FakeUploadFile("new.txt", b"new")))

    assert exc_info.value.status_code == 500
    assert old_path.read_bytes() == b"old"
    assert [path.name for path in old_path.parents[1].iterdir()] == ["v1"]
    assert store.tasks == {}


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
