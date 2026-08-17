from __future__ import annotations

import asyncio
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from KBzhy.app.api import documents
from KBzhy.app.core.indexing_worker import IndexingWorker
from KBzhy.app.core.document_models import KnowledgeChunk, content_hash
from KBzhy.app.core.metadata_store import MySQLMetadataStore


class FakeUploadFile:
    def __init__(self, filename: str, content: bytes):
        self.filename = filename
        self._content = content

    async def read(self):
        return self._content


def make_structured_chunk(chunk_id, chunk_type, version=2):
    content = chunk_id
    return KnowledgeChunk(
        chunk_id=chunk_id,
        document_id="doc1",
        document_version=version,
        parent_chunk_id="parent-new" if chunk_type == "child" else None,
        chunk_type=chunk_type,
        content=content,
        retrieval_text=content,
        content_hash=content_hash(content),
        section_path=(),
        page_start=None,
        page_end=None,
        position=0,
        token_count=1,
        index_version=1,
    )


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
            "filename": filename,
            "file_type": file_type,
            "status": "staging",
        }
        document.update(
            status="queued",
            task_id=task_id,
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

    def get_document_version(self, doc_id, version):
        item = self.versions.get((doc_id, version))
        return dict(item) if item else None

    def update_document_version_status(self, doc_id, version, status):
        self.versions[(doc_id, version)]["status"] = status

    def complete_indexing_task(self, task_id, chunk_count):
        task = self.tasks[task_id]
        document = self.documents[task["doc_id"]]
        version = self.versions[(task["doc_id"], task["document_version"])]
        if (
            document.get("task_id") != task_id
            or document.get("status") == "deleting"
            or task.get("status") != "indexing"
            or version.get("status") != "staging"
        ):
            raise RuntimeError("indexing task is no longer current")
        for (doc_id, _), item in self.versions.items():
            if doc_id == task["doc_id"] and item.get("status") == "active":
                item["status"] = "inactive"
        version["status"] = "active"
        document.update(
            filename=version.get("filename"),
            file_type=version.get("file_type"),
            storage_path=version.get("storage_path"),
            content_hash=version.get("content_hash"),
            current_version=task["document_version"],
            active_index_version=task["index_version"],
            status="ready",
            chunk_count=chunk_count,
            error_message=None,
        )
        task.update(status="ready", error_message=None)

    def is_indexing_completion_committed(self, task_id):
        task = self.tasks.get(task_id)
        if not task or task.get("status") != "ready":
            return False
        document = self.documents.get(task["doc_id"])
        version = self.versions.get((task["doc_id"], task.get("document_version")))
        return bool(
            document
            and document.get("status") == "ready"
            and document.get("task_id") == task_id
            and int(document.get("current_version") or 0) == int(task.get("document_version") or 0)
            and version
            and version.get("status") == "active"
        )

    def finish_indexing_task(self, task_id, status, error_message):
        task = self.tasks[task_id]
        if task.get("status") not in {"queued", "parsing", "chunking", "indexing"}:
            return False
        document = self.documents.get(task["doc_id"])
        version = self.versions.get((task["doc_id"], task.get("document_version")))
        owns_document = bool(
            document
            and document.get("task_id") == task_id
            and document.get("status") != "deleting"
        )
        task.update(status=status, error_message=error_message)
        if version and version.get("status") == "staging":
            version["status"] = status
        if owns_document:
            document.update(
                status="ready" if int(document.get("current_version") or 0) > 0 else "failed",
                error_message=error_message,
            )
        return True

    def requeue_indexing_task(self, task_id):
        task = self.tasks[task_id]
        document = self.documents.get(task["doc_id"])
        version = self.versions.get((task["doc_id"], task.get("document_version")))
        if (
            not document
            or document.get("task_id") != task_id
            or document.get("status") == "deleting"
            or not version
            or version.get("status") != "staging"
        ):
            return False
        task.update(status="queued", error_message=None)
        document.update(status="queued", error_message=None)
        return True

    def claim_task_recovery(
        self, task_id, owner, now, lease_until, expected_updated_at=None
    ):
        task = self.tasks.get(task_id)
        if not task:
            return False
        document = self.documents.get(task["doc_id"])
        version = self.versions.get((task["doc_id"], task.get("document_version")))
        current_lease = task.get("recovery_lease_until")
        if (
            task.get("status") not in {"queued", "parsing", "chunking", "indexing"}
            or not document
            or document.get("task_id") != task_id
            or document.get("status") == "deleting"
            or not version
            or version.get("status") != "staging"
            or (
                expected_updated_at is not None
                and task.get("updated_at") != expected_updated_at
            )
            or (task.get("recovery_owner") and current_lease and current_lease > now)
        ):
            return False
        task.update(
            recovery_owner=owner, recovery_lease_until=lease_until, updated_at=now
        )
        return True

    def complete_task_recovery(self, task_id, owner):
        task = self.tasks.get(task_id)
        if not task or task.get("recovery_owner") != owner:
            return False
        document = self.documents.get(task["doc_id"])
        version = self.versions.get((task["doc_id"], task.get("document_version")))
        if (
            not document
            or document.get("task_id") != task_id
            or document.get("status") == "deleting"
            or not version
            or version.get("status") != "staging"
        ):
            return False
        task.update(
            status="queued", error_message=None,
            updated_at=datetime.now(),
        )
        document.update(status="queued", error_message=None)
        return True

    def set_indexing_phase(self, task_id, phase):
        task = self.tasks[task_id]
        document = self.documents.get(task["doc_id"])
        allowed_statuses = {
            "parsing": {"parsing"},
            "chunking": {"parsing"},
            "indexing": {"parsing", "chunking"},
        }
        if (
            not document
            or document.get("task_id") != task_id
            or document.get("status") == "deleting"
            or task.get("status") not in allowed_statuses[phase]
        ):
            return False
        document.update(status=phase, error_message=None)
        task["status"] = phase
        return True

    def update_document(self, doc_id, **changes):
        self.documents[doc_id].update(changes)

    def update_task(self, task_id, **changes):
        self.tasks[task_id].update(changes)

    def claim_task(self, task_id, recovery_owner=None):
        task = self.tasks.get(task_id)
        if not task or task.get("status") != "queued":
            return False
        lease_until = task.get("recovery_lease_until")
        if (
            task.get("recovery_owner")
            and lease_until
            and lease_until > datetime.now()
            and task.get("recovery_owner") != recovery_owner
        ):
            return False
        task.update(
            status="parsing", recovery_owner=None, recovery_lease_until=None
        )
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
    assert store.documents["doc1"]["filename"] == "old.txt"
    assert store.documents["doc1"]["file_type"] == ".txt"
    assert store.documents["doc1"]["storage_path"] == str(old_path)
    assert store.documents["doc1"]["chunk_count"] == 2
    staging = store.versions[("doc1", 2)]
    assert Path(staging["storage_path"]).read_bytes() == b"new"
    assert response.task_id in staging["storage_path"]
    assert store.tasks[response.task_id]["document_version"] == 2
    assert response.chunk_count == 2
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
    store.tasks["task1"].update(document_version=1, index_version=1)
    store.versions[("doc1", 1)] = {
        "filename": "bad.txt", "file_type": ".txt", "storage_path": str(file_path), "status": "staging",
    }
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
    assert store.versions[("doc1", 1)]["status"] == "failed"
    assert removed == [("bad.txt", "kb1", "doc1", "task1")]


def test_indexing_worker_failed_task_removes_only_its_safe_version_directory(monkeypatch, tmp_path):
    monkeypatch.setattr("KBzhy.app.core.indexing_worker.UPLOAD_STORAGE_DIR", str(tmp_path))
    active_path = tmp_path / "kb1" / "doc1" / "v1" / "old.txt"
    failed_path = tmp_path / "kb1" / "doc1" / "task2" / "new.txt"
    active_path.parent.mkdir(parents=True)
    failed_path.parent.mkdir(parents=True)
    active_path.write_bytes(b"old")
    failed_path.write_bytes(b"new")
    store = InMemoryStore()
    store.documents["doc1"] = {
        "id": "doc1", "kb_id": "kb1", "filename": "old.txt", "file_type": ".txt",
        "storage_path": str(active_path), "content_hash": "old-hash", "current_version": 1,
        "status": "queued", "task_id": "task2", "chunk_count": 2,
    }
    store.tasks["task2"] = {
        "task_id": "task2", "doc_id": "doc1", "kb_id": "kb1", "status": "queued",
        "document_version": 2, "index_version": 1,
    }
    store.versions[("doc1", 2)] = {
        "filename": "new.txt", "file_type": ".txt", "storage_path": str(failed_path), "status": "staging",
    }

    class Engine:
        def index_document(self, *args, **kwargs):
            raise RuntimeError("failed")

        def remove_document(self, *args, **kwargs):
            pass

    IndexingWorker(store=store, engine_factory=lambda: Engine(), autostart=False).process_task("task2")

    assert active_path.read_bytes() == b"old"
    assert not failed_path.parent.exists()
    assert store.documents["doc1"]["filename"] == "old.txt"
    assert store.documents["doc1"]["status"] == "ready"
    assert store.versions[("doc1", 2)]["status"] == "failed"


def test_indexing_worker_does_not_remove_version_path_outside_expected_boundary(tmp_path):
    unsafe_path = tmp_path / "shared" / "new.txt"
    unsafe_path.parent.mkdir()
    unsafe_path.write_bytes(b"new")
    store = InMemoryStore()
    store.documents["doc1"] = {
        "id": "doc1", "kb_id": "kb1", "filename": None, "file_type": None,
        "storage_path": None, "content_hash": None, "current_version": 0,
        "status": "queued", "task_id": "task1", "chunk_count": 0,
    }
    store.tasks["task1"] = {
        "task_id": "task1", "doc_id": "doc1", "kb_id": "kb1", "status": "queued",
        "document_version": 1, "index_version": 1,
    }
    store.versions[("doc1", 1)] = {
        "filename": "new.txt", "file_type": ".txt", "storage_path": str(unsafe_path), "status": "staging",
    }

    class Engine:
        def index_document(self, *args, **kwargs):
            raise RuntimeError("failed")

        def remove_document(self, *args, **kwargs):
            pass

    IndexingWorker(store=store, engine_factory=lambda: Engine(), autostart=False).process_task("task1")

    assert unsafe_path.read_bytes() == b"new"
    assert store.documents["doc1"]["status"] == "failed"
    assert store.documents["doc1"]["current_version"] == 0
    assert store.documents["doc1"]["filename"] is None
    assert store.documents["doc1"]["storage_path"] is None
    assert store.versions[("doc1", 1)]["status"] == "failed"


def test_indexing_worker_does_not_remove_matching_hierarchy_outside_upload_root(monkeypatch, tmp_path):
    upload_root = tmp_path / "configured-uploads"
    outside_path = tmp_path / "outside" / "kb1" / "doc1" / "task1" / "new.txt"
    outside_path.parent.mkdir(parents=True)
    outside_path.write_bytes(b"new")
    monkeypatch.setattr("KBzhy.app.core.indexing_worker.UPLOAD_STORAGE_DIR", str(upload_root))
    store = InMemoryStore()
    store.documents["doc1"] = {
        "id": "doc1", "kb_id": "kb1", "filename": None, "file_type": None,
        "storage_path": None, "content_hash": None, "current_version": 0,
        "status": "queued", "task_id": "task1", "chunk_count": 0,
    }
    store.tasks["task1"] = {
        "task_id": "task1", "doc_id": "doc1", "kb_id": "kb1", "status": "queued",
        "document_version": 1, "index_version": 1,
    }
    store.versions[("doc1", 1)] = {
        "filename": "new.txt", "file_type": ".txt", "storage_path": str(outside_path), "status": "staging",
    }

    class Engine:
        def index_document(self, *args, **kwargs):
            raise RuntimeError("failed")

        def remove_document(self, *args, **kwargs):
            pass

    IndexingWorker(store=store, engine_factory=lambda: Engine(), autostart=False).process_task("task1")

    assert outside_path.read_bytes() == b"new"


def test_late_failure_after_activation_never_deletes_active_version(monkeypatch, tmp_path):
    monkeypatch.setattr("KBzhy.app.core.indexing_worker.UPLOAD_STORAGE_DIR", str(tmp_path))
    active_path = tmp_path / "kb1" / "doc1" / "task2" / "new.txt"
    active_path.parent.mkdir(parents=True)
    active_path.write_bytes(b"new")

    class ActivatedThenRaisedStore(InMemoryStore):
        def complete_indexing_task(self, task_id, chunk_count):
            super().complete_indexing_task(task_id, chunk_count)
            raise RuntimeError("connection dropped after commit")

        def finish_indexing_task(self, task_id, status, error_message):
            if self.tasks[task_id]["status"] == "ready":
                return False
            return super().finish_indexing_task(task_id, status, error_message)

    store = ActivatedThenRaisedStore()
    store.documents["doc1"] = {
        "id": "doc1", "kb_id": "kb1", "filename": "old.txt", "file_type": ".txt",
        "storage_path": None, "content_hash": "old", "current_version": 1,
        "status": "queued", "task_id": "task2", "chunk_count": 1,
    }
    store.tasks["task2"] = {
        "task_id": "task2", "doc_id": "doc1", "kb_id": "kb1", "status": "queued",
        "document_version": 2, "index_version": 1,
    }
    store.versions[("doc1", 2)] = {
        "filename": "new.txt", "file_type": ".txt", "storage_path": str(active_path),
        "content_hash": "new", "status": "staging",
    }

    removed = []

    class Engine:
        def index_document(self, *args, **kwargs):
            return 3

        def remove_document(self, *args, **kwargs):
            removed.append((args, kwargs))

    IndexingWorker(store=store, engine_factory=lambda: Engine(), autostart=False).process_task("task2")

    assert active_path.read_bytes() == b"new"
    assert store.tasks["task2"]["status"] == "ready"
    assert store.versions[("doc1", 2)]["status"] == "active"
    assert store.documents["doc1"]["current_version"] == 2
    assert removed == []


def test_post_commit_exception_uses_authoritative_completion_check_not_stale_shared_reads(
    monkeypatch, tmp_path
):
    monkeypatch.setattr("KBzhy.app.core.indexing_worker.UPLOAD_STORAGE_DIR", str(tmp_path))
    active_path = tmp_path / "kb1" / "doc1" / "task2" / "new.txt"
    active_path.parent.mkdir(parents=True)
    active_path.write_bytes(b"new")

    class StaleSnapshotAfterCommitStore(InMemoryStore):
        def __init__(self):
            super().__init__()
            self.completion_checks = 0
            self.return_stale_shared_reads = False

        def complete_indexing_task(self, task_id, chunk_count):
            super().complete_indexing_task(task_id, chunk_count)
            self.return_stale_shared_reads = True
            raise RuntimeError("connection dropped after commit")

        def get_task(self, task_id):
            if self.return_stale_shared_reads:
                return {**self.tasks[task_id], "status": "indexing"}
            return super().get_task(task_id)

        def get_document(self, kb_id, doc_id):
            if self.return_stale_shared_reads:
                return {**self.documents[doc_id], "status": "indexing", "current_version": 1}
            return super().get_document(kb_id, doc_id)

        def get_document_version(self, doc_id, version):
            if self.return_stale_shared_reads:
                return {**self.versions[(doc_id, version)], "status": "staging"}
            return super().get_document_version(doc_id, version)

        def is_indexing_completion_committed(self, task_id):
            self.completion_checks += 1
            return InMemoryStore.is_indexing_completion_committed(self, task_id)

    store = StaleSnapshotAfterCommitStore()
    store.documents["doc1"] = {
        "id": "doc1", "kb_id": "kb1", "filename": "old.txt", "file_type": ".txt",
        "storage_path": None, "content_hash": "old", "current_version": 1,
        "status": "queued", "task_id": "task2", "chunk_count": 1,
    }
    store.tasks["task2"] = {
        "task_id": "task2", "doc_id": "doc1", "kb_id": "kb1", "status": "queued",
        "document_version": 2, "index_version": 1,
    }
    store.versions[("doc1", 2)] = {
        "filename": "new.txt", "file_type": ".txt", "storage_path": str(active_path),
        "content_hash": "new", "status": "staging",
    }
    removed = []

    class Engine:
        def index_document(self, *args, **kwargs):
            return 3

        def remove_document(self, *args, **kwargs):
            removed.append((args, kwargs))

    IndexingWorker(store=store, engine_factory=lambda: Engine(), autostart=False).process_task("task2")

    assert store.completion_checks == 1
    assert active_path.read_bytes() == b"new"
    assert removed == []


def test_post_commit_exception_cleans_up_when_fresh_completion_check_is_incomplete(
    monkeypatch, tmp_path
):
    monkeypatch.setattr("KBzhy.app.core.indexing_worker.UPLOAD_STORAGE_DIR", str(tmp_path))
    staging_path = tmp_path / "kb1" / "doc1" / "task2" / "new.txt"
    staging_path.parent.mkdir(parents=True)
    staging_path.write_bytes(b"new")

    class IncompleteCompletionStore(InMemoryStore):
        def complete_indexing_task(self, task_id, chunk_count):
            raise RuntimeError("commit failed")

        def is_indexing_completion_committed(self, task_id):
            return False

    store = IncompleteCompletionStore()
    store.documents["doc1"] = {
        "id": "doc1", "kb_id": "kb1", "filename": "old.txt", "file_type": ".txt",
        "storage_path": None, "content_hash": "old", "current_version": 1,
        "status": "queued", "task_id": "task2", "chunk_count": 1,
    }
    store.tasks["task2"] = {
        "task_id": "task2", "doc_id": "doc1", "kb_id": "kb1", "status": "queued",
        "document_version": 2, "index_version": 1,
    }
    store.versions[("doc1", 2)] = {
        "filename": "new.txt", "file_type": ".txt", "storage_path": str(staging_path),
        "content_hash": "new", "status": "staging",
    }
    removed = []

    class Engine:
        def index_document(self, *args, **kwargs):
            return 3

        def remove_document(self, *args, **kwargs):
            removed.append((args, kwargs))

    IndexingWorker(store=store, engine_factory=lambda: Engine(), autostart=False).process_task("task2")

    assert removed
    assert store.tasks["task2"]["status"] == "failed"
    assert not staging_path.parent.exists()


def test_post_commit_exception_preserves_state_when_authoritative_check_itself_fails(
    monkeypatch, tmp_path
):
    monkeypatch.setattr("KBzhy.app.core.indexing_worker.UPLOAD_STORAGE_DIR", str(tmp_path))
    staging_path = tmp_path / "kb1" / "doc1" / "task2" / "new.txt"
    staging_path.parent.mkdir(parents=True)
    staging_path.write_bytes(b"new")

    class FailingCompletionCheckStore(InMemoryStore):
        def complete_indexing_task(self, task_id, chunk_count):
            raise RuntimeError("commit result unknown")

        def is_indexing_completion_committed(self, task_id):
            raise RuntimeError("fresh read failed")

    store = FailingCompletionCheckStore()
    store.documents["doc1"] = {
        "id": "doc1", "kb_id": "kb1", "filename": "old.txt", "file_type": ".txt",
        "storage_path": None, "content_hash": "old", "current_version": 1,
        "status": "queued", "task_id": "task2", "chunk_count": 1,
    }
    store.tasks["task2"] = {
        "task_id": "task2", "doc_id": "doc1", "kb_id": "kb1", "status": "queued",
        "document_version": 2, "index_version": 1,
    }
    store.versions[("doc1", 2)] = {
        "filename": "new.txt", "file_type": ".txt", "storage_path": str(staging_path),
        "content_hash": "new", "status": "staging",
    }
    removed = []

    class Engine:
        def index_document(self, *args, **kwargs):
            return 3

        def remove_document(self, *args, **kwargs):
            removed.append((args, kwargs))

    IndexingWorker(store=store, engine_factory=lambda: Engine(), autostart=False).process_task("task2")

    assert removed == []
    assert store.tasks["task2"]["status"] == "indexing"
    assert store.documents["doc1"]["status"] == "indexing"
    assert store.versions[("doc1", 2)]["status"] == "staging"
    assert staging_path.read_bytes() == b"new"


def test_indexing_worker_reads_file_metadata_from_task_document_version(tmp_path):
    active_path = tmp_path / "v1" / "old.txt"
    staging_path = tmp_path / "task2" / "new.txt"
    active_path.parent.mkdir()
    staging_path.parent.mkdir()
    active_path.write_bytes(b"old")
    staging_path.write_bytes(b"new")
    store = InMemoryStore()
    store.documents["doc1"] = {
        "id": "doc1", "kb_id": "kb1", "filename": "old.txt", "file_type": ".txt",
        "storage_path": str(active_path), "content_hash": "old-hash", "current_version": 1,
        "status": "queued", "task_id": "task2", "chunk_count": 2,
    }
    store.tasks["task2"] = {
        "task_id": "task2", "doc_id": "doc1", "kb_id": "kb1", "status": "queued",
        "document_version": 2, "index_version": 1,
    }
    store.versions[("doc1", 2)] = {
        "filename": "new.txt", "file_type": ".txt", "storage_path": str(staging_path),
        "content_hash": "new-hash", "status": "staging",
    }

    class Engine:
        def index_document(self, path, kb_id, display_name=None, doc_id=None, task_id=None):
            assert path == str(staging_path)
            assert display_name == "new.txt"
            return 1

    IndexingWorker(store=store, engine_factory=lambda: Engine(), autostart=False).process_task("task2")

    assert store.tasks["task2"]["status"] == "ready"
    assert store.documents["doc1"]["filename"] == "new.txt"
    assert store.documents["doc1"]["storage_path"] == str(staging_path)
    assert store.documents["doc1"]["content_hash"] == "new-hash"
    assert store.documents["doc1"]["current_version"] == 2
    assert store.versions[("doc1", 2)]["status"] == "active"


def test_indexing_worker_skips_stale_task_when_document_has_newer_task(monkeypatch, tmp_path):
    monkeypatch.setattr("KBzhy.app.core.indexing_worker.UPLOAD_STORAGE_DIR", str(tmp_path))
    active_path = tmp_path / "kb1" / "doc1" / "v1" / "new.txt"
    stale_path = tmp_path / "kb1" / "doc1" / "task1" / "old.txt"
    active_path.parent.mkdir(parents=True)
    stale_path.parent.mkdir(parents=True)
    active_path.write_text("new content", encoding="utf-8")
    stale_path.write_text("old content", encoding="utf-8")
    store = InMemoryStore()
    store.documents["doc1"] = {
        "id": "doc1",
        "kb_id": "kb1",
        "filename": "new.txt",
        "file_type": ".txt",
        "status": "queued",
        "chunk_count": 0,
        "task_id": "task2",
        "storage_path": str(active_path),
        "current_version": 1,
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
    }
    store.tasks["task1"] = {
        "task_id": "task1", "doc_id": "doc1", "kb_id": "kb1", "status": "queued",
        "document_version": 2, "index_version": 1,
    }
    store.versions[("doc1", 2)] = {
        "filename": "old.txt", "file_type": ".txt", "storage_path": str(stale_path), "status": "staging",
    }
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
    assert store.versions[("doc1", 2)]["status"] == "stale"
    assert not stale_path.parent.exists()
    assert active_path.read_text(encoding="utf-8") == "new content"


def test_indexing_worker_marks_task_failed_when_version_metadata_is_missing():
    store = InMemoryStore()
    store.documents["doc1"] = {
        "id": "doc1", "kb_id": "kb1", "filename": None, "file_type": None,
        "storage_path": None, "content_hash": None, "current_version": 0,
        "status": "queued", "task_id": "task1", "chunk_count": 0,
    }
    store.tasks["task1"] = {
        "task_id": "task1", "doc_id": "doc1", "kb_id": "kb1", "status": "queued",
        "document_version": 1, "index_version": 1,
    }

    IndexingWorker(store=store, engine_factory=lambda: None, autostart=False).process_task("task1")

    assert store.tasks["task1"]["status"] == "failed"
    assert store.documents["doc1"]["status"] == "failed"
    assert store.documents["doc1"]["current_version"] == 0
    assert store.documents["doc1"]["filename"] is None
    assert store.documents["doc1"]["storage_path"] is None


def test_old_task_late_failure_does_not_change_newer_document_state(tmp_path):
    stale_path = tmp_path / "kb1" / "doc1" / "task1" / "old.txt"
    stale_path.parent.mkdir(parents=True)
    stale_path.write_bytes(b"old")
    store = InMemoryStore()
    store.documents["doc1"] = {
        "id": "doc1", "kb_id": "kb1", "filename": "new.txt", "status": "indexing",
        "task_id": "task2", "current_version": 1, "chunk_count": 7,
        "storage_path": str(tmp_path / "kb1" / "doc1" / "v1" / "new.txt"),
    }
    store.tasks["task1"] = {
        "task_id": "task1", "doc_id": "doc1", "kb_id": "kb1", "status": "queued",
        "document_version": 2, "index_version": 1,
    }
    store.versions[("doc1", 2)] = {
        "filename": "old.txt", "storage_path": str(stale_path), "status": "staging",
    }

    IndexingWorker(store=store, engine_factory=lambda: None, autostart=False).process_task("task1")

    assert store.tasks["task1"]["status"] == "stale"
    assert store.documents["doc1"]["status"] == "indexing"
    assert store.documents["doc1"]["task_id"] == "task2"
    assert store.documents["doc1"]["chunk_count"] == 7


def test_worker_does_not_revive_deleting_document(tmp_path):
    version_path = tmp_path / "kb1" / "doc1" / "task1" / "new.txt"
    version_path.parent.mkdir(parents=True)
    version_path.write_bytes(b"new")
    store = InMemoryStore()
    store.documents["doc1"] = {
        "id": "doc1", "kb_id": "kb1", "task_id": "task1", "status": "deleting",
        "current_version": 1, "storage_path": None, "chunk_count": 4,
    }
    store.tasks["task1"] = {
        "task_id": "task1", "doc_id": "doc1", "kb_id": "kb1", "status": "queued",
        "document_version": 2, "index_version": 1,
    }
    store.versions[("doc1", 2)] = {
        "filename": "new.txt", "storage_path": str(version_path), "status": "staging",
    }

    IndexingWorker(
        store=store,
        engine_factory=lambda: (_ for _ in ()).throw(AssertionError("deleting document must not index")),
        autostart=False,
    ).process_task("task1")

    assert store.documents["doc1"]["status"] == "deleting"
    assert store.documents["doc1"]["chunk_count"] == 4
    assert store.tasks["task1"]["status"] == "stale"
    assert store.versions[("doc1", 2)]["status"] == "stale"


def test_worker_losing_ownership_after_claim_never_changes_newer_document(tmp_path):
    version_path = tmp_path / "kb1" / "doc1" / "task1" / "new.txt"
    version_path.parent.mkdir(parents=True)
    version_path.write_bytes(b"new")
    store = InMemoryStore()
    store.documents["doc1"] = {
        "id": "doc1", "kb_id": "kb1", "task_id": "task1", "status": "queued",
        "current_version": 1, "storage_path": None, "chunk_count": 4,
    }
    store.tasks["task1"] = {
        "task_id": "task1", "doc_id": "doc1", "kb_id": "kb1", "status": "queued",
        "document_version": 2, "index_version": 1,
    }
    store.versions[("doc1", 2)] = {
        "filename": "new.txt", "storage_path": str(version_path), "status": "staging",
    }

    def claim_and_replace(task_id, recovery_owner=None):
        store.tasks[task_id]["status"] = "parsing"
        store.documents["doc1"].update(task_id="task2", status="queued", chunk_count=8)
        return True

    store.claim_task = claim_and_replace
    worker = IndexingWorker(
        store=store,
        engine_factory=lambda: (_ for _ in ()).throw(AssertionError("stale task must not index")),
        autostart=False,
    )
    worker.process_task("task1")

    assert store.documents["doc1"]["task_id"] == "task2"
    assert store.documents["doc1"]["status"] == "queued"
    assert store.documents["doc1"]["chunk_count"] == 8
    assert store.tasks["task1"]["status"] == "stale"


def test_recovery_does_not_requeue_newer_or_deleting_document(tmp_path):
    store = InMemoryStore()
    store.list_recoverable_tasks = lambda: [store.tasks["old"], store.tasks["deleting"]]
    for doc_id, task_id, status in (("doc-old", "new", "indexing"), ("doc-del", "deleting", "deleting")):
        path = tmp_path / "kb1" / doc_id / ("old" if doc_id == "doc-old" else task_id) / "x.txt"
        path.parent.mkdir(parents=True)
        path.write_bytes(b"x")
        old_task = "old" if doc_id == "doc-old" else "deleting"
        store.documents[doc_id] = {
            "id": doc_id, "kb_id": "kb1", "task_id": task_id, "status": status,
            "current_version": 1, "storage_path": None,
        }
        store.tasks[old_task] = {
            "task_id": old_task, "doc_id": doc_id, "kb_id": "kb1", "status": "indexing",
            "document_version": 2, "index_version": 1,
        }
        store.versions[(doc_id, 2)] = {"filename": "x.txt", "storage_path": str(path), "status": "staging"}

    worker = IndexingWorker(store=store, engine_factory=lambda: (_ for _ in ()).throw(AssertionError()), autostart=False)
    worker.recover_unfinished_tasks()

    assert store.tasks["old"]["status"] == "stale"
    assert store.tasks["deleting"]["status"] == "stale"
    assert store.documents["doc-old"]["status"] == "indexing"
    assert store.documents["doc-del"]["status"] == "deleting"
    assert worker._queue.empty()


def test_recovery_does_not_remove_vectors_when_task_becomes_ready_after_listing(tmp_path):
    path = tmp_path / "kb1" / "doc1" / "task1" / "doc.txt"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"content")
    store = InMemoryStore()
    store.documents["doc1"] = {
        "id": "doc1", "kb_id": "kb1", "task_id": "task1", "status": "indexing",
        "current_version": 0, "storage_path": None,
    }
    store.tasks["task1"] = {
        "task_id": "task1", "doc_id": "doc1", "kb_id": "kb1", "status": "indexing",
        "document_version": 1, "index_version": 1,
    }
    store.versions[("doc1", 1)] = {
        "filename": "doc.txt", "storage_path": str(path), "status": "staging",
    }
    task_snapshot = dict(store.tasks["task1"])
    store.list_recoverable_tasks = lambda: [task_snapshot]

    def finish_concurrently(_task_id, _owner, _now, _lease_until, _updated_at=None):
        store.tasks["task1"]["status"] = "ready"
        store.documents["doc1"].update(status="ready", current_version=1)
        store.versions[("doc1", 1)]["status"] = "active"
        return False

    store.claim_task_recovery = finish_concurrently
    removed = []

    class Engine:
        def remove_document(self, *args, **kwargs):
            removed.append((args, kwargs))

    worker = IndexingWorker(store=store, engine_factory=lambda: Engine(), autostart=False)
    worker.recover_unfinished_tasks()

    assert removed == []
    assert store.tasks["task1"]["status"] == "ready"
    assert store.documents["doc1"]["status"] == "ready"
    assert store.documents["doc1"]["current_version"] == 1
    assert store.versions[("doc1", 1)]["status"] == "active"
    assert worker._queue.empty()


def test_recovery_removes_interrupted_task_vectors_after_requeue_claim(tmp_path):
    path = tmp_path / "kb1" / "doc1" / "task1" / "doc.txt"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"content")
    store = InMemoryStore()
    store.documents["doc1"] = {
        "id": "doc1", "kb_id": "kb1", "task_id": "task1", "status": "indexing",
        "current_version": 0, "storage_path": None,
    }
    store.tasks["task1"] = {
        "task_id": "task1", "doc_id": "doc1", "kb_id": "kb1", "status": "indexing",
        "document_version": 1, "index_version": 1,
    }
    store.versions[("doc1", 1)] = {
        "filename": "doc.txt", "storage_path": str(path), "status": "staging",
    }
    store.list_recoverable_tasks = lambda: [dict(store.tasks["task1"])]
    removed = []

    class Engine:
        def remove_document(self, *args, **kwargs):
            removed.append((args, kwargs))

    worker = IndexingWorker(store=store, engine_factory=lambda: Engine(), autostart=False)
    worker.recover_unfinished_tasks()

    assert len(removed) == 1
    assert store.tasks["task1"]["status"] == "queued"
    assert store.documents["doc1"]["status"] == "queued"
    assert worker._queue.get_nowait() == "task1"


def test_two_workers_only_lease_cleanup_and_enqueue_task_once(tmp_path):
    path = tmp_path / "doc.txt"
    path.write_text("content", encoding="utf-8")
    store = InMemoryStore()
    store.documents["doc1"] = {
        "id": "doc1", "kb_id": "kb1", "task_id": "task1",
        "status": "indexing", "current_version": 0,
    }
    store.tasks["task1"] = {
        "task_id": "task1", "doc_id": "doc1", "kb_id": "kb1",
        "status": "indexing", "document_version": 1, "index_version": 1,
        "updated_at": datetime(2026, 8, 17, 10, 0, 0),
    }
    store.versions[("doc1", 1)] = {
        "filename": "doc.txt", "storage_path": str(path), "status": "staging",
    }
    store.list_recoverable_tasks = lambda: [dict(store.tasks["task1"])]
    remove_calls = []

    class Engine:
        def remove_document(self, *args, **kwargs):
            remove_calls.append((args, kwargs))

    worker_a = IndexingWorker(
        store=store, engine_factory=lambda: Engine(), autostart=False,
        recovery_owner="worker-a",
    )
    worker_b = IndexingWorker(
        store=store, engine_factory=lambda: Engine(), autostart=False,
        recovery_owner="worker-b",
    )

    worker_a.recover_unfinished_tasks()
    worker_b.recover_unfinished_tasks()

    assert len(remove_calls) == 1
    assert worker_a._queue.qsize() + worker_b._queue.qsize() == 1


def test_recovery_cleanup_failure_keeps_lease_and_does_not_enqueue(tmp_path):
    path = tmp_path / "doc.txt"
    path.write_text("content", encoding="utf-8")
    store = InMemoryStore()
    store.documents["doc1"] = {
        "id": "doc1", "kb_id": "kb1", "task_id": "task1",
        "status": "indexing", "current_version": 0,
    }
    store.tasks["task1"] = {
        "task_id": "task1", "doc_id": "doc1", "kb_id": "kb1",
        "status": "indexing", "document_version": 1, "index_version": 1,
    }
    store.versions[("doc1", 1)] = {
        "filename": "doc.txt", "storage_path": str(path), "status": "staging",
    }
    store.list_recoverable_tasks = lambda: [dict(store.tasks["task1"])]

    class Engine:
        def remove_document(self, *args, **kwargs):
            raise RuntimeError("chroma unavailable")

    worker = IndexingWorker(
        store=store, engine_factory=lambda: Engine(), autostart=False,
        recovery_owner="worker-a",
    )

    worker.recover_unfinished_tasks()

    assert worker._queue.empty()
    assert store.tasks["task1"]["status"] == "indexing"
    assert store.tasks["task1"]["recovery_owner"] == "worker-a"


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


def test_structured_worker_stages_activates_then_cleans_old_children(tmp_path):
    source = tmp_path / "doc.txt"
    source.write_text("content", encoding="utf-8")
    store = InMemoryStore()
    store.documents["doc1"] = {
        "id": "doc1", "kb_id": "kb1", "filename": "doc.txt", "file_type": ".txt",
        "status": "queued", "chunk_count": 0, "task_id": "task1",
        "storage_path": str(source), "current_version": 1,
    }
    store.versions[("doc1", 2)] = {
        "doc_id": "doc1", "version": 2, "filename": "doc.txt",
        "storage_path": str(source), "status": "staging",
    }
    store.tasks["task1"] = {
        "task_id": "task1", "doc_id": "doc1", "kb_id": "kb1", "status": "queued",
        "document_version": 2, "index_version": 1,
    }
    parent = KnowledgeChunk(
        chunk_id="parent-new", document_id="doc1", document_version=2,
        parent_chunk_id=None, chunk_type="parent", content="parent",
        retrieval_text="parent", content_hash=content_hash("parent"), section_path=(),
        page_start=None, page_end=None, position=0, token_count=1, index_version=1,
    )
    child = KnowledgeChunk(
        chunk_id="child-new", document_id="doc1", document_version=2,
        parent_chunk_id="parent-new", chunk_type="child", content="child",
        retrieval_text="child", content_hash=content_hash("child"), section_path=(),
        page_start=None, page_end=None, position=0, token_count=1, index_version=1,
    )
    events = []

    class Repository:
        def list_active_children(self, document_id):
            return [SimpleNamespace(chunk_id="child-old")]

        def replace_staging(self, task_id, document_id, version, chunks, parsed_artifact_path=None):
            events.append(("mysql-stage", [item.chunk_id for item in chunks], parsed_artifact_path))

        def activate_version(self, document_id, version, task_id):
            events.append(("activate", document_id, version, task_id))
            store.tasks[task_id]["status"] = "ready"
            store.documents[document_id].update(status="ready", current_version=version)
            store.versions[(document_id, version)]["status"] = "active"

        def discard_task(self, task_id):
            events.append(("discard", task_id))

    class Engine:
        def parse_document_for_index(self, *args, **kwargs):
            assert store.tasks["task1"]["status"] == "parsing"
            events.append(("parse", kwargs["document_version"]))
            return SimpleNamespace(artifact_path="parsed/v2.json")

        def chunk_document_for_index(self, parsed_artifact, *, index_version):
            assert store.tasks["task1"]["status"] == "chunking"
            events.append(("chunk", index_version))
            return SimpleNamespace(chunks=(parent, child), artifact_path=parsed_artifact.artifact_path)

        def stage_document_children(self, kb_id, document_id, children):
            events.append(("vector-stage", [item.chunk_id for item in children]))

        def remove_children(self, kb_id, chunk_ids):
            events.append(("remove", list(chunk_ids)))

    worker = IndexingWorker(
        store=store, engine_factory=lambda: Engine(), autostart=False,
        chunk_repository=Repository(),
    )
    worker.process_task("task1")

    assert events == [
        ("parse", 2),
        ("chunk", 1),
        ("mysql-stage", ["parent-new", "child-new"], "parsed/v2.json"),
        ("vector-stage", ["child-new"]),
        ("activate", "doc1", 2, "task1"),
        ("remove", ["child-old"]),
    ]
    assert store.documents["doc1"]["status"] == "ready"


def test_structured_worker_stage_failure_cleans_new_ids_but_keeps_old_active(tmp_path):
    source = tmp_path / "doc.txt"
    source.write_text("content", encoding="utf-8")
    store = InMemoryStore()
    store.documents["doc1"] = {
        "id": "doc1", "kb_id": "kb1", "filename": "doc.txt", "file_type": ".txt",
        "status": "queued", "chunk_count": 4, "task_id": "task1",
        "storage_path": str(source), "current_version": 1,
    }
    store.versions[("doc1", 2)] = {
        "doc_id": "doc1", "version": 2, "filename": "doc.txt",
        "storage_path": str(source), "status": "staging",
    }
    store.tasks["task1"] = {
        "task_id": "task1", "doc_id": "doc1", "kb_id": "kb1", "status": "queued",
        "document_version": 2, "index_version": 1,
    }
    parent = make_structured_chunk("parent-new", "parent")
    child = make_structured_chunk("child-new", "child")
    removed = []
    discarded = []

    class Repository:
        def list_active_children(self, document_id):
            return [SimpleNamespace(chunk_id="child-old")]

        def replace_staging(self, *args, **kwargs):
            pass

        def activate_version(self, *args):
            raise AssertionError("failed vector stage must not activate")

        def discard_task(self, task_id):
            discarded.append(task_id)

    class Engine:
        def parse_document_for_index(self, *args, **kwargs):
            return SimpleNamespace(artifact_path="parsed/v2.json")

        def chunk_document_for_index(self, parsed_artifact, *, index_version):
            return SimpleNamespace(chunks=(parent, child), artifact_path=parsed_artifact.artifact_path)

        def stage_document_children(self, *args):
            raise RuntimeError("embedding failed")

        def remove_children(self, kb_id, chunk_ids):
            removed.append(list(chunk_ids))

    worker = IndexingWorker(
        store=store, engine_factory=lambda: Engine(), autostart=False,
        chunk_repository=Repository(),
    )
    worker.process_task("task1")

    assert removed == [["child-new"]]
    assert discarded == ["task1"]
    assert "child-old" not in removed[0]
    assert store.documents["doc1"]["current_version"] == 1
    assert store.tasks["task1"]["status"] == "failed"


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
    document_sql, document_params = next(
        (sql, params) for sql, params in executed
        if "INSERT IGNORE INTO documents" in " ".join(sql.split())
    )
    assert "current_version" in document_sql
    assert document_params[6:8] == (1, 1)
    version_sql, version_params = next(
        (sql, params) for sql, params in executed
        if "INSERT IGNORE INTO document_versions" in " ".join(sql.split())
    )
    assert "'active'" in version_sql
    assert version_params[1:3] == ("doc1", "guide.pdf")
    assert ("COMMIT", None) in executed
    assert not (data_dir / "kb_meta.json").exists()
    assert not (data_dir / "doc_registry.json").exists()
    assert (data_dir / "kb_meta.json.migrated").exists()
    assert (data_dir / "doc_registry.json.migrated").exists()


def test_legacy_json_migration_keeps_files_when_version_insert_fails(monkeypatch, tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "kb_meta.json").write_text(
        '{"kb1":{"name":"Legacy KB"}}', encoding="utf-8"
    )
    (data_dir / "doc_registry.json").write_text(
        '{"kb1":{"doc1":{"filename":"guide.pdf","status":"ready"}}}',
        encoding="utf-8",
    )

    calls = []

    class FailingCursor:
        def execute(self, sql, params=None):
            calls.append((sql, params))
            if "INSERT IGNORE INTO document_versions" in " ".join(sql.split()):
                raise RuntimeError("version insert failed")

        def close(self):
            pass

    class FakeConn:
        def cursor(self):
            return FailingCursor()

        def commit(self):
            calls.append(("COMMIT", None))

        def rollback(self):
            calls.append(("ROLLBACK", None))

    monkeypatch.setattr("KBzhy.app.core.metadata_store.DATA_DIR", str(data_dir))
    store = MySQLMetadataStore.__new__(MySQLMetadataStore)
    store._conn = FakeConn()

    store._migrate_legacy_json_if_present()

    assert ("ROLLBACK", None) in calls
    assert (data_dir / "kb_meta.json").exists()
    assert (data_dir / "doc_registry.json").exists()
    assert not (data_dir / "kb_meta.json.migrated").exists()
    assert not (data_dir / "doc_registry.json.migrated").exists()
