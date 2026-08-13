from __future__ import annotations

import asyncio
import hashlib

import pytest
from fastapi import HTTPException

from KBzhy.app.api import documents


class FakeUploadFile:
    def __init__(self, filename: str, content: bytes):
        self.filename = filename
        self._content = content

    async def read(self):
        return self._content


class FakeStore:
    def __init__(self):
        self.kbs = {}
        self.documents = {}
        self.tasks = {}

    def knowledge_base_exists(self, kb_id):
        return kb_id in self.kbs

    def create_document(self, data):
        self.documents[data["id"]] = dict(data)

    def create_task(self, data):
        self.tasks[data["task_id"]] = dict(data)

    def get_document(self, kb_id, doc_id):
        doc = self.documents.get(doc_id)
        if doc and doc["kb_id"] == kb_id:
            return dict(doc)
        return None

    def find_document_by_hash(self, kb_id, content_hash, exclude_document_id=None):
        for doc in self.documents.values():
            if (
                doc["kb_id"] == kb_id
                and doc["id"] != exclude_document_id
                and doc.get("content_hash") == content_hash
                and doc.get("status") != "deleting"
            ):
                return dict(doc)
        return None


def test_upload_rejects_unknown_knowledge_base(monkeypatch):
    store = FakeStore()
    monkeypatch.setattr(documents, "get_metadata_store", lambda: store)
    monkeypatch.setattr(documents, "get_engine", lambda: (_ for _ in ()).throw(AssertionError("engine should not be used")))

    with pytest.raises(HTTPException) as exc:
        asyncio.run(documents.upload_document("missing-kb", FakeUploadFile("a.txt", b"hello")))

    assert exc.value.status_code == 404


def test_upload_rejects_empty_file(monkeypatch):
    store = FakeStore()
    store.kbs["kb1"] = {"kb_id": "kb1"}
    monkeypatch.setattr(documents, "get_metadata_store", lambda: store)

    with pytest.raises(HTTPException) as exc:
        asyncio.run(documents.upload_document("kb1", FakeUploadFile("empty.txt", b"")))

    assert exc.value.status_code == 400


def test_upload_duplicate_error_identifies_existing_document(monkeypatch, tmp_path):
    store = FakeStore()
    content = b"duplicate content"
    store.kbs["kb1"] = {"kb_id": "kb1"}
    store.documents["doc-existing"] = {
        "id": "doc-existing", "kb_id": "kb1", "filename": "original.txt",
        "status": "ready", "content_hash": hashlib.sha256(content).hexdigest(),
    }
    monkeypatch.setattr(documents, "get_metadata_store", lambda: store)
    monkeypatch.setattr(documents, "UPLOAD_STORAGE_DIR", str(tmp_path))

    with pytest.raises(HTTPException) as exc:
        asyncio.run(documents.upload_document("kb1", FakeUploadFile("copy.txt", content)))

    assert exc.value.status_code == 409
    assert exc.value.detail == {
        "message": "知识库中已存在相同内容的文档",
        "document_id": "doc-existing",
    }
    assert not tmp_path.exists() or not list(tmp_path.iterdir())


def test_update_rejects_empty_filename_before_writing(monkeypatch, tmp_path):
    store = FakeStore()
    store.documents["doc1"] = {
        "id": "doc1", "kb_id": "kb1", "filename": "old.txt", "file_type": ".txt",
        "status": "ready", "chunk_count": 1,
    }
    monkeypatch.setattr(documents, "get_metadata_store", lambda: store)
    monkeypatch.setattr(documents, "UPLOAD_STORAGE_DIR", str(tmp_path))

    with pytest.raises(HTTPException) as exc:
        asyncio.run(documents.update_document("kb1", "doc1", FakeUploadFile("", b"new")))

    assert exc.value.status_code == 400
    assert list(tmp_path.iterdir()) == []


def test_get_document_chunks_returns_chunks_for_ready_document(monkeypatch):
    store = FakeStore()
    store.documents["doc1"] = {
        "id": "doc1",
        "filename": "guide.pdf",
        "file_type": ".pdf",
        "kb_id": "kb1",
        "status": "ready",
        "chunk_count": 2,
        "created_at": "2026-06-11T10:00:00",
        "updated_at": "2026-06-11T10:00:00",
    }

    class ChunkEngine:
        def list_document_chunks(self, kb_id, source=None, doc_id=None):
            assert kb_id == "kb1"
            assert source == "guide.pdf"
            assert doc_id == "doc1"
            return [
                {
                    "chunk_index": 2,
                    "content": "second chunk",
                    "metadata": {"source": "guide.pdf", "page": 3},
                },
                {
                    "chunk_index": 1,
                    "content": "first chunk",
                    "metadata": {"source": "guide.pdf", "page": 1},
                },
            ]

    monkeypatch.setattr(documents, "get_metadata_store", lambda: store)
    monkeypatch.setattr(documents, "get_engine", lambda: ChunkEngine())

    response = documents.get_document_chunks("kb1", "doc1")

    assert response.kb_id == "kb1"
    assert response.document_id == "doc1"
    assert response.filename == "guide.pdf"
    assert response.total == 2
    assert [chunk.chunk_index for chunk in response.chunks] == [1, 2]
    assert response.chunks[0].content == "first chunk"


def test_get_document_chunks_returns_empty_for_unready_document(monkeypatch):
    store = FakeStore()
    store.documents["doc1"] = {
        "id": "doc1",
        "filename": "guide.pdf",
        "file_type": ".pdf",
        "kb_id": "kb1",
        "status": "indexing",
        "chunk_count": 0,
        "created_at": "2026-06-11T10:00:00",
        "updated_at": "2026-06-11T10:00:00",
    }
    monkeypatch.setattr(documents, "get_metadata_store", lambda: store)
    monkeypatch.setattr(documents, "get_engine", lambda: (_ for _ in ()).throw(AssertionError("engine should not be used")))

    response = documents.get_document_chunks("kb1", "doc1")

    assert response.total == 0
    assert response.chunks == []


def test_get_document_chunks_rejects_unknown_document(monkeypatch):
    store = FakeStore()
    monkeypatch.setattr(documents, "get_metadata_store", lambda: store)
    monkeypatch.setattr(documents, "get_engine", lambda: (_ for _ in ()).throw(AssertionError("engine should not be used")))

    with pytest.raises(HTTPException) as exc:
        documents.get_document_chunks("kb1", "missing")

    assert exc.value.status_code == 404
