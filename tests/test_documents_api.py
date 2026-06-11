from __future__ import annotations

import asyncio
import glob
import os
import tempfile

import pytest
from fastapi import HTTPException

from KBzhy.app.api import documents


class FakeUploadFile:
    def __init__(self, filename: str, content: bytes):
        self.filename = filename
        self._content = content

    async def read(self):
        return self._content


def test_upload_rejects_unknown_knowledge_base(monkeypatch):
    monkeypatch.setattr(documents, "_doc_registry", {})
    monkeypatch.setattr(documents, "_kb_meta", {})
    monkeypatch.setattr(documents, "get_engine", lambda: (_ for _ in ()).throw(AssertionError("engine should not be used")))

    with pytest.raises(HTTPException) as exc:
        asyncio.run(documents.upload_document("missing-kb", FakeUploadFile("a.txt", b"hello")))

    assert exc.value.status_code == 404


def test_upload_removes_temp_file_when_indexing_fails(monkeypatch):
    class FailingEngine:
        def index_document(self, tmp_path, kb_id, display_name=None):
            assert os.path.exists(tmp_path)
            raise RuntimeError("index failed")

    monkeypatch.setattr(documents, "_doc_registry", {"kb1": {}})
    monkeypatch.setattr(documents, "_kb_meta", {"kb1": {"name": "KB"}})
    monkeypatch.setattr(documents, "get_engine", lambda: FailingEngine())
    monkeypatch.setattr(documents, "_save_registry", lambda: None)

    before = set(glob.glob(os.path.join(tempfile.gettempdir(), "*.txt")))
    with pytest.raises(HTTPException):
        asyncio.run(documents.upload_document("kb1", FakeUploadFile("leak-check.txt", b"hello")))
    after = set(glob.glob(os.path.join(tempfile.gettempdir(), "*.txt")))

    assert after == before


def test_get_document_chunks_returns_chunks_for_existing_document(monkeypatch):
    class ChunkEngine:
        def list_document_chunks(self, kb_id, source):
            assert kb_id == "kb1"
            assert source == "guide.pdf"
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

    monkeypatch.setattr(
        documents,
        "_doc_registry",
        {
            "kb1": {
                "doc1": {
                    "filename": "guide.pdf",
                    "kb_id": "kb1",
                    "status": "ready",
                    "chunk_count": 2,
                    "created_at": "2026-06-11T10:00:00",
                    "updated_at": "2026-06-11T10:00:00",
                }
            }
        },
    )
    monkeypatch.setattr(documents, "get_engine", lambda: ChunkEngine())

    response = documents.get_document_chunks("kb1", "doc1")

    assert response.kb_id == "kb1"
    assert response.document_id == "doc1"
    assert response.filename == "guide.pdf"
    assert response.total == 2
    assert [chunk.chunk_index for chunk in response.chunks] == [1, 2]
    assert response.chunks[0].content == "first chunk"


def test_get_document_chunks_rejects_unknown_document(monkeypatch):
    monkeypatch.setattr(documents, "_doc_registry", {"kb1": {}})
    monkeypatch.setattr(documents, "get_engine", lambda: (_ for _ in ()).throw(AssertionError("engine should not be used")))

    with pytest.raises(HTTPException) as exc:
        documents.get_document_chunks("kb1", "missing")

    assert exc.value.status_code == 404
