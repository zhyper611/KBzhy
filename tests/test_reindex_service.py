from dataclasses import dataclass

import pytest

from KBzhy.app.core.document_models import KnowledgeChunk
from KBzhy.app.core.reindex_service import ReindexCompletionUnknown, ReindexService


def make_chunks(doc_id: str, version: int = 1):
    parent = KnowledgeChunk(
        chunk_id=f"{doc_id}-parent", document_id=doc_id, document_version=version,
        parent_chunk_id=None, chunk_type="parent", content="parent",
        retrieval_text="parent", content_hash="parent-hash", section_path=(),
        page_start=None, page_end=None, position=0, token_count=1, index_version=1,
    )
    child = KnowledgeChunk(
        chunk_id=f"{doc_id}-child", document_id=doc_id, document_version=version,
        parent_chunk_id=parent.chunk_id, chunk_type="child", content="child",
        retrieval_text="child", content_hash="child-hash", section_path=(),
        page_start=None, page_end=None, position=0, token_count=1, index_version=1,
    )
    return (parent, child)


@dataclass
class Prepared:
    chunks: tuple
    artifact_path: str


class Store:
    def __init__(self):
        self.kb = {"kb_id": "kb1", "active_collection_name": "kbzhy_kb1"}
        self.documents = [
            {"id": "doc-1", "filename": "one.txt", "storage_path": "/one.txt"},
            {"id": "doc-2", "filename": "two.txt", "storage_path": "/two.txt"},
        ]
        self.activated = None
        self.aborted = []
        self.activation_error = None
        self.verification_error = None

    def get_kb(self, kb_id):
        return dict(self.kb) if kb_id == "kb1" else None

    def list_documents(self, kb_id, page, page_size, status=None):
        return len(self.documents), list(self.documents) if page == 1 else []

    def create_reindex_task(self, kb_id, doc_id, task_id, index_version, now):
        document = next(item for item in self.documents if item["id"] == doc_id)
        return {
            **document, "document_version": 1, "owner_task_id": f"owner-{doc_id}"
        }

    def activate_reindex(self, kb_id, collection_name, manifests):
        self.activated = (kb_id, collection_name, list(manifests))
        self.kb["active_collection_name"] = collection_name
        if self.activation_error:
            raise self.activation_error

    def is_reindex_committed(self, kb_id, collection_name, task_ids):
        if self.verification_error:
            raise self.verification_error
        return self.kb["active_collection_name"] == collection_name

    def abort_reindex(self, task_ids):
        self.aborted.append(list(task_ids))


class Repository:
    def __init__(self):
        self.staged = []

    def replace_reindex_staging(self, task_id, doc_id, version, chunks):
        self.staged.append((task_id, doc_id, version, tuple(chunks)))


class Engine:
    def __init__(self, fail_doc=None):
        self.fail_doc = fail_doc
        self.staged = []
        self.deleted = []
        self.removed_artifacts = []

    def prepare_document_index(self, path, kb_id, **kwargs):
        doc_id = kwargs["document_id"]
        if doc_id == self.fail_doc:
            raise RuntimeError("parse failed")
        return Prepared(make_chunks(doc_id), f"/artifacts/{doc_id}.json")

    def stage_collection_children(self, collection, kb_id, doc_id, children):
        self.staged.append((collection, kb_id, doc_id, tuple(children)))

    def delete_collection(self, collection):
        self.deleted.append(collection)

    def remove_parsed_artifact(self, path):
        self.removed_artifacts.append(path)


def make_service(store=None, engine=None):
    store = store or Store()
    engine = engine or Engine()
    repository = Repository()
    return ReindexService(store, engine, repository), store, engine, repository


def test_reindex_switches_collection_only_after_every_document_is_staged():
    service, store, engine, repository = make_service()

    result = service.reindex("kb1")

    assert result.status == "ready"
    assert result.document_count == 2
    assert result.child_count == 2
    assert store.kb["active_collection_name"] == result.collection_name
    assert len(repository.staged) == 2
    assert [item[2] for item in engine.staged] == ["doc-1", "doc-2"]
    assert len(store.activated[2]) == 2


def test_reindex_failure_keeps_old_collection_and_cleans_temporary_state():
    service, store, engine, _ = make_service(engine=Engine(fail_doc="doc-2"))

    with pytest.raises(RuntimeError, match="parse failed"):
        service.reindex("kb1")

    assert store.kb["active_collection_name"] == "kbzhy_kb1"
    assert len(engine.deleted) == 1
    assert len(store.aborted) == 1
    assert engine.removed_artifacts == ["/artifacts/doc-1.json"]


def test_cleanup_refuses_to_delete_active_collection():
    service, store, engine, _ = make_service()

    with pytest.raises(ValueError, match="active collection"):
        service.cleanup_collection("kb1", "kbzhy_kb1")

    assert engine.deleted == []


def test_cleanup_treats_default_collection_as_active_when_pointer_is_null():
    service, store, engine, _ = make_service()
    store.kb["active_collection_name"] = None

    with pytest.raises(ValueError, match="active collection"):
        service.cleanup_collection("kb1", "kbzhy_kb1")

    assert engine.deleted == []


def test_cleanup_deletes_only_explicit_inactive_collection():
    service, store, engine, _ = make_service()
    store.kb["active_collection_name"] = "kbzhy_kb1_v2_current"

    service.cleanup_collection("kb1", "kbzhy_kb1")

    assert engine.deleted == ["kbzhy_kb1"]


def test_cleanup_refuses_collection_owned_by_another_knowledge_base():
    service, _, engine, _ = make_service()

    with pytest.raises(ValueError, match="does not belong"):
        service.cleanup_collection("kb1", "kbzhy_kb2_v2_old")

    assert engine.deleted == []


def test_reindex_rejects_empty_knowledge_base_without_switching():
    store = Store()
    store.documents = []
    service, _, engine, _ = make_service(store=store)

    with pytest.raises(ValueError, match="no ready documents"):
        service.reindex("kb1")

    assert store.kb["active_collection_name"] == "kbzhy_kb1"
    assert engine.deleted == []


def test_activation_acknowledgement_failure_uses_authoritative_committed_state():
    store = Store()
    store.activation_error = RuntimeError("connection lost after commit")
    service, _, engine, _ = make_service(store=store)

    result = service.reindex("kb1")

    assert result.status == "ready"
    assert store.kb["active_collection_name"] == result.collection_name
    assert engine.deleted == []
    assert store.aborted == []


def test_unknown_activation_result_preserves_collection_and_staging_for_recovery():
    store = Store()
    store.activation_error = RuntimeError("connection lost")
    store.verification_error = RuntimeError("database unavailable")
    service, _, engine, _ = make_service(store=store)

    with pytest.raises(ReindexCompletionUnknown):
        service.reindex("kb1")

    assert engine.deleted == []
    assert store.aborted == []
    assert engine.removed_artifacts == []
