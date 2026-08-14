from __future__ import annotations

import threading

import pytest

from KBzhy.app.core.document_models import KnowledgeChunk, RetrievalCandidate
from KBzhy.app.core.retriever import Retriever, rrf_fuse
from KBzhy.app.core.splitter import Chunk


class FakeVectorStore:
    def __init__(self):
        self.deleted_ids = []
        self.added_documents = []
        self.documents = ["alpha policy text", "beta unrelated text", "gamma unrelated text"]
        self.metadatas = [
            {"source": "policy.md", "page": 3},
            {"source": "other.md", "page": 1},
            {"source": "third.md", "page": 1},
        ]

    def add_documents(self, docs, ids=None):
        self.added_documents.extend(docs)
        self.added_ids = list(ids or [])

    def get(self, where=None):
        if where == {"source": "policy.md"}:
            return {"ids": ["chunk-1"]}
        return {"documents": self.documents, "metadatas": self.metadatas}

    def delete(self, ids):
        self.deleted_ids.extend(ids)


def make_retriever(fake_vs):
    retriever = Retriever.__new__(Retriever)
    retriever._vectorstores = {}
    retriever._bm25_indices = {}
    retriever._lock = threading.Lock()
    retriever._get_vectorstore = lambda kb_id: fake_vs
    return retriever


def make_knowledge_chunk(chunk_id, *, chunk_type="child", version=2, position=0):
    return KnowledgeChunk(
        chunk_id=chunk_id,
        document_id="doc-1",
        document_version=version,
        parent_chunk_id="parent-1" if chunk_type == "child" else None,
        chunk_type=chunk_type,
        content=f"content-{chunk_id}",
        retrieval_text=f"Section\n\ncontent-{chunk_id}",
        content_hash=f"hash-{chunk_id}",
        section_path=("Section",),
        page_start=1,
        page_end=2,
        position=position,
        token_count=3,
        index_version=1,
        metadata={"source": "policy.md"},
    )


def make_candidate(chunk_id, raw_score=0.0):
    return RetrievalCandidate(
        chunk_id=chunk_id,
        content=f"content-{chunk_id}",
        metadata={"chunk_id": chunk_id},
        vector_score=raw_score,
        bm25_score=raw_score,
    )


def test_rrf_merges_same_chunk_by_stable_id():
    fused = rrf_fuse(
        [make_candidate("a"), make_candidate("b")],
        [make_candidate("b"), make_candidate("c")],
        k=60,
    )

    assert fused[0].chunk_id == "b"
    assert fused[0].vector_rank == 2
    assert fused[0].bm25_rank == 1


def test_rrf_uses_rank_instead_of_raw_score_scale():
    fused = rrf_fuse(
        [make_candidate("low", 0.01), make_candidate("high", 9999)], [], k=60
    )

    assert [item.chunk_id for item in fused] == ["low", "high"]


def test_hybrid_search_fetches_each_route_independently_and_filters_before_rrf():
    retriever = Retriever.__new__(Retriever)
    retriever.vector_fetch_k = 30
    retriever.bm25_fetch_k = 25
    retriever.rrf_k = 60
    retriever.rrf_candidate_k = 40
    calls = []
    retriever._vector_search = lambda query, kb_id, k: calls.append(("vector", k)) or [
        ("old", {"chunk_id": "old", "doc_id": "doc-1", "document_version": 1}, 0.99)
    ]
    retriever._bm25_search = lambda query, kb_id, k: calls.append(("bm25", k)) or [
        ("new", {"chunk_id": "new", "doc_id": "doc-1", "document_version": 2}, 0.01)
    ]
    retriever._active_version_resolver = lambda doc_ids: {"doc-1": 2}

    result = retriever._hybrid_search("query", "kb1", 5)

    assert sorted(calls) == [("bm25", 25), ("vector", 30)]
    assert [item.chunk_id for item in result] == ["new"]
    assert result[0].bm25_rank == 1


def test_hybrid_search_uses_surviving_route_when_other_route_raises():
    retriever = Retriever.__new__(Retriever)
    retriever.vector_fetch_k = 30
    retriever.bm25_fetch_k = 30
    retriever.rrf_k = 60
    retriever.rrf_candidate_k = 40
    retriever._vector_search = lambda *_args: (_ for _ in ()).throw(RuntimeError("down"))
    retriever._bm25_search = lambda *_args: [
        ("text", {"chunk_id": "child-1"}, 0.5)
    ]
    retriever._active_version_resolver = lambda doc_ids: {}

    result = retriever._hybrid_search("query", "kb1", 5)

    assert [item.chunk_id for item in result] == ["child-1"]


def test_stage_document_children_uses_stable_ids_and_only_indexes_children():
    fake_vs = FakeVectorStore()
    retriever = make_retriever(fake_vs)
    children = [make_knowledge_chunk("child-1"), make_knowledge_chunk("child-2", position=1)]

    retriever.stage_document_children("kb1", "doc-1", children)

    assert fake_vs.added_ids == ["child-1", "child-2"]
    assert [doc.page_content for doc in fake_vs.added_documents] == [
        "Section\n\ncontent-child-1",
        "Section\n\ncontent-child-2",
    ]
    assert fake_vs.added_documents[0].metadata["document_version"] == 2
    assert fake_vs.added_documents[0].metadata["section_path"] == '["Section"]'


def test_stage_document_children_is_idempotent_in_bm25():
    fake_vs = FakeVectorStore()
    retriever = make_retriever(fake_vs)
    child = make_knowledge_chunk("child-1")

    retriever.stage_document_children("kb1", "doc-1", [child])
    retriever.stage_document_children("kb1", "doc-1", [child])

    _, entries = retriever._bm25_indices["kb1"]
    assert [entry["metadata"]["chunk_id"] for entry in entries] == ["child-1"]


def test_remove_children_deletes_only_explicit_ids_from_both_indices():
    fake_vs = FakeVectorStore()
    retriever = make_retriever(fake_vs)
    retriever.stage_document_children(
        "kb1",
        "doc-1",
        [make_knowledge_chunk("child-1"), make_knowledge_chunk("child-2", position=1)],
    )

    retriever.remove_children("kb1", ["child-1"])

    assert fake_vs.deleted_ids == ["child-1"]
    _, entries = retriever._bm25_indices["kb1"]
    assert [entry["metadata"]["chunk_id"] for entry in entries] == ["child-2"]


def test_active_collection_switch_replaces_vectorstore_and_invalidates_bm25():
    retriever = Retriever.__new__(Retriever)
    active = {"name": "kbzhy_kb1"}
    stores = {}
    retriever._active_collection_resolver = lambda kb_id: active["name"]
    retriever._vectorstores = {}
    retriever._vectorstore_names = {}
    retriever._bm25_indices = {"kb1": (object(), [{"content": "old"}])}
    retriever._get_named_vectorstore = lambda name: stores.setdefault(name, object())

    old_store = retriever._get_vectorstore("kb1")
    active["name"] = "kbzhy_kb1_v2_20260814"
    new_store = retriever._get_vectorstore("kb1")

    assert old_store is stores["kbzhy_kb1"]
    assert new_store is stores["kbzhy_kb1_v2_20260814"]
    assert new_store is not old_store
    assert "kb1" not in retriever._bm25_indices


def test_active_collection_lookup_failure_keeps_last_known_collection():
    retriever = Retriever.__new__(Retriever)
    retriever._vectorstore_names = {"kb1": "kbzhy_kb1_v2_stable"}

    def fail(_kb_id):
        raise RuntimeError("database unavailable")

    retriever._active_collection_resolver = fail

    assert retriever._collection_name("kb1") == "kbzhy_kb1_v2_stable"


def test_active_collection_lookup_failure_without_cache_fails_closed():
    retriever = Retriever.__new__(Retriever)
    retriever._vectorstore_names = {}
    retriever._active_collection_resolver = lambda _kb_id: (_ for _ in ()).throw(
        RuntimeError("database unavailable")
    )

    with pytest.raises(RuntimeError, match="database unavailable"):
        retriever._collection_name("kb1")


def test_stage_collection_children_targets_explicit_collection_without_bm25_mutation():
    retriever = Retriever.__new__(Retriever)
    fake_vs = FakeVectorStore()
    retriever._bm25_indices = {"kb1": (None, [{"content": "active"}])}
    requested = []

    def get_named(name):
        requested.append(name)
        return fake_vs

    retriever._get_named_vectorstore = get_named

    retriever.stage_collection_children(
        "kbzhy_kb1_v2_temp", "kb1", "doc-1", [make_knowledge_chunk("child-1")]
    )

    assert requested == ["kbzhy_kb1_v2_temp"]
    assert fake_vs.added_ids == ["child-1"]
    assert retriever._bm25_indices["kb1"][1] == [{"content": "active"}]


def test_versioned_candidates_must_match_mysql_active_version():
    retriever = Retriever.__new__(Retriever)
    retriever._active_version_resolver = lambda doc_ids: {"doc-1": 2}
    candidates = [
        ("old", {"doc_id": "doc-1", "document_version": 1}, 0.9),
        ("new", {"doc_id": "doc-1", "document_version": 2}, 0.8),
        ("legacy-same-doc", {"doc_id": "doc-1"}, 0.75),
        ("legacy", {"source": "legacy.md"}, 0.7),
    ]

    filtered = retriever._filter_active_candidates(candidates)

    assert [item[0] for item in filtered] == ["new", "legacy"]


def test_active_version_lookup_failure_drops_candidates_with_document_identity():
    retriever = Retriever.__new__(Retriever)

    def fail(_doc_ids):
        raise RuntimeError("database unavailable")

    retriever._active_version_resolver = fail
    candidates = [
        ("versioned", {"doc_id": "doc-1", "document_version": 2}, 0.9),
        ("legacy-same-doc", {"doc_id": "doc-1"}, 0.8),
        ("unregistered-legacy", {"source": "legacy.md"}, 0.7),
    ]

    filtered = retriever._filter_active_candidates(candidates)

    assert [item[0] for item in filtered] == ["unregistered-legacy"]


def test_remove_document_opens_persisted_vectorstore_when_not_cached():
    fake_vs = FakeVectorStore()
    retriever = make_retriever(fake_vs)

    retriever.remove_document("kb1", source="policy.md")

    assert fake_vs.deleted_ids == ["chunk-1"]


def test_bm25_search_preserves_chunk_metadata():
    fake_vs = FakeVectorStore()
    retriever = make_retriever(fake_vs)
    chunks = [
        Chunk("alpha policy text", metadata={"source": "policy.md", "page": 3}),
        Chunk("beta unrelated text", metadata={"source": "other.md", "page": 1}),
        Chunk("gamma unrelated text", metadata={"source": "third.md", "page": 1}),
    ]

    retriever.add_documents(chunks, "kb1")
    results = retriever._bm25_search("alpha", "kb1", 1)

    assert results[0][1] == {"source": "policy.md", "page": 3, "kb_id": "kb1"}


def test_retrieve_filters_scores_after_rerank():
    retriever = Retriever.__new__(Retriever)
    retriever.top_k = 2
    retriever.threshold = 0.35
    retriever._is_complex = lambda query: False
    retriever._hybrid_search = lambda query, kb_id, top_k, request_id=None: [
        ("high after rerank", {"source": "a.md"}, 0.95),
        ("low after rerank", {"source": "b.md"}, 0.92),
    ]
    retriever._mmr = lambda query, candidates, top_k: [
        {"content": content, "metadata": metadata, "score": score}
        for content, metadata, score in candidates
    ]

    def fake_rerank(query, candidates, method):
        candidates[0]["score"] = 0.8
        candidates[1]["score"] = 0.2
        return candidates

    retriever._rerank = fake_rerank

    results = retriever.retrieve(
        "query",
        "kb1",
        top_k=2,
        rerank_method="model",
        enable_expansion=False,
        enable_decomposition=False,
        threshold=0.7,
    )

    assert [item["content"] for item in results] == ["high after rerank"]


def test_mmr_embeds_candidate_documents_in_one_batch():
    class FakeEmbeddings:
        def __init__(self):
            self.query_calls = 0
            self.document_calls = []

        def embed_query(self, text):
            self.query_calls += 1
            return [1.0, 0.0]

        def embed_documents(self, texts):
            self.document_calls.append(list(texts))
            return [
                [1.0, 0.0],
                [0.9, 0.1],
                [0.0, 1.0],
            ]

    retriever = Retriever.__new__(Retriever)
    retriever.embeddings = FakeEmbeddings()
    candidates = [
        ("alpha", {"source": "a.md"}, 0.9),
        ("beta", {"source": "b.md"}, 0.8),
        ("gamma", {"source": "c.md"}, 0.7),
    ]

    results = retriever._mmr("query", candidates, top_k=2)

    assert len(results) == 2
    assert retriever.embeddings.query_calls == 1
    assert retriever.embeddings.document_calls == [["alpha", "beta", "gamma"]]


def test_mmr_logs_batch_embedding_failure_reason(caplog):
    class FailingEmbeddings:
        def embed_query(self, text):
            return [1.0, 0.0]

        def embed_documents(self, texts):
            raise RuntimeError("400 Bad Request: input too long")

    retriever = Retriever.__new__(Retriever)
    retriever.embeddings = FailingEmbeddings()
    candidates = [
        ("alpha", {"source": "a.md"}, 0.9),
        ("beta", {"source": "b.md"}, 0.8),
        ("gamma", {"source": "c.md"}, 0.7),
    ]

    with caplog.at_level("WARNING", logger="KBzhy.app.core.retriever"):
        results = retriever._mmr("query", candidates, top_k=2)

    assert [item["content"] for item in results] == ["alpha", "beta"]
    assert "MMR embedding 失败，回退到相关性排序" in caplog.text
    assert "400 Bad Request: input too long" in caplog.text


def test_mmr_batches_candidate_embeddings_by_provider_limit():
    class FakeEmbeddings:
        def __init__(self):
            self.document_calls = []

        def embed_query(self, text):
            return [1.0, 0.0]

        def embed_documents(self, texts):
            self.document_calls.append(list(texts))
            return [[1.0, 0.0] for _ in texts]

    retriever = Retriever.__new__(Retriever)
    retriever.embeddings = FakeEmbeddings()
    candidates = [
        (f"content-{i}", {"source": f"{i}.md"}, 1.0 - i * 0.01)
        for i in range(12)
    ]

    results = retriever._mmr("query", candidates, top_k=5)

    assert len(results) == 5
    assert [len(batch) for batch in retriever.embeddings.document_calls] == [10, 2]
