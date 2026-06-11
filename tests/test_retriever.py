from __future__ import annotations

import threading

from KBzhy.app.core.retriever import Retriever
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

    def add_documents(self, docs):
        self.added_documents.extend(docs)

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
