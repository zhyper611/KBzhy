from __future__ import annotations

import httpx
import pytest
from contextlib import nullcontext
from pathlib import Path
from fastapi import HTTPException

from KBzhy.app.api import chat
from KBzhy.app.core.context_assembler import ContextUnit
from KBzhy.app.core.document_models import RetrievalCandidate
from KBzhy.app.core.rag_engine import KNOWLEDGE_QA_REFUSAL, KNOWLEDGE_QA_SYSTEM_PROMPT, RAGEngine


class FakeMemory:
    def __init__(self, context=None):
        self.messages = []
        self._context = context or []

    def get_context(self):
        return self._context

    def add_message(self, role, content, sources=None):
        item = {"role": role, "content": content}
        if sources is not None:
            item["sources"] = sources
        self.messages.append(item)


class FakeMemoryManager:
    def __init__(self, memory):
        self.memory = memory

    def get(self, session_id):
        return self.memory


class EmptyRetriever:
    def retrieve(self, *args, **kwargs):
        return []


class FixedRetriever:
    def retrieve(self, *args, **kwargs):
        return [
            {
                "content": "irrelevant retrieved content",
                "metadata": {"source": "education.md", "page": 1},
                "score": 0.9,
            }
        ]


def test_chat_generates_from_assembled_context_but_sources_remain_original_hits():
    hit = RetrievalCandidate(
        chunk_id="hit-1",
        content="short child hit",
        metadata={"doc_id": "doc-1", "source": "policy.md", "page": 2},
        rerank_score=0.9,
    )

    class Retriever:
        def retrieve(self, *args, **kwargs):
            return [hit]

    parent = ContextUnit(
        chunk_id="parent-1",
        document_id="doc-1",
        content="expanded parent context",
        metadata={"source": "policy.md"},
        context_role="parent",
        origin_chunk_id="hit-1",
    )

    class Assembler:
        def assemble(self, candidates, final_k):
            assert candidates == [hit]
            return [parent]

    engine = RAGEngine.__new__(RAGEngine)
    engine.memory_manager = FakeMemoryManager(None)
    engine.retriever = Retriever()
    engine.context_assembler = Assembler()
    engine.llm_model = "test-model"
    captured = {}

    def generate(messages, temperature):
        captured["messages"] = messages
        return "answer"

    engine._call_llm_sync = generate
    engine._manage_context_window = lambda messages: nullcontext()

    result = engine.chat(
        question="question", kb_id="kb1", top_k=5,
        enable_expansion=False, enable_rewrite=False,
    )

    assert "expanded parent context" in captured["messages"][0]["content"]
    assert "short child hit" not in captured["messages"][0]["content"]
    assert [source["content"] for source in result["sources"]] == ["short child hit"]


def test_chat_records_refusal_in_session_memory():
    memory = FakeMemory()
    engine = RAGEngine.__new__(RAGEngine)
    engine.memory_manager = FakeMemoryManager(memory)
    engine.retriever = EmptyRetriever()
    engine.llm_model = "test-model"

    def fail_if_called(*args, **kwargs):
        raise AssertionError("empty retrieval should not call LLM")

    engine._call_llm_sync = fail_if_called

    result = engine.chat(question="未知问题", kb_id="kb1", session_id="s1")

    assert result["answer"] == KNOWLEDGE_QA_REFUSAL
    assert memory.messages == [
        {"role": "user", "content": "未知问题"},
        {"role": "assistant", "content": KNOWLEDGE_QA_REFUSAL, "sources": []},
    ]


def test_chat_clears_sources_when_llm_refuses_after_retrieval():
    memory = FakeMemory()
    engine = RAGEngine.__new__(RAGEngine)
    engine.memory_manager = FakeMemoryManager(memory)
    engine.retriever = FixedRetriever()
    engine.llm_model = "test-model"
    engine._call_llm_sync = lambda *args, **kwargs: f"{KNOWLEDGE_QA_REFUSAL} 建议补充对应文档。"
    engine._manage_context_window = lambda messages: nullcontext()

    result = engine.chat(question="out of scope question", kb_id="kb1", session_id="s1")

    assert result["sources"] == []
    assert memory.messages[-1]["sources"] == []


def test_chat_stream_clears_sources_when_llm_refuses_after_retrieval():
    memory = FakeMemory()
    engine = RAGEngine.__new__(RAGEngine)
    engine.memory_manager = FakeMemoryManager(memory)
    engine.retriever = FixedRetriever()
    engine.llm_model = "test-model"
    engine._call_llm_stream = lambda *args, **kwargs: iter([KNOWLEDGE_QA_REFUSAL, " 建议补充对应文档。"])
    engine._manage_context_window = lambda messages: nullcontext()

    chunks = list(engine.chat_stream(question="out of scope question", kb_id="kb1", session_id="s1"))

    assert chunks[-1] == "[SOURCES][]"
    assert memory.messages[-1]["sources"] == []


def test_build_messages_uses_strict_knowledge_qa_prompt():
    engine = RAGEngine.__new__(RAGEngine)
    results = [{"content": "孟子主张仁政。", "metadata": {"source": "doc.md"}, "score": 0.9}]

    messages = engine._build_messages("孟子的思想是什么", results, memory=None)

    assert messages[0]["role"] == "system"
    assert KNOWLEDGE_QA_SYSTEM_PROMPT in messages[0]["content"]
    assert "只基于已提供的知识库检索上下文与对话历史作答" in messages[0]["content"]
    assert KNOWLEDGE_QA_REFUSAL in messages[0]["content"]
    assert "参考资料" in messages[0]["content"]


def test_index_docs_adds_doc_and_task_metadata_to_chunks():
    class Splitter:
        def split(self, content, doc_type, metadata):
            from KBzhy.app.core.splitter import Chunk

            return [Chunk("chunk", metadata=dict(metadata))]

    class Retriever:
        def __init__(self):
            self.chunks = []

        def add_documents(self, chunks, kb_id):
            self.chunks.extend(chunks)

    class ParsedDoc:
        content = "content"
        metadata = {"file_type": "text"}

    engine = RAGEngine.__new__(RAGEngine)
    engine.splitter = Splitter()
    engine.retriever = Retriever()

    count = engine._index_docs([ParsedDoc()], "kb1", doc_id="doc1", task_id="task1")

    assert count == 1
    assert engine.retriever.chunks[0].metadata["doc_id"] == "doc1"
    assert engine.retriever.chunks[0].metadata["task_id"] == "task1"
    assert engine.retriever.chunks[0].metadata["chunk_index"] == 1


def test_prepare_document_index_uses_structured_parser_and_chunker(tmp_path):
    from KBzhy.app.core.document_models import ParsedDocument

    parsed = ParsedDocument(
        document_id="doc1", version=2, title="source", language="und",
        metadata={"kb_id": "kb1", "source": "source.md"},
    )

    class Parser:
        def parse_structured(self, path, *, document_id, version, kb_id):
            assert (path, document_id, version, kb_id) == ("source.md", "doc1", 2, "kb1")
            return parsed

        def save_artifact(self, value):
            assert value.metadata["source"] == "display.md"
            return tmp_path / "v2.json"

    class Chunker:
        def split(self, value, index_version):
            assert value.metadata["source"] == "display.md"
            assert index_version == 3
            return ["parent", "child"]

    engine = RAGEngine.__new__(RAGEngine)
    engine.parser = Parser()
    engine.structural_chunker = Chunker()

    prepared = engine.prepare_document_index(
        "source.md", "kb1", document_id="doc1", document_version=2,
        index_version=3, display_name="display.md",
    )

    assert prepared.chunks == ("parent", "child")
    assert prepared.artifact_path == str(Path(tmp_path / "v2.json"))


def test_prepare_query_skips_rewrite_for_clear_question_by_default():
    memory = FakeMemory(context=[
        {"role": "user", "content": "上一轮问题"},
        {"role": "assistant", "content": "上一轮回答"},
    ])
    engine = RAGEngine.__new__(RAGEngine)
    called = False

    def fail_if_called(question, history):
        nonlocal called
        called = True
        return "rewritten"

    engine._contextualize_query = fail_if_called

    query = engine._prepare_retrieval_query("孟子的主要思想", memory, enable_rewrite=False)

    assert query == "孟子的主要思想"
    assert called is False


def test_prepare_query_rewrites_context_dependent_question():
    memory = FakeMemory(context=[
        {"role": "user", "content": "孟子的主要思想"},
        {"role": "assistant", "content": "孟子强调仁政。"},
    ])
    engine = RAGEngine.__new__(RAGEngine)
    engine._contextualize_query = lambda question, history: "孟子仁政的具体含义"

    query = engine._prepare_retrieval_query("这个具体是什么意思", memory, enable_rewrite=False)

    assert query == "孟子仁政的具体含义"


def test_response_preview_handles_unread_streaming_response():
    class UnreadResponse:
        @property
        def text(self):
            raise httpx.ResponseNotRead()

    preview = RAGEngine._response_preview(UnreadResponse())

    assert preview == "<streaming response body not read>"


def test_chat_api_rejects_unknown_kb(monkeypatch):
    class Store:
        def knowledge_base_exists(self, kb_id):
            return False

    monkeypatch.setattr(chat, "get_metadata_store", lambda: Store())

    with pytest.raises(HTTPException) as exc:
        chat._ensure_kb_exists("missing")

    assert exc.value.status_code == 404
