from __future__ import annotations

import httpx

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


def test_build_messages_uses_strict_knowledge_qa_prompt():
    engine = RAGEngine.__new__(RAGEngine)
    results = [{"content": "孟子主张仁政。", "metadata": {"source": "doc.md"}, "score": 0.9}]

    messages = engine._build_messages("孟子的思想是什么", results, memory=None)

    assert messages[0]["role"] == "system"
    assert KNOWLEDGE_QA_SYSTEM_PROMPT in messages[0]["content"]
    assert "只基于已提供的知识库检索上下文与对话历史作答" in messages[0]["content"]
    assert KNOWLEDGE_QA_REFUSAL in messages[0]["content"]
    assert "参考资料" in messages[0]["content"]


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
