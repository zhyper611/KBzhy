from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from KBzhy.app.core.document_models import (
    DocumentElement,
    KnowledgeChunk,
    content_hash,
    stable_chunk_id,
)
from KBzhy.app.core.token_counter import TokenCounter


def test_content_hash_treats_text_as_utf8_bytes():
    content = "RAG 知识库"

    assert content_hash(content) == content_hash(content.encode("utf-8"))
    assert len(content_hash(content)) == 64


def test_stable_chunk_id_is_repeatable_and_version_sensitive():
    first = stable_chunk_id("document-1", 1, 3, "chunk content")
    repeated = stable_chunk_id("document-1", 1, 3, "chunk content")
    next_version = stable_chunk_id("document-1", 2, 3, "chunk content")

    assert first == repeated
    assert first != next_version
    assert len(first) == 64


def test_document_element_is_frozen():
    element = DocumentElement(
        element_id="element-1",
        element_type="paragraph",
        text="正文",
        order=0,
    )

    with pytest.raises(FrozenInstanceError):
        element.text = "修改后的正文"


def test_knowledge_chunk_child_builds_retrieval_text_and_hashes():
    chunk = KnowledgeChunk.child(
        document_id="document-1",
        document_version=2,
        content="向量检索正文",
        section_path=("第一章", "检索"),
        position=4,
        token_count=8,
        index_version="v2",
    )

    assert chunk.chunk_type == "child"
    assert chunk.retrieval_text == "第一章 > 检索\n\n向量检索正文"
    assert chunk.content_hash == content_hash("向量检索正文")
    assert chunk.chunk_id == stable_chunk_id("document-1", 2, 4, "向量检索正文")
    assert len(chunk.chunk_id) == 64
    assert len(chunk.content_hash) == 64


def test_knowledge_chunk_child_without_section_uses_content_for_retrieval():
    chunk = KnowledgeChunk.child(
        document_id="document-1",
        document_version=1,
        content="无标题正文",
        position=0,
        token_count=4,
        index_version="v2",
    )

    assert chunk.retrieval_text == chunk.content


def test_token_counter_truncate_never_exceeds_budget():
    counter = TokenCounter()
    text = "知识库检索需要稳定而精确的 token 预算。" * 20

    truncated = counter.truncate(text, max_tokens=17)

    assert counter.count(truncated) <= 17
    assert truncated


def test_token_counter_truncate_handles_zero_and_negative_budgets():
    counter = TokenCounter()

    assert counter.truncate("任意文本", max_tokens=0) == ""
    with pytest.raises(ValueError):
        counter.truncate("任意文本", max_tokens=-1)


def test_token_counter_counts_special_token_text_as_document_content():
    counter = TokenCounter()

    assert counter.count("正文包含 <|endoftext|> 字样") > 0


def test_token_counter_truncate_returns_a_valid_text_prefix():
    counter = TokenCounter()
    text = "知识库检索需要稳定而精确"

    truncated = counter.truncate(text, max_tokens=2)

    assert text.startswith(truncated)
