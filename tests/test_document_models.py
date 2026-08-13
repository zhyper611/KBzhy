from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import get_type_hints

import pytest

from KBzhy.app.core.document_models import (
    DocumentElement,
    KnowledgeChunk,
    ParsedDocument,
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


@pytest.mark.parametrize(
    ("document_id", "version", "position", "content"),
    [
        ("document-2", 1, 3, "chunk content"),
        ("document-1", 2, 3, "chunk content"),
        ("document-1", 1, 4, "chunk content"),
        ("document-1", 1, 3, "changed content"),
    ],
)
def test_stable_chunk_id_changes_when_any_identity_input_changes(
    document_id: str,
    version: int,
    position: int,
    content: str,
):
    baseline = stable_chunk_id("document-1", 1, 3, "chunk content")

    assert stable_chunk_id(document_id, version, position, content) != baseline


def test_document_element_is_frozen():
    element = DocumentElement(
        element_id="element-1",
        element_type="paragraph",
        text="正文",
        order=0,
    )

    with pytest.raises(FrozenInstanceError):
        element.text = "修改后的正文"


def test_document_element_accepts_bounding_box_mapping():
    bounding_box = {"x": 10.0, "y": 20.0, "width": 300.0, "height": 40.0}

    element = DocumentElement(
        element_id="element-1",
        element_type="paragraph",
        text="正文",
        order=0,
        bounding_box=bounding_box,
    )

    assert element.bounding_box == bounding_box
    assert get_type_hints(DocumentElement)["bounding_box"] == dict[str, float] | None


def test_document_element_defensively_copies_mutable_inputs_and_is_unhashable():
    section_path = ["第一章"]
    bounding_box = {"x": 10.0}
    metadata = {"labels": ["initial"]}

    element = DocumentElement(
        element_id="element-1",
        element_type="paragraph",
        text="正文",
        order=0,
        section_path=section_path,
        bounding_box=bounding_box,
        metadata=metadata,
    )
    section_path.append("外部修改")
    bounding_box["x"] = 99.0
    metadata["labels"].append("external")

    assert element.section_path == ("第一章",)
    assert element.bounding_box == {"x": 10.0}
    assert element.metadata == {"labels": ["initial"]}
    with pytest.raises(TypeError):
        hash(element)


def test_parsed_document_defensively_copies_mutable_inputs_and_is_unhashable():
    element = DocumentElement("element-1", "paragraph", "正文", 0)
    elements = [element]
    metadata = {"source": {"name": "original"}}

    document = ParsedDocument(
        document_id="document-1",
        version=1,
        title="标题",
        language="zh-CN",
        elements=elements,
        metadata=metadata,
    )
    elements.clear()
    metadata["source"]["name"] = "changed"

    assert document.elements == (element,)
    assert document.metadata == {"source": {"name": "original"}}
    with pytest.raises(TypeError):
        hash(document)


def test_knowledge_chunk_child_builds_retrieval_text_and_hashes():
    chunk = KnowledgeChunk.child(
        document_id="document-1",
        document_version=2,
        content="向量检索正文",
        section_path=("第一章", "检索"),
        position=4,
        token_count=8,
        parent_chunk_id="parent-1",
    )

    assert chunk.chunk_type == "child"
    assert chunk.retrieval_text == "第一章 > 检索\n\n向量检索正文"
    assert chunk.content_hash == content_hash("向量检索正文")
    assert chunk.chunk_id == stable_chunk_id("document-1", 2, 4, "向量检索正文")
    assert len(chunk.chunk_id) == 64
    assert len(chunk.content_hash) == 64
    assert chunk.index_version == 1
    assert get_type_hints(KnowledgeChunk)["index_version"] is int


def test_knowledge_chunk_child_without_section_uses_content_for_retrieval():
    chunk = KnowledgeChunk.child(
        document_id="document-1",
        document_version=1,
        content="无标题正文",
        position=0,
        token_count=4,
        parent_chunk_id="parent-1",
    )

    assert chunk.retrieval_text == chunk.content


def test_knowledge_chunk_child_normalizes_list_section_path_to_tuple():
    chunk = KnowledgeChunk.child(
        document_id="document-1",
        document_version=1,
        content="列表路径正文",
        position=1,
        token_count=5,
        section_path=["第二章", "生成"],
        parent_chunk_id="parent-1",
    )

    assert chunk.section_path == ("第二章", "生成")
    assert chunk.retrieval_text == "第二章 > 生成\n\n列表路径正文"


def test_knowledge_chunk_child_requires_parent_chunk_id():
    with pytest.raises(TypeError):
        KnowledgeChunk.child(
            document_id="document-1",
            document_version=1,
            content="正文",
            position=0,
            token_count=2,
        )


def test_knowledge_chunk_child_rejects_empty_parent_chunk_id():
    with pytest.raises(ValueError, match="parent_chunk_id"):
        KnowledgeChunk.child(
            document_id="document-1",
            document_version=1,
            content="正文",
            position=0,
            token_count=2,
            parent_chunk_id="",
        )


def test_knowledge_chunk_defensively_copies_mutable_inputs_and_is_unhashable():
    section_path = ["第一章"]
    metadata = {"source": {"page": 1}}

    chunk = KnowledgeChunk.child(
        document_id="document-1",
        document_version=1,
        content="正文",
        position=0,
        token_count=2,
        parent_chunk_id="parent-1",
        section_path=section_path,
        metadata=metadata,
    )
    section_path.append("外部修改")
    metadata["source"]["page"] = 2

    assert chunk.section_path == ("第一章",)
    assert chunk.metadata == {"source": {"page": 1}}
    with pytest.raises(TypeError):
        hash(chunk)


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
    text = "正文包含 <|endoftext|> 字样"

    assert counter.count(text) == len(counter.encoding.encode_ordinary(text))


def test_token_counter_truncate_returns_a_valid_text_prefix():
    counter = TokenCounter()
    text = "知识库检索需要稳定而精确"

    truncated = counter.truncate(text, max_tokens=2)

    assert text.startswith(truncated)


@pytest.mark.parametrize("text", ["😀表情符号截断", "𠮷野家罕见汉字截断"])
def test_token_counter_truncate_unicode_returns_a_valid_prefix(text: str):
    counter = TokenCounter()

    truncated = counter.truncate(text, max_tokens=2)

    assert text.startswith(truncated)
    assert counter.count(truncated) <= 2


def test_token_counter_rejects_unknown_encoding():
    with pytest.raises(ValueError):
        TokenCounter("encoding-that-does-not-exist")
