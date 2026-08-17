from __future__ import annotations

import pytest

from KBzhy import config
from KBzhy.app.core.document_models import DocumentElement, ParsedDocument
from KBzhy.app.core.splitter import SmartSplitter, StructuralChunker
from KBzhy.app.core.token_counter import TokenCounter


def test_clause_split_keeps_clause_numbers():
    content = "\n".join(
        [
            "第一条 总则内容",
            "第二条 用户应遵守平台规则",
            "第三条 管理员负责审核",
            "第四条 数据应定期备份",
            "第五条 违规将被处理",
        ]
    )
    splitter = SmartSplitter(chunk_size=20, chunk_overlap=0)

    chunks = splitter.split(content, doc_type="text")
    joined = "\n".join(chunk.content for chunk in chunks)

    assert "第一条" in joined
    assert "第五条" in joined


def _parsed(*elements: DocumentElement, document_id: str = "doc-1", version: int = 2):
    return ParsedDocument(
        document_id=document_id,
        version=version,
        title="产品指南",
        language="zh-CN",
        elements=elements,
        metadata={"source": "guide.md", "kb_id": "kb-1"},
    )


def _element(
    element_id: str,
    text: str,
    order: int,
    section_path: tuple[str, ...] = (),
    *,
    element_type: str = "paragraph",
    page: int | None = None,
):
    return DocumentElement(
        element_id=element_id,
        element_type=element_type,
        text=text,
        order=order,
        page=page,
        section_path=section_path,
        metadata={"origin": element_id},
    )


def test_structural_chunks_keep_parent_boundaries_and_merge_only_inside_parent():
    parsed = _parsed(
        _element("root", "前言", 0),
        _element("a1", "第一节", 1, ("第一章",)),
        _element("a2", "第一节正文", 2, ("第一章",)),
        _element("b1", "第二节", 3, ("第二章",)),
        _element("b2", "第二节正文", 4, ("第二章",)),
    )
    chunker = StructuralChunker(child_token_limit=100, parent_token_limit=200)

    chunks = chunker.split(parsed, index_version=3)
    parents = {chunk.chunk_id: chunk for chunk in chunks if chunk.chunk_type == "parent"}
    children = [chunk for chunk in chunks if chunk.chunk_type == "child"]

    assert len(parents) == 3
    assert all(child.parent_chunk_id in parents for child in children)
    assert all(not ("第一节" in child.content and "第二节" in child.content) for child in children)
    assert [child.section_path for child in children] == [(), ("第一章",), ("第二章",)]


def test_non_contiguous_equal_section_paths_form_separate_parents_without_reordering():
    parsed = _parsed(
        _element("a1", "A 的第一段", 0, ("A",)),
        _element("b1", "B 的正文", 1, ("B",)),
        _element("a2", "A 的补充", 2, ("A",)),
    )

    chunks = StructuralChunker().split(parsed, index_version=1)
    parents = [chunk for chunk in chunks if chunk.chunk_type == "parent"]

    assert [chunk.section_path for chunk in parents] == [("A",), ("B",), ("A",)]
    assert [chunk.content for chunk in parents] == ["A 的第一段", "B 的正文", "A 的补充"]


def test_retrieval_text_adds_breadcrumb_without_changing_content():
    original = "表格与正文的原始内容"
    parsed = _parsed(_element("e1", original, 0, ("指南", "安装"), page=4))

    child = next(
        chunk
        for chunk in StructuralChunker().split(parsed, index_version=7)
        if chunk.chunk_type == "child"
    )

    assert child.content == original
    assert child.retrieval_text == f"指南 > 安装\n\n{original}"
    assert child.page_start == child.page_end == 4
    assert child.index_version == 7


def test_table_and_code_content_never_contains_internal_marker_characters():
    table = "| 字段 | 类型 |\n| --- | --- |\n| id | int |"
    code = "```python\nprint('line 1')\nprint('line 2')\n```"
    parsed = _parsed(
        _element("table", table, 0, ("格式",), element_type="table"),
        _element("code", code, 1, ("格式",), element_type="code"),
    )

    chunks = StructuralChunker(child_token_limit=200).split(parsed, index_version=1)
    content = "\n".join(chunk.content for chunk in chunks)

    assert "␟TABLE␟" not in content
    assert "␤" not in content
    assert table in content
    assert code in content


def test_chunk_ids_are_deterministic_and_isolated_by_document_version_and_document():
    parsed = _parsed(_element("e1", "稳定内容", 0, ("章节",)))
    chunker = StructuralChunker()

    first = chunker.split(parsed, index_version=1)
    repeated = chunker.split(parsed, index_version=9)
    next_version = chunker.split(_parsed(*parsed.elements, version=3), index_version=1)
    other_document = chunker.split(
        _parsed(*parsed.elements, document_id="doc-2", version=2), index_version=1
    )

    assert [chunk.chunk_id for chunk in first] == [chunk.chunk_id for chunk in repeated]
    assert {chunk.chunk_id for chunk in first}.isdisjoint(
        chunk.chunk_id for chunk in next_version
    )
    assert {chunk.chunk_id for chunk in first}.isdisjoint(
        chunk.chunk_id for chunk in other_document
    )
    assert len({chunk.chunk_id for chunk in first}) == len(first)


def test_parent_and_child_chunk_ids_use_unambiguous_type_namespaces():
    parent = next(
        chunk
        for chunk in StructuralChunker().split(
            _parsed(_element("parent", "payload", 0)),
            index_version=1,
        )
        if chunk.chunk_type == "parent"
    )
    child = next(
        chunk
        for chunk in StructuralChunker().split(
            _parsed(_element("child", "parent\0payload", 0)),
            index_version=1,
        )
        if chunk.chunk_type == "child"
    )

    assert parent.chunk_id != child.chunk_id


def test_structural_chunker_defaults_use_parent_and_child_token_config():
    chunker = StructuralChunker()

    assert chunker.child_token_limit == config.CHILD_CHUNK_TOKENS
    assert chunker.parent_token_limit == config.PARENT_CHUNK_TOKENS


def test_only_oversized_single_element_is_split_and_every_child_respects_limit():
    short_a = "短段一"
    short_b = "短段二"
    long_code = "\n".join(f"print({number})" for number in range(80))
    parsed = _parsed(
        _element("short-a", short_a, 0, ("章节",)),
        _element("short-b", short_b, 1, ("章节",)),
        _element("long", long_code, 2, ("章节",), element_type="code"),
    )
    chunker = StructuralChunker(child_token_limit=25, parent_token_limit=500)

    children = [
        chunk for chunk in chunker.split(parsed, index_version=1) if chunk.chunk_type == "child"
    ]

    assert children[0].content == f"{short_a}\n\n{short_b}"
    assert "".join(chunk.content for chunk in children[1:]) == long_code
    assert all(chunk.token_count <= chunker.child_token_limit for chunk in children)


def test_unicode_character_larger_than_child_budget_is_rejected():
    parsed = _parsed(_element("emoji", "😀", 0))

    with pytest.raises(ValueError, match="too small.*Unicode character"):
        StructuralChunker(child_token_limit=1).split(parsed, index_version=1)


def test_normal_unicode_is_split_losslessly_within_child_budget():
    content = "你好😀世界"
    chunker = StructuralChunker(child_token_limit=3)

    children = [
        chunk
        for chunk in chunker.split(
            _parsed(_element("unicode", content, 0)),
            index_version=1,
        )
        if chunk.chunk_type == "child"
    ]

    assert "".join(chunk.content for chunk in children) == content
    assert all(chunk.token_count <= chunker.child_token_limit for chunk in children)


def test_positions_pages_tokens_and_metadata_are_stable_and_correct():
    parsed = _parsed(
        _element("e2", "后出现但 order 大", 20, ("章节",), page=5),
        _element("e1", "先出现且 order 小", 10, ("章节",), page=2),
        _element("e3", "无页码", 30, ("章节",), page=None),
    )
    chunker = StructuralChunker(child_token_limit=8, parent_token_limit=200)

    chunks = chunker.split(parsed, index_version=4)
    parents = [chunk for chunk in chunks if chunk.chunk_type == "parent"]
    children = [chunk for chunk in chunks if chunk.chunk_type == "child"]

    assert [chunk.position for chunk in parents] == list(range(len(parents)))
    assert [chunk.position for chunk in children] == list(range(len(children)))
    assert parents[0].content.startswith("先出现且 order 小")
    assert parents[0].page_start == 2
    assert parents[0].page_end == 5
    assert parents[0].metadata["source"] == "guide.md"
    assert parents[0].metadata["element_ids"] == ["e1", "e2", "e3"]
    assert all(chunk.token_count == chunker.token_counter.count(chunk.content) for chunk in chunks)


def test_parent_limit_splits_at_element_boundaries():
    parsed = _parsed(
        _element("e1", "alpha " * 15, 0, ("A",)),
        _element("e2", "beta " * 15, 1, ("A",)),
    )
    chunker = StructuralChunker(child_token_limit=100, parent_token_limit=20)

    parents = [
        chunk for chunk in chunker.split(parsed, index_version=1) if chunk.chunk_type == "parent"
    ]

    assert all(parent.token_count <= chunker.parent_token_limit for parent in parents)
    assert "".join(
        parent.content for parent in parents if parent.metadata["element_ids"] == ["e1"]
    ) == "alpha " * 15
    assert "".join(
        parent.content for parent in parents if parent.metadata["element_ids"] == ["e2"]
    ) == "beta " * 15


class _LengthCounter:
    def __init__(self):
        self.calls = 0

    def count(self, text: str) -> int:
        self.calls += 1
        return len(text)


@pytest.mark.parametrize("element_type", ["code", "table", "list"])
def test_line_structures_split_only_at_complete_lines_when_each_line_fits(element_type):
    content = "".join(f"{element_type}-{number}\n" for number in range(12))
    counter = _LengthCounter()
    chunker = StructuralChunker(
        child_token_limit=24,
        parent_token_limit=10_000,
        token_counter=counter,
    )

    children = [
        chunk
        for chunk in chunker.split(
            _parsed(_element("structured", content, 0, element_type=element_type)),
            index_version=1,
        )
        if chunk.chunk_type == "child"
    ]

    assert "".join(chunk.content for chunk in children) == content
    assert all(chunk.content.endswith("\n") for chunk in children)
    assert all(chunk.token_count <= chunker.child_token_limit for chunk in children)


def test_oversized_table_exposes_header_in_chunk_metadata_without_content_markers():
    content = "| name | type |\n" + "".join(
        f"| field-{number} | int |\n" for number in range(8)
    )
    chunker = StructuralChunker(
        child_token_limit=35,
        parent_token_limit=10_000,
        token_counter=_LengthCounter(),
    )

    children = [
        chunk
        for chunk in chunker.split(
            _parsed(_element("table", content, 0, element_type="table")),
            index_version=1,
        )
        if chunk.chunk_type == "child"
    ]

    assert len(children) > 1
    assert all(chunk.metadata["table_header"] == "| name | type |" for chunk in children)
    assert all("␟TABLE␟" not in chunk.content for chunk in children)
    assert "".join(chunk.content for chunk in children) == content


def test_paragraph_prefers_sentence_boundaries_before_hard_splitting():
    content = "第一句完整。第二句完整！第三句完整？"
    chunker = StructuralChunker(
        child_token_limit=7,
        parent_token_limit=100,
        token_counter=_LengthCounter(),
    )

    children = [
        chunk
        for chunk in chunker.split(
            _parsed(_element("paragraph", content, 0)),
            index_version=1,
        )
        if chunk.chunk_type == "child"
    ]

    assert [chunk.content for chunk in children] == ["第一句完整。", "第二句完整！", "第三句完整？"]


def test_parent_limit_is_strict_when_smaller_than_child_limit_and_links_stay_local():
    content = "".join(f"row-{number}\n" for number in range(10))
    chunker = StructuralChunker(
        child_token_limit=100,
        parent_token_limit=18,
        token_counter=_LengthCounter(),
    )

    chunks = chunker.split(
        _parsed(_element("code", content, 0, ("section",), element_type="code")),
        index_version=1,
    )
    parents = [chunk for chunk in chunks if chunk.chunk_type == "parent"]
    children = [chunk for chunk in chunks if chunk.chunk_type == "child"]

    assert "".join(parent.content for parent in parents) == content
    assert all(parent.token_count <= chunker.parent_token_limit for parent in parents)
    assert {child.parent_chunk_id for child in children} == {parent.chunk_id for parent in parents}
    assert {
        parent.chunk_id: "".join(
            child.content for child in children if child.parent_chunk_id == parent.chunk_id
        )
        for parent in parents
    } == {parent.chunk_id: parent.content for parent in parents}


def test_hard_split_preserves_combining_and_zwj_sequences_losslessly():
    graphemes = ["A\u0301", "👩\u200d💻", "B\ufe0f", "👨\u200d👩\u200d👧"]
    content = "".join(graphemes * 5)
    chunker = StructuralChunker(
        child_token_limit=9,
        parent_token_limit=1_000,
        token_counter=_LengthCounter(),
    )

    children = [
        chunk
        for chunk in chunker.split(_parsed(_element("unicode", content, 0)), index_version=1)
        if chunk.chunk_type == "child"
    ]

    assert "".join(chunk.content for chunk in children) == content
    assert all(not chunk.content.startswith(("\u0301", "\u200d", "\ufe0f")) for chunk in children)
    assert all(not chunk.content.endswith("\u200d") for chunk in children)
    assert all(chunk.token_count <= chunker.child_token_limit for chunk in children)


@pytest.mark.parametrize("limit", [2, 3, 4])
@pytest.mark.parametrize(
    ("cluster", "forbidden_start", "forbidden_end"),
    [
        ("👍🏽", ("🏽",), ("👍",)),
        ("🇨🇳", ("🇳",), ("🇨",)),
    ],
)
def test_hard_split_keeps_emoji_modifiers_and_flag_pairs_atomic(
    limit,
    cluster,
    forbidden_start,
    forbidden_end,
):
    content = cluster * 2
    chunker = StructuralChunker(
        child_token_limit=limit,
        parent_token_limit=100,
        token_counter=_LengthCounter(),
    )

    children = [
        chunk
        for chunk in chunker.split(_parsed(_element("emoji", content, 0)), index_version=1)
        if chunk.chunk_type == "child"
    ]

    assert "".join(chunk.content for chunk in children) == content
    assert all(not chunk.content.startswith(forbidden_start) for chunk in children)
    assert all(not chunk.content.endswith(forbidden_end) for chunk in children)
    assert all(chunk.token_count <= limit for chunk in children)


class _RecordingEncoding:
    def __init__(self, encoding):
        self._encoding = encoding
        self.max_input_chars = 0
        self.calls = 0

    def encode(self, text, **kwargs):
        self.calls += 1
        self.max_input_chars = max(self.max_input_chars, len(text))
        return self._encoding.encode(text, **kwargs)

    def __getattr__(self, name):
        return getattr(self._encoding, name)


def test_real_token_counter_never_encodes_the_complete_large_element():
    content = "流式分块 memory bound " * 3_000
    counter = TokenCounter()
    recording_encoding = _RecordingEncoding(counter.encoding)
    counter.encoding = recording_encoding
    chunker = StructuralChunker(
        child_token_limit=64,
        parent_token_limit=256,
        token_counter=counter,
    )

    chunks = chunker.split(_parsed(_element("large-real", content, 0)), index_version=1)
    parents = [chunk for chunk in chunks if chunk.chunk_type == "parent"]

    assert "".join(chunk.content for chunk in parents) == content
    assert all(chunk.token_count <= chunker.parent_token_limit for chunk in parents)
    assert recording_encoding.max_input_chars <= 4_096


def test_default_limits_process_highly_compressed_text_in_bounded_encoding_windows():
    content = ("x" + " " * 4_095) * 5
    counter = TokenCounter()
    recording_encoding = _RecordingEncoding(counter.encoding)
    counter.encoding = recording_encoding
    chunker = StructuralChunker(token_counter=counter)

    chunks = chunker.split(_parsed(_element("compressed", content, 0)), index_version=1)
    parents = [chunk for chunk in chunks if chunk.chunk_type == "parent"]

    assert "".join(chunk.content for chunk in parents) == content
    assert all(chunk.token_count <= 2_000 for chunk in parents)
    assert recording_encoding.max_input_chars <= 4_096
    assert recording_encoding.calls <= 100


def test_oversized_grapheme_is_rejected_before_any_encoding_call():
    content = "a" + "\u0301" * 4_096
    counter = TokenCounter()
    recording_encoding = _RecordingEncoding(counter.encoding)
    counter.encoding = recording_encoding
    chunker = StructuralChunker(token_counter=counter)

    with pytest.raises(ValueError, match="grapheme"):
        chunker.split(_parsed(_element("oversized-grapheme", content, 0)), index_version=1)

    assert recording_encoding.calls == 0


def test_large_single_element_uses_linear_number_of_count_calls():
    content = "x" * 20_000
    counter = _LengthCounter()
    chunker = StructuralChunker(
        child_token_limit=5,
        parent_token_limit=25_000,
        token_counter=counter,
    )

    children = [
        chunk
        for chunk in chunker.split(_parsed(_element("large", content, 0)), index_version=1)
        if chunk.chunk_type == "child"
    ]

    assert "".join(chunk.content for chunk in children) == content
    assert counter.calls <= len(content) + len(children) * 12


class _NonMonotonicCounter:
    def count(self, text: str) -> int:
        if text.endswith("a") and len(text) % 4 == 1:
            return 99
        return len(text)


def test_hard_split_does_not_binary_search_non_monotonic_character_prefix_counts():
    content = "baaaabaaaa"
    counter = _NonMonotonicCounter()
    chunker = StructuralChunker(
        child_token_limit=4,
        parent_token_limit=100,
        token_counter=counter,
    )

    children = [
        chunk
        for chunk in chunker.split(_parsed(_element("fake", content, 0)), index_version=1)
        if chunk.chunk_type == "child"
    ]

    assert "".join(chunk.content for chunk in children) == content
    assert all(counter.count(chunk.content) <= chunker.child_token_limit for chunk in children)


def test_empty_document_returns_no_chunks():
    assert StructuralChunker().split(_parsed(), index_version=1) == []


@pytest.mark.parametrize("index_version", [0, -1, True, 1.5, "1"])
def test_invalid_index_version_is_rejected(index_version):
    with pytest.raises(ValueError, match="index_version"):
        StructuralChunker().split(_parsed(), index_version=index_version)
