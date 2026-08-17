"""文本切分器 — 按文档类型选用不同策略，保留结构化内容"""

from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

from KBzhy.app.core.document_models import (
    DocumentElement,
    KnowledgeChunk,
    ParsedDocument,
    content_hash,
    stable_chunk_id,
)
from KBzhy.app.core.token_counter import TokenCounter
from KBzhy.config import (
    CHILD_CHUNK_TOKENS,
    CHINESE_SEPARATORS,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    PARENT_CHUNK_TOKENS,
)

logger = logging.getLogger(__name__)


class Chunk:
    """切分后的文本块"""

    def __init__(self, content: str, metadata: dict[str, Any] | None = None):
        self.content = content
        self.metadata = metadata or {}

    def __repr__(self) -> str:
        return f"Chunk(len={len(self.content)})"


@dataclass(frozen=True)
class _ElementFragment:
    element: DocumentElement
    text: str


@dataclass(frozen=True)
class _ElementGroup:
    fragments: tuple[_ElementFragment, ...]
    section_path: tuple[str, ...]


class StructuralChunker:
    _MAX_ENCODING_WINDOW_CHARS = 4_096

    """按章节和结构化 Element 生成 Parent-Child Chunk。"""

    def __init__(
        self,
        child_token_limit: int = CHILD_CHUNK_TOKENS,
        parent_token_limit: int = PARENT_CHUNK_TOKENS,
        token_counter: TokenCounter | None = None,
    ):
        self._validate_limit(child_token_limit, "child_token_limit")
        self._validate_limit(parent_token_limit, "parent_token_limit")
        self.child_token_limit = child_token_limit
        self.parent_token_limit = parent_token_limit
        self.token_counter = token_counter or TokenCounter()

    def split(
        self,
        parsed: ParsedDocument,
        index_version: int,
    ) -> list[KnowledgeChunk]:
        if isinstance(index_version, bool) or not isinstance(index_version, int) or index_version <= 0:
            raise ValueError("index_version must be a positive integer")

        elements = [
            element
            for _, element in sorted(
                enumerate(parsed.elements),
                key=lambda item: (item[1].order, item[0]),
            )
            if element.text.strip()
        ]
        if not elements:
            return []

        chunks: list[KnowledgeChunk] = []
        parent_position = 0
        child_position = 0
        for group in self._parent_groups(elements):
            parent_content = self._join_fragments(group.fragments)
            parent_id = stable_chunk_id(
                parsed.document_id,
                parsed.version,
                parent_position,
                f"parent\0{parent_content}",
            )
            page_start, page_end = self._page_range(group.fragments)
            parent_metadata = self._metadata(parsed, group.fragments)
            breadcrumb = " > ".join(group.section_path)
            parent_retrieval_text = (
                f"{breadcrumb}\n\n{parent_content}" if breadcrumb else parent_content
            )
            chunks.append(
                KnowledgeChunk(
                    chunk_id=parent_id,
                    document_id=parsed.document_id,
                    document_version=parsed.version,
                    parent_chunk_id=None,
                    chunk_type="parent",
                    content=parent_content,
                    retrieval_text=parent_retrieval_text,
                    content_hash=content_hash(parent_content),
                    section_path=group.section_path,
                    page_start=page_start,
                    page_end=page_end,
                    position=parent_position,
                    token_count=self.token_counter.count(parent_content),
                    index_version=index_version,
                    metadata=parent_metadata,
                )
            )

            for child_content, child_fragments in self._child_groups(group.fragments):
                child_page_start, child_page_end = self._page_range(child_fragments)
                child_content_hash = content_hash(child_content)
                breadcrumb = " > ".join(group.section_path)
                child_retrieval_text = (
                    f"{breadcrumb}\n\n{child_content}" if breadcrumb else child_content
                )
                chunks.append(
                    KnowledgeChunk(
                        chunk_id=stable_chunk_id(
                            parsed.document_id,
                            parsed.version,
                            child_position,
                            f"child\0{child_content}",
                        ),
                        document_id=parsed.document_id,
                        document_version=parsed.version,
                        parent_chunk_id=parent_id,
                        chunk_type="child",
                        content=child_content,
                        retrieval_text=child_retrieval_text,
                        content_hash=child_content_hash,
                        section_path=group.section_path,
                        page_start=child_page_start,
                        page_end=child_page_end,
                        position=child_position,
                        token_count=self.token_counter.count(child_content),
                        index_version=index_version,
                        metadata=self._metadata(parsed, child_fragments),
                    )
                )
                child_position += 1
            parent_position += 1
        return chunks

    def _parent_groups(self, elements: list[DocumentElement]) -> list[_ElementGroup]:
        groups: list[_ElementGroup] = []
        current: list[_ElementFragment] = []
        section_path: tuple[str, ...] | None = None

        def close_current() -> None:
            if current:
                groups.append(_ElementGroup(tuple(current), section_path or ()))
                current.clear()

        for element in elements:
            if section_path is not None and element.section_path != section_path:
                close_current()
            section_path = element.section_path
            for text in self._split_element_text(
                element.text,
                element.element_type,
                self.parent_token_limit,
            ):
                fragment = _ElementFragment(element, text)
                candidate = (*current, fragment)
                if (
                    current
                    and self._exceeds_limit(
                        self._join_fragments(candidate),
                        self.parent_token_limit,
                    )
                ):
                    close_current()
                current.append(fragment)
        close_current()
        return groups

    def _child_groups(
        self,
        fragments: tuple[_ElementFragment, ...],
    ) -> list[tuple[str, tuple[_ElementFragment, ...]]]:
        groups: list[tuple[str, tuple[_ElementFragment, ...]]] = []
        current: list[_ElementFragment] = []

        def close_current() -> None:
            if current:
                groups.append((self._join_fragments(current), tuple(current)))
                current.clear()

        for source in fragments:
            for text in self._split_element_text(
                source.text,
                source.element.element_type,
                self.child_token_limit,
            ):
                fragment = _ElementFragment(source.element, text)
                candidate = (*current, fragment)
                if (
                    current
                    and self._exceeds_limit(
                        self._join_fragments(candidate),
                        self.child_token_limit,
                    )
                ):
                    close_current()
                current.append(fragment)
        close_current()
        return groups

    def _split_element_text(
        self,
        text: str,
        element_type: str,
        limit: int,
    ) -> list[str]:
        units = self._semantic_units(text, element_type)
        atomic_pieces = [
            piece
            for unit in units
            for piece in self._fit_atomic_unit(unit, limit)
        ]
        pieces: list[str] = []
        current = ""
        for piece in atomic_pieces:
            candidate = current + piece
            if current and self._exceeds_limit(candidate, limit):
                pieces.append(current)
                current = piece
            else:
                current = candidate
        if current:
            pieces.append(current)
        return pieces

    @staticmethod
    def _semantic_units(text: str, element_type: str) -> list[str]:
        if element_type in {"code", "table", "list"}:
            return text.splitlines(keepends=True) or [text]
        units = re.split(r"(?<=[。！？!?；;])|(?<=\n)", text)
        return [unit for unit in units if unit]

    def _fit_atomic_unit(self, text: str, limit: int) -> list[str]:
        return self._split_by_graphemes(text, limit)

    def _split_by_graphemes(self, text: str, limit: int) -> list[str]:
        if getattr(self.token_counter, "encoding", None) is not None:
            return self._split_with_encoding_windows(text, limit)

        source = iter(self._iter_graphemes(text))
        buffer: list[str] = []
        pieces: list[str] = []
        exhausted = False
        max_window_chars = self._MAX_ENCODING_WINDOW_CHARS

        while buffer or not exhausted:
            probe = 0
            candidate = ""
            last_fitting: int | None = None
            consecutive_failures = 0

            while consecutive_failures < 4:
                if probe == len(buffer):
                    try:
                        buffer.append(next(source))
                    except StopIteration:
                        exhausted = True
                        break
                cluster = buffer[probe]
                if len(cluster) > max_window_chars:
                    raise ValueError(
                        "a single Unicode grapheme exceeds the encoding window"
                    )
                if (
                    candidate
                    and last_fitting is not None
                    and len(candidate) + len(cluster) > max_window_chars
                ):
                    break
                candidate += cluster
                probe += 1
                if self.token_counter.count(candidate) <= limit:
                    last_fitting = probe
                    consecutive_failures = 0
                else:
                    consecutive_failures += 1

            if last_fitting is None:
                raise ValueError(
                    "token limit is too small for a single Unicode character or grapheme"
                )
            pieces.append("".join(buffer[:last_fitting]))
            del buffer[:last_fitting]
        return pieces

    def _split_with_encoding_windows(self, text: str, limit: int) -> list[str]:
        source = iter(self._iter_graphemes(text))
        pending: str | None = None
        pieces: list[str] = []
        exhausted = False

        while pending is not None or not exhausted:
            clusters: list[str] = []
            window_chars = 0

            if pending is not None:
                cluster = pending
                pending = None
            else:
                try:
                    cluster = next(source)
                except StopIteration:
                    break

            while True:
                if len(cluster) > self._MAX_ENCODING_WINDOW_CHARS:
                    raise ValueError(
                        "a single Unicode grapheme exceeds the encoding window"
                    )
                if (
                    clusters
                    and window_chars + len(cluster) > self._MAX_ENCODING_WINDOW_CHARS
                ):
                    pending = cluster
                    break
                clusters.append(cluster)
                window_chars += len(cluster)
                try:
                    cluster = next(source)
                except StopIteration:
                    exhausted = True
                    break

            pieces.extend(self._split_encoding_window(clusters, limit))

        return pieces

    def _split_encoding_window(self, clusters: list[str], limit: int) -> list[str]:
        encoding = self.token_counter.encoding
        pieces: list[str] = []
        start = 0

        while start < len(clusters):
            candidate = "".join(clusters[start:])
            tokens = encoding.encode(candidate, disallowed_special=())
            if len(tokens) <= limit:
                pieces.append(candidate)
                break

            prefix_bytes = b"".join(
                encoding.decode_single_token_bytes(token)
                for token in tokens[:limit]
            )
            prefix_length = len(prefix_bytes.decode("utf-8", errors="ignore"))
            end = start
            consumed_chars = 0
            while end < len(clusters):
                next_length = consumed_chars + len(clusters[end])
                if next_length > prefix_length:
                    break
                consumed_chars = next_length
                end += 1

            if end == start:
                first = clusters[start]
                if self.token_counter.count(first) > limit:
                    raise ValueError(
                        "token limit is too small for a single Unicode character or grapheme"
                    )
                end += 1

            piece = "".join(clusters[start:end])
            while end > start and self.token_counter.count(piece) > limit:
                end -= 1
                piece = "".join(clusters[start:end])
            if end == start:
                raise ValueError(
                    "token limit is too small for a single Unicode character or grapheme"
                )
            pieces.append(piece)
            start = end

        return pieces

    def _exceeds_limit(self, text: str, limit: int) -> bool:
        if (
            getattr(self.token_counter, "encoding", None) is not None
            and len(text) > self._MAX_ENCODING_WINDOW_CHARS
        ):
            return True
        return self.token_counter.count(text) > limit

    @staticmethod
    def _graphemes(text: str) -> list[str]:
        return list(StructuralChunker._iter_graphemes(text))

    @staticmethod
    def _iter_graphemes(text: str):
        if not text:
            return

        start = 0
        previous = text[0]
        regional_count = 1 if StructuralChunker._is_regional_indicator(previous) else 0
        for index in range(1, len(text)):
            character = text[index]
            is_regional = StructuralChunker._is_regional_indicator(character)
            joins_previous = (
                StructuralChunker._is_grapheme_extension(character)
                or character == "\u200d"
                or previous == "\u200d"
                or (is_regional and regional_count % 2 == 1)
            )
            if not joins_previous:
                yield text[start:index]
                start = index
                regional_count = 0

            if is_regional:
                regional_count += 1
            elif not StructuralChunker._is_grapheme_extension(character):
                regional_count = 0
            previous = character
        yield text[start:]

    @staticmethod
    def _is_grapheme_extension(character: str) -> bool:
        return (
            unicodedata.combining(character) != 0
            or unicodedata.category(character) in {"Mc", "Me", "Mn"}
            or "\ufe00" <= character <= "\ufe0f"
            or "\U000e0100" <= character <= "\U000e01ef"
            or "\U0001f3fb" <= character <= "\U0001f3ff"
            or "\U000e0020" <= character <= "\U000e007f"
        )

    @staticmethod
    def _is_regional_indicator(character: str) -> bool:
        return "\U0001f1e6" <= character <= "\U0001f1ff"

    @staticmethod
    def _join_fragments(
        fragments: list[_ElementFragment] | tuple[_ElementFragment, ...],
    ) -> str:
        content = ""
        previous: DocumentElement | None = None
        for fragment in fragments:
            if content and previous is not fragment.element:
                content += "\n\n"
            content += fragment.text
            previous = fragment.element
        return content

    @staticmethod
    def _page_range(
        fragments: tuple[_ElementFragment, ...],
    ) -> tuple[int | None, int | None]:
        pages = [
            fragment.element.page
            for fragment in fragments
            if fragment.element.page is not None
        ]
        return (min(pages), max(pages)) if pages else (None, None)

    @staticmethod
    def _metadata(
        parsed: ParsedDocument,
        fragments: tuple[_ElementFragment, ...],
    ) -> dict[str, Any]:
        elements: list[DocumentElement] = []
        for fragment in fragments:
            if not elements or elements[-1] is not fragment.element:
                elements.append(fragment.element)
        metadata = dict(parsed.metadata)
        metadata.update(
            {
                "element_ids": [element.element_id for element in elements],
                "element_types": [element.element_type for element in elements],
                "element_metadata": [dict(element.metadata) for element in elements],
            }
        )
        table_elements = [element for element in elements if element.element_type == "table"]
        if len(table_elements) == 1:
            lines = table_elements[0].text.splitlines()
            if lines:
                metadata["table_header"] = lines[0]
        return metadata

    @staticmethod
    def _validate_limit(value: int, name: str) -> None:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")


class SmartSplitter:
    """智能分块：按文档类型选策略，保留结构化内容"""

    # FAQ 特征：短行 + 问号结尾 + 答案紧跟
    _QA_PATTERN = re.compile(
        r"(?:Q[：:.\d]*|问[：:]|问题[：:\d]*|FAQ).*?[\s\S]*?"
        r"(?=(?:Q[：:.\d]*|问[：:]|问题[：:\d]*|FAQ)|$)",
        re.IGNORECASE,
    )

    # 条款特征：第X条/X./X.)/1.1 等
    _CLAUSE_PATTERN = re.compile(
        r"(?:第[一二三四五六七八九十百千\d]+[条章节款]|[\(（]?\d+[\)）\.\、])[^\n]*",
    )
    _CLAUSE_START_PATTERN = re.compile(
        r"^\s*(?:第[一二三四五六七八九十百千\d]+[条章节款]|[\(（]?\d+[\)）\.\、])"
    )

    # 列表特征
    _LIST_PATTERN = re.compile(r"(?:^|\n)(?:\d+[\.\)、]|[-*+•])[ \t]+[^\n]+")

    def __init__(
        self,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
        separators: list[str] | None = None,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or CHINESE_SEPARATORS

    def split(self, content: str, doc_type: str = "text", metadata: dict[str, Any] | None = None) -> list[Chunk]:
        """按文档类型选择策略切分"""
        meta = metadata or {}
        meta["doc_type"] = doc_type

        if doc_type == "excel":
            chunks = self._split_by_rows(content, meta)
        elif doc_type in ("word", "pdf"):
            chunks = self._split_by_paragraph(content, meta)
        else:
            strategy = self._pick_strategy(content)
            if strategy == "qa":
                chunks = self._split_by_qa(content, meta)
            elif strategy == "clause":
                chunks = self._split_by_clause(content, meta)
            else:
                chunks = self._split_recursive(content, meta)

        # 后处理：合并过短的块
        chunks = self._merge_short(chunks)
        logger.info(
            "切分完成: 文档类型=%s, 块数=%d, 平均长度=%d",
            doc_type,
            len(chunks),
            sum(len(c.content) for c in chunks) // max(len(chunks), 1),
        )
        return chunks

    def _pick_strategy(self, content: str) -> str:
        """自动识别内容类型"""
        lines = content.strip().split("\n")
        qa_lines = sum(1 for l in lines if l.strip().endswith("?") or l.strip().endswith("？") or l.startswith("Q") or l.startswith("问"))
        if qa_lines >= 3:
            return "qa"
        if self._CLAUSE_PATTERN.findall(content):
            if len(self._CLAUSE_PATTERN.findall(content)) >= 5:
                return "clause"
        return "default"

    # ── 按段落切分（Word/PDF）───────────────────

    def _split_by_paragraph(self, content: str, meta: dict) -> list[Chunk]:
        """按换行分段落，段落内再递归切分"""
        paragraphs = content.split("\n\n")
        chunks: list[Chunk] = []
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            if len(para) <= self.chunk_size:
                chunks.append(Chunk(content=para, metadata=dict(meta)))
            else:
                sub = self._split_recursive(para, meta)
                chunks.extend(sub)
        return chunks

    # ── 按行切分（Excel）────────────────────────

    def _split_by_rows(self, content: str, meta: dict) -> list[Chunk]:
        """Excel 按表头+数据行分组"""
        lines = [l.strip() for l in content.split("\n") if l.strip()]
        if not lines:
            return []

        chunks: list[Chunk] = []
        header = lines[0]
        batch = [header]
        current_size = len(header)
        for line in lines[1:]:
            if current_size + len(line) + 1 > self.chunk_size:
                chunks.append(Chunk(content="\n".join(batch), metadata=dict(meta)))
                batch = [header]
                current_size = len(header)
            batch.append(line)
            current_size += len(line) + 1
        if len(batch) > 1:
            chunks.append(Chunk(content="\n".join(batch), metadata=dict(meta)))
        return chunks

    # ── FAQ 切分 ────────────────────────────────

    def _split_by_qa(self, content: str, meta: dict) -> list[Chunk]:
        """按 Q&A 对切分，保持问答完整性"""
        pairs = self._QA_PATTERN.findall(content)
        if not pairs:
            return self._split_recursive(content, meta)

        chunks: list[Chunk] = []
        current: list[str] = []
        current_size = 0
        for pair in pairs:
            pair = pair.strip()
            if current_size + len(pair) > self.chunk_size and current:
                chunks.append(Chunk(content="\n".join(current), metadata=dict(meta)))
                current = []
                current_size = 0
            current.append(pair)
            current_size += len(pair)
        if current:
            chunks.append(Chunk(content="\n".join(current), metadata=dict(meta)))
        return chunks

    # ── 条款切分 ────────────────────────────────

    def _split_by_clause(self, content: str, meta: dict) -> list[Chunk]:
        """按条款号切分"""
        clauses: list[str] = []
        current: list[str] = []
        for line in content.splitlines():
            if self._CLAUSE_START_PATTERN.match(line) and current:
                clauses.append("\n".join(current).strip())
                current = [line]
            else:
                current.append(line)
        if current:
            clauses.append("\n".join(current).strip())
        clauses = [c for c in clauses if c]
        if not clauses:
            return self._split_recursive(content, meta)

        chunks: list[Chunk] = []
        current = []
        current_size = 0
        for clause in clauses:
            if current_size + len(clause) > self.chunk_size and current:
                chunks.append(Chunk(content="\n".join(current), metadata=dict(meta)))
                current = []
                current_size = 0
            current.append(clause)
            current_size += len(clause)
        if current:
            chunks.append(Chunk(content="\n".join(current), metadata=dict(meta)))
        return chunks

    # ── 递归字符切分（默认）─────────────────────

    def _split_recursive(self, content: str, meta: dict) -> list[Chunk]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators,
        )
        sub_meta = dict(meta)
        texts = splitter.split_text(content)
        return [Chunk(content=t, metadata=dict(sub_meta)) for t in texts]

    # ── 合短块 ──────────────────────────────────

    def _merge_short(self, chunks: list[Chunk], min_len: int = 50) -> list[Chunk]:
        if not chunks:
            return chunks
        merged: list[Chunk] = []
        buf = chunks[0]
        for ch in chunks[1:]:
            if len(buf.content) < min_len:
                buf.content = buf.content + "\n" + ch.content
                buf.metadata.update(ch.metadata)
            else:
                merged.append(buf)
                buf = ch
        merged.append(buf)
        return merged
