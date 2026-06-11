"""文本切分器 — 按文档类型选用不同策略，保留结构化内容"""

from __future__ import annotations

import logging
import re
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

from KBzhy.config import CHUNK_SIZE, CHUNK_OVERLAP, CHINESE_SEPARATORS

logger = logging.getLogger(__name__)


class Chunk:
    """切分后的文本块"""

    def __init__(self, content: str, metadata: dict[str, Any] | None = None):
        self.content = content
        self.metadata = metadata or {}

    def __repr__(self) -> str:
        return f"Chunk(len={len(self.content)})"


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

    # 表格行特征：| 分隔
    _TABLE_PATTERN = re.compile(r"\|.+\|")

    # 代码块特征
    _CODE_PATTERN = re.compile(r"```[\s\S]*?```")

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

        # 预处理：标记结构化内容
        content = self._mark_structured(content)

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

    # ── 标记结构化内容 ─────────────────────────

    def _mark_structured(self, content: str) -> str:
        """给表格/代码块加特殊标记，防止被切碎"""
        # 代码块标记为原子单元
        content = self._CODE_PATTERN.sub(
            lambda m: m.group().replace("\n", "␤"), content
        )
        # 表格标记为原子单元
        table_lines: list[str] = []
        result: list[str] = []
        in_table = False
        for line in content.split("\n"):
            if self._TABLE_PATTERN.match(line):
                if not in_table:
                    in_table = True
                    table_lines = []
                table_lines.append(line)
            else:
                if in_table:
                    in_table = False
                    result.append("␟TABLE␟" + "␤".join(table_lines) + "␟TABLE_END␟")
                result.append(line)
        if in_table:
            result.append("␟TABLE␟" + "␤".join(table_lines) + "␟TABLE_END␟")
        return "\n".join(result)

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
