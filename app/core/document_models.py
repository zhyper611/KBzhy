from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Literal


ElementType = Literal["heading", "paragraph", "list", "table", "code"]
ChunkType = Literal["parent", "child"]
BoundingBox = tuple[float, float, float, float]


def content_hash(content: str | bytes) -> str:
    raw_content = content.encode("utf-8") if isinstance(content, str) else content
    return hashlib.sha256(raw_content).hexdigest()


def stable_chunk_id(document_id: str, version: int, position: int, content: str) -> str:
    identity = json.dumps(
        [document_id, version, position, content_hash(content)],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return content_hash(identity)


@dataclass(frozen=True)
class DocumentElement:
    element_id: str
    element_type: ElementType
    text: str
    order: int
    page: int | None = None
    section_path: tuple[str, ...] = ()
    bounding_box: BoundingBox | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ParsedDocument:
    document_id: str
    version: int
    title: str
    language: str
    elements: tuple[DocumentElement, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class KnowledgeChunk:
    chunk_id: str
    document_id: str
    document_version: int
    parent_chunk_id: str | None
    chunk_type: ChunkType
    content: str
    retrieval_text: str
    content_hash: str
    section_path: tuple[str, ...]
    page_start: int | None
    page_end: int | None
    position: int
    token_count: int
    index_version: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def child(
        cls,
        *,
        document_id: str,
        document_version: int,
        content: str,
        position: int,
        token_count: int,
        index_version: str,
        section_path: tuple[str, ...] = (),
        parent_chunk_id: str | None = None,
        page_start: int | None = None,
        page_end: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> KnowledgeChunk:
        path_text = " > ".join(section_path)
        retrieval_text = f"{path_text}\n\n{content}" if path_text else content
        return cls(
            chunk_id=stable_chunk_id(document_id, document_version, position, content),
            document_id=document_id,
            document_version=document_version,
            parent_chunk_id=parent_chunk_id,
            chunk_type="child",
            content=content,
            retrieval_text=retrieval_text,
            content_hash=content_hash(content),
            section_path=section_path,
            page_start=page_start,
            page_end=page_end,
            position=position,
            token_count=token_count,
            index_version=index_version,
            metadata=dict(metadata) if metadata is not None else {},
        )
