from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Literal


ElementType = Literal["heading", "paragraph", "list", "table", "code"]
ChunkType = Literal["parent", "child"]


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
    __hash__ = None

    element_id: str
    element_type: ElementType
    text: str
    order: int
    page: int | None = None
    section_path: tuple[str, ...] = ()
    bounding_box: dict[str, float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "section_path", tuple(self.section_path))
        object.__setattr__(self, "bounding_box", deepcopy(self.bounding_box))
        object.__setattr__(self, "metadata", deepcopy(self.metadata))


@dataclass(frozen=True)
class ParsedDocument:
    __hash__ = None

    document_id: str
    version: int
    title: str
    language: str
    elements: tuple[DocumentElement, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "elements", tuple(self.elements))
        object.__setattr__(self, "metadata", deepcopy(self.metadata))


@dataclass(frozen=True)
class KnowledgeChunk:
    __hash__ = None

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
    index_version: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.chunk_type == "child" and not self.parent_chunk_id:
            raise ValueError("parent_chunk_id is required for child chunks")
        object.__setattr__(self, "section_path", tuple(self.section_path))
        object.__setattr__(self, "metadata", deepcopy(self.metadata))

    @classmethod
    def child(
        cls,
        *,
        document_id: str,
        document_version: int,
        content: str,
        position: int,
        token_count: int,
        parent_chunk_id: str,
        index_version: int = 1,
        section_path: list[str] | tuple[str, ...] = (),
        page_start: int | None = None,
        page_end: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> KnowledgeChunk:
        if not parent_chunk_id:
            raise ValueError("parent_chunk_id is required for child chunks")
        path = tuple(section_path)
        path_text = " > ".join(path)
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
            section_path=path,
            page_start=page_start,
            page_end=page_end,
            position=position,
            token_count=token_count,
            index_version=index_version,
            metadata=dict(metadata) if metadata is not None else {},
        )


@dataclass(frozen=True)
class RetrievalCandidate:
    __hash__ = None

    chunk_id: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    vector_rank: int | None = None
    vector_score: float | None = None
    bm25_rank: int | None = None
    bm25_score: float | None = None
    rrf_score: float = 0.0
    rerank_score: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", deepcopy(self.metadata))

    @property
    def score(self) -> float:
        return self.rerank_score if self.rerank_score is not None else self.rrf_score
