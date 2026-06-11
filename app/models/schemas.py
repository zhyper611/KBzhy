"""Pydantic 请求/响应数据模型"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ──────────────────────────────────────────────
#  枚举
# ──────────────────────────────────────────────


class DocStatus(str, Enum):
    UPLOADED = "uploaded"
    PARSING = "parsing"
    READY = "ready"
    FAILED = "failed"


class RerankMethod(str, Enum):
    MODEL = "model"
    LLM = "llm"
    KEYWORD = "keyword"


class ChainType(str, Enum):
    STUFF = "stuff"
    MAP_REDUCE = "map_reduce"
    REFINE = "refine"


# ──────────────────────────────────────────────
#  知识库
# ──────────────────────────────────────────────


class KnowledgeBaseCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=100, description="知识库名称")
    description: str = Field(default="", max_length=500)


class KnowledgeBaseInfo(BaseModel):
    kb_id: str
    name: str
    description: str = ""
    doc_count: int = 0
    created_at: str


# ──────────────────────────────────────────────
#  会话
# ──────────────────────────────────────────────


class SessionCreate(BaseModel):
    title: str = Field(default="新会话", max_length=200)
    kb_id: str | None = None


class SessionResponse(BaseModel):
    session_id: str
    title: str
    created_at: str
    message_count: int
    kb_id: str | None = None
    kb_name: str = ""


# ──────────────────────────────────────────────
#  聊天
# ──────────────────────────────────────────────


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=5000)
    kb_id: str | None = Field(default=None, description="知识库 ID，不传则搜索全部")
    top_k: int = Field(default=5, ge=1, le=20)
    chain_type: ChainType = Field(default=ChainType.STUFF)
    temperature: float = Field(default=0.5, ge=0, le=1)
    stream: bool = Field(default=False)
    rerank_method: str = Field(default="model", description="重排序方法：model / llm / keyword")
    similarity_threshold: float = Field(default=0.35, ge=0, le=1, description="相似度阈值，低于此值的片段会被过滤")
    enable_expansion: bool = Field(default=False, description="是否启用查询扩展")
    enable_rewrite: bool = Field(default=False, description="是否强制启用多轮查询改写；关闭时仅在明显指代上下文时自动改写")


class SourceDocument(BaseModel):
    content: str
    source: str = ""
    page: int | None = None
    score: float = 0.0


class ChatResponse(BaseModel):
    answer: str
    session_id: str | None = None
    sources: list[SourceDocument] = Field(default_factory=list)
    hallucination_flags: list[str] = Field(default_factory=list)


class HistoryMessage(BaseModel):
    role: str
    content: str
    sources: list[dict[str, Any]] | None = None


class SessionMessagesResponse(BaseModel):
    session_id: str
    messages: list[HistoryMessage]


class HallucinationFlag(BaseModel):
    sentence: str
    supported: bool
    source: str = ""


# ──────────────────────────────────────────────
#  文档管理
# ──────────────────────────────────────────────


class DocumentInfo(BaseModel):
    id: str
    filename: str
    file_type: str
    kb_id: str
    status: DocStatus
    chunk_count: int = 0
    created_at: str
    updated_at: str


class DocumentChunkInfo(BaseModel):
    chunk_index: int
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    source: str = ""
    page: int | None = None


class DocumentChunksResponse(BaseModel):
    kb_id: str
    document_id: str
    filename: str
    total: int
    chunks: list[DocumentChunkInfo]


class DocumentUpdateResponse(BaseModel):
    id: str
    filename: str
    kb_id: str
    status: DocStatus
    chunk_count: int
    message: str


class DocumentListResponse(BaseModel):
    total: int
    page: int
    page_size: int
    kb_id: str
    documents: list[DocumentInfo]


# ──────────────────────────────────────────────
#  通用
# ──────────────────────────────────────────────


class ErrorResponse(BaseModel):
    code: int
    message: str
    detail: dict[str, Any] | None = None


class HealthResponse(BaseModel):
    status: str
    version: str
    components: dict[str, str]
