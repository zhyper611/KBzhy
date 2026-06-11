"""聊天与会话管理 API"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse

from KBzhy.app.models.schemas import (
    ChatRequest,
    ChatResponse,
    SessionCreate,
    SessionResponse,
    SessionMessagesResponse,
    SourceDocument,
    ErrorResponse,
)
from KBzhy.config import REDIS_HOST, REDIS_PORT, REDIS_DB, REDIS_PASSWORD
from KBzhy.app.core.rag_engine import RAGEngine
from KBzhy.app.core.engine import get_rag_engine
from KBzhy.app.api.documents import _kb_meta

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["chat"])

_SESSION_META_PREFIX = "kbzhy:session_meta:"
# 内存缓存：session_id → {title, kb_id, created_at}
_session_cache: dict[str, dict] = {}
_redis = None


def _get_redis():
    global _redis
    if _redis is not None:
        return _redis if _redis else None
    try:
        import redis
        _redis = redis.Redis(
            host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB,
            password=REDIS_PASSWORD or None,
            socket_connect_timeout=3, decode_responses=True,
        )
        _redis.ping()
    except Exception:
        _redis = False
    return _redis if _redis else None


def _save_meta(session_id: str, meta: dict):
    _session_cache[session_id] = meta
    r = _get_redis()
    if r:
        try:
            r.setex(_SESSION_META_PREFIX + session_id, 86400 * 30, json.dumps(meta, ensure_ascii=False))
        except Exception:
            pass


def _delete_meta(session_id: str):
    _session_cache.pop(session_id, None)
    r = _get_redis()
    if r:
        try:
            r.delete(_SESSION_META_PREFIX + session_id)
        except Exception:
            pass


def _load_metas() -> dict[str, dict]:
    r = _get_redis()
    if not r:
        return dict(_session_cache)
    try:
        metas: dict[str, dict] = {}
        for key in r.scan_iter(_SESSION_META_PREFIX + "*"):
            sid = key[len(_SESSION_META_PREFIX):]
            raw = r.get(key)
            if raw:
                metas[sid] = json.loads(raw)
        # 合并内存缓存中没有持久化的数据
        for sid, meta in _session_cache.items():
            if sid not in metas:
                metas[sid] = meta
        _session_cache.clear()
        _session_cache.update(metas)
        return metas
    except Exception:
        return dict(_session_cache)


def get_engine() -> RAGEngine:
    return get_rag_engine()


def _sse_data(data: str) -> str:
    lines = data.split("\n")
    return "".join(f"data: {line}\n" for line in lines) + "\n"


def _auto_title(session_id: str, question: str):
    """首次发言时自动用问题内容作为会话标题"""
    meta = _session_cache.get(session_id) or (_load_metas().get(session_id))
    if not meta or meta.get("title") != "新会话":
        return
    engine = get_engine()
    memory = engine.memory_manager.get(session_id)
    if len(memory.get_context()) == 0:
        title = question.strip()[:30]
        if len(question.strip()) > 30:
            title += "..."
        meta["title"] = title
        _save_meta(session_id, meta)


# ── 会话管理 ───────────────────────────────────


@router.post("/sessions", response_model=SessionResponse)
def create_session(body: SessionCreate):
    session_id = uuid.uuid4().hex[:16]
    engine = get_engine()
    engine.memory_manager.get(session_id)
    meta = {
        "title": body.title,
        "kb_id": body.kb_id,
        "created_at": datetime.now().isoformat(),
    }
    _save_meta(session_id, meta)
    return SessionResponse(
        session_id=session_id,
        title=body.title,
        kb_id=body.kb_id,
        created_at=meta["created_at"],
        message_count=0,
    )


@router.get("/sessions", response_model=list[SessionResponse])
def list_sessions(kb_id: str | None = Query(default=None)):
    engine = get_engine()
    result = []
    for sid, meta in _load_metas().items():
        if kb_id and meta.get("kb_id") != kb_id:
            continue
        msg_count = 0
        try:
            memory = engine.memory_manager.get(sid)
            msg_count = len(memory.get_context())
        except Exception:
            pass
        session_kb_id = meta.get("kb_id")
        kb_name = _kb_meta.get(session_kb_id, {}).get("name", "") if session_kb_id else ""
        result.append(SessionResponse(
            session_id=sid,
            title=meta.get("title", ""),
            kb_id=session_kb_id,
            kb_name=kb_name,
            created_at=meta.get("created_at", ""),
            message_count=msg_count,
        ))
    result.sort(key=lambda s: s.created_at, reverse=True)
    return result


@router.delete("/sessions/{session_id}")
def delete_session(session_id: str):
    engine = get_engine()
    engine.memory_manager.delete(session_id)
    _delete_meta(session_id)
    return {"message": f"会话 {session_id} 已删除"}


@router.get("/sessions/{session_id}/messages", response_model=SessionMessagesResponse)
def get_session_messages(session_id: str):
    engine = get_engine()
    memory = engine.memory_manager.get(session_id)
    messages = memory.get_context()
    if not messages:
        messages = memory.get_history()
    return SessionMessagesResponse(session_id=session_id, messages=messages)


# ── 单轮问答 ───────────────────────────────────


@router.post("/chat", response_model=ChatResponse)
def single_chat(body: ChatRequest):
    engine = get_engine()
    try:
        result = engine.chat(
            question=body.question,
            kb_id=body.kb_id,
            top_k=body.top_k,
            temperature=body.temperature,
            chain_type=body.chain_type.value,
            rerank_method=body.rerank_method,
            similarity_threshold=body.similarity_threshold,
            enable_expansion=body.enable_expansion,
            enable_rewrite=body.enable_rewrite,
        )
        return ChatResponse(
            answer=result["answer"],
            session_id=result.get("session_id"),
            sources=[
                SourceDocument(
                    content=s["content"],
                    source=s.get("source", ""),
                    page=s.get("page"),
                    score=s.get("score", 0),
                )
                for s in result.get("sources", [])
            ],
            hallucination_flags=result.get("hallucination_flags", []),
        )
    except Exception as exc:
        logger.exception("单轮问答失败: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/chat/stream")
def single_chat_stream(body: ChatRequest):
    engine = get_engine()

    def generate():
        try:
            for chunk in engine.chat_stream(
                question=body.question,
                kb_id=body.kb_id,
                top_k=body.top_k,
                temperature=body.temperature,
                rerank_method=body.rerank_method,
                similarity_threshold=body.similarity_threshold,
                enable_expansion=body.enable_expansion,
                enable_rewrite=body.enable_rewrite,
            ):
                yield _sse_data(chunk)
            yield _sse_data("[DONE]")
        except Exception as exc:
            logger.exception("流式对话失败: %s", exc)
            yield _sse_data(f"[ERROR] {exc}")

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ── 多轮对话 ───────────────────────────────────


@router.post("/chat/{session_id}", response_model=ChatResponse)
def multi_chat(session_id: str, body: ChatRequest):
    _auto_title(session_id, body.question)
    engine = get_engine()
    try:
        result = engine.chat(
            question=body.question,
            kb_id=body.kb_id,
            session_id=session_id,
            top_k=body.top_k,
            temperature=body.temperature,
            chain_type=body.chain_type.value,
            rerank_method=body.rerank_method,
            similarity_threshold=body.similarity_threshold,
            enable_expansion=body.enable_expansion,
            enable_rewrite=body.enable_rewrite,
        )
        return ChatResponse(
            answer=result["answer"],
            session_id=session_id,
            sources=[
                SourceDocument(
                    content=s["content"],
                    source=s.get("source", ""),
                    page=s.get("page"),
                    score=s.get("score", 0),
                )
                for s in result.get("sources", [])
            ],
            hallucination_flags=result.get("hallucination_flags", []),
        )
    except Exception as exc:
        logger.exception("多轮对话失败: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/chat/{session_id}/stream")
def multi_chat_stream(session_id: str, body: ChatRequest):
    _auto_title(session_id, body.question)
    engine = get_engine()

    def generate():
        try:
            for chunk in engine.chat_stream(
                question=body.question,
                kb_id=body.kb_id,
                session_id=session_id,
                top_k=body.top_k,
                temperature=body.temperature,
                rerank_method=body.rerank_method,
                similarity_threshold=body.similarity_threshold,
                enable_expansion=body.enable_expansion,
                enable_rewrite=body.enable_rewrite,
            ):
                yield _sse_data(chunk)
            yield _sse_data("[DONE]")
        except Exception as exc:
            logger.exception("流式对话失败: %s", exc)
            yield _sse_data(f"[ERROR] {exc}")

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
