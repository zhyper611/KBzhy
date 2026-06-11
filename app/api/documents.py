"""知识库 & 文档管理 API"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import uuid
from datetime import datetime

from fastapi import APIRouter, HTTPException, UploadFile, File, Query

from KBzhy.app.models.schemas import (
    KnowledgeBaseCreate,
    KnowledgeBaseInfo,
    DocumentInfo,
    DocumentChunkInfo,
    DocumentChunksResponse,
    DocumentListResponse,
    DocumentUpdateResponse,
    DocStatus,
)
from KBzhy.app.core.rag_engine import RAGEngine
from KBzhy.app.core.engine import get_rag_engine
from KBzhy.config import DATA_DIR, MAX_UPLOAD_SIZE

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["knowledge-bases & documents"])

# kb_id → doc_id → info
_doc_registry: dict[str, dict[str, dict]] = {}
# kb_id → {name, description, created_at}
_kb_meta: dict[str, dict] = {}
_REGISTRY_FILE = os.path.join(DATA_DIR, "doc_registry.json")
_KB_META_FILE = os.path.join(DATA_DIR, "kb_meta.json")


def get_engine() -> RAGEngine:
    return get_rag_engine()


def _load_registry():
    global _doc_registry, _kb_meta
    _doc_registry = _load_json_safe(_REGISTRY_FILE, {})
    if _doc_registry:
        logger.info("已加载文档注册表: %d 个知识库", len(_doc_registry))
    _kb_meta = _load_json_safe(_KB_META_FILE, {})
    if _kb_meta:
        logger.info("已加载 KB 元数据: %d 条", len(_kb_meta))


def _load_json_safe(path: str, default: dict) -> dict:
    """安全加载 JSON：先试主文件，损坏则试备份"""
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("注册表文件损坏 (%s): %s，尝试从备份恢复", path, exc)
    bak = path + ".bak"
    if os.path.exists(bak):
        try:
            with open(bak, "r", encoding="utf-8") as f:
                data = json.load(f)
            logger.info("已从备份恢复: %s", path)
            return data
        except Exception as exc2:
            logger.warning("备份同样损坏 (%s): %s", bak, exc2)
    return default


def _save_registry():
    try:
        os.makedirs(os.path.dirname(_REGISTRY_FILE), exist_ok=True)
        _write_json_atomic(_REGISTRY_FILE, _doc_registry)
        _write_json_atomic(_KB_META_FILE, _kb_meta)
    except Exception as exc:
        logger.warning("保存注册表失败: %s", exc)


def _write_json_atomic(path: str, data: dict):
    """原子写入：先写临时文件，成功后 rename（同文件系统下原子操作）"""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    # 先备份旧文件
    if os.path.exists(path):
        try:
            os.replace(path, path + ".bak")
        except OSError:
            pass
    os.replace(tmp, path)


_load_registry()


# ── 知识库 CRUD ─────────────────────────────────


@router.post("/knowledge-bases", response_model=KnowledgeBaseInfo)
def create_knowledge_base(body: KnowledgeBaseCreate):
    engine = get_engine()
    kb_id = uuid.uuid4().hex[:12]
    engine.create_kb(kb_id)

    now = datetime.now().isoformat()
    _doc_registry.setdefault(kb_id, {})
    _kb_meta[kb_id] = {
        "name": body.name,
        "description": body.description,
        "created_at": now,
    }
    _save_registry()

    logger.info("知识库已创建: %s (%s)", body.name, kb_id)

    return KnowledgeBaseInfo(
        kb_id=kb_id,
        name=body.name,
        description=body.description,
        doc_count=0,
        created_at=now,
    )


@router.get("/knowledge-bases", response_model=list[KnowledgeBaseInfo])
def list_knowledge_bases():
    engine = get_engine()
    kb_ids = engine.list_kbs()
    # 兜底：ChromaDB 不可用时从本地注册表恢复 KB 列表
    if not kb_ids and _kb_meta:
        kb_ids = list(_kb_meta.keys())
        logger.warning("ChromaDB 返回空，从 kb_meta.json 恢复 %d 个知识库", len(kb_ids))
    result = []
    for kb_id in kb_ids:
        docs = _doc_registry.get(kb_id, {})
        meta = _kb_meta.get(kb_id, {})
        ready_docs = sum(1 for d in docs.values() if d.get("status") == "ready")
        result.append(KnowledgeBaseInfo(
            kb_id=kb_id,
            name=meta.get("name", f"知识库-{kb_id}"),
            description=meta.get("description", ""),
            doc_count=ready_docs,
            created_at=meta.get("created_at", min((d.get("created_at", "") for d in docs.values()), default="")),
        ))
    return result


@router.delete("/knowledge-bases/{kb_id}")
def delete_knowledge_base(kb_id: str):
    engine = get_engine()

    # 清理关联的会话
    from KBzhy.app.api.chat import _load_metas, _delete_meta
    deleted_sessions = 0
    for sid, meta in _load_metas().items():
        if meta.get("kb_id") == kb_id:
            engine.memory_manager.delete(sid)
            _delete_meta(sid)
            deleted_sessions += 1
    if deleted_sessions:
        logger.info("级联删除了 %d 个关联会话", deleted_sessions)

    engine.delete_kb(kb_id)
    _doc_registry.pop(kb_id, None)
    _kb_meta.pop(kb_id, None)
    _save_registry()
    logger.info("知识库已删除: %s", kb_id)
    return {"message": f"知识库 {kb_id} 已删除，同时清理了 {deleted_sessions} 个关联会话"}


# ── 文档上传 ───────────────────────────────────


@router.post("/knowledge-bases/{kb_id}/documents/upload", response_model=DocumentInfo)
async def upload_document(kb_id: str, file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="文件名不能为空")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="文件内容为空")
    if len(content) > MAX_UPLOAD_SIZE:
        raise HTTPException(status_code=413, detail=f"文件大小超过限制 ({MAX_UPLOAD_SIZE // 1024 // 1024}MB)")

    if kb_id not in _kb_meta and kb_id not in _doc_registry:
        raise HTTPException(status_code=404, detail="知识库不存在")
    if kb_id not in _doc_registry:
        _doc_registry[kb_id] = {}

    engine = get_engine()
    doc_id = f"{file.filename}-{datetime.now().strftime('%Y%m%d%H%M%S')}"

    _doc_registry[kb_id][doc_id] = {
        "filename": file.filename,
        "kb_id": kb_id,
        "status": DocStatus.PARSING.value,
        "chunk_count": 0,
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
    }
    _save_registry()

    try:
        ext = os.path.splitext(file.filename)[1].lower()
        tmp_path = None
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        try:
            chunk_count = engine.index_document(tmp_path, kb_id, display_name=file.filename)
        finally:
            if tmp_path:
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass

        _doc_registry[kb_id][doc_id]["status"] = DocStatus.READY.value
        _doc_registry[kb_id][doc_id]["chunk_count"] = chunk_count
        _doc_registry[kb_id][doc_id]["updated_at"] = datetime.now().isoformat()
        _save_registry()

        logger.info("文档索引完成: %s → KB %s, %d chunks", file.filename, kb_id, chunk_count)

        return DocumentInfo(
            id=doc_id,
            filename=file.filename,
            file_type=ext,
            kb_id=kb_id,
            status=DocStatus.READY,
            chunk_count=chunk_count,
            created_at=_doc_registry[kb_id][doc_id]["created_at"],
            updated_at=_doc_registry[kb_id][doc_id]["updated_at"],
        )

    except Exception as exc:
        _doc_registry[kb_id][doc_id]["status"] = DocStatus.FAILED.value
        _doc_registry[kb_id][doc_id]["updated_at"] = datetime.now().isoformat()
        _save_registry()
        logger.exception("文档上传失败: %s", exc)
        raise HTTPException(status_code=500, detail=f"文档处理失败: {exc}")


# ── 更新文档 ───────────────────────────────────


@router.put("/knowledge-bases/{kb_id}/documents/{doc_id}", response_model=DocumentUpdateResponse)
async def update_document(kb_id: str, doc_id: str, file: UploadFile = File(...)):
    kb_docs = _doc_registry.get(kb_id, {})
    if doc_id not in kb_docs:
        raise HTTPException(status_code=404, detail="文档不存在")

    engine = get_engine()
    old_filename = kb_docs[doc_id]["filename"]

    try:
        engine.remove_document(old_filename, kb_id)
    except Exception as exc:
        logger.warning("删除旧 chunks 失败: %s", exc)

    content = await file.read()
    if len(content) > MAX_UPLOAD_SIZE:
        raise HTTPException(status_code=413, detail=f"文件大小超过限制 ({MAX_UPLOAD_SIZE // 1024 // 1024}MB)")
    ext = os.path.splitext(file.filename)[1].lower()

    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        chunk_count = engine.index_document(tmp_path, kb_id, display_name=file.filename)
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass

    kb_docs[doc_id].update({
        "filename": file.filename,
        "status": DocStatus.READY.value,
        "chunk_count": chunk_count,
        "updated_at": datetime.now().isoformat(),
    })
    _save_registry()

    return DocumentUpdateResponse(
        id=doc_id,
        filename=file.filename,
        kb_id=kb_id,
        status=DocStatus.READY,
        chunk_count=chunk_count,
        message=f"文档已更新，共 {chunk_count} 个文本块",
    )


# ── 文档列表 ───────────────────────────────────


@router.get("/knowledge-bases/{kb_id}/documents", response_model=DocumentListResponse)
def list_documents(
    kb_id: str,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    status: str | None = Query(default=None),
):
    kb_docs = _doc_registry.get(kb_id, {})
    docs = list(kb_docs.items())  # (doc_id, info) pairs
    if status:
        docs = [(did, d) for did, d in docs if d.get("status") == status]

    docs.sort(key=lambda item: item[1].get("updated_at", ""), reverse=True)
    total = len(docs)
    start = (page - 1) * page_size
    end = start + page_size
    page_docs = docs[start:end]

    return DocumentListResponse(
        total=total,
        page=page,
        page_size=page_size,
        kb_id=kb_id,
        documents=[
            DocumentInfo(
                id=did,
                filename=d["filename"],
                file_type=os.path.splitext(d["filename"])[1],
                kb_id=kb_id,
                status=DocStatus(d["status"]),
                chunk_count=d.get("chunk_count", 0),
                created_at=d["created_at"],
                updated_at=d["updated_at"],
            )
            for did, d in page_docs
        ],
    )


@router.get("/knowledge-bases/{kb_id}/documents/{doc_id}/chunks", response_model=DocumentChunksResponse)
def get_document_chunks(kb_id: str, doc_id: str):
    kb_docs = _doc_registry.get(kb_id, {})
    if doc_id not in kb_docs:
        raise HTTPException(status_code=404, detail="文档不存在")

    filename = kb_docs[doc_id]["filename"]
    chunks = get_engine().list_document_chunks(kb_id, filename)
    chunks = sorted(chunks, key=lambda item: item.get("chunk_index", 0))

    return DocumentChunksResponse(
        kb_id=kb_id,
        document_id=doc_id,
        filename=filename,
        total=len(chunks),
        chunks=[
            DocumentChunkInfo(
                chunk_index=chunk.get("chunk_index") if isinstance(chunk.get("chunk_index"), int) else index,
                content=chunk.get("content", ""),
                metadata=chunk.get("metadata") or {},
                source=(chunk.get("metadata") or {}).get("source", filename),
                page=(chunk.get("metadata") or {}).get("page"),
            )
            for index, chunk in enumerate(chunks, start=1)
        ],
    )


# ── 删除文档 ───────────────────────────────────


@router.delete("/knowledge-bases/{kb_id}/documents/{doc_id}")
def delete_document(kb_id: str, doc_id: str):
    kb_docs = _doc_registry.get(kb_id, {})
    if doc_id not in kb_docs:
        raise HTTPException(status_code=404, detail="文档不存在")

    engine = get_engine()
    filename = kb_docs[doc_id]["filename"]

    try:
        engine.remove_document(filename, kb_id)
    except Exception as exc:
        logger.warning("删除文档 chunks 失败: %s", exc)

    del kb_docs[doc_id]
    _save_registry()
    logger.info("文档已删除: %s (KB: %s)", filename, kb_id)
    return {"message": f"文档 {filename} 已删除"}
