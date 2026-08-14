from __future__ import annotations

import hashlib
import logging
import os
import re
import shutil
import uuid
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile, File, Query

from KBzhy.app.core.engine import get_rag_engine
from KBzhy.app.core.indexing_worker import get_indexing_worker
from KBzhy.app.core.metadata_store import (
    DocumentContentUnchanged,
    DocumentNotFoundError,
    DuplicateDocumentError,
    KnowledgeBaseNotFoundError,
    MetadataStoreUnavailable,
    get_metadata_store,
    now_iso,
)
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
from KBzhy.config import MAX_UPLOAD_SIZE, UPLOAD_STORAGE_DIR

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["knowledge-bases & documents"])


def get_engine():
    return get_rag_engine()


def _store():
    try:
        return get_metadata_store()
    except MetadataStoreUnavailable as exc:
        logger.error("元数据存储不可用: %s", exc)
        raise HTTPException(status_code=503, detail="MySQL 元数据存储不可用，请检查数据库配置和连接") from exc


def _metadata_read(operation, *, failure_detail: str):
    try:
        return operation()
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("document metadata read failed")
        raise HTTPException(status_code=500, detail=failure_detail) from exc


def _to_document_info(data: dict) -> DocumentInfo:
    return DocumentInfo(
        id=data["id"],
        filename=data["filename"],
        file_type=data.get("file_type") or os.path.splitext(data["filename"])[1],
        kb_id=data["kb_id"],
        status=DocStatus(data["status"]),
        chunk_count=data.get("chunk_count", 0),
        task_id=data.get("task_id"),
        error_message=data.get("error_message"),
        created_at=data["created_at"],
        updated_at=data["updated_at"],
    )


_WINDOWS_RESERVED_NAMES = {"CON", "PRN", "AUX", "NUL"} | {
    f"{prefix}{number}" for prefix in ("COM", "LPT") for number in range(1, 10)
}
_WINDOWS_SUPERSCRIPT_DIGITS = str.maketrans("¹²³", "123")


def _safe_filename(filename: str) -> str:
    basename = Path(filename).name
    stem = basename.split(".", 1)[0].upper().translate(_WINDOWS_SUPERSCRIPT_DIGITS)
    utf16_units = len(basename.encode("utf-16-le")) // 2
    invalid = (
        not basename
        or basename in {".", ".."}
        or basename != filename
        or utf16_units > 255
        or stem in _WINDOWS_RESERVED_NAMES
        or basename.endswith((".", " "))
        or re.search(r'[<>:"/\\|?*]', basename)
        or any(ord(char) < 32 for char in basename)
    )
    if invalid:
        raise HTTPException(status_code=400, detail="文件名不合法")
    return basename


def _save_upload_file(
    kb_id: str,
    doc_id: str,
    filename: str,
    content: bytes,
    version_directory: str = "v1",
) -> str:
    base_dir = Path(UPLOAD_STORAGE_DIR) / kb_id / doc_id / version_directory
    safe_filename = _safe_filename(filename)
    target = base_dir / safe_filename
    resolved_base = base_dir.resolve()
    resolved_target = target.resolve()
    if resolved_target.parent != resolved_base:
        raise HTTPException(status_code=400, detail="文件名不合法")
    resolved_base.mkdir(parents=True, exist_ok=True)
    with resolved_target.open("xb") as output:
        output.write(content)
    return str(resolved_target)


@router.post("/knowledge-bases", response_model=KnowledgeBaseInfo)
def create_knowledge_base(body: KnowledgeBaseCreate):
    store = _store()
    engine = get_engine()
    kb_id = uuid.uuid4().hex[:12]
    engine.create_kb(kb_id)

    created_at = datetime.now().isoformat()
    try:
        store.create_knowledge_base(kb_id, body.name, body.description, created_at)
    except Exception as exc:
        try:
            engine.delete_kb(kb_id)
        except Exception:
            logger.exception("MySQL 写入失败后回滚 Chroma 知识库失败: %s", kb_id)
        logger.exception("创建知识库元数据失败: kb=%s", kb_id)
        raise HTTPException(status_code=500, detail="创建知识库失败") from exc

    return KnowledgeBaseInfo(
        kb_id=kb_id,
        name=body.name,
        description=body.description,
        doc_count=0,
        created_at=created_at,
    )


@router.get("/knowledge-bases", response_model=list[KnowledgeBaseInfo])
def list_knowledge_bases():
    store = _store()
    items = _metadata_read(store.list_knowledge_bases, failure_detail="获取知识库列表失败")
    return [
        KnowledgeBaseInfo(
            kb_id=item["kb_id"],
            name=item["name"],
            description=item.get("description", ""),
            doc_count=item.get("doc_count", 0),
            created_at=item["created_at"],
        )
        for item in items
    ]


@router.delete("/knowledge-bases/{kb_id}")
def delete_knowledge_base(kb_id: str):
    store = _store()
    engine = get_engine()

    from KBzhy.app.api.chat import _delete_meta, _load_metas

    deleted_sessions = 0
    for sid, meta in _load_metas().items():
        if meta.get("kb_id") == kb_id:
            engine.memory_manager.delete(sid)
            _delete_meta(sid)
            deleted_sessions += 1

    try:
        engine.delete_kb(kb_id)
        store.delete_knowledge_base(kb_id)
        shutil.rmtree(Path(UPLOAD_STORAGE_DIR) / kb_id, ignore_errors=True)
    except Exception as exc:
        logger.exception("删除知识库失败: kb=%s", kb_id)
        raise HTTPException(status_code=500, detail="删除知识库失败") from exc

    return {"message": f"知识库 {kb_id} 已删除，同时清理 {deleted_sessions} 个关联会话"}


@router.post("/knowledge-bases/{kb_id}/documents/upload", response_model=DocumentInfo)
async def upload_document(kb_id: str, file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="文件名不能为空")

    content = await file.read()
    content_hash = hashlib.sha256(content).hexdigest()
    if not content:
        raise HTTPException(status_code=400, detail="文件内容为空")
    if len(content) > MAX_UPLOAD_SIZE:
        raise HTTPException(status_code=413, detail=f"文件大小超过限制 ({MAX_UPLOAD_SIZE // 1024 // 1024}MB)")

    store = _store()
    if not _metadata_read(
        lambda: store.knowledge_base_exists(kb_id), failure_detail="文档入队失败"
    ):
        raise HTTPException(status_code=404, detail="知识库不存在")
    duplicate = _metadata_read(
        lambda: store.find_document_by_hash(kb_id, content_hash),
        failure_detail="文档入队失败",
    )
    if duplicate:
        raise HTTPException(
            status_code=409,
            detail={"message": "知识库中已存在相同内容的文档", "document_id": duplicate["id"]},
        )

    doc_id = uuid.uuid4().hex
    task_id = uuid.uuid4().hex
    filename = _safe_filename(file.filename)
    ext = os.path.splitext(filename)[1].lower()
    now = now_iso()

    try:
        storage_path = _save_upload_file(kb_id, doc_id, filename, content)
        doc_data = {
            "id": doc_id,
            "filename": filename,
            "file_type": ext,
            "kb_id": kb_id,
            "status": DocStatus.QUEUED.value,
            "chunk_count": 0,
            "task_id": task_id,
            "storage_path": storage_path,
            "content_hash": content_hash,
            "error_message": None,
            "created_at": now,
            "updated_at": now,
        }
        task_data = {
            "task_id": task_id,
            "doc_id": doc_id,
            "kb_id": kb_id,
            "status": "queued",
            "error_message": None,
            "created_at": now,
            "updated_at": now,
        }
        if hasattr(store, "create_document_with_task"):
            store.create_document_with_task(doc_data, task_data)
        else:
            store.create_document(doc_data)
            store.create_task(task_data)
    except DuplicateDocumentError as exc:
        shutil.rmtree(Path(UPLOAD_STORAGE_DIR) / kb_id / doc_id, ignore_errors=True)
        raise HTTPException(
            status_code=409,
            detail={"message": "知识库中已存在相同内容的文档", "document_id": exc.document_id},
        ) from exc
    except KnowledgeBaseNotFoundError as exc:
        shutil.rmtree(Path(UPLOAD_STORAGE_DIR) / kb_id / doc_id, ignore_errors=True)
        raise HTTPException(status_code=404, detail="知识库不存在") from exc
    except HTTPException:
        raise
    except Exception as exc:
        shutil.rmtree(Path(UPLOAD_STORAGE_DIR) / kb_id / doc_id, ignore_errors=True)
        logger.exception("文档入队失败: kb=%s doc=%s", kb_id, doc_id)
        raise HTTPException(status_code=500, detail="文档入队失败") from exc

    try:
        get_indexing_worker().enqueue(task_id)
    except Exception as exc:
        logger.warning("文档任务已持久化但内存入队失败，等待恢复: task=%s error=%s", task_id, exc)
    logger.info("文档已入队: kb=%s doc=%s task=%s filename=%s", kb_id, doc_id, task_id, filename)
    return _to_document_info(doc_data)


@router.put("/knowledge-bases/{kb_id}/documents/{doc_id}", response_model=DocumentUpdateResponse)
async def update_document(kb_id: str, doc_id: str, file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="文件名不能为空")

    store = _store()
    current = _metadata_read(
        lambda: store.get_document(kb_id, doc_id),
        failure_detail="文档更新入队失败",
    )
    if not current:
        raise HTTPException(status_code=404, detail="文档不存在")

    content = await file.read()
    content_hash = hashlib.sha256(content).hexdigest()
    if not content:
        raise HTTPException(status_code=400, detail="文件内容为空")
    if len(content) > MAX_UPLOAD_SIZE:
        raise HTTPException(status_code=413, detail=f"文件大小超过限制 ({MAX_UPLOAD_SIZE // 1024 // 1024}MB)")

    duplicate = _metadata_read(
        lambda: store.find_document_by_hash(kb_id, content_hash, exclude_document_id=doc_id),
        failure_detail="文档更新入队失败",
    )
    if duplicate:
        raise HTTPException(
            status_code=409,
            detail={"message": "知识库中已存在相同内容的文档", "document_id": duplicate["id"]},
        )
    duplicate = _metadata_read(
        lambda: store.find_document_by_hash(kb_id, content_hash),
        failure_detail="文档更新入队失败",
    )
    if duplicate:
        return DocumentUpdateResponse(
            id=doc_id,
            filename=current["filename"],
            kb_id=kb_id,
            status=DocStatus(current["status"]),
            chunk_count=current.get("chunk_count", 0),
            task_id=current.get("task_id"),
            error_message=current.get("error_message"),
            message="文档内容未变化，无需重新索引",
        )

    task_id = uuid.uuid4().hex
    filename = _safe_filename(file.filename)
    ext = os.path.splitext(filename)[1].lower()
    now = now_iso()

    try:
        storage_path = _save_upload_file(kb_id, doc_id, filename, content, task_id)
        store.create_document_version_and_task(
            doc_id, kb_id, content_hash, storage_path, filename, ext, task_id, now
        )
    except DocumentContentUnchanged:
        shutil.rmtree(Path(UPLOAD_STORAGE_DIR) / kb_id / doc_id / task_id, ignore_errors=True)
        return DocumentUpdateResponse(
            id=doc_id,
            filename=current["filename"],
            kb_id=kb_id,
            status=DocStatus(current["status"]),
            chunk_count=current.get("chunk_count", 0),
            task_id=current.get("task_id"),
            error_message=current.get("error_message"),
            message="文档内容未变化，无需重新索引",
        )
    except DuplicateDocumentError as exc:
        shutil.rmtree(Path(UPLOAD_STORAGE_DIR) / kb_id / doc_id / task_id, ignore_errors=True)
        raise HTTPException(
            status_code=409,
            detail={"message": "知识库中已存在相同内容的文档", "document_id": exc.document_id},
        ) from exc
    except DocumentNotFoundError as exc:
        shutil.rmtree(Path(UPLOAD_STORAGE_DIR) / kb_id / doc_id / task_id, ignore_errors=True)
        raise HTTPException(status_code=404, detail="文档不存在") from exc
    except KnowledgeBaseNotFoundError as exc:
        shutil.rmtree(Path(UPLOAD_STORAGE_DIR) / kb_id / doc_id / task_id, ignore_errors=True)
        raise HTTPException(status_code=404, detail="知识库不存在") from exc
    except Exception as exc:
        shutil.rmtree(Path(UPLOAD_STORAGE_DIR) / kb_id / doc_id / task_id, ignore_errors=True)
        logger.exception("文档更新入队失败: kb=%s doc=%s", kb_id, doc_id)
        raise HTTPException(status_code=500, detail="文档更新入队失败") from exc

    try:
        get_indexing_worker().enqueue(task_id)
    except Exception as exc:
        logger.warning("文档更新任务已持久化但内存入队失败，等待恢复: task=%s error=%s", task_id, exc)

    return DocumentUpdateResponse(
        id=doc_id,
        filename=filename,
        kb_id=kb_id,
        status=DocStatus.QUEUED,
        chunk_count=0,
        task_id=task_id,
        message="文档已重新入队，后台正在处理",
    )


@router.get("/knowledge-bases/{kb_id}/documents", response_model=DocumentListResponse)
def list_documents(
    kb_id: str,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    status: str | None = Query(default=None),
):
    store = _store()
    total, docs = _metadata_read(
        lambda: store.list_documents(kb_id, page=page, page_size=page_size, status=status),
        failure_detail="获取文档列表失败",
    )
    return DocumentListResponse(
        total=total,
        page=page,
        page_size=page_size,
        kb_id=kb_id,
        documents=[_to_document_info(doc) for doc in docs],
    )


@router.get("/knowledge-bases/{kb_id}/documents/{doc_id}/chunks", response_model=DocumentChunksResponse)
def get_document_chunks(kb_id: str, doc_id: str):
    store = _store()
    doc = _metadata_read(
        lambda: store.get_document(kb_id, doc_id), failure_detail="获取文档分块失败"
    )
    if not doc:
        raise HTTPException(status_code=404, detail="文档不存在")
    if doc["status"] != DocStatus.READY.value:
        return DocumentChunksResponse(
            kb_id=kb_id,
            document_id=doc_id,
            filename=doc["filename"],
            total=0,
            chunks=[],
        )

    chunks = get_engine().list_document_chunks(kb_id, source=doc["filename"], doc_id=doc_id)
    chunks = sorted(chunks, key=lambda item: item.get("chunk_index", 0))

    return DocumentChunksResponse(
        kb_id=kb_id,
        document_id=doc_id,
        filename=doc["filename"],
        total=len(chunks),
        chunks=[
            DocumentChunkInfo(
                chunk_index=chunk.get("chunk_index") if isinstance(chunk.get("chunk_index"), int) else index,
                content=chunk.get("content", ""),
                metadata=chunk.get("metadata") or {},
                source=(chunk.get("metadata") or {}).get("source", doc["filename"]),
                page=(chunk.get("metadata") or {}).get("page"),
            )
            for index, chunk in enumerate(chunks, start=1)
        ],
    )


@router.delete("/knowledge-bases/{kb_id}/documents/{doc_id}")
def delete_document(kb_id: str, doc_id: str):
    store = _store()
    doc = _metadata_read(
        lambda: store.get_document(kb_id, doc_id), failure_detail="删除文档失败"
    )
    if not doc:
        raise HTTPException(status_code=404, detail="文档不存在")

    try:
        store.update_document(doc_id, status=DocStatus.DELETING.value)
        get_engine().remove_document(doc["filename"], kb_id, doc_id=doc_id)
        store.delete_document_record(doc_id)
        shutil.rmtree(Path(UPLOAD_STORAGE_DIR) / kb_id / doc_id, ignore_errors=True)
    except Exception as exc:
        try:
            store.update_document(doc_id, status=DocStatus.FAILED.value, error_message="document deletion failed")
        except Exception:
            logger.exception("回写文档删除失败状态失败: kb=%s doc=%s", kb_id, doc_id)
        logger.exception("删除文档失败: kb=%s doc=%s", kb_id, doc_id)
        raise HTTPException(status_code=500, detail="删除文档失败") from exc

    return {"message": f"文档 {doc['filename']} 已删除"}
