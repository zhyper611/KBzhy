from __future__ import annotations

import logging
import queue
import shutil
import threading
from pathlib import Path
from typing import Callable

from KBzhy.app.core.engine import get_rag_engine
from KBzhy.app.core.chunk_repository import ChunkRepository
from KBzhy.app.core.metadata_store import get_metadata_store, now_iso
from KBzhy.config import UPLOAD_STORAGE_DIR

logger = logging.getLogger(__name__)


class IndexingWorker:
    def __init__(
        self,
        store=None,
        engine_factory: Callable | None = None,
        autostart: bool = True,
        chunk_repository=None,
    ):
        self.store = store or get_metadata_store()
        self.engine_factory = engine_factory or get_rag_engine
        self.chunk_repository = chunk_repository
        if self.chunk_repository is None and hasattr(self.store, "create_connection"):
            self.chunk_repository = ChunkRepository(self.store)
        self._queue: queue.Queue[str] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        if autostart:
            self.start()

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._run, name="kbzhy-indexing-worker", daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._queue.put("")
        if self._thread:
            self._thread.join(timeout=5)

    def enqueue(self, task_id: str):
        self._queue.put(task_id)

    def recover_unfinished_tasks(self):
        for task in self.store.list_recoverable_tasks():
            task_id = task["task_id"]
            doc = self.store.get_document(task["kb_id"], task["doc_id"])
            if not doc:
                self.store.update_task(task_id, status="failed", error_message="document metadata missing")
                continue
            if doc.get("task_id") != task_id or doc.get("status") == "deleting":
                changed = self._mark_terminal(task, doc, "stale", "document has a newer indexing task")
                version = self._optional_task_version(task)
                if changed and version:
                    self._discard_version(task, version, "stale", doc, status_already_updated=True)
                continue
            try:
                version = self._task_version(task, doc)
            except RuntimeError as exc:
                self._fail_unusable_version(task, doc, str(exc))
                continue
            if hasattr(self.store, "requeue_indexing_task"):
                if not self.store.requeue_indexing_task(task_id):
                    changed = self._mark_terminal(task, doc, "stale", "document changed during recovery")
                    if changed:
                        self._discard_version(task, version, "stale", doc, status_already_updated=True)
                    continue
            else:
                latest = self.store.get_document(task["kb_id"], task["doc_id"])
                if not latest or latest.get("task_id") != task_id or latest.get("status") == "deleting":
                    changed = self._mark_terminal(task, doc, "stale", "document changed during recovery")
                    if changed:
                        self._discard_version(task, version, "stale", doc, status_already_updated=True)
                    continue
                self.store.update_document(doc["id"], status="queued", error_message=None)
                self.store.update_task(task_id, status="queued", error_message=None)
            try:
                engine = self.engine_factory()
                if self.chunk_repository is not None and hasattr(engine, "remove_children"):
                    staged_child_ids = [
                        chunk.chunk_id
                        for chunk in self.chunk_repository.list_by_task(task_id)
                        if chunk.chunk_type == "child"
                    ]
                    engine.remove_children(doc["kb_id"], staged_child_ids)
                    self.chunk_repository.discard_task(task_id)
                    self._remove_prepared_artifact(
                        engine, version.get("parsed_artifact_path")
                    )
                else:
                    engine.remove_document(
                        version["filename"], doc["kb_id"],
                        doc_id=doc["id"], task_id=task_id,
                    )
            except Exception as exc:
                logger.warning("恢复任务时清理旧向量失败: task=%s error=%s", task_id, exc)
            self.enqueue(task_id)

    def _run(self):
        while not self._stop.is_set():
            task_id = self._queue.get()
            if not task_id:
                continue
            try:
                self.process_task(task_id)
            except Exception:
                logger.exception("索引任务执行异常: %s", task_id)
            finally:
                self._queue.task_done()

    def process_task(self, task_id: str):
        task = self.store.get_task(task_id)
        if not task:
            logger.warning("索引任务不存在: %s", task_id)
            return

        doc = self.store.get_document(task["kb_id"], task["doc_id"])
        if not doc:
            self.store.update_task(task_id, status="failed", error_message="document metadata missing")
            return
        if doc.get("task_id") != task_id or doc.get("status") == "deleting":
            changed = self._mark_terminal(task, doc, "stale", "document is deleting or has a newer indexing task")
            version = self._optional_task_version(task)
            if changed and version:
                self._discard_version(task, version, "stale", doc, status_already_updated=True)
            logger.info("跳过陈旧索引任务: task=%s doc=%s current_task=%s", task_id, doc["id"], doc.get("task_id"))
            return
        try:
            version = self._task_version(task, doc)
        except RuntimeError as exc:
            self._fail_unusable_version(task, doc, str(exc))
            return
        if hasattr(self.store, "claim_task") and not self.store.claim_task(task_id):
            logger.info("索引任务已被其他 worker 认领或状态已变化: task=%s", task_id)
            return

        if hasattr(self.store, "set_indexing_phase"):
            if not self.store.set_indexing_phase(task_id, "parsing"):
                changed = self._mark_terminal(task, doc, "stale", "document changed after task claim")
                if changed:
                    self._discard_version(task, version, "stale", doc, status_already_updated=True)
                return
        else:
            self.store.update_document(doc["id"], status="parsing", error_message=None)

        engine = self.engine_factory()
        structured_mode = bool(
            self.chunk_repository is not None
            and hasattr(engine, "parse_document_for_index")
            and hasattr(engine, "chunk_document_for_index")
            and hasattr(engine, "stage_document_children")
            and hasattr(engine, "remove_children")
        )
        staged_child_ids: list[str] = []
        prepared_artifact_path: str | None = None
        try:
            if structured_mode:
                parsed_artifact = engine.parse_document_for_index(
                    version["storage_path"],
                    doc["kb_id"],
                    document_id=doc["id"],
                    document_version=int(task["document_version"]),
                    display_name=version["filename"],
                )
                prepared_artifact_path = parsed_artifact.artifact_path
                if hasattr(self.store, "set_indexing_phase"):
                    if not self.store.set_indexing_phase(task_id, "chunking"):
                        changed = self._mark_terminal(
                            task, doc, "stale", "document changed before chunking"
                        )
                        if changed:
                            self._remove_prepared_artifact(engine, prepared_artifact_path)
                            self._discard_version(
                                task, version, "stale", doc, status_already_updated=True
                            )
                        return
                else:
                    self.store.update_task(task_id, status="chunking")
                    self.store.update_document(doc["id"], status="chunking")

                prepared = engine.chunk_document_for_index(
                    parsed_artifact,
                    index_version=int(task.get("index_version") or 1),
                )
                chunks = list(prepared.chunks)
                children = [chunk for chunk in chunks if chunk.chunk_type == "child"]
                if not children:
                    raise RuntimeError("structured indexing produced no child chunks")
                old_child_ids = [
                    chunk.chunk_id
                    for chunk in self.chunk_repository.list_active_children(doc["id"])
                ]
                self.chunk_repository.replace_staging(
                    task_id,
                    doc["id"],
                    int(task["document_version"]),
                    chunks,
                    parsed_artifact_path=prepared.artifact_path,
                )
                if hasattr(self.store, "set_indexing_phase"):
                    if not self.store.set_indexing_phase(task_id, "indexing"):
                        changed = self._mark_terminal(
                            task, doc, "stale", "document changed before indexing"
                        )
                        if changed:
                            self.chunk_repository.discard_task(task_id)
                            self._remove_prepared_artifact(engine, prepared_artifact_path)
                            self._discard_version(
                                task, version, "stale", doc, status_already_updated=True
                            )
                        return
                else:
                    self.store.update_task(task_id, status="indexing")
                    self.store.update_document(doc["id"], status="indexing")

                staged_child_ids = [chunk.chunk_id for chunk in children]
                engine.stage_document_children(doc["kb_id"], doc["id"], children)
                latest_doc = self.store.get_document(task["kb_id"], task["doc_id"])
                if (
                    not latest_doc
                    or latest_doc.get("task_id") != task_id
                    or latest_doc.get("status") == "deleting"
                ):
                    self._rollback_structured_index(
                        engine,
                        doc["kb_id"],
                        staged_child_ids,
                        task_id,
                        prepared_artifact_path,
                        self.chunk_repository,
                    )
                    changed = self._mark_terminal(
                        task, latest_doc or doc, "stale", "document changed before commit"
                    )
                    if changed:
                        self._discard_version(
                            task,
                            version,
                            "stale",
                            latest_doc or doc,
                            status_already_updated=True,
                        )
                    return

                self.chunk_repository.activate_version(
                    doc["id"], int(task["document_version"]), task_id
                )
                try:
                    engine.remove_children(doc["kb_id"], old_child_ids)
                except Exception as cleanup_exc:
                    logger.warning(
                        "active version committed but old child cleanup failed: task=%s error=%s",
                        task_id,
                        cleanup_exc,
                    )
                logger.info(
                    "structured indexing task completed: task=%s doc=%s children=%d",
                    task_id,
                    doc["id"],
                    len(children),
                )
                return

            if hasattr(self.store, "set_indexing_phase"):
                if not self.store.set_indexing_phase(task_id, "indexing"):
                    changed = self._mark_terminal(task, doc, "stale", "document changed before indexing")
                    if changed:
                        self._discard_version(task, version, "stale", doc, status_already_updated=True)
                    return
            else:
                self.store.update_task(task_id, status="indexing")
                self.store.update_document(doc["id"], status="indexing")
            chunk_count = engine.index_document(
                version["storage_path"],
                doc["kb_id"],
                display_name=version["filename"],
                doc_id=doc["id"],
                task_id=task_id,
            )

            latest_doc = self.store.get_document(task["kb_id"], task["doc_id"])
            if not latest_doc or latest_doc.get("task_id") != task_id or latest_doc.get("status") == "deleting":
                try:
                    engine.remove_document(version["filename"], doc["kb_id"], doc_id=doc["id"], task_id=task_id)
                finally:
                    changed = self._mark_terminal(task, latest_doc or doc, "stale", "document changed before commit")
                    if changed:
                        self._discard_version(task, version, "stale", latest_doc or doc, status_already_updated=True)
                return

            if hasattr(self.store, "complete_indexing_task"):
                self.store.complete_indexing_task(task_id, chunk_count)
            else:
                self.store.update_document(
                    doc["id"], status="ready", chunk_count=chunk_count,
                    error_message=None, updated_at=now_iso(),
                )
                self.store.update_task(task_id, status="ready", error_message=None)
            logger.info("索引任务完成: task=%s doc=%s chunks=%d", task_id, doc["id"], chunk_count)
        except Exception as exc:
            completion_state = self._completion_state(task)
            if completion_state is True:
                logger.warning("索引任务提交后返回异常，权威状态已确认成功: task=%s error=%s", task_id, exc)
                return
            if completion_state is None:
                logger.exception("索引任务提交结果无法确认，保留现场等待恢复: task=%s", task_id)
                return
            error = str(exc)
            try:
                if structured_mode:
                    cleanup_error = self._rollback_structured_index(
                        engine,
                        doc["kb_id"],
                        staged_child_ids,
                        task_id,
                        prepared_artifact_path,
                        self.chunk_repository,
                    )
                    if cleanup_error:
                        error = f"{error}; rollback failed: {cleanup_error}"
                else:
                    engine.remove_document(
                        version["filename"], doc["kb_id"],
                        doc_id=doc["id"], task_id=task_id,
                    )
            except Exception as cleanup_exc:
                logger.warning("索引失败后回滚向量失败: task=%s error=%s", task_id, cleanup_exc)
                error = f"{error}; rollback failed: {cleanup_exc}"
            changed = self._mark_terminal(task, doc, "failed", error)
            if changed:
                self._discard_version(task, version, "failed", doc, status_already_updated=True)
            logger.exception("索引任务失败: task=%s doc=%s", task_id, doc["id"])

    @staticmethod
    def _rollback_structured_index(
        engine,
        kb_id: str,
        child_ids: list[str],
        task_id: str,
        artifact_path: str | None,
        repository=None,
    ) -> str | None:
        errors = []
        try:
            engine.remove_children(kb_id, child_ids)
        except Exception as exc:
            errors.append(str(exc))
        if repository is not None:
            try:
                repository.discard_task(task_id)
            except Exception as exc:
                errors.append(str(exc))
        IndexingWorker._remove_prepared_artifact(engine, artifact_path)
        return "; ".join(errors) or None

    @staticmethod
    def _remove_prepared_artifact(engine, artifact_path: str | None) -> None:
        if artifact_path and hasattr(engine, "remove_parsed_artifact"):
            try:
                engine.remove_parsed_artifact(artifact_path)
            except Exception as exc:
                logger.warning("failed to remove parsed artifact %s: %s", artifact_path, exc)

    def _completion_state(self, task: dict) -> bool | None:
        try:
            return bool(self.store.is_indexing_completion_committed(task["task_id"]))
        except Exception as exc:
            logger.warning("索引异常后读取权威状态失败: task=%s error=%s", task["task_id"], exc)
            return None

    def _task_version(self, task: dict, document: dict) -> dict:
        version_number = task.get("document_version")
        if version_number is None or not hasattr(self.store, "get_document_version"):
            return document
        version = self.store.get_document_version(task["doc_id"], version_number)
        if not version:
            raise RuntimeError("document version metadata missing")
        return version

    def _fail_unusable_version(self, task: dict, document: dict, error: str) -> None:
        self._mark_terminal(task, document, "failed", error)

    def _mark_terminal(self, task: dict, document: dict, status: str, error: str) -> bool:
        if hasattr(self.store, "finish_indexing_task"):
            return bool(self.store.finish_indexing_task(task["task_id"], status, error))
        latest = self.store.get_document(task["kb_id"], task["doc_id"])
        owns_document = bool(
            latest and latest.get("task_id") == task["task_id"] and latest.get("status") != "deleting"
        )
        if owns_document:
            fallback_status = "ready" if int(latest.get("current_version") or 0) > 0 else "failed"
            self.store.update_document(document["id"], status=fallback_status, error_message=error)
        self.store.update_task(task["task_id"], status=status, error_message=error)
        version_number = task.get("document_version")
        if version_number is not None and hasattr(self.store, "update_document_version_status"):
            self.store.update_document_version_status(task["doc_id"], version_number, status)
        return owns_document

    def _optional_task_version(self, task: dict) -> dict | None:
        version_number = task.get("document_version")
        if version_number is None or not hasattr(self.store, "get_document_version"):
            return None
        return self.store.get_document_version(task["doc_id"], version_number)

    def _discard_version(
        self, task: dict, version: dict, status: str, document: dict, *, status_already_updated: bool = False
    ):
        version_number = task.get("document_version")
        latest_document = self.store.get_document(task["kb_id"], task["doc_id"])
        latest_version = self._optional_task_version(task)
        if latest_version and latest_version.get("status") == "active":
            return
        if latest_document and version_number is not None:
            if int(latest_document.get("current_version") or 0) == int(version_number):
                return
            latest_active_path = latest_document.get("storage_path")
            if latest_active_path and version.get("storage_path"):
                if Path(latest_active_path).resolve() == Path(version["storage_path"]).resolve():
                    return
        if not status_already_updated and version_number is not None and hasattr(self.store, "update_document_version_status"):
            self.store.update_document_version_status(task["doc_id"], version_number, status)
        version_path = version.get("storage_path")
        active_path = document.get("storage_path") if int(document.get("current_version") or 0) > 0 else None
        if not version_path or (active_path and Path(version_path).resolve() == Path(active_path).resolve()):
            return
        directory = self._safe_version_directory(task, version_path)
        if directory is not None and directory.exists():
            shutil.rmtree(directory, ignore_errors=True)

    @staticmethod
    def _safe_version_directory(task: dict, storage_path: str) -> Path | None:
        path = Path(storage_path).resolve()
        directory = path.parent
        upload_root = Path(UPLOAD_STORAGE_DIR).resolve()
        try:
            directory.relative_to(upload_root)
        except ValueError:
            return None
        version = task.get("document_version")
        allowed_directories = {str(task["task_id"])}
        if version is not None:
            allowed_directories.add(f"v{version}")
        if (
            directory.name not in allowed_directories
            or directory.parent.name != str(task["doc_id"])
            or directory.parent.parent.name != str(task["kb_id"])
        ):
            return None
        return directory


_indexing_worker: IndexingWorker | None = None


def get_indexing_worker() -> IndexingWorker:
    global _indexing_worker
    if _indexing_worker is None:
        _indexing_worker = IndexingWorker()
    return _indexing_worker
