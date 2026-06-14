from __future__ import annotations

import logging
import queue
import threading
from typing import Callable

from KBzhy.app.core.engine import get_rag_engine
from KBzhy.app.core.metadata_store import get_metadata_store, now_iso

logger = logging.getLogger(__name__)


class IndexingWorker:
    def __init__(self, store=None, engine_factory: Callable | None = None, autostart: bool = True):
        self.store = store or get_metadata_store()
        self.engine_factory = engine_factory or get_rag_engine
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
            try:
                self.engine_factory().remove_document(doc["filename"], doc["kb_id"], doc_id=doc["id"], task_id=task_id)
            except Exception as exc:
                logger.warning("恢复任务时清理旧向量失败: task=%s error=%s", task_id, exc)
            self.store.update_document(doc["id"], status="queued", error_message=None)
            self.store.update_task(task_id, status="queued", error_message=None)
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
        if doc.get("task_id") != task_id:
            self.store.update_task(task_id, status="stale", error_message="document has a newer indexing task")
            logger.info("跳过陈旧索引任务: task=%s doc=%s current_task=%s", task_id, doc["id"], doc.get("task_id"))
            return
        if hasattr(self.store, "claim_task") and not self.store.claim_task(task_id):
            logger.info("索引任务已被其他 worker 认领或状态已变化: task=%s", task_id)
            return

        engine = self.engine_factory()
        try:
            self.store.update_document(doc["id"], status="parsing", error_message=None)

            self.store.update_task(task_id, status="indexing")
            self.store.update_document(doc["id"], status="indexing")
            chunk_count = engine.index_document(
                doc["storage_path"],
                doc["kb_id"],
                display_name=doc["filename"],
                doc_id=doc["id"],
                task_id=task_id,
            )

            latest_doc = self.store.get_document(task["kb_id"], task["doc_id"])
            if not latest_doc or latest_doc.get("task_id") != task_id or latest_doc.get("status") == "deleting":
                try:
                    engine.remove_document(doc["filename"], doc["kb_id"], doc_id=doc["id"], task_id=task_id)
                finally:
                    self.store.update_task(task_id, status="stale", error_message="document changed before commit")
                return

            self.store.update_document(
                doc["id"],
                status="ready",
                chunk_count=chunk_count,
                error_message=None,
                updated_at=now_iso(),
            )
            self.store.update_task(task_id, status="ready", error_message=None)
            logger.info("索引任务完成: task=%s doc=%s chunks=%d", task_id, doc["id"], chunk_count)
        except Exception as exc:
            error = str(exc)
            try:
                engine.remove_document(doc["filename"], doc["kb_id"], doc_id=doc["id"], task_id=task_id)
            except Exception as cleanup_exc:
                logger.warning("索引失败后回滚向量失败: task=%s error=%s", task_id, cleanup_exc)
                error = f"{error}; rollback failed: {cleanup_exc}"
            self.store.update_document(doc["id"], status="failed", error_message=error)
            self.store.update_task(task_id, status="failed", error_message=error)
            logger.exception("索引任务失败: task=%s doc=%s", task_id, doc["id"])


_indexing_worker: IndexingWorker | None = None


def get_indexing_worker() -> IndexingWorker:
    global _indexing_worker
    if _indexing_worker is None:
        _indexing_worker = IndexingWorker()
    return _indexing_worker
