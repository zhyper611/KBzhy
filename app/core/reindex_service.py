from __future__ import annotations

import argparse
import logging
import re
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime

from KBzhy.app.core.chunk_repository import ChunkRepository
from KBzhy.app.core.engine import get_rag_engine
from KBzhy.app.core.metadata_store import get_metadata_store, now_iso

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ReindexResult:
    kb_id: str
    collection_name: str
    document_count: int
    child_count: int
    status: str = "ready"


class ReindexCompletionUnknown(RuntimeError):
    pass


class ReindexService:
    def __init__(self, store=None, engine=None, chunk_repository=None, index_version: int = 1):
        self.store = store or get_metadata_store()
        self.engine = engine or get_rag_engine()
        self.chunk_repository = chunk_repository or ChunkRepository(self.store)
        self.index_version = index_version

    def reindex(self, kb_id: str) -> ReindexResult:
        kb = self.store.get_kb(kb_id)
        if not kb:
            raise ValueError(f"knowledge base does not exist: {kb_id}")
        documents = self._ready_documents(kb_id)
        if not documents:
            raise ValueError("knowledge base has no ready documents")
        collection_name = self._new_collection_name(kb_id)
        manifests: list[dict] = []
        task_ids: list[str] = []
        artifact_paths: list[str] = []
        try:
            for document in documents:
                task_id = f"reindex-{uuid.uuid4().hex}"
                task_ids.append(task_id)
                snapshot = self.store.create_reindex_task(
                    kb_id,
                    document["id"],
                    task_id,
                    self.index_version,
                    now_iso(),
                )
                version = int(snapshot["document_version"])
                prepared = self.engine.prepare_document_index(
                    snapshot["storage_path"],
                    kb_id,
                    document_id=document["id"],
                    document_version=version,
                    index_version=self.index_version,
                    display_name=snapshot["filename"],
                )
                artifact_paths.append(prepared.artifact_path)
                chunks = list(prepared.chunks)
                children = [chunk for chunk in chunks if chunk.chunk_type == "child"]
                if not children:
                    raise RuntimeError(f"document produced no child chunks: {document['id']}")
                self.chunk_repository.replace_reindex_staging(
                    task_id, document["id"], version, chunks
                )
                self.engine.stage_collection_children(
                    collection_name, kb_id, document["id"], children
                )
                manifests.append({
                    "document_id": document["id"],
                    "document_version": version,
                    "owner_task_id": snapshot.get("owner_task_id"),
                    "task_id": task_id,
                    "index_version": self.index_version,
                    "child_count": len(children),
                    "artifact_path": prepared.artifact_path,
                })
        except Exception:
            self._cleanup_failed_reindex(collection_name, task_ids, artifact_paths)
            raise
        try:
            self.store.activate_reindex(kb_id, collection_name, manifests)
        except Exception as activation_error:
            try:
                committed = self.store.is_reindex_committed(
                    kb_id, collection_name, task_ids
                )
            except Exception as verification_error:
                logger.error(
                    "reindex activation result is unknown; preserving temporary state: kb=%s collection=%s error=%s",
                    kb_id, collection_name, verification_error,
                )
                raise ReindexCompletionUnknown(
                    "reindex activation result could not be verified"
                ) from activation_error
            if not committed:
                self._cleanup_failed_reindex(collection_name, task_ids, artifact_paths)
                raise
            logger.warning(
                "reindex activation committed but acknowledgement failed: kb=%s collection=%s",
                kb_id, collection_name,
            )
        return ReindexResult(
            kb_id=kb_id,
            collection_name=collection_name,
            document_count=len(manifests),
            child_count=sum(item["child_count"] for item in manifests),
        )

    def cleanup_collection(self, kb_id: str, collection_name: str) -> None:
        kb = self.store.get_kb(kb_id)
        if not kb:
            raise ValueError(f"knowledge base does not exist: {kb_id}")
        base_name = f"kbzhy_{self._safe_kb_id(kb_id)}"
        if collection_name != base_name and not collection_name.startswith(f"{base_name}_v2_"):
            raise ValueError("collection does not belong to knowledge base")
        active_collection = (
            self.store.get_active_collection_name(kb_id)
            if hasattr(self.store, "get_active_collection_name")
            else kb.get("active_collection_name")
        )
        active_collection = active_collection or base_name
        if active_collection == collection_name:
            raise ValueError("cannot delete active collection")
        self.engine.delete_collection(collection_name)

    def _ready_documents(self, kb_id: str) -> list[dict]:
        documents = []
        page = 1
        while True:
            total, batch = self.store.list_documents(kb_id, page, 100, status="ready")
            documents.extend(batch)
            if len(documents) >= total or not batch:
                return documents
            page += 1

    def _cleanup_failed_reindex(
        self,
        collection_name: str,
        task_ids: list[str],
        artifact_paths: list[str],
    ) -> None:
        try:
            self.engine.delete_collection(collection_name)
        except Exception as exc:
            logger.warning("failed to delete temporary reindex collection: %s", exc)
        try:
            self.store.abort_reindex(task_ids)
        except Exception as exc:
            logger.warning("failed to discard reindex staging rows: %s", exc)
        for artifact_path in artifact_paths:
            try:
                self.engine.remove_parsed_artifact(artifact_path)
            except Exception as exc:
                logger.warning("failed to delete reindex artifact %s: %s", artifact_path, exc)

    @staticmethod
    def _new_collection_name(kb_id: str) -> str:
        safe_kb_id = ReindexService._safe_kb_id(kb_id)
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
        return f"kbzhy_{safe_kb_id}_v2_{timestamp}"

    @staticmethod
    def _safe_kb_id(kb_id: str) -> str:
        return re.sub(r"[^a-zA-Z0-9_-]", "-", kb_id).strip("-_") or "kb"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Safely reindex a knowledge base")
    parser.add_argument("--kb-id", required=True)
    args = parser.parse_args(argv)
    try:
        result = ReindexService().reindex(args.kb_id)
    except Exception as exc:
        logger.exception("knowledge base reindex failed: kb=%s", args.kb_id)
        print(str(exc), file=sys.stderr)
        return 1
    print(
        f"reindexed {result.document_count} documents and {result.child_count} children "
        f"into {result.collection_name}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
