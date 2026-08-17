"""混合检索 + MMR + Reranker + 查询扩展 + 子问题拆解（多知识库支持）"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from itertools import zip_longest
from typing import Any

import httpx
import numpy as np
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from openai import OpenAI
from rank_bm25 import BM25Okapi

from KBzhy.config import (
    API_KEY,
    API_BASE,
    LLM_MODEL,
    EMBEDDING_MODEL,
    RERANKER_MODEL,
    TOP_K,
    FETCH_K,
    SIMILARITY_THRESHOLD,
    BM25_WEIGHT,
    VECTOR_WEIGHT,
    VECTOR_FETCH_K,
    BM25_FETCH_K,
    RRF_K,
    RRF_CANDIDATE_K,
    RERANK_CANDIDATE_K,
    MODEL_RERANK_THRESHOLD,
    KEYWORD_RERANK_THRESHOLD,
    CONNECT_TIMEOUT,
    READ_TIMEOUT,
    CHROMA_PERSIST_DIR,
)
from KBzhy.app.core.document_models import (
    KnowledgeChunk,
    RetrievalCandidate,
    RerankResult,
)
from KBzhy.app.core.splitter import Chunk
from KBzhy.app.core.timing import timed_stage

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = (CONNECT_TIMEOUT, READ_TIMEOUT)
_KB_COLLECTION_PREFIX = "kbzhy_"
_EMBEDDING_BATCH_SIZE = 10


def rrf_fuse(
    vector_candidates: list[RetrievalCandidate],
    bm25_candidates: list[RetrievalCandidate],
    *,
    k: int = RRF_K,
    limit: int | None = None,
) -> list[RetrievalCandidate]:
    if k <= 0:
        raise ValueError("rrf k must be positive")
    fused: dict[str, RetrievalCandidate] = {}
    first_seen: dict[str, int] = {}
    sequence = 0
    for channel, candidates in (("vector", vector_candidates), ("bm25", bm25_candidates)):
        seen_in_channel = set()
        for rank, candidate in enumerate(candidates, start=1):
            if candidate.chunk_id in seen_in_channel:
                continue
            seen_in_channel.add(candidate.chunk_id)
            if candidate.chunk_id not in first_seen:
                first_seen[candidate.chunk_id] = sequence
                sequence += 1
            existing = fused.get(candidate.chunk_id, candidate)
            score = existing.rrf_score + 1.0 / (k + rank)
            if channel == "vector":
                existing = replace(
                    existing,
                    vector_rank=rank,
                    vector_score=candidate.vector_score,
                    rrf_score=score,
                )
            else:
                existing = replace(
                    existing,
                    bm25_rank=rank,
                    bm25_score=candidate.bm25_score,
                    rrf_score=score,
                )
            fused[candidate.chunk_id] = existing
    ranked = sorted(
        fused.values(),
        key=lambda item: (-item.rrf_score, first_seen[item.chunk_id], item.chunk_id),
    )
    return ranked[:limit] if limit is not None else ranked

# 共享 httpx 客户端：禁用 HTTP/2 避免 SSL EOF 错误
_http_client = httpx.Client(
    http2=False,
    timeout=httpx.Timeout(READ_TIMEOUT, connect=CONNECT_TIMEOUT),
    limits=httpx.Limits(max_keepalive_connections=5, max_connections=20),
)


class _BailianEmbeddings(Embeddings):
    """阿里云百炼 Embedding 封装（兼容 LangChain Embeddings 接口）"""

    def __init__(self, model: str, api_key: str, base_url: str, timeout: tuple[int, int]):
        self.model = model
        self._client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        embeddings: list[list[float]] = []
        for start in range(0, len(texts), _EMBEDDING_BATCH_SIZE):
            batch = texts[start:start + _EMBEDDING_BATCH_SIZE]
            result = self._client.embeddings.create(model=self.model, input=batch)
            embeddings.extend(item.embedding for item in result.data)
        return embeddings

    def embed_query(self, text: str) -> list[float]:
        result = self._client.embeddings.create(model=self.model, input=text)
        return result.data[0].embedding


class Retriever:
    """混合检索器：BM25 + 向量 + MMR + Reranker，支持多知识库隔离"""

    def __init__(
        self,
        persist_dir: str | None = None,
        embedding_model: str | None = None,
        top_k: int = TOP_K,
        fetch_k: int = FETCH_K,
        threshold: float = SIMILARITY_THRESHOLD,
        vector_fetch_k: int = VECTOR_FETCH_K,
        bm25_fetch_k: int = BM25_FETCH_K,
        rrf_k: int = RRF_K,
        rrf_candidate_k: int = RRF_CANDIDATE_K,
        rerank_candidate_k: int = RERANK_CANDIDATE_K,
        model_rerank_threshold: float = MODEL_RERANK_THRESHOLD,
        keyword_rerank_threshold: float = KEYWORD_RERANK_THRESHOLD,
    ):
        self.persist_dir = persist_dir or CHROMA_PERSIST_DIR
        self.top_k = top_k
        self.fetch_k = fetch_k
        self.threshold = threshold
        self.vector_fetch_k = vector_fetch_k
        self.bm25_fetch_k = bm25_fetch_k
        self.rrf_k = rrf_k
        self.rrf_candidate_k = rrf_candidate_k
        self.rerank_candidate_k = rerank_candidate_k
        self.model_rerank_threshold = model_rerank_threshold
        self.keyword_rerank_threshold = keyword_rerank_threshold
        self._lock = threading.Lock()

        emb_model = embedding_model or EMBEDDING_MODEL
        self.embeddings = _BailianEmbeddings(
            model=emb_model,
            api_key=API_KEY,
            base_url=API_BASE,
            timeout=_DEFAULT_TIMEOUT,
        )
        # kb_id → Chroma 实例
        self._vectorstores: dict[str, Chroma] = {}
        self._vectorstore_names: dict[str, str] = {}
        # kb_id → (BM25Okapi, list[{content, metadata}])
        self._bm25_indices: dict[str, tuple[BM25Okapi | None, list[dict[str, Any]]]] = {}
        self._active_version_resolver = self._load_active_versions
        self._active_collection_resolver = self._load_active_collection

    # ── 知识库管理 ────────────────────────────────

    def _collection_name(self, kb_id: str) -> str:
        fallback = f"{_KB_COLLECTION_PREFIX}{kb_id}"
        try:
            return self._active_collection_resolver(kb_id) or fallback
        except Exception as exc:
            logger.error("failed to resolve active collection: kb=%s error=%s", kb_id, exc)
            if kb_id in self._vectorstore_names:
                return self._vectorstore_names[kb_id]
            raise

    def _get_named_vectorstore(self, collection_name: str) -> Chroma:
        return Chroma(
            persist_directory=self.persist_dir,
            embedding_function=self.embeddings,
            collection_name=collection_name,
        )

    def _get_vectorstore(self, kb_id: str) -> Chroma:
        collection_name = self._collection_name(kb_id)
        if self._vectorstore_names.get(kb_id) != collection_name:
            self._vectorstores[kb_id] = self._get_named_vectorstore(collection_name)
            self._vectorstore_names[kb_id] = collection_name
            self._bm25_indices.pop(kb_id, None)
        return self._vectorstores[kb_id]

    def create_kb(self, kb_id: str):
        """初始化知识库的 collection"""
        self._get_vectorstore(kb_id)
        self._bm25_indices.setdefault(kb_id, (None, []))
        logger.info("知识库已创建: %s", kb_id)

    def delete_kb(self, kb_id: str):
        """删除知识库的 collection 和 BM25 索引"""
        # 先从内存缓存中删除
        if kb_id in self._vectorstores:
            try:
                self._vectorstores[kb_id].delete_collection()
            except Exception as exc:
                logger.warning("删除 ChromaDB collection 失败: %s", exc)
            del self._vectorstores[kb_id]
            self._vectorstore_names.pop(kb_id, None)
        # 确保 ChromaDB 持久化 collection 也被删除（处理重启后缓存丢失的情况）
        try:
            import chromadb
            client = chromadb.PersistentClient(path=self.persist_dir)
            try:
                client.delete_collection(self._collection_name(kb_id))
            except Exception:
                pass
        except Exception as exc:
            logger.warning("清理 ChromaDB 持久化 collection 失败: %s", exc)
        self._bm25_indices.pop(kb_id, None)
        logger.info("知识库已删除: %s", kb_id)

    def list_kbs(self) -> list[str]:
        """列出所有已创建的知识库 ID"""
        import chromadb
        client = chromadb.PersistentClient(path=self.persist_dir)
        kb_ids = set()
        for col in client.list_collections():
            if col.name.startswith(_KB_COLLECTION_PREFIX):
                kb_ids.add(col.name[len(_KB_COLLECTION_PREFIX):])
        for kb_id in self._bm25_indices:
            kb_ids.add(kb_id)
        return sorted(kb_ids)

    def get_kb_doc_count(self, kb_id: str) -> int:
        """获取知识库中的 chunk 数量"""
        try:
            vs = self._get_vectorstore(kb_id)
            return vs._collection.count()
        except Exception:
            return 0

    def list_document_chunks(self, kb_id: str, source: str | None = None, doc_id: str | None = None) -> list[dict[str, Any]]:
        """Return all stored chunks for one source document."""
        try:
            vs = self._get_vectorstore(kb_id)
            where = {"doc_id": doc_id} if doc_id else {"source": source}
            data = vs.get(where=where)
        except Exception as exc:
            logger.warning("获取文档 chunks 失败: %s", exc)
            return []

        documents = data.get("documents") or []
        metadatas = data.get("metadatas") or []
        chunks: list[dict[str, Any]] = []
        for index, (content, metadata) in enumerate(zip_longest(documents, metadatas, fillvalue={}), start=1):
            if not content:
                continue
            meta = dict(metadata or {})
            raw_index = meta.get("chunk_index")
            chunk_index = raw_index if isinstance(raw_index, int) else index
            chunks.append({
                "chunk_index": chunk_index,
                "content": content,
                "metadata": meta,
            })
        return sorted(chunks, key=lambda item: item["chunk_index"])

    # ── 索引 ────────────────────────────────────

    def add_documents(self, chunks: list[Chunk], kb_id: str):
        """将切分后的文本块加入指定知识库的向量库和 BM25 索引"""
        from langchain_core.documents import Document as LCDocument

        for ch in chunks:
            ch.metadata["kb_id"] = kb_id

        docs = [
            LCDocument(page_content=ch.content, metadata=ch.metadata)
            for ch in chunks
        ]
        vs = self._get_vectorstore(kb_id)
        vs.add_documents(docs)

        with self._lock:
            _, bm25_docs = self._bm25_indices.get(kb_id, (None, []))
            bm25_docs = list(bm25_docs)
            bm25_docs.extend([
                {"content": ch.content, "metadata": dict(ch.metadata)}
                for ch in chunks
            ])
            self._bm25_indices[kb_id] = (None, bm25_docs)
            self._rebuild_bm25(kb_id)
        logger.info("知识库 %s 已索引 %d 个文本块", kb_id, len(chunks))

    def stage_document_children(
        self,
        kb_id: str,
        document_id: str,
        new_children: list[KnowledgeChunk],
    ) -> None:
        if any(
            chunk.chunk_type != "child" or chunk.document_id != document_id
            for chunk in new_children
        ):
            raise ValueError("only children belonging to the target document may be staged")
        if not new_children:
            return

        from langchain_core.documents import Document as LCDocument

        ids = [chunk.chunk_id for chunk in new_children]
        documents = []
        entries = []
        for chunk in new_children:
            metadata = self._normalize_chroma_metadata(chunk.metadata)
            metadata.update({
                "kb_id": kb_id,
                "chunk_id": chunk.chunk_id,
                "doc_id": chunk.document_id,
                "document_version": chunk.document_version,
                "parent_chunk_id": chunk.parent_chunk_id or "",
                "section_path": json.dumps(
                    list(chunk.section_path), ensure_ascii=False, separators=(",", ":")
                ),
                "position": chunk.position,
                "index_version": chunk.index_version,
            })
            if chunk.page_start is not None:
                metadata["page_start"] = chunk.page_start
                metadata.setdefault("page", chunk.page_start)
            if chunk.page_end is not None:
                metadata["page_end"] = chunk.page_end
            documents.append(
                LCDocument(page_content=chunk.retrieval_text, metadata=metadata)
            )
            entries.append({"content": chunk.retrieval_text, "metadata": metadata})

        self._get_vectorstore(kb_id).add_documents(documents, ids=ids)
        id_set = set(ids)
        with self._lock:
            _, existing = self._bm25_indices.get(kb_id, (None, []))
            retained = [
                entry
                for entry in existing
                if self._bm25_metadata(entry).get("chunk_id") not in id_set
            ]
            self._bm25_indices[kb_id] = (None, retained + entries)
            self._rebuild_bm25(kb_id)

    def stage_collection_children(
        self,
        collection_name: str,
        kb_id: str,
        document_id: str,
        new_children: list[KnowledgeChunk],
    ) -> None:
        if any(
            chunk.chunk_type != "child" or chunk.document_id != document_id
            for chunk in new_children
        ):
            raise ValueError("only children belonging to the target document may be staged")
        if not new_children:
            return
        from langchain_core.documents import Document as LCDocument

        ids = [chunk.chunk_id for chunk in new_children]
        documents = []
        for chunk in new_children:
            metadata = self._normalize_chroma_metadata(chunk.metadata)
            metadata.update({
                "kb_id": kb_id,
                "chunk_id": chunk.chunk_id,
                "doc_id": chunk.document_id,
                "document_version": chunk.document_version,
                "parent_chunk_id": chunk.parent_chunk_id or "",
                "section_path": json.dumps(list(chunk.section_path), ensure_ascii=False),
                "position": chunk.position,
                "index_version": chunk.index_version,
            })
            if chunk.page_start is not None:
                metadata["page_start"] = chunk.page_start
                metadata.setdefault("page", chunk.page_start)
            if chunk.page_end is not None:
                metadata["page_end"] = chunk.page_end
            documents.append(LCDocument(page_content=chunk.retrieval_text, metadata=metadata))
        self._get_named_vectorstore(collection_name).add_documents(documents, ids=ids)

    def delete_collection(self, collection_name: str) -> None:
        import chromadb

        client = chromadb.PersistentClient(path=self.persist_dir)
        client.delete_collection(collection_name)

    def remove_children(self, kb_id: str, chunk_ids: list[str]) -> None:
        ids = list(dict.fromkeys(chunk_ids))
        if not ids:
            return
        self._get_vectorstore(kb_id).delete(ids=ids)
        id_set = set(ids)
        with self._lock:
            _, existing = self._bm25_indices.get(kb_id, (None, []))
            retained = [
                entry
                for entry in existing
                if self._bm25_metadata(entry).get("chunk_id") not in id_set
            ]
            if retained:
                self._bm25_indices[kb_id] = (None, retained)
                self._rebuild_bm25(kb_id)
            else:
                self._bm25_indices.pop(kb_id, None)

    @staticmethod
    def _normalize_chroma_metadata(metadata: dict[str, Any]) -> dict[str, str | int | float | bool]:
        normalized: dict[str, str | int | float | bool] = {}
        for key, value in metadata.items():
            if value is None:
                continue
            if isinstance(value, (str, int, float, bool)):
                normalized[str(key)] = value
            elif isinstance(value, (dict, list, tuple)):
                normalized[str(key)] = json.dumps(
                    value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                )
            else:
                normalized[str(key)] = str(value)
        return normalized

    def remove_document(
        self,
        kb_id: str,
        source: str | None = None,
        doc_id: str | None = None,
        task_id: str | None = None,
    ):
        """从指定知识库中删除文档的 chunks"""
        try:
            vs = self._get_vectorstore(kb_id)
        except Exception as exc:
            logger.warning("打开知识库 %s 的 Chroma collection 失败: %s", kb_id, exc)
            return
        where = {"task_id": task_id} if task_id else {"doc_id": doc_id} if doc_id else {"source": source} if source else None
        if where:
            results = vs.get(where=where)
            ids_to_delete = results.get("ids", [])
            if ids_to_delete:
                vs.delete(ids=ids_to_delete)
                logger.info("知识库 %s 已删除 %s 的 %d 个 chunks", kb_id, task_id or doc_id or source, len(ids_to_delete))

        # 同时清理 BM25 索引 — 从 ChromaDB 重建以排除已删除文档
        if where and kb_id in self._bm25_indices:
            with self._lock:
                remaining_entries = self._entries_from_chroma(vs.get())
                if remaining_entries:
                    self._bm25_indices[kb_id] = (None, remaining_entries)
                    self._rebuild_bm25(kb_id)
                else:
                    self._bm25_indices.pop(kb_id, None)

    def _rebuild_bm25(self, kb_id: str):
        import jieba
        entry = self._bm25_indices.get(kb_id)
        if not entry:
            return
        _, bm25_docs = entry
        if not bm25_docs:
            return
        tokenized = [list(jieba.cut(self._bm25_content(doc))) for doc in bm25_docs]
        self._bm25_indices[kb_id] = (BM25Okapi(tokenized), bm25_docs)

    @staticmethod
    def _bm25_content(entry: str | dict[str, Any]) -> str:
        if isinstance(entry, dict):
            return str(entry.get("content", ""))
        return str(entry)

    @staticmethod
    def _bm25_metadata(entry: str | dict[str, Any]) -> dict:
        if isinstance(entry, dict):
            return dict(entry.get("metadata") or {})
        return {}

    def _entries_from_chroma(self, data: dict | None) -> list[dict[str, Any]]:
        if not data:
            return []
        documents = data.get("documents") or []
        metadatas = data.get("metadatas") or []
        entries: list[dict[str, Any]] = []
        for content, metadata in zip_longest(documents, metadatas, fillvalue={}):
            if content:
                entries.append({"content": content, "metadata": dict(metadata or {})})
        return entries

    def _load_bm25_from_vectorstore(self, kb_id: str):
        try:
            vs = self._get_vectorstore(kb_id)
            entries = self._entries_from_chroma(vs.get())
        except Exception as exc:
            logger.warning("从 ChromaDB 重建 BM25 失败: %s", exc)
            return
        if not entries:
            return
        with self._lock:
            self._bm25_indices[kb_id] = (None, entries)
            self._rebuild_bm25(kb_id)

    # ── 检索主流程 ──────────────────────────────

    def retrieve(
        self,
        query: str,
        kb_id: str,
        top_k: int | None = None,
        rerank_method: str = "model",
        enable_expansion: bool = True,
        enable_decomposition: bool = True,
        threshold: float | None = None,
        on_status: callable = None,
        request_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """混合检索主入口"""
        with timed_stage(logger, "retrieve_total", request_id=request_id, kb_id=kb_id, top_k=top_k or self.top_k):
            return self._retrieve(
                query=query,
                kb_id=kb_id,
                top_k=top_k,
                rerank_method=rerank_method,
                enable_expansion=enable_expansion,
                enable_decomposition=enable_decomposition,
                threshold=threshold,
                on_status=on_status,
                request_id=request_id,
            )

    def _retrieve(
        self,
        query: str,
        kb_id: str,
        top_k: int | None,
        rerank_method: str,
        enable_expansion: bool,
        enable_decomposition: bool,
        threshold: float | None,
        on_status: callable = None,
        request_id: str | None = None,
    ) -> list[RetrievalCandidate]:
        tk = top_k or self.top_k

        queries = [query]
        if enable_expansion:
            if on_status:
                on_status("expand", "正在扩展查询...")
            with timed_stage(logger, "query_expand", request_id=request_id):
                expanded = self._expand_query(query)
            if expanded:
                queries = expanded
                logger.info("查询扩展: %s → %d 个表述", query, len(expanded))

        if enable_decomposition and self._is_complex(query):
            if on_status:
                on_status("expand", "正在拆解子问题...")
            with timed_stage(logger, "query_decompose", request_id=request_id):
                sub = self._decompose_query(query)
            if sub:
                queries = list(set(queries + sub))
                logger.info("子问题拆解: 共 %d 个查询", len(queries))

        if on_status:
            on_status("retrieve", "正在检索相关知识...")

        all_candidates: dict[str, RetrievalCandidate] = {}
        with timed_stage(logger, "hybrid_search_all", request_id=request_id, query_count=len(queries)):
            for q in queries:
                candidates = self._hybrid_search(q, kb_id, tk, request_id=request_id)
                for candidate in candidates:
                    if not isinstance(candidate, RetrievalCandidate):
                        content, meta, score = candidate
                        candidate = replace(
                            self._candidate_from_result(content, meta), rrf_score=score
                        )
                    existing = all_candidates.get(candidate.chunk_id)
                    if existing is None or candidate.rrf_score > existing.rrf_score:
                        all_candidates[candidate.chunk_id] = candidate

        if not all_candidates:
            return []

        if on_status:
            on_status("rerank", "正在重排序结果...")

        rerank_input = sorted(
            all_candidates.values(), key=lambda item: item.rrf_score, reverse=True
        )[:self.rerank_candidate_k]
        with timed_stage(logger, "rerank", request_id=request_id, method=rerank_method, candidates=len(rerank_input)):
            rerank_result = self._rerank(query, rerank_input, rerank_method)
        if rerank_result.method in {"model", "llm"}:
            applied_threshold = (
                threshold if threshold is not None else self.model_rerank_threshold
            )
        else:
            applied_threshold = self.keyword_rerank_threshold
        final_filtered = [
            item for item in rerank_result.items if item.score >= applied_threshold
        ]
        if not final_filtered:
            logger.info("all reranked results were below threshold %.3f", applied_threshold)
            return []
        return final_filtered

    # ── 混合检索 ────────────────────────────────

    def _hybrid_search(self, query: str, kb_id: str, top_k: int, request_id: str | None = None) -> list[RetrievalCandidate]:
        searches = {
            "vector": (self._vector_search, self.vector_fetch_k),
            "bm25": (self._bm25_search, self.bm25_fetch_k),
        }
        results: dict[str, list[tuple[str, dict, float]]] = {"vector": [], "bm25": []}

        def run_route(channel, search, fetch_k):
            with timed_stage(
                logger, f"{channel}_search", request_id=request_id,
                kb_id=kb_id, k=fetch_k,
            ):
                return search(query, kb_id, fetch_k)

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = {
                executor.submit(run_route, channel, search, fetch_k): channel
                for channel, (search, fetch_k) in searches.items()
            }
            for future in as_completed(futures):
                channel = futures[future]
                try:
                    results[channel] = future.result()
                except Exception as exc:
                    logger.error("%s retrieval route failed: %s", channel, exc)

        vec_results = results["vector"]
        bm25_results = results["bm25"]

        versioned_candidates = vec_results + bm25_results
        doc_ids = {
            str(metadata.get("doc_id"))
            for _, metadata, _ in versioned_candidates
            if metadata.get("doc_id")
        }
        active_versions = None
        if doc_ids:
            try:
                active_versions = self._active_version_resolver(sorted(doc_ids))
            except Exception as exc:
                logger.error("failed to resolve active document versions: %s", exc)
                active_versions = {doc_id: -1 for doc_id in doc_ids}
        vec_results = self._filter_active_candidates(vec_results, active_versions)
        bm25_results = self._filter_active_candidates(bm25_results, active_versions)

        vector_candidates = [
            self._candidate_from_result(content, metadata, vector_score=score)
            for content, metadata, score in vec_results
        ]
        bm25_candidates = [
            self._candidate_from_result(content, metadata, bm25_score=score)
            for content, metadata, score in bm25_results
        ]
        return rrf_fuse(
            vector_candidates,
            bm25_candidates,
            k=self.rrf_k,
            limit=self.rrf_candidate_k,
        )

    @staticmethod
    def _candidate_from_result(
        content: str,
        metadata: dict,
        *,
        vector_score: float | None = None,
        bm25_score: float | None = None,
    ) -> RetrievalCandidate:
        chunk_id = metadata.get("chunk_id")
        if not chunk_id:
            identity = json.dumps(
                [content, metadata], ensure_ascii=False, sort_keys=True, default=str
            )
            chunk_id = f"legacy-{hashlib.sha256(identity.encode('utf-8')).hexdigest()}"
        return RetrievalCandidate(
            chunk_id=str(chunk_id),
            content=content,
            metadata=metadata,
            vector_score=vector_score,
            bm25_score=bm25_score,
        )

    def _filter_active_candidates(
        self,
        candidates: list[tuple[str, dict, float]],
        active_versions: dict[str, int] | None = None,
    ) -> list[tuple[str, dict, float]]:
        candidate_doc_ids = {
            str(metadata.get("doc_id"))
            for _, metadata, _ in candidates
            if metadata.get("doc_id")
        }
        if not candidate_doc_ids:
            return candidates
        if active_versions is None:
            try:
                active_versions = self._active_version_resolver(sorted(candidate_doc_ids))
            except Exception as exc:
                logger.error("failed to resolve active document versions: %s", exc)
                active_versions = {doc_id: -1 for doc_id in candidate_doc_ids}

        filtered = []
        for candidate in candidates:
            metadata = candidate[1]
            doc_id = metadata.get("doc_id")
            version = metadata.get("document_version")
            if not doc_id:
                filtered.append(candidate)
                continue
            active_version = active_versions.get(str(doc_id))
            if version is None:
                if active_version is None:
                    filtered.append(candidate)
                continue
            try:
                candidate_version = int(version)
            except (TypeError, ValueError):
                continue
            if active_version == candidate_version:
                filtered.append(candidate)
        return filtered

    @staticmethod
    def _load_active_versions(document_ids: list[str]) -> dict[str, int]:
        from KBzhy.app.core.chunk_repository import ChunkRepository
        from KBzhy.app.core.metadata_store import get_metadata_store

        return ChunkRepository(get_metadata_store()).get_active_versions(document_ids)

    @staticmethod
    def _load_active_collection(kb_id: str) -> str | None:
        from KBzhy.app.core.metadata_store import get_metadata_store

        return get_metadata_store().get_active_collection_name(kb_id)

    def _vector_search(self, query: str, kb_id: str, k: int) -> list[tuple[str, dict, float]]:
        """向量相似度检索"""
        try:
            vs = self._get_vectorstore(kb_id)
            results = vs.similarity_search_with_score(query, k=k)
        except Exception as exc:
            logger.error("向量检索失败: %s", exc)
            return []
        return [
            (doc.page_content, doc.metadata, self._normalize_l2(score))
            for doc, score in results
        ]

    def _bm25_search(self, query: str, kb_id: str, k: int) -> list[tuple[str, dict, float]]:
        """BM25 关键词检索"""
        import jieba
        entry = self._bm25_indices.get(kb_id)
        if not entry:
            self._load_bm25_from_vectorstore(kb_id)
            entry = self._bm25_indices.get(kb_id)
            if not entry:
                return []
        bm25, bm25_docs = entry
        if bm25 is None or not bm25_docs:
            return []
        tokenized = list(jieba.cut(query))
        scores = bm25.get_scores(tokenized)
        if not scores.size:
            return []
        top_indices = np.argsort(scores)[::-1][:k]
        return [
            (self._bm25_content(bm25_docs[i]), self._bm25_metadata(bm25_docs[i]), float(scores[i]))
            for i in top_indices
            if scores[i] > 0
        ]

    @staticmethod
    def _normalize_l2(l2_distance: float) -> float:
        """L2 距离 → 0~1 相似度"""
        return round(1.0 / (1.0 + l2_distance), 4)

    # ── MMR 多样性 ─────────────────────────────

    def _mmr(
        self,
        query: str,
        candidates: list[tuple[str, dict, float]],
        top_k: int,
        lambda_param: float = 0.7,
    ) -> list[dict[str, Any]]:
        """最大边际相关性（Maximum Marginal Relevance）"""
        if len(candidates) <= top_k:
            return [
                {"content": c, "metadata": m, "score": s}
                for c, m, s in candidates
            ]

        items = [{"content": c, "metadata": m, "score": s} for c, m, s in candidates]

        try:
            query_vec = np.array(self.embeddings.embed_query(query))
            contents = [it["content"] for it in items]
            vectors: list[list[float]] = []
            for i in range(0, len(contents), _EMBEDDING_BATCH_SIZE):
                vectors.extend(self.embeddings.embed_documents(contents[i:i + _EMBEDDING_BATCH_SIZE]))
            item_vecs = [np.array(vec) for vec in vectors]
        except Exception as exc:
            logger.warning(
                "MMR embedding 失败，回退到相关性排序: candidates=%d top_k=%d error=%s",
                len(items),
                top_k,
                exc,
            )
            return sorted(items, key=lambda x: x["score"], reverse=True)[:top_k]

        selected: list[int] = []
        remaining = list(range(len(items)))

        for _ in range(min(top_k, len(items))):
            if not selected:
                idx = max(remaining, key=lambda i: items[i]["score"])
            else:
                best = -1
                best_score = -float("inf")
                for i in remaining:
                    relevance = items[i]["score"]
                    redundancy = max(
                        float(np.dot(item_vecs[i], item_vecs[j]))
                        for j in selected
                    )
                    mmr_score = lambda_param * relevance - (1 - lambda_param) * redundancy
                    if mmr_score > best_score:
                        best_score = mmr_score
                        best = i
                idx = best
            selected.append(idx)
            remaining.remove(idx)

        return [items[i] for i in selected]

    # ── Reranker ────────────────────────────────

    _model_rerank_failed_at: float = 0.0

    def _rerank(
        self,
        query: str,
        candidates: list[RetrievalCandidate],
        method: str,
    ) -> RerankResult:
        """重排序：按选定方法执行，失败回退到关键词评分"""
        if not candidates:
            return RerankResult((), method, False)

        if method == "model":
            if time.time() - Retriever._model_rerank_failed_at > 60:
                results = self._rerank_model(query, candidates)
                if results:
                    return RerankResult(tuple(results), "model", True)
                Retriever._model_rerank_failed_at = time.time()
                logger.info("Reranker 模型不可用，60s 内回退到关键词评分")
            return RerankResult(tuple(self._rerank_keyword(query, candidates)), "keyword", False)

        if method == "llm":
            try:
                return RerankResult(tuple(self._rerank_llm(query, candidates)), "llm", True)
            except Exception:
                logger.info("LLM 打分失败，回退到关键词评分")

        return RerankResult(tuple(self._rerank_keyword(query, candidates)), "keyword", False)

    def _rerank_model(
        self, query: str, candidates: list[RetrievalCandidate]
    ) -> list[RetrievalCandidate]:
        """调用专用 Reranker 模型（DashScope 原生 API）"""

        docs = [candidate.content for candidate in candidates]
        url = "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank"
        try:
            resp = _http_client.post(
                url,
                headers={
                    "Authorization": f"Bearer {API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": RERANKER_MODEL,
                    "input": {
                        "query": query,
                        "documents": docs,
                    },
                    "parameters": {
                        "top_n": len(docs),
                    },
                },
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            logger.warning("Reranker API 调用失败: %s", exc)
            return []

        results = data.get("output", {}).get("results", [])
        if not results:
            return []

        scored = list(candidates)
        for r in results:
            idx = r.get("index", -1)
            score = r.get("relevance_score", 0.0)
            if 0 <= idx < len(candidates):
                scored[idx] = replace(candidates[idx], rerank_score=float(score))

        return sorted(scored, key=lambda item: item.score, reverse=True)

    def _rerank_llm(
        self, query: str, candidates: list[RetrievalCandidate]
    ) -> list[RetrievalCandidate]:
        """LLM 逐条打分"""

        scored = []
        for item in candidates:
            prompt = (
                f"评估以下文档与问题的相关性，只返回0-10的整数分数。\n"
                f"问题：{query}\n"
                f"文档：{item.content[:500]}"
            )
            try:
                resp = _http_client.post(
                    f"{API_BASE.rstrip('/')}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {API_KEY}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": LLM_MODEL,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": 10,
                    },
                )
                resp.raise_for_status()
                data = resp.json()
                text = data["choices"][0]["message"]["content"]
                numbers = re.findall(r"\d+", text)
                score = min(int(numbers[0]), 10) / 10.0 if numbers else 0.5
            except Exception:
                score = 0.5
            scored.append(replace(item, rerank_score=score))

        return sorted(scored, key=lambda item: item.score, reverse=True)

    def _rerank_keyword(
        self, query: str, candidates: list[RetrievalCandidate]
    ) -> list[RetrievalCandidate]:
        """关键词覆盖率评分"""
        import jieba
        stop = {"", " ", "？", "?", "，", "。", "的", "了", "是", "在", "有", "和"}
        query_words = set(jieba.cut(query)) - stop

        scored = []
        for item in candidates:
            hits = sum(1 for word in query_words if word in item.content)
            score = hits / max(len(query_words), 1)
            scored.append(replace(item, rerank_score=score))

        return sorted(scored, key=lambda item: item.score, reverse=True)

    # ── 查询扩展 ──────────────────────────────

    def _expand_query(self, query: str) -> list[str] | None:
        """LLM 改写查询为多个不同表述，提高召回率"""

        prompt = (
            f"请将以下问题改写成2-3个不同的表述方式，保持原意不变，每个一行以'- '开头。\n"
            f"问题：{query}\n"
            f"改写："
        )
        try:
            resp = _http_client.post(
                f"{API_BASE.rstrip('/')}/chat/completions",
                headers={
                    "Authorization": f"Bearer {API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": LLM_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 300,
                },
            )
            resp.raise_for_status()
            text = resp.json()["choices"][0]["message"]["content"]
            variants = [l.strip("- ").strip() for l in text.split("\n") if l.strip().startswith("-")]
            return variants if variants else None
        except Exception as exc:
            logger.warning("查询扩展失败: %s", exc)
            return None

    # ── 子问题拆解 ──────────────────────────────

    @staticmethod
    def _is_complex(query: str) -> bool:
        """判断是否为复杂对比类问题"""
        signals = [
            "比较", "对比", "区别", "不同", "异同", "优缺点",
            "哪个更", "分别", "各自", "vs", "VS", "和.*哪个",
            "还是", "要么", "或者.*或者", "首先.*然后.*最后",
        ]
        return any(re.search(s, query) for s in signals)

    def _decompose_query(self, query: str) -> list[str] | None:
        """LLM 拆解复杂问题为子问题"""

        prompt = (
            f"请将以下复杂问题拆解为2-4个独立的子问题，每个子问题单独一行，以'- '开头。\n"
            f"问题：{query}\n"
            f"子问题："
        )
        try:
            resp = _http_client.post(
                f"{API_BASE.rstrip('/')}/chat/completions",
                headers={
                    "Authorization": f"Bearer {API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": LLM_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 500,
                },
            )
            resp.raise_for_status()
            data = resp.json()
            text = data["choices"][0]["message"]["content"]
            sub = [l.strip("- ").strip() for l in text.split("\n") if l.strip().startswith("-")]
            return sub if sub else None
        except Exception as exc:
            logger.warning("子问题拆解失败: %s", exc)
            return None
