"""混合检索 + MMR + Reranker + 查询扩展 + 子问题拆解（多知识库支持）"""

from __future__ import annotations

import logging
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    CONNECT_TIMEOUT,
    READ_TIMEOUT,
    CHROMA_PERSIST_DIR,
)
from KBzhy.app.core.splitter import Chunk
from KBzhy.app.core.timing import timed_stage

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = (CONNECT_TIMEOUT, READ_TIMEOUT)
_KB_COLLECTION_PREFIX = "kbzhy_"
_EMBEDDING_BATCH_SIZE = 10

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
        result = self._client.embeddings.create(model=self.model, input=texts)
        return [d.embedding for d in result.data]

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
    ):
        self.persist_dir = persist_dir or CHROMA_PERSIST_DIR
        self.top_k = top_k
        self.fetch_k = fetch_k
        self.threshold = threshold
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
        # kb_id → (BM25Okapi, list[{content, metadata}])
        self._bm25_indices: dict[str, tuple[BM25Okapi | None, list[dict[str, Any]]]] = {}

    # ── 知识库管理 ────────────────────────────────

    def _collection_name(self, kb_id: str) -> str:
        return f"{_KB_COLLECTION_PREFIX}{kb_id}"

    def _get_vectorstore(self, kb_id: str) -> Chroma:
        if kb_id not in self._vectorstores:
            self._vectorstores[kb_id] = Chroma(
                persist_directory=self.persist_dir,
                embedding_function=self.embeddings,
                collection_name=self._collection_name(kb_id),
            )
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

    def list_document_chunks(self, kb_id: str, source: str) -> list[dict[str, Any]]:
        """Return all stored chunks for one source document."""
        try:
            vs = self._get_vectorstore(kb_id)
            data = vs.get(where={"source": source})
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

    def remove_document(self, kb_id: str, source: str | None = None):
        """从指定知识库中删除文档的 chunks"""
        try:
            vs = self._get_vectorstore(kb_id)
        except Exception as exc:
            logger.warning("打开知识库 %s 的 Chroma collection 失败: %s", kb_id, exc)
            return
        if source:
            results = vs.get(where={"source": source})
            ids_to_delete = results.get("ids", [])
            if ids_to_delete:
                vs.delete(ids=ids_to_delete)
                logger.info("知识库 %s 已删除 %s 的 %d 个 chunks", kb_id, source, len(ids_to_delete))

        # 同时清理 BM25 索引 — 从 ChromaDB 重建以排除已删除文档
        if source and kb_id in self._bm25_indices:
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
    ) -> list[dict[str, Any]]:
        tk = top_k or self.top_k
        th = threshold if threshold is not None else self.threshold

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

        all_candidates: list[tuple[str, dict, float]] = []
        seen = set()
        with timed_stage(logger, "hybrid_search_all", request_id=request_id, query_count=len(queries)):
            for q in queries:
                candidates = self._hybrid_search(q, kb_id, tk, request_id=request_id)
                for content, meta, score in candidates:
                    key = content[:100]
                    if key not in seen:
                        seen.add(key)
                        all_candidates.append((content, meta, score))

        if not all_candidates:
            return []

        filtered = [(c, m, s) for c, m, s in all_candidates if s >= th]
        if not filtered:
            logger.info("所有结果低于阈值 %.3f，触发拒答", th)
            return []

        if on_status:
            on_status("rerank", "正在重排序结果...")

        with timed_stage(logger, "mmr", request_id=request_id, candidates=len(filtered), top_k=tk):
            mmr_results = self._mmr(query, filtered, tk)
        with timed_stage(logger, "rerank", request_id=request_id, method=rerank_method, candidates=len(mmr_results)):
            final = self._rerank(query, mmr_results, rerank_method)
        final_filtered = [item for item in final if item.get("score", 0) >= th]
        if not final_filtered:
            logger.info("重排后所有结果低于阈值 %.3f，触发拒答", th)
            return []
        return final_filtered

    # ── 混合检索 ────────────────────────────────

    def _hybrid_search(self, query: str, kb_id: str, top_k: int, request_id: str | None = None) -> list[tuple[str, dict, float]]:
        """BM25 + 向量检索加权融合"""
        fetch = max(top_k * 3, self.fetch_k)

        with timed_stage(logger, "vector_search", request_id=request_id, kb_id=kb_id, k=fetch):
            vec_results = self._vector_search(query, kb_id, fetch)
        with timed_stage(logger, "bm25_search", request_id=request_id, kb_id=kb_id, k=fetch):
            bm25_results = self._bm25_search(query, kb_id, fetch)

        combined: dict[str, tuple[str, dict, float]] = {}
        for content, meta, score in vec_results:
            combined[content[:200]] = (content, meta, score * VECTOR_WEIGHT)
        for content, meta, score in bm25_results:
            key = content[:200]
            if key in combined:
                _, m, s = combined[key]
                combined[key] = (content, m, s + score * BM25_WEIGHT)
            else:
                combined[key] = (content, meta, score * BM25_WEIGHT)

        ranked = sorted(combined.values(), key=lambda x: x[2], reverse=True)
        return ranked[:fetch]

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
        candidates: list[dict[str, Any]],
        method: str,
    ) -> list[dict[str, Any]]:
        """重排序：按选定方法执行，失败回退到关键词评分"""
        if not candidates:
            return []

        if method == "model":
            if time.time() - Retriever._model_rerank_failed_at > 60:
                results = self._rerank_model(query, candidates)
                if results:
                    return results
                Retriever._model_rerank_failed_at = time.time()
                logger.info("Reranker 模型不可用，60s 内回退到关键词评分")
            return self._rerank_keyword(query, candidates)

        if method == "llm":
            try:
                return self._rerank_llm(query, candidates)
            except Exception:
                logger.info("LLM 打分失败，回退到关键词评分")

        return self._rerank_keyword(query, candidates)

    def _rerank_model(self, query: str, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """调用专用 Reranker 模型（DashScope 原生 API）"""

        docs = [c["content"] for c in candidates]
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

        for r in results:
            idx = r.get("index", -1)
            score = r.get("relevance_score", 0.0)
            if 0 <= idx < len(candidates):
                candidates[idx]["score"] = float(score)

        return sorted(candidates, key=lambda x: x["score"], reverse=True)

    def _rerank_llm(self, query: str, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """LLM 逐条打分"""

        for item in candidates:
            prompt = (
                f"评估以下文档与问题的相关性，只返回0-10的整数分数。\n"
                f"问题：{query}\n"
                f"文档：{item['content'][:500]}"
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
                item["score"] = min(int(numbers[0]), 10) / 10.0 if numbers else 0.5
            except Exception:
                if "score" not in item or item["score"] == 0.0:
                    item["score"] = 0.5

        return sorted(candidates, key=lambda x: x["score"], reverse=True)

    def _rerank_keyword(self, query: str, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """关键词覆盖率评分"""
        import jieba
        stop = {"", " ", "？", "?", "，", "。", "的", "了", "是", "在", "有", "和"}
        query_words = set(jieba.cut(query)) - stop

        for item in candidates:
            hits = sum(1 for w in query_words if w in item["content"])
            item["score"] = hits / max(len(query_words), 1)

        return sorted(candidates, key=lambda x: x["score"], reverse=True)

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
