"""RAG 引擎 — 全流程编排：检索 → 幻觉检测 → 生成 → 溯源"""

from __future__ import annotations

import json
import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Generator

import httpx

from KBzhy.config import (
    API_KEY,
    API_BASE,
    LLM_MODEL,
    TEMPERATURE,
    CONNECT_TIMEOUT,
    READ_TIMEOUT,
    MAX_CONTEXT_TOKENS,
    CHROMA_PERSIST_DIR,
)
from KBzhy.app.core.parser import DocumentParser
from KBzhy.app.core.splitter import SmartSplitter
from KBzhy.app.core.retriever import Retriever, _http_client
from KBzhy.app.core.memory import MemoryManager
from KBzhy.app.core.timing import timed_stage

logger = logging.getLogger(__name__)

KNOWLEDGE_QA_SYSTEM_PROMPT = (
    "你是一个严谨的知识库问答助手。请严格遵循以下规则：\n"
    "1) 只基于已提供的知识库检索上下文与对话历史作答，不得凭空补充事实。\n"
    "2) 当上下文证据不足、没有明确结论、或与问题不相关时，必须明确回答：\n"
    "   “我在当前知识库中没有找到足够信息回答这个问题。”\n"
    "3) 不要编造来源、时间、数据、链接、文件名、术语定义或业务规则。\n"
    "4) 如果问题超出知识库范围，可给出简短的澄清建议（如建议用户补充文档或更具体关键词），\n"
    "   但不得把建议伪装成事实答案。\n"
    "5) 优先输出简洁、可执行、与问题直接相关的结论；有多条依据时按要点列出。\n"
    "6) 若上下文存在冲突，请明确指出“知识库信息存在冲突”，并分别说明冲突点。"
)

KNOWLEDGE_QA_REFUSAL = "我在当前知识库中没有找到足够信息回答这个问题。"

COMBINE_PROMPT = (
    "你是一个严谨的知识库问答助手。\n"
    "以下是多个参考资料的摘要信息，请基于这些摘要回答用户问题。\n"
    f"如果所有摘要都不包含相关信息，请说「{KNOWLEDGE_QA_REFUSAL}」。\n\n"
    "{summaries}"
)


# ── RAG 引擎 ───────────────────────────────────

class RAGEngine:
    """RAG 引擎：编排检索、生成、记忆全流程"""

    def __init__(
        self,
        persist_dir: str | None = None,
        llm_model: str | None = None,
        embedding_model: str | None = None,
    ):
        self.persist_dir = persist_dir or CHROMA_PERSIST_DIR
        self.llm_model = llm_model or LLM_MODEL
        self.parser = DocumentParser(vlm_model=self.llm_model)
        self.splitter = SmartSplitter()
        self.retriever = Retriever(
            persist_dir=self.persist_dir,
            embedding_model=embedding_model,
        )
        self.memory_manager = MemoryManager()

    # ── LLM 调用 ────────────────────────────────

    def _call_llm_sync(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int = 4096,
    ) -> str:
        temp = temperature if temperature is not None else TEMPERATURE
        try:
            resp = _http_client.post(
                f"{API_BASE.rstrip('/')}/chat/completions",
                headers={
                    "Authorization": f"Bearer {API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.llm_model,
                    "messages": messages,
                    "temperature": temp,
                    "max_tokens": max_tokens,
                },
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except httpx.HTTPStatusError as exc:
            logger.error("LLM API HTTP %d: %s", exc.response.status_code, self._response_preview(exc.response))
            raise RuntimeError(f"LLM API 调用失败 (HTTP {exc.response.status_code})") from exc
        except httpx.RequestError as exc:
            logger.error("LLM API 请求错误: %s", exc)
            raise RuntimeError(f"无法连接 LLM 服务: {exc}") from exc

    @staticmethod
    def _response_preview(response: httpx.Response) -> str:
        try:
            return response.text[:200]
        except httpx.ResponseNotRead:
            return "<streaming response body not read>"

    def _call_llm_stream(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
    ) -> Generator[str, None, None]:
        temp = temperature if temperature is not None else TEMPERATURE
        url = f"{API_BASE.rstrip('/')}/chat/completions"
        payload = {
            "model": self.llm_model,
            "messages": messages,
            "temperature": temp,
            "stream": True,
        }
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        }

        last_error = None
        for attempt in range(2):
            try:
                with _http_client.stream("POST", url, headers=headers, json=payload) as resp:
                    resp.raise_for_status()
                    for line in resp.iter_lines():
                        if line.startswith("data: "):
                            data_str = line[6:]
                            if data_str.strip() == "[DONE]":
                                return
                            try:
                                chunk = json.loads(data_str)
                                delta = chunk["choices"][0].get("delta", {})
                                content = delta.get("content", "")
                                if content:
                                    yield content
                            except (json.JSONDecodeError, KeyError, IndexError):
                                continue
                    return
            except (httpx.RemoteProtocolError, httpx.ReadError, httpx.ConnectError) as exc:
                last_error = exc
                if attempt == 0:
                    logger.warning("LLM Stream 连接异常，1秒后重试: %s", exc)
                    time.sleep(1)
            except httpx.HTTPStatusError as exc:
                logger.error("LLM Stream HTTP %d: %s", exc.response.status_code, self._response_preview(exc.response))
                yield f"\n[错误: LLM 调用失败 (HTTP {exc.response.status_code})]"
                return
            except httpx.RequestError as exc:
                logger.error("LLM Stream 请求错误: %s", exc)
                yield f"\n[错误: 无法连接 LLM 服务]"
                return

        logger.error("LLM Stream 重试后仍失败: %s", last_error)
        yield "\n[错误: LLM 服务暂时不可用，请稍后重试]"

    # ── 知识库管理 ────────────────────────────────

    def create_kb(self, kb_id: str):
        self.retriever.create_kb(kb_id)

    def delete_kb(self, kb_id: str):
        self.retriever.delete_kb(kb_id)

    def list_kbs(self) -> list[str]:
        return self.retriever.list_kbs()

    def get_kb_doc_count(self, kb_id: str) -> int:
        return self.retriever.get_kb_doc_count(kb_id)

    def list_document_chunks(self, kb_id: str, source: str) -> list[dict[str, Any]]:
        return self.retriever.list_document_chunks(kb_id, source)

    # ── 文档索引 ────────────────────────────────

    def index_document(self, file_path: str, kb_id: str, display_name: str | None = None) -> int:
        docs = self.parser.parse(file_path)
        if display_name:
            for doc in docs:
                doc.metadata["source"] = display_name
        return self._index_docs(docs, kb_id)

    def index_bytes(self, content: bytes, filename: str, kb_id: str) -> int:
        docs = self.parser.parse_bytes(content, filename)
        return self._index_docs(docs, kb_id)

    def _index_docs(self, docs, kb_id: str) -> int:
        total = 0
        for doc in docs:
            file_type = doc.metadata.get("file_type", "text")
            chunks = self.splitter.split(doc.content, doc_type=file_type, metadata=doc.metadata)
            self.retriever.add_documents(chunks, kb_id)
            total += len(chunks)
        return total

    def remove_document(self, filename: str, kb_id: str):
        self.retriever.remove_document(kb_id, source=filename)

    # ── 对话（支持 stuff / map_reduce / refine）───

    def chat(
        self,
        question: str,
        kb_id: str | None = None,
        session_id: str | None = None,
        top_k: int = 5,
        temperature: float | None = None,
        chain_type: str = "stuff",
        rerank_method: str = "model",
        similarity_threshold: float = 0.35,
        enable_expansion: bool = False,
        enable_rewrite: bool = False,
    ) -> dict[str, Any]:
        request_id = uuid.uuid4().hex[:8]
        with timed_stage(logger, "chat_total", request_id=request_id, session_id=session_id, kb_id=kb_id or "default"):
            return self._chat(
                question=question,
                kb_id=kb_id,
                session_id=session_id,
                top_k=top_k,
                temperature=temperature,
                chain_type=chain_type,
                rerank_method=rerank_method,
                similarity_threshold=similarity_threshold,
                enable_expansion=enable_expansion,
                enable_rewrite=enable_rewrite,
                request_id=request_id,
            )

    def _chat(
        self,
        question: str,
        kb_id: str | None,
        session_id: str | None,
        top_k: int,
        temperature: float | None,
        chain_type: str,
        rerank_method: str,
        similarity_threshold: float,
        enable_expansion: bool,
        enable_rewrite: bool,
        request_id: str,
    ) -> dict[str, Any]:
        memory = self.memory_manager.get(session_id) if session_id else None
        with timed_stage(logger, "prepare_query", request_id=request_id, session_id=session_id):
            retrieval_query = self._prepare_retrieval_query(
                question,
                memory,
                request_id=request_id,
                enable_rewrite=enable_rewrite,
            )
        results = self.retriever.retrieve(
            retrieval_query,
            kb_id or "default",
            top_k=top_k,
            rerank_method=rerank_method,
            threshold=similarity_threshold,
            enable_expansion=enable_expansion,
            request_id=request_id,
        )

        temp = temperature if temperature is not None else TEMPERATURE

        if not results:
            answer = KNOWLEDGE_QA_REFUSAL
            if memory:
                memory.add_message("user", question)
                memory.add_message("assistant", answer, sources=[])
            return {"answer": answer, "session_id": session_id, "sources": [], "hallucination_flags": []}

        if chain_type == "map_reduce":
            answer = self._generate_map_reduce(question, results, temp)
        elif chain_type == "refine":
            answer = self._generate_refine(question, results, temp)
        else:
            messages = self._build_messages(question, results, memory)
            with timed_stage(logger, "llm_generate_sync", request_id=request_id, model=self.llm_model):
                answer = self._call_llm_sync(messages, temp)

        sources = [
            {
                "content": r["content"],
                "source": r.get("metadata", {}).get("source", ""),
                "page": r.get("metadata", {}).get("page"),
                "score": r.get("score", 0),
            }
            for r in results
        ]
        if memory:
            memory.add_message("user", question)
            memory.add_message("assistant", answer, sources=sources)

        flags = self._detect_hallucinations(answer, results)

        return {
            "answer": answer,
            "session_id": session_id,
            "sources": sources,
            "hallucination_flags": flags,
        }

    def chat_stream(
        self,
        question: str,
        kb_id: str | None = None,
        session_id: str | None = None,
        top_k: int = 5,
        temperature: float | None = None,
        rerank_method: str = "model",
        similarity_threshold: float = 0.35,
        enable_expansion: bool = False,
        enable_rewrite: bool = False,
    ) -> Generator[str, None, None]:
        request_id = uuid.uuid4().hex[:8]
        stream_start = time.perf_counter()
        memory = self.memory_manager.get(session_id) if session_id else None

        if memory and memory.get_context() and (enable_rewrite or self._needs_contextual_rewrite(question)):
            yield '[STATUS]{"stage":"rewrite","message":"正在改写查询..."}'
        with timed_stage(logger, "prepare_query", request_id=request_id, session_id=session_id):
            retrieval_query = self._prepare_retrieval_query(
                question,
                memory,
                request_id=request_id,
                enable_rewrite=enable_rewrite,
            )

        # 通过回调收集检索管线内部的状态
        _statuses: list[str] = []
        def _on_status(stage: str, msg: str):
            _statuses.append(json.dumps({"stage": stage, "message": msg}, ensure_ascii=False))

        results = self.retriever.retrieve(
            retrieval_query, kb_id or "default",
            top_k=top_k, rerank_method=rerank_method,
            threshold=similarity_threshold, enable_expansion=enable_expansion,
            on_status=_on_status,
            request_id=request_id,
        )

        for s in _statuses:
            yield f"[STATUS]{s}"

        temp = temperature if temperature is not None else TEMPERATURE

        if not results:
            yield '[STATUS]{"stage":"generate","message":"正在生成回答..."}'

            full_text = KNOWLEDGE_QA_REFUSAL
            yield full_text
            if memory:
                memory.add_message("user", question)
                memory.add_message("assistant", full_text, sources=[])
            yield "[SOURCES]" + json.dumps([], ensure_ascii=False)
            return

        messages = self._build_messages(question, results, memory)

        yield '[STATUS]{"stage":"generate","message":"正在生成回答..."}'

        with self._manage_context_window(messages):
            full: list[str] = []
            first_token_seen = False
            with timed_stage(logger, "llm_generate_stream", request_id=request_id, model=self.llm_model):
                for chunk in self._call_llm_stream(messages, temp):
                    if not first_token_seen:
                        first_token_seen = True
                        logger.info(
                            "[PERF] stage=stream_first_token elapsed_ms=%.2f request_id=%s",
                            (time.perf_counter() - stream_start) * 1000,
                            request_id,
                        )
                    full.append(chunk)
                    yield chunk
            full_text = "".join(full)
            sources = [
                {
                    "content": r["content"],
                    "source": r.get("metadata", {}).get("source", ""),
                    "page": r.get("metadata", {}).get("page"),
                    "score": r.get("score", 0),
                }
                for r in results
            ]
            if memory:
                memory.add_message("user", question)
                memory.add_message("assistant", full_text, sources=sources)

            yield "[SOURCES]" + json.dumps(sources, ensure_ascii=False)
        logger.info(
            "[PERF] stage=chat_stream_total elapsed_ms=%.2f request_id=%s session_id=%s kb_id=%s",
            (time.perf_counter() - stream_start) * 1000,
            request_id,
            session_id,
            kb_id or "default",
        )

    # ── 生成策略 ────────────────────────────────

    def _generate_map_reduce(
        self,
        question: str,
        results: list[dict[str, Any]],
        temperature: float,
    ) -> str:
        """Map-Reduce: 每个文档块独立生成摘要，再合并汇总"""
        summaries: list[str] = []

        def _map_one(item: dict, idx: int) -> str:
            context = f"[参考资料{idx + 1}]\n{item['content']}"
            msgs = [
                {"role": "system", "content": KNOWLEDGE_QA_SYSTEM_PROMPT},
                {"role": "user", "content": f"参考资料：\n{context}\n\n问题：{question}\n\n基于上述资料提取与问题相关的信息（如不相关请回复'不相关'）："},
            ]
            return self._call_llm_sync(msgs, temperature, max_tokens=1024)

        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = {pool.submit(_map_one, item, i): i for i, item in enumerate(results)}
            errors: list[str] = []
            for f in as_completed(futures):
                try:
                    s = f.result()
                    if "不相关" not in s:
                        summaries.append(s)
                except Exception as exc:
                    errors.append(str(exc))
                    logger.warning("Map-Reduce 子任务失败: %s", exc)

        if not summaries:
            if errors and len(errors) == len(futures):
                return f"Map-Reduce 生成失败：所有 {len(futures)} 个子任务均未能完成，可能是 LLM 服务暂时不可用。"
            return KNOWLEDGE_QA_REFUSAL

        combined = "\n\n---\n\n".join(f"[摘要{i + 1}] {s}" for i, s in enumerate(summaries))
        msgs = [
            {"role": "system", "content": KNOWLEDGE_QA_SYSTEM_PROMPT},
            {"role": "user", "content": f"{COMBINE_PROMPT.format(summaries=combined)}\n\n用户问题：{question}"},
        ]
        return self._call_llm_sync(msgs, temperature)

    def _generate_refine(
        self,
        question: str,
        results: list[dict[str, Any]],
        temperature: float,
    ) -> str:
        """Refine: 逐文档块迭代优化答案"""

        # 初始答案从第一个块生成
        context_0 = f"[参考资料1]\n{results[0]['content']}"
        msgs = [
            {"role": "system", "content": KNOWLEDGE_QA_SYSTEM_PROMPT},
            {"role": "user", "content": f"参考资料：\n{context_0}\n\n问题：{question}"},
        ]
        answer = self._call_llm_sync(msgs, temperature)

        # 用后续块逐步优化
        for i, item in enumerate(results[1:], 2):
            context = f"[参考资料{i}]\n{item['content']}"
            refine_msgs = [
                {"role": "system", "content": KNOWLEDGE_QA_SYSTEM_PROMPT},
                {"role": "user", "content": (
                    f"你已有以下初步回答：\n{answer}\n\n"
                    f"现在获得了一份新参考资料：\n{context}\n\n"
                    f"原问题：{question}\n\n"
                    f"请基于新资料优化你的回答（如果新资料不相关，直接返回原回答，不要改动）："
                )},
            ]
            try:
                refined = self._call_llm_sync(refine_msgs, temperature)
                answer = refined
            except Exception:
                continue

        return answer

    # ── 上下文窗口管理 ──────────────────────────

    def _manage_context_window(self, messages: list[dict[str, str]]):
        """上下文窗口管理器：超限时自动摘要压缩历史消息"""

        class _WindowManager:
            def __init__(self, engine: RAGEngine, msgs: list[dict[str, str]]):
                self.engine = engine
                self.msgs = msgs
                self._original: list[dict[str, str]] | None = None

            def __enter__(self):
                history = [m for m in self.msgs if m["role"] in ("user", "assistant")]
                if not history:
                    return self

                total_chars = sum(len(m["content"]) for m in self.msgs)
                estimated_tokens = total_chars // 2
                if estimated_tokens <= MAX_CONTEXT_TOKENS:
                    return self

                # 取前半部分旧消息做摘要
                split = max(2, len(history) // 2)
                old = history[:split]
                recent = history[split:]

                old_text = "\n".join(f"{m['role']}: {m['content'][:300]}" for m in old)
                try:
                    summary = self.engine._call_llm_sync(
                        [{"role": "user", "content": f"请用2-3句话概括以下对话的要点：\n{old_text}"}],
                        temperature=0,
                        max_tokens=500,
                    )
                except Exception:
                    summary = "（对话摘要生成失败）"

                self._original = list(self.msgs)
                self.msgs.clear()
                self.msgs.append({"role": "system", "content": f"[前序对话摘要] {summary}"})
                for m in recent:
                    self.msgs.append(m)
                # 保留原始的 system prompt
                if self._original:
                    sys_msg = next((m for m in self._original if m["role"] == "system" and "知识库" in m.get("content", "")), None)
                    if sys_msg:
                        self.msgs.insert(0, sys_msg)

                logger.info("上下文窗口压缩: %d → ~%d tokens", estimated_tokens, total_chars // 4)
                return self

            def __exit__(self, *args):
                pass

        return _WindowManager(self, messages)

    # ── 消息构建 ────────────────────────────────

    def _build_messages(
        self,
        question: str,
        results: list[dict[str, Any]],
        memory,
    ) -> list[dict[str, str]]:
        context_text = self._build_context(results)
        system_content = f"{KNOWLEDGE_QA_SYSTEM_PROMPT}\n\n参考资料：\n{context_text}"

        messages: list[dict[str, str]] = [{"role": "system", "content": system_content}]
        if memory:
            for m in memory.get_context():
                messages.append({"role": m["role"], "content": m["content"]})
        messages.append({"role": "user", "content": question})
        return messages

    # ── 上下文构建 ──────────────────────────────

    def _build_context(self, results: list[dict[str, Any]]) -> str:
        parts: list[str] = []
        for i, r in enumerate(results, 1):
            src = r.get("metadata", {}).get("source", "未知")
            page = r.get("metadata", {}).get("page", "")
            loc = f"（{src}" + (f" 第{page}页" if page else "") + "）"
            parts.append(f"[参考资料{i}]{loc}\n{r['content']}")
        return "\n\n".join(parts)

    # ── 查询准备 ────────────────────────────────

    def _prepare_retrieval_query(
        self,
        question: str,
        memory,
        request_id: str | None = None,
        enable_rewrite: bool = False,
    ) -> str:
        """结合对话历史改写查询，并检查是否需要压缩上下文"""
        if not memory:
            return question
        history = memory.get_context()
        if not history:
            return question
        if not enable_rewrite and not self._needs_contextual_rewrite(question):
            logger.info("跳过查询改写: %s", question)
            return question
        with timed_stage(logger, "query_rewrite", request_id=request_id):
            return self._contextualize_query(question, history)

    @staticmethod
    def _needs_contextual_rewrite(question: str) -> bool:
        q = question.strip().lower()
        context_markers = [
            "这个", "那个", "这些", "那些", "上述", "上面", "前面", "刚才", "之前",
            "继续", "它", "他", "她", "其", "该", "此", "这", "那",
            "this", "that", "these", "those", "it", "they", "them", "above", "previous",
            "continue", "same",
        ]
        return any(marker in q for marker in context_markers)

    def _contextualize_query(self, question: str, history: list[dict[str, str]]) -> str:
        history_text = "\n".join([
            f"{'用户' if m['role'] == 'user' else '助手'}: {m['content']}"
            for m in history[-6:]
        ])
        prompt = (
            f"对话历史：\n{history_text}\n\n"
            f"用户最新问题：{question}\n\n"
            f"将用户问题改写为独立完整的检索查询（直接返回改写后的问题，不要解释）："
        )
        try:
            answer = self._call_llm_sync([{"role": "user", "content": prompt}], temperature=0, max_tokens=300)
            logger.info("查询改写: %s → %s", question, answer)
            return answer
        except Exception:
            return question

    # ── 幻觉检测 ─────────────────────────────────

    def _detect_hallucinations(self, answer: str, sources: list[dict[str, Any]]) -> list[str]:
        import re
        import jieba

        sentences = re.split(r"[。！？\n]", answer)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        source_text = " ".join([s["content"] for s in sources])

        flags: list[str] = []
        for sent in sentences:
            words = [w for w in jieba.cut(sent) if len(w) > 1]
            if len(words) < 3:
                continue
            hits = sum(1 for w in words if w in source_text)
            if hits / len(words) < 0.4:
                flags.append(f"可能不准确: {sent}")
        return flags
