from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from typing import Literal

from KBzhy.app.core.document_models import KnowledgeChunk, RetrievalCandidate
from KBzhy.app.core.token_counter import TokenCounter
from KBzhy.config import (
    CONTEXT_NEIGHBOR_WINDOW,
    CONTEXT_PER_DOCUMENT_LIMIT,
    CONTEXT_SINGLE_SOURCE_TOKEN_BUDGET,
    CONTEXT_TOKEN_BUDGET,
)

logger = logging.getLogger(__name__)

ContextRole = Literal["hit", "parent", "neighbor"]


@dataclass(frozen=True)
class ContextUnit:
    chunk_id: str
    document_id: str
    content: str
    metadata: dict
    context_role: ContextRole
    origin_chunk_id: str
    rerank_score: float | None = None
    parent_chunk_id: str | None = None
    position: int | None = None

    def __getitem__(self, key: str):
        if key == "content":
            return self.content
        if key == "metadata":
            return self.metadata
        if key == "score":
            return self.rerank_score or 0.0
        raise KeyError(key)

    def get(self, key: str, default=None):
        try:
            return self[key]
        except KeyError:
            return default


class ContextAssembler:
    def __init__(
        self,
        repository,
        token_counter: TokenCounter | None = None,
        token_budget: int = CONTEXT_TOKEN_BUDGET,
        per_document_limit: int = CONTEXT_PER_DOCUMENT_LIMIT,
        neighbor_window: int = CONTEXT_NEIGHBOR_WINDOW,
        single_source_budget: int = CONTEXT_SINGLE_SOURCE_TOKEN_BUDGET,
    ):
        if token_budget <= 0:
            raise ValueError("token_budget must be positive")
        if per_document_limit <= 0:
            raise ValueError("per_document_limit must be positive")
        if neighbor_window < 0:
            raise ValueError("neighbor_window must be non-negative")
        self.repository = repository
        self.token_counter = token_counter or TokenCounter()
        self.token_budget = token_budget
        self.per_document_limit = per_document_limit
        self.neighbor_window = neighbor_window
        self.single_source_budget = single_source_budget

    def assemble(
        self, candidates: list[RetrievalCandidate], final_k: int
    ) -> list[ContextUnit]:
        if final_k <= 0 or not candidates:
            return []
        selected = self._apply_document_quota(candidates)[:final_k]
        hits = [self._hit_unit(candidate) for candidate in selected]
        supplements = self._expand_context(selected, {unit.chunk_id for unit in hits})
        return self._fit_token_budget(hits + supplements)

    def _apply_document_quota(
        self, candidates: list[RetrievalCandidate]
    ) -> list[RetrievalCandidate]:
        counts: dict[str, int] = {}
        selected = []
        seen_chunks = set()
        for candidate in candidates:
            if candidate.chunk_id in seen_chunks:
                continue
            document_key = str(
                candidate.metadata.get("doc_id")
                or candidate.metadata.get("source")
                or candidate.chunk_id
            )
            if counts.get(document_key, 0) >= self.per_document_limit:
                continue
            seen_chunks.add(candidate.chunk_id)
            counts[document_key] = counts.get(document_key, 0) + 1
            selected.append(candidate)
        return selected

    def _expand_context(
        self,
        candidates: list[RetrievalCandidate],
        used_chunk_ids: set[str],
    ) -> list[ContextUnit]:
        supplements = []
        used_parent_ids = set()
        for candidate in candidates:
            try:
                family = self.repository.get_context_family(
                    candidate.chunk_id, self.neighbor_window
                )
            except Exception as exc:
                logger.warning(
                    "context family lookup failed: chunk=%s error=%s",
                    candidate.chunk_id, exc,
                )
                continue
            parent = family.parent
            if parent and parent.chunk_id not in used_parent_ids:
                try:
                    parent_fits = (
                        self.token_counter.count(parent.retrieval_text)
                        <= self.single_source_budget
                    )
                except Exception as exc:
                    logger.warning(
                        "parent token count failed: chunk=%s error=%s",
                        parent.chunk_id, exc,
                    )
                    parent_fits = False
                if parent_fits:
                    supplements.append(
                        self._chunk_unit(parent, "parent", candidate.chunk_id)
                    )
                    used_parent_ids.add(parent.chunk_id)
            for child in family.children:
                if child.chunk_id in used_chunk_ids:
                    continue
                used_chunk_ids.add(child.chunk_id)
                supplements.append(
                    self._chunk_unit(child, "neighbor", candidate.chunk_id)
                )
        return supplements

    def _fit_token_budget(self, units: list[ContextUnit]) -> list[ContextUnit]:
        if not units:
            return []
        selected = []
        remaining = self.token_budget
        try:
            for unit in units:
                tokens = self.token_counter.count(unit.content)
                if tokens <= remaining:
                    selected.append(unit)
                    remaining -= tokens
                    continue
                if not selected:
                    truncated = self.token_counter.truncate(unit.content, remaining)
                    if truncated:
                        selected.append(replace(unit, content=truncated))
                    return selected
        except Exception as exc:
            logger.warning("context token budgeting failed: %s", exc)
            return [units[0]]
        return selected

    @staticmethod
    def _hit_unit(candidate: RetrievalCandidate) -> ContextUnit:
        return ContextUnit(
            chunk_id=candidate.chunk_id,
            document_id=str(candidate.metadata.get("doc_id") or ""),
            content=candidate.content,
            metadata=dict(candidate.metadata),
            context_role="hit",
            origin_chunk_id=candidate.chunk_id,
            rerank_score=candidate.rerank_score,
            parent_chunk_id=candidate.metadata.get("parent_chunk_id"),
            position=candidate.metadata.get("position"),
        )

    @staticmethod
    def _chunk_unit(
        chunk: KnowledgeChunk,
        role: Literal["parent", "neighbor"],
        origin_chunk_id: str,
    ) -> ContextUnit:
        return ContextUnit(
            chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
            content=chunk.retrieval_text,
            metadata=dict(chunk.metadata),
            context_role=role,
            origin_chunk_id=origin_chunk_id,
            rerank_score=None,
            parent_chunk_id=chunk.parent_chunk_id,
            position=chunk.position,
        )
