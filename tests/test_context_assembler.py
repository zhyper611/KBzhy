from KBzhy.app.core.chunk_repository import ContextFamily
from KBzhy.app.core.context_assembler import ContextAssembler
from KBzhy.app.core.document_models import KnowledgeChunk, RetrievalCandidate


class WordCounter:
    def count(self, text):
        return len(text.split())

    def truncate(self, text, max_tokens):
        return " ".join(text.split()[:max_tokens])


class Repository:
    def __init__(self, families=None, error=None):
        self.families = families or {}
        self.error = error

    def get_context_family(self, chunk_id, neighbor_window=1):
        if self.error:
            raise self.error
        return self.families.get(chunk_id, ContextFamily(None, ()))


def chunk(chunk_id, *, doc_id="doc-1", kind="child", parent_id="parent-1", position=0, text=None):
    content = text or chunk_id
    return KnowledgeChunk(
        chunk_id=chunk_id,
        document_id=doc_id,
        document_version=1,
        parent_chunk_id=parent_id if kind == "child" else None,
        chunk_type=kind,
        content=content,
        retrieval_text=content,
        content_hash=f"hash-{chunk_id}",
        section_path=(),
        page_start=None,
        page_end=None,
        position=position,
        token_count=len(content.split()),
        index_version=1,
        metadata={"source": f"{doc_id}.txt"},
    )


def candidate(chunk_id, *, doc_id="doc-1", score=0.9, content=None, parent_id="parent-1"):
    return RetrievalCandidate(
        chunk_id=chunk_id,
        content=content or chunk_id,
        metadata={
            "doc_id": doc_id,
            "parent_chunk_id": parent_id,
            "position": 0,
            "source": f"{doc_id}.txt",
        },
        rrf_score=0.1,
        rerank_score=score,
    )


def assembler(repository=None, **kwargs):
    return ContextAssembler(
        repository or Repository(),
        WordCounter(),
        token_budget=kwargs.pop("token_budget", 100),
        single_source_budget=kwargs.pop("single_source_budget", 30),
        **kwargs,
    )


def test_limits_original_hits_per_document_to_three():
    candidates = [candidate(f"hit-{index}") for index in range(5)]

    units = assembler(per_document_limit=3).assemble(candidates, final_k=8)

    assert sum(unit.context_role == "hit" for unit in units) == 3


def test_document_quota_still_allows_hits_from_other_documents():
    candidates = [candidate(f"a-{index}") for index in range(4)] + [
        candidate("b-1", doc_id="doc-2")
    ]

    units = assembler(per_document_limit=3).assemble(candidates, final_k=8)

    assert [unit.chunk_id for unit in units if unit.context_role == "hit"] == [
        "a-0", "a-1", "a-2", "b-1"
    ]


def test_deduplicates_parent_shared_by_multiple_hits():
    parent = chunk("parent-1", kind="parent", text="shared parent")
    repository = Repository({
        "hit-1": ContextFamily(parent, (chunk("hit-1"),)),
        "hit-2": ContextFamily(parent, (chunk("hit-2", position=1),)),
    })

    units = assembler(repository).assemble(
        [candidate("hit-1"), candidate("hit-2")], final_k=8
    )

    assert sum(unit.context_role == "parent" for unit in units) == 1


def test_neighbors_do_not_inherit_hit_score_and_track_origin():
    neighbor = chunk("neighbor", position=1)
    repository = Repository({
        "hit-1": ContextFamily(None, (chunk("hit-1"), neighbor))
    })

    units = assembler(repository).assemble([candidate("hit-1", score=0.95)], final_k=8)
    unit = next(item for item in units if item.context_role == "neighbor")

    assert unit.rerank_score is None
    assert unit.origin_chunk_id == "hit-1"


def test_parent_is_added_only_when_it_fits_single_source_budget():
    fitting = chunk("parent-1", kind="parent", text="one two three")
    oversized = chunk("parent-2", kind="parent", text="one two three four")
    repository = Repository({
        "hit-1": ContextFamily(fitting, ()),
        "hit-2": ContextFamily(oversized, ()),
    })

    units = assembler(repository, single_source_budget=3).assemble(
        [candidate("hit-1"), candidate("hit-2", parent_id="parent-2")], final_k=8
    )

    assert [unit.chunk_id for unit in units if unit.context_role == "parent"] == ["parent-1"]


def test_repository_failure_falls_back_to_original_hit():
    repository = Repository(error=RuntimeError("mysql unavailable"))

    units = assembler(repository).assemble([candidate("hit-1")], final_k=8)

    assert [unit.chunk_id for unit in units] == ["hit-1"]


def test_context_never_exceeds_token_budget():
    parent = chunk("parent", kind="parent", text="p1 p2 p3 p4")
    neighbor = chunk("neighbor", text="n1 n2 n3 n4")
    repository = Repository({"hit": ContextFamily(parent, (neighbor,))})

    units = assembler(repository, token_budget=6).assemble(
        [candidate("hit", content="h1 h2 h3")], final_k=8
    )

    assert sum(WordCounter().count(unit.content) for unit in units) <= 6
    assert units[0].chunk_id == "hit"


def test_oversized_highest_hit_is_truncated_instead_of_dropped():
    units = assembler(token_budget=3).assemble(
        [candidate("top", content="one two three four five")], final_k=8
    )

    assert len(units) == 1
    assert units[0].chunk_id == "top"
    assert units[0].content == "one two three"


def test_token_counter_failure_keeps_highest_ranked_hit():
    class FailingCounter:
        def count(self, text):
            raise RuntimeError("encoding unavailable")

        def truncate(self, text, max_tokens):
            raise RuntimeError("encoding unavailable")

    instance = ContextAssembler(Repository(), FailingCounter(), token_budget=3)

    units = instance.assemble(
        [candidate("top"), candidate("second", score=0.8)], final_k=8
    )

    assert [unit.chunk_id for unit in units] == ["top"]
