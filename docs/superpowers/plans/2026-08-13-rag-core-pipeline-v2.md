# RAG Core Pipeline v2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在保留 MySQL、ChromaDB、内存 BM25 和现有 API/SSE 协议的前提下，实现可追溯的版本化索引、结构化 Parent-Child 切块、RRF 宽召回重排和上下文补全。

**Architecture:** MySQL 成为文档版本、Parent/Child 内容和相邻关系的事实来源；ChromaDB 与 BM25 只索引 Child。在线检索以稳定 `chunk_id` 合并向量和 BM25 候选，经 RRF、宽候选 Rerank 后交给独立 `ContextAssembler` 补充 Parent 或相邻块。

**Tech Stack:** Python 3.10+、FastAPI、PyMySQL、PyMuPDF、python-docx、ChromaDB、rank-bm25、OpenAI-compatible Embedding/Rerank、pytest。

---

## 执行约束

- 当前工作区已有用户未提交修改：`app/core/metadata_store.py`、`app/core/rag_engine.py`、`tests/test_async_indexing.py`、`tests/test_rag_engine.py` 和 `data/`。每个任务开始前先读当前内容和 `git diff`，不得覆盖这些改动。
- 所有代码修改使用 TDD：先写失败测试，再写最小实现。
- 计划中的 Commit 只有在用户明确授权后才能执行；未授权时跳过 Commit 步骤并保留工作区改动。
- PowerShell 测试命令统一从项目根目录执行：

```powershell
$env:PYTHONPATH = (Resolve-Path '..').Path
pytest -q
```

## 文件结构

### 新增

```text
app/core/document_models.py       结构化文档、Chunk、候选模型
app/core/token_counter.py         Token 计数抽象
app/core/parsers/__init__.py      格式解析器包
app/core/parsers/pdf_parser.py    PDF 结构化解析
app/core/parsers/word_parser.py   Word 结构化解析
app/core/parsers/text_parser.py   Markdown/TXT 结构化解析
app/core/chunk_repository.py      文档版本和 Parent/Child 仓储
app/core/context_assembler.py     Parent/相邻块补全与预算控制
app/core/reindex_service.py       旧 collection 到 v2 的迁移
tests/test_document_models.py
tests/test_structured_parsers.py
tests/test_chunk_repository.py
tests/test_context_assembler.py
tests/test_reindex_service.py
```

### 修改

```text
config.py
requirements.txt
app/core/metadata_store.py
app/core/parser.py
app/core/splitter.py
app/core/retriever.py
app/core/rag_engine.py
app/core/indexing_worker.py
app/api/documents.py
app/models/schemas.py
tests/test_splitter.py
tests/test_retriever.py
tests/test_rag_engine.py
tests/test_async_indexing.py
tests/test_documents_api.py
README.md
```

---

### Task 1: 核心模型、Hash 与 TokenCounter

**Files:**
- Create: `app/core/document_models.py`
- Create: `app/core/token_counter.py`
- Modify: `config.py`
- Modify: `requirements.txt`
- Test: `tests/test_document_models.py`

- [ ] **Step 1: 写稳定 ID、Hash 和 Token 预算失败测试**

```python
from KBzhy.app.core.document_models import KnowledgeChunk, stable_chunk_id
from KBzhy.app.core.token_counter import TokenCounter


def test_stable_chunk_id_is_repeatable_and_versioned():
    first = stable_chunk_id("doc1", 2, 3, "same content")
    second = stable_chunk_id("doc1", 2, 3, "same content")
    changed = stable_chunk_id("doc1", 3, 3, "same content")
    assert first == second
    assert first != changed


def test_knowledge_chunk_builds_heading_aware_retrieval_text():
    chunk = KnowledgeChunk.child(
        document_id="doc1",
        document_version=1,
        parent_chunk_id="parent1",
        content="政治经济制度决定教育性质。",
        section_path=["教育概述", "教育与社会"],
        position=1,
        token_count=12,
    )
    assert chunk.retrieval_text == "教育概述 > 教育与社会\n\n政治经济制度决定教育性质。"
    assert len(chunk.content_hash) == 64


def test_token_counter_truncates_without_exceeding_budget():
    counter = TokenCounter()
    value = counter.truncate("教育制度" * 100, max_tokens=20)
    assert counter.count(value) <= 20
```

- [ ] **Step 2: 运行测试并确认因模块不存在而失败**

Run: `$env:PYTHONPATH=(Resolve-Path '..').Path; pytest tests/test_document_models.py -q`

Expected: FAIL，提示 `KBzhy.app.core.document_models` 不存在。

- [ ] **Step 3: 实现不可变核心模型和稳定 ID**

`document_models.py` 至少定义：

```python
from dataclasses import dataclass, field
from hashlib import sha256
from typing import Any, Literal


def content_hash(content: str | bytes) -> str:
    raw = content.encode("utf-8") if isinstance(content, str) else content
    return sha256(raw).hexdigest()


def stable_chunk_id(document_id: str, version: int, position: int, content: str) -> str:
    raw = f"{document_id}:{version}:{position}:{content_hash(content)}"
    return sha256(raw.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class DocumentElement:
    element_id: str
    element_type: Literal["heading", "paragraph", "list", "table", "code"]
    text: str
    order: int
    page: int | None = None
    section_path: tuple[str, ...] = ()
    bounding_box: dict[str, float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ParsedDocument:
    document_id: str
    version: int
    title: str
    language: str
    elements: tuple[DocumentElement, ...]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class KnowledgeChunk:
    chunk_id: str
    document_id: str
    document_version: int
    parent_chunk_id: str | None
    chunk_type: Literal["parent", "child"]
    content: str
    retrieval_text: str
    content_hash: str
    section_path: tuple[str, ...]
    page_start: int | None
    page_end: int | None
    position: int
    token_count: int
    index_version: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def child(cls, *, document_id, document_version, parent_chunk_id, content,
              section_path, position, token_count, page_start=None, page_end=None,
              index_version=1, metadata=None):
        path = tuple(section_path)
        prefix = " > ".join(path)
        retrieval_text = f"{prefix}\n\n{content}" if prefix else content
        return cls(
            chunk_id=stable_chunk_id(document_id, document_version, position, content),
            document_id=document_id,
            document_version=document_version,
            parent_chunk_id=parent_chunk_id,
            chunk_type="child",
            content=content,
            retrieval_text=retrieval_text,
            content_hash=content_hash(content),
            section_path=path,
            page_start=page_start,
            page_end=page_end,
            position=position,
            token_count=token_count,
            index_version=index_version,
            metadata=metadata or {},
        )
```

- [ ] **Step 4: 实现 TokenCounter 并增加配置**

使用 `tiktoken>=0.8.0`，`TokenCounter` 封装 `count()` 和 `truncate()`；配置增加 `PARENT_CHUNK_TOKENS=2000`、`CHILD_CHUNK_TOKENS=400`、`CONTEXT_TOKEN_BUDGET=6000`、`TOKEN_ENCODING=cl100k_base`。禁止让 splitter 直接依赖具体 tokenizer。

- [ ] **Step 5: 运行新测试和全量测试**

Run: `$env:PYTHONPATH=(Resolve-Path '..').Path; pytest tests/test_document_models.py -q; pytest -q`

Expected: 新测试 PASS；全量测试保持 43 个既有测试全部通过。

- [ ] **Step 6: Commit（需要用户授权）**

```powershell
git add config.py requirements.txt app/core/document_models.py app/core/token_counter.py tests/test_document_models.py
git commit -m "feat: add structured document models"
```

---

### Task 2: MySQL 幂等 Schema Migration 与 ChunkRepository

**Files:**
- Modify: `app/core/metadata_store.py`
- Create: `app/core/chunk_repository.py`
- Create: `tests/test_chunk_repository.py`
- Modify: `tests/test_metadata_store.py`

- [ ] **Step 1: 写 Schema 与仓储失败测试**

测试必须断言：

```python
def test_schema_contains_version_and_chunk_tables(fake_mysql):
    store = MySQLMetadataStore(connection=fake_mysql)
    sql = "\n".join(fake_mysql.executed_sql)
    assert "active_collection_name" in sql
    assert "CREATE TABLE IF NOT EXISTS document_versions" in sql
    assert "CREATE TABLE IF NOT EXISTS document_chunks" in sql


def test_replace_staging_chunks_is_idempotent(chunk_repository):
    chunks = make_parent_and_child_chunks()
    chunk_repository.replace_staging("task1", "doc1", 2, chunks)
    chunk_repository.replace_staging("task1", "doc1", 2, chunks)
    assert chunk_repository.list_by_task("task1") == chunks


def test_get_context_family_returns_parent_and_neighbors(chunk_repository):
    chunk_repository.replace_staging("task1", "doc1", 1, make_three_children())
    family = chunk_repository.get_context_family("child-2", neighbor_window=1)
    assert [item.position for item in family.children] == [1, 2, 3]
    assert family.parent.chunk_type == "parent"
```

- [ ] **Step 2: 运行测试并确认失败**

Run: `$env:PYTHONPATH=(Resolve-Path '..').Path; pytest tests/test_chunk_repository.py tests/test_metadata_store.py -q`

Expected: FAIL，缺少新表和 `ChunkRepository`。

- [ ] **Step 3: 增加幂等字段迁移**

在 `metadata_store.py` 中保留现有 `_ensure_schema()`，增加 `_ensure_column(table, column, ddl)`，通过 `INFORMATION_SCHEMA.COLUMNS` 判断后执行 `ALTER TABLE`。新增：

```text
knowledge_bases.active_collection_name
documents.content_hash/current_version/parser_version/active_index_version/parsed_artifact_path
document_index_tasks.document_version/index_version/attempt_count
```

创建 `document_versions` 和 `document_chunks`，所有 SQL 使用参数绑定；不得改动用户已有的 legacy JSON 归档逻辑。

- [ ] **Step 4: 实现 ChunkRepository 的事务接口**

至少提供以下精确接口：`replace_staging(task_id, document_id, version, chunks)`、`activate_version(document_id, version)`、`discard_task(task_id)`、`list_active_children(document_id=None)`、`get_context_family(chunk_id, neighbor_window=1)`、`get_active_versions(document_ids)`。返回值分别为 `None`、`None`、`None`、`list[KnowledgeChunk]`、`ContextFamily` 和 `dict[str, int]`。

`replace_staging()` 必须在一个 MySQL 事务中先删除同 `task_id` staging 行再批量插入，Parent 和 Child 都写入 MySQL。

- [ ] **Step 5: 运行仓储测试和全量测试**

Run: `$env:PYTHONPATH=(Resolve-Path '..').Path; pytest tests/test_chunk_repository.py tests/test_metadata_store.py tests/test_async_indexing.py -q; pytest -q`

Expected: 全部 PASS，legacy JSON 迁移归档测试仍通过。

- [ ] **Step 6: Commit（需要用户授权）**

```powershell
git add app/core/metadata_store.py app/core/chunk_repository.py tests/test_chunk_repository.py tests/test_metadata_store.py
git commit -m "feat: persist document versions and chunks"
```

---

### Task 3: 上传 Hash、重复检测与版本创建

**Files:**
- Modify: `app/api/documents.py`
- Modify: `app/core/metadata_store.py`
- Modify: `app/models/schemas.py`
- Modify: `tests/test_documents_api.py`
- Modify: `tests/test_async_indexing.py`

- [ ] **Step 1: 写重复上传与 no-op 更新测试**

```python
def test_duplicate_upload_returns_409_with_existing_document(client, store):
    store.find_document_by_hash.return_value = {"id": "existing"}
    response = client.post("/api/knowledge-bases/kb1/documents/upload", files=pdf_file())
    assert response.status_code == 409
    assert response.json()["detail"]["document_id"] == "existing"


def test_unchanged_update_does_not_enqueue_new_task(client, worker, store):
    store.get_document.return_value = {"id": "doc1", "content_hash": sha256(PDF).hexdigest()}
    response = client.put("/api/knowledge-bases/kb1/documents/doc1", files=pdf_file())
    assert response.status_code == 200
    assert response.json()["message"] == "文档内容未变化，无需重新索引"
    worker.enqueue.assert_not_called()
```

- [ ] **Step 2: 运行测试并确认失败**

Run: `$env:PYTHONPATH=(Resolve-Path '..').Path; pytest tests/test_documents_api.py tests/test_async_indexing.py -q`

Expected: FAIL，当前接口没有 Hash 重复判断。

- [ ] **Step 3: 实现 API Hash 流程**

上传内容读取后立即计算 SHA-256；`find_document_by_hash(kb_id, hash)` 命中时抛出结构化 409。更新相同 Hash 时返回现有 `DocumentUpdateResponse`，不删除文件、不移除旧索引、不创建任务。

内容变化时由 `create_document_version_and_task()` 在单事务内增加版本并创建任务，避免文档版本和任务分开提交。

- [ ] **Step 4: 运行 API、异步索引和全量测试**

Run: `$env:PYTHONPATH=(Resolve-Path '..').Path; pytest tests/test_documents_api.py tests/test_async_indexing.py -q; pytest -q`

Expected: PASS。

- [ ] **Step 5: Commit（需要用户授权）**

```powershell
git add app/api/documents.py app/core/metadata_store.py app/models/schemas.py tests/test_documents_api.py tests/test_async_indexing.py
git commit -m "feat: add content hash based indexing"
```

---

### Task 4: StructuredParser 入口和解析产物持久化

**Files:**
- Modify: `app/core/parser.py`
- Create: `app/core/parsers/__init__.py`
- Create: `tests/test_structured_parsers.py`
- Modify: `config.py`

- [ ] **Step 1: 写解析入口和 JSON round-trip 测试**

```python
def test_parser_returns_parsed_document(tmp_path):
    parsed = DocumentParser().parse_structured(markdown_fixture(), document_id="doc1", version=2)
    assert parsed.document_id == "doc1"
    assert parsed.version == 2
    assert parsed.elements


def test_parsed_artifact_round_trip(tmp_path):
    parser = DocumentParser(artifact_dir=tmp_path)
    parsed = parser.parse_structured(markdown_fixture(), document_id="doc1", version=1)
    path = parser.save_artifact(parsed)
    assert parser.load_artifact(path) == parsed
```

- [ ] **Step 2: 运行并确认失败**

Run: `$env:PYTHONPATH=(Resolve-Path '..').Path; pytest tests/test_structured_parsers.py -q`

Expected: FAIL，现有 parser 返回 `list[Document]`。

- [ ] **Step 3: 改造解析入口但保留兼容适配器**

新增 `DocumentParser.parse_structured()` 返回 `ParsedDocument`；现有 `parse()` 在 Task 7 完成接线前保持原返回值，确保每个任务结束时全量测试可通过。为未深度改造格式提供 `_legacy_to_elements()`，把旧 `Document` 转成 Paragraph/Table Element。解析产物保存到：

```text
data/parsed/{kb_id}/{doc_id}/v{version}.json
```

JSON 写入采用临时文件加 `os.replace()`，避免进程中断留下半文件。

- [ ] **Step 4: 运行解析测试和全量测试**

Run: `$env:PYTHONPATH=(Resolve-Path '..').Path; pytest tests/test_structured_parsers.py tests/test_rag_engine.py -q`

Expected: 新接口 PASS；旧索引调用将在后续 Task 适配前以明确测试替身隔离。

- [ ] **Step 5: Commit（需要用户授权）**

```powershell
git add config.py app/core/parser.py app/core/parsers/__init__.py tests/test_structured_parsers.py
git commit -m "refactor: introduce structured parser model"
```

---

### Task 5: PDF、Word、Markdown/TXT 结构化解析器

**Files:**
- Create: `app/core/parsers/pdf_parser.py`
- Create: `app/core/parsers/word_parser.py`
- Create: `app/core/parsers/text_parser.py`
- Modify: `app/core/parser.py`
- Modify: `tests/test_structured_parsers.py`
- Add fixtures under: `tests/fixtures/`

- [ ] **Step 1: 为三类格式写结构断言**

测试覆盖：PDF 页码与 block 顺序、重复页眉页脚移除；Word Heading section path 与表格；Markdown Heading、代码围栏和表格；TXT 段落降级。示例：

```python
def test_markdown_parser_preserves_heading_path_and_code():
    parsed = parse_markdown("# API\n## 创建用户\n```python\ncreate_user()\n```")
    code = next(e for e in parsed.elements if e.element_type == "code")
    assert code.section_path == ("API", "创建用户")
    assert code.text == "```python\ncreate_user()\n```"


def test_scanned_pdf_is_rejected_instead_of_indexing_placeholder(scanned_pdf):
    with pytest.raises(DocumentParseError, match="未提取到可索引文字"):
        parse_pdf(scanned_pdf)
```

- [ ] **Step 2: 运行并确认失败**

Run: `$env:PYTHONPATH=(Resolve-Path '..').Path; pytest tests/test_structured_parsers.py -q`

- [ ] **Step 3: 实现 PDF Parser**

使用 `page.get_text("dict")` 生成 block；根据字体大小和短行规则识别 Heading；使用规范化文本的跨页出现次数识别页眉页脚；无有效 Element 时抛出 `DocumentParseError`，禁止索引错误占位文本。

- [ ] **Step 4: 实现 Word 和 Markdown/TXT Parser**

Word 按段落样式更新 section path，表格转为每行都带表头的 Markdown；Markdown 使用状态机保护 fenced code，再识别 Heading、列表和表格；TXT 只做段落与标题启发式，不虚构 section path。

- [ ] **Step 5: 运行解析专项和全量测试**

Run: `$env:PYTHONPATH=(Resolve-Path '..').Path; pytest tests/test_structured_parsers.py -q; pytest -q`

Expected: PASS。

- [ ] **Step 6: Commit（需要用户授权）**

```powershell
git add app/core/parser.py app/core/parsers tests/test_structured_parsers.py tests/fixtures
git commit -m "feat: parse document structure"
```

---

### Task 6: Parent-Child StructuralChunker

**Files:**
- Modify: `app/core/splitter.py`
- Modify: `tests/test_splitter.py`

- [ ] **Step 1: 写 Parent-Child、边界和内容完整性测试**

```python
def test_child_chunks_never_cross_parent_boundary(chunker, parsed_two_sections):
    chunks = chunker.split(parsed_two_sections, index_version=1)
    parents = {item.chunk_id for item in chunks if item.chunk_type == "parent"}
    assert all(item.parent_chunk_id in parents for item in chunks if item.chunk_type == "child")


def test_short_children_merge_only_within_same_parent(chunker, parsed_two_sections):
    children = [item for item in chunker.split(parsed_two_sections, 1) if item.chunk_type == "child"]
    assert all("第一节" not in item.content or "第二节" not in item.content for item in children)


def test_retrieval_text_contains_section_breadcrumb(chunker, parsed_two_sections):
    child = next(item for item in chunker.split(parsed_two_sections, 1) if item.chunk_type == "child")
    assert " > ".join(child.section_path) in child.retrieval_text


def test_table_and_code_have_no_internal_marker_characters(chunker, parsed_with_table_and_code):
    content = "\n".join(item.content for item in chunker.split(parsed_with_table_and_code, 1))
    assert "␟TABLE␟" not in content
    assert "␤" not in content


def test_same_input_produces_same_chunk_ids(chunker, parsed_two_sections):
    first = chunker.split(parsed_two_sections, 1)
    second = chunker.split(parsed_two_sections, 1)
    assert [item.chunk_id for item in first] == [item.chunk_id for item in second]


def test_each_child_stays_within_token_limit(chunker, parsed_long_section):
    children = [item for item in chunker.split(parsed_long_section, 1) if item.chunk_type == "child"]
    assert max(item.token_count for item in children) <= chunker.child_token_limit
```

- [ ] **Step 2: 运行并确认旧 SmartSplitter 无法满足新返回模型**

Run: `$env:PYTHONPATH=(Resolve-Path '..').Path; pytest tests/test_splitter.py -q`

- [ ] **Step 3: 实现 StructuralChunker**

`split(parsed: ParsedDocument, index_version: int) -> list[KnowledgeChunk]`：先按 section path 聚合 Parent，再按 Element 原子边界构建 Child，只有超长 Element 才使用 TokenCounter 兜底切分。Parent position 与 Child position 分别单调，Child `parent_chunk_id` 必填。

删除旧 `_mark_structured()` 的占位字符做法；为旧调用保留薄适配方法，但 v2 索引只调用新接口。

- [ ] **Step 4: 运行切块、解析和全量测试**

Run: `$env:PYTHONPATH=(Resolve-Path '..').Path; pytest tests/test_splitter.py tests/test_structured_parsers.py tests/test_document_models.py -q; pytest -q`

Expected: PASS。

- [ ] **Step 5: Commit（需要用户授权）**

```powershell
git add app/core/splitter.py tests/test_splitter.py
git commit -m "feat: add parent child chunking"
```

---

### Task 7: 版本化索引、Child Upsert 与失败补偿

**Files:**
- Modify: `app/core/retriever.py`
- Modify: `app/core/rag_engine.py`
- Modify: `app/core/indexing_worker.py`
- Modify: `app/core/chunk_repository.py`
- Modify: `tests/test_async_indexing.py`
- Modify: `tests/test_rag_engine.py`
- Modify: `tests/test_retriever.py`

- [ ] **Step 1: 写索引幂等、激活切换和清理失败测试**

覆盖：仅 Child 进入 Chroma/BM25；稳定 ID 作为 Chroma ID；相同任务重复执行不重复；新版本写入失败时旧 active Child 从未删除；MySQL 激活后旧版本候选会被过滤；旧 Child 清理失败不影响新版本查询。

- [ ] **Step 2: 运行并确认失败**

Run: `$env:PYTHONPATH=(Resolve-Path '..').Path; pytest tests/test_async_indexing.py tests/test_rag_engine.py tests/test_retriever.py -q`

- [ ] **Step 3: 实现 Retriever Child upsert 接口**

```python
def stage_document_children(
    self,
    kb_id: str,
    document_id: str,
    new_children: list[KnowledgeChunk],
) -> None:
    """以稳定 ID 幂等写入新 Child，不删除旧 active Child。"""

def remove_children(self, kb_id: str, chunk_ids: list[str]) -> None:
    """激活新版本后按显式 ID 清理旧 Child。"""
```

Chroma 写入显式传入 `ids=[chunk.chunk_id, ...]`；metadata 包含 `chunk_id`、`doc_id`、`document_version`、`parent_chunk_id`、`section_path`、页码和 position。BM25 entries 同样以 `chunk_id` 为唯一键重建。

- [ ] **Step 4: 改造 IndexingWorker 状态机**

严格执行 `parsing → chunking → embedding/indexing → ready`；先保存 staging MySQL Chunk，再将新 Child 写入 Chroma/BM25，验证后 activate MySQL version，最后清理旧 Child。异常路径调用 `discard_task()` 并按新 Chunk ID 清理部分写入；旧 active Child 在整个失败路径中保持不变。

保留用户已有的 stale task 和 legacy migration 行为。

- [ ] **Step 5: 运行索引专项和全量测试**

Run: `$env:PYTHONPATH=(Resolve-Path '..').Path; pytest tests/test_async_indexing.py tests/test_rag_engine.py tests/test_retriever.py -q; pytest -q`

Expected: PASS。

- [ ] **Step 6: Commit（需要用户授权）**

```powershell
git add app/core/retriever.py app/core/rag_engine.py app/core/indexing_worker.py app/core/chunk_repository.py tests/test_async_indexing.py tests/test_rag_engine.py tests/test_retriever.py
git commit -m "feat: add versioned idempotent indexing"
```

---

### Task 8: 旧知识库安全重索引服务

**Files:**
- Create: `app/core/reindex_service.py`
- Create: `tests/test_reindex_service.py`
- Modify: `app/core/metadata_store.py`

- [ ] **Step 1: 写迁移成功、失败和显式清理测试**

```python
def test_reindex_switches_active_collection_only_after_all_documents_succeed(service, store):
    result = service.reindex("kb1")
    assert result.status == "ready"
    assert store.get_kb("kb1")["active_collection_name"] == result.collection_name


def test_reindex_failure_keeps_previous_active_collection(service, store, failing_engine):
    store.set_active_collection("kb1", "kbzhy_kb1")
    service.engine = failing_engine
    with pytest.raises(RuntimeError):
        service.reindex("kb1")
    assert store.get_kb("kb1")["active_collection_name"] == "kbzhy_kb1"


def test_cleanup_refuses_to_delete_active_collection(service, store):
    store.set_active_collection("kb1", "kbzhy_kb1_v2")
    with pytest.raises(ValueError, match="active collection"):
        service.cleanup_collection("kb1", "kbzhy_kb1_v2")
```

- [ ] **Step 2: 运行并确认服务不存在**

Run: `$env:PYTHONPATH=(Resolve-Path '..').Path; pytest tests/test_reindex_service.py -q`

- [ ] **Step 3: 实现 ReindexService**

为 KB 创建 `kbzhy_{kb_id}_v2_{timestamp}`，按 ready 文档重新解析和索引；全部成功后单事务更新 `active_collection_name`。失败时删除临时 collection，旧 collection 不变。旧 collection 清理必须是单独显式方法，并校验目标不是 active collection。

- [ ] **Step 4: 增加受控命令行入口**

支持 `python -m KBzhy.app.core.reindex_service --kb-id <id>`。命令同步运行并以退出码表示成功或失败；不增加 API 和新任务表，避免把运维入口扩展为本轮业务协议。

- [ ] **Step 5: 运行迁移、API 和全量测试**

Run: `$env:PYTHONPATH=(Resolve-Path '..').Path; pytest tests/test_reindex_service.py -q; pytest -q`

- [ ] **Step 6: Commit（需要用户授权）**

```powershell
git add app/core/reindex_service.py app/core/metadata_store.py tests/test_reindex_service.py
git commit -m "feat: add safe knowledge base reindexing"
```

---

### Task 9: RetrievalCandidate、双路宽召回与 RRF

**Files:**
- Modify: `app/core/document_models.py`
- Modify: `app/core/retriever.py`
- Modify: `config.py`
- Modify: `tests/test_retriever.py`

- [ ] **Step 1: 写 RRF 的确定性测试**

```python
def test_rrf_merges_same_chunk_by_id():
    vector = [candidate("a"), candidate("b")]
    bm25 = [candidate("b"), candidate("c")]
    fused = rrf_fuse(vector, bm25, k=60)
    assert fused[0].chunk_id == "b"
    assert fused[0].vector_rank == 2
    assert fused[0].bm25_rank == 1


def test_rrf_ignores_raw_score_scale():
    high = candidate("a", raw_score=9999)
    low = candidate("b", raw_score=0.01)
    assert rrf_fuse([low, high], [], k=60)[0].chunk_id == "b"


def test_retrieve_does_not_apply_threshold_before_rerank(retriever):
    retriever._vector_search = lambda *args: [candidate("a", raw_score=0.01)]
    retriever._bm25_search = lambda *args: []
    retriever._rerank = lambda items, **kwargs: rerank_result(items, scores=[0.9])
    assert retriever.retrieve("query", "kb1", threshold=0.8)[0].chunk_id == "a"


def test_vector_and_bm25_each_fetch_configured_top_n(retriever):
    retriever.retrieve("query", "kb1")
    retriever._vector_search.assert_called_once_with("query", "kb1", 30)
    retriever._bm25_search.assert_called_once_with("query", "kb1", 30)


def test_candidates_from_inactive_document_version_are_removed_before_rrf(retriever, repository):
    repository.get_active_versions.return_value = {"doc1": 2}
    retriever._vector_search.return_value = [candidate("old", doc_id="doc1", version=1)]
    retriever._bm25_search.return_value = [candidate("new", doc_id="doc1", version=2)]
    result = retriever.retrieve("query", "kb1")
    assert [item.chunk_id for item in result] == ["new"]
```

- [ ] **Step 2: 运行并确认旧加权融合测试失败**

Run: `$env:PYTHONPATH=(Resolve-Path '..').Path; pytest tests/test_retriever.py -q`

- [ ] **Step 3: 实现 Candidate 和纯函数 RRF**

增加 `RetrievalCandidate`，分别保存 rank、raw score、RRF score 和 rerank score。`rrf_fuse()` 只依赖排名，并以 `chunk_id` 合并。

配置替换为：

```text
VECTOR_FETCH_K=30
BM25_FETCH_K=30
RRF_K=60
RRF_CANDIDATE_K=40
RERANK_CANDIDATE_K=30
```

旧 `BM25_WEIGHT`、`VECTOR_WEIGHT` 保留一个版本的弃用兼容但不再参与 v2 排名。

- [ ] **Step 4: 并发执行双路召回并返回 Candidate**

使用现有 `ThreadPoolExecutor` 同时调用 vector/BM25；任一路失败时记录日志并使用另一路，不因单路失败返回空结果。融合前从 MySQL 批量读取候选文档的当前版本，丢弃 `document_version` 不匹配的候选；去重必须基于 `chunk_id`。

- [ ] **Step 5: 运行检索专项和全量测试**

Run: `$env:PYTHONPATH=(Resolve-Path '..').Path; pytest tests/test_retriever.py -q; pytest -q`

- [ ] **Step 6: Commit（需要用户授权）**

```powershell
git add config.py app/core/document_models.py app/core/retriever.py tests/test_retriever.py
git commit -m "feat: fuse hybrid retrieval with rrf"
```

---

### Task 10: 宽候选 Rerank 与独立降级阈值

**Files:**
- Modify: `app/core/retriever.py`
- Modify: `config.py`
- Modify: `tests/test_retriever.py`

- [ ] **Step 1: 写候选数量、模型阈值和回退测试**

```python
def test_reranker_receives_top_30_rrf_candidates(retriever):
    retriever._rrf_candidates = make_candidates(40)
    retriever.retrieve("query", "kb1")
    assert len(retriever.reranker.last_candidates) == 30


def test_model_threshold_is_applied_after_rerank(retriever):
    retriever.reranker.return_value = rerank_result(make_candidates(2), scores=[0.7, 0.2], method="model")
    result = retriever.retrieve("query", "kb1", threshold=0.5)
    assert [item.rerank_score for item in result] == [0.7]


def test_keyword_fallback_does_not_use_model_threshold(retriever):
    retriever.reranker.return_value = rerank_result(make_candidates(2), scores=[0.3, 0.2], method="keyword")
    result = retriever.retrieve("query", "kb1", threshold=0.8)
    assert len(result) == 2


def test_rerank_failure_preserves_rrf_candidate_order_when_keyword_has_no_signal(retriever):
    retriever.reranker.raise_model_error = True
    original = make_candidates(3)
    retriever._rrf_candidates = original
    assert [item.chunk_id for item in retriever.retrieve("query", "kb1")] == [item.chunk_id for item in original]
```

- [ ] **Step 2: 运行并确认当前流程只重排 top_k**

Run: `$env:PYTHONPATH=(Resolve-Path '..').Path; pytest tests/test_retriever.py -q`

- [ ] **Step 3: 重构 RerankResult**

让 `_rerank()` 返回 `RerankResult(items, method, threshold_applied)`；模型模式使用 `MODEL_RERANK_THRESHOLD`，关键词回退使用 `KEYWORD_RERANK_THRESHOLD` 或只排序。最终 `top_k` 只在 ContextAssembler 之前截取，不能在 Rerank 前截断。

- [ ] **Step 4: 运行检索和 RAG 回归**

Run: `$env:PYTHONPATH=(Resolve-Path '..').Path; pytest tests/test_retriever.py tests/test_rag_engine.py -q; pytest -q`

- [ ] **Step 5: Commit（需要用户授权）**

```powershell
git add config.py app/core/retriever.py tests/test_retriever.py
git commit -m "feat: rerank wide candidate sets"
```

---

### Task 11: ContextAssembler

**Files:**
- Create: `app/core/context_assembler.py`
- Create: `tests/test_context_assembler.py`
- Modify: `config.py`

- [ ] **Step 1: 写配额、Parent、相邻块和预算测试**

```python
def test_limits_hits_per_document_to_three(assembler):
    units = assembler.assemble(candidates_from_document("doc1", count=5), final_k=8)
    assert sum(unit.document_id == "doc1" and unit.context_role == "hit" for unit in units) == 3


def test_deduplicates_same_parent(assembler):
    units = assembler.assemble(two_hits_with_same_parent(), final_k=8)
    assert sum(unit.context_role == "parent" for unit in units) == 1


def test_adds_neighbors_without_inheriting_hit_score(assembler):
    units = assembler.assemble([single_hit(score=0.9)], final_k=8)
    neighbor = next(unit for unit in units if unit.context_role == "neighbor")
    assert neighbor.rerank_score is None
    assert neighbor.origin_chunk_id == "hit-1"


def test_uses_parent_when_it_fits_single_source_budget(assembler):
    units = assembler.assemble([single_hit(score=0.9)], final_k=8)
    assert any(unit.context_role == "parent" for unit in units)


def test_falls_back_to_hit_child_when_repository_fails(assembler):
    assembler.repository.get_context_family.side_effect = RuntimeError("mysql unavailable")
    units = assembler.assemble([single_hit(score=0.9)], final_k=8)
    assert [unit.chunk_id for unit in units] == ["hit-1"]


def test_context_never_exceeds_token_budget(assembler, token_counter):
    units = assembler.assemble(make_large_candidates(), final_k=12)
    assert sum(token_counter.count(unit.content) for unit in units) <= assembler.token_budget
```

- [ ] **Step 2: 运行并确认模块不存在**

Run: `$env:PYTHONPATH=(Resolve-Path '..').Path; pytest tests/test_context_assembler.py -q`

- [ ] **Step 3: 实现 ContextAssembler**

```python
class ContextAssembler:
    def __init__(self, repository: ChunkRepository, token_counter: TokenCounter,
                 token_budget: int, per_document_limit: int = 3):
        self.repository = repository
        self.token_counter = token_counter
        self.token_budget = token_budget
        self.per_document_limit = per_document_limit

    def assemble(self, candidates: list[RetrievalCandidate], final_k: int) -> list[ContextUnit]:
        selected = self._apply_document_quota(candidates)[:final_k]
        expanded = self._expand_context(selected)
        return self._fit_token_budget(expanded)
```

同一文件内实现 `_apply_document_quota()`、`_expand_context()` 和 `_fit_token_budget()`；三者分别只负责配额、上下文补全和预算，不互相改变候选分数。

算法顺序固定为：Rerank 顺序 → 每文档命中配额 → parent 去重 → Parent/相邻块补充 → Token 预算。补充块记录 `origin_chunk_id` 和 `context_role`，但不伪造 rerank 分数。

- [ ] **Step 4: 运行 ContextAssembler 与全量测试**

Run: `$env:PYTHONPATH=(Resolve-Path '..').Path; pytest tests/test_context_assembler.py -q; pytest -q`

- [ ] **Step 5: Commit（需要用户授权）**

```powershell
git add config.py app/core/context_assembler.py tests/test_context_assembler.py
git commit -m "feat: assemble parent aware context"
```

---

### Task 12: RAGEngine 接线、兼容输出与端到端回归

**Files:**
- Modify: `app/core/rag_engine.py`
- Modify: `app/core/engine.py`
- Modify: `app/api/chat.py`
- Modify: `tests/test_rag_engine.py`
- Modify: `tests/test_chat_stream.py`
- Modify: `tests/test_documents_api.py`

- [ ] **Step 1: 写 Engine 接线和兼容性测试**

```python
def test_engine_generates_from_assembled_context_not_raw_candidates(engine):
    engine.retriever.retrieve.return_value = [candidate("hit")]
    engine.context_assembler.assemble.return_value = [context_unit("parent text")]
    engine.chat("question", "kb1")
    assert "parent text" in engine._call_llm_sync.call_args.args[0][0]["content"]


def test_no_candidates_still_returns_existing_refusal(engine):
    engine.retriever.retrieve.return_value = []
    assert engine.chat("question", "kb1")["answer"] == KNOWLEDGE_QA_REFUSAL


def test_sources_only_include_original_hits_not_neighbor_context(engine):
    engine.retriever.retrieve.return_value = [candidate("hit")]
    engine.context_assembler.assemble.return_value = [context_unit("neighbor", role="neighbor")]
    result = engine.chat("question", "kb1")
    assert [source["content"] for source in result["sources"]] == ["hit"]


def test_chat_response_schema_is_unchanged(client):
    body = client.post("/api/chat", json={"question": "q", "kb_id": "kb1"}).json()
    assert set(body) == {"answer", "session_id", "sources", "hallucination_flags"}


def test_sse_sources_event_is_unchanged(engine):
    events = list(engine.chat_stream("question", "kb1"))
    assert events[-1].startswith("[SOURCES]")
```

- [ ] **Step 2: 运行并确认 raw retrieval results 仍直接进入生成**

Run: `$env:PYTHONPATH=(Resolve-Path '..').Path; pytest tests/test_rag_engine.py tests/test_chat_stream.py -q`

- [ ] **Step 3: 接入 ContextAssembler**

`RAGEngine` 构造时注入 parser、splitter、repository、retriever、assembler；`chat()` 和 `chat_stream()` 都执行同一条检索与组装逻辑。`_build_context()` 接收 `ContextUnit`，而 sources 仍映射原始命中 Candidate，保持 API Schema 和 SSE `[SOURCES]` 事件不变。

保留用户已添加的 `_is_refusal_answer()` 及拒答后清空 sources 行为。

- [ ] **Step 4: 运行后端和前端静态回归**

Run: `$env:PYTHONPATH=(Resolve-Path '..').Path; pytest -q`

Expected: 所有测试 PASS。

Run: `npm test --if-present`

Workdir: `frontend`

Expected: 若未配置 test script，命令正常退出；不得新增前端协议改动。

- [ ] **Step 5: Commit（需要用户授权）**

```powershell
git add app/core/rag_engine.py app/core/engine.py app/api/chat.py tests/test_rag_engine.py tests/test_chat_stream.py tests/test_documents_api.py
git commit -m "feat: integrate rag core pipeline v2"
```

---

### Task 13: 文档、迁移演练与最终验证

**Files:**
- Modify: `README.md`
- Modify: `.env.example`
- Verify: all modified files

- [ ] **Step 1: 更新配置和运维文档**

README 必须说明：v2 数据模型、支持格式范围、重复上传行为、重索引 API、旧 collection 清理流程、RRF/Rerank/Context 配置和故障恢复。`.env.example` 增加所有新配置但不包含真实密钥。

- [ ] **Step 2: 在临时目录执行迁移演练**

复制测试 fixture 到临时数据目录，创建旧 collection，执行 reindex，验证：旧 collection 在切换前可用、失败不切换、成功后 active collection 改变、显式清理不能删除 active collection。

- [ ] **Step 3: 运行完整测试**

Run: `$env:PYTHONPATH=(Resolve-Path '..').Path; pytest -q`

Expected: 全部 PASS，无新增 warning；现有 jieba/setuptools warning 可记录但不作为本次失败。

- [ ] **Step 4: 运行编译和导入检查**

Run: `$env:PYTHONPATH=(Resolve-Path '..').Path; python -m compileall -q app; python -c "from KBzhy.main import app; print(app.title)"`

Expected: 退出码 0，输出 `KBzhy RAG Knowledge Base QA System`。

- [ ] **Step 5: 审查工作区范围**

Run: `git status --short; git diff --stat; git diff --check`

Expected: 只有计划内文件和用户原有改动；`git diff --check` 无空白错误。

- [ ] **Step 6: Commit（需要用户授权）**

```powershell
git add README.md .env.example
git commit -m "docs: document rag pipeline v2"
```

## 最终验收清单

- [ ] 同一文档 Hash 不重复创建索引。
- [ ] 索引任务可重入，稳定 Chunk ID 不产生重复数据。
- [ ] 更新失败时上一 active 版本仍可检索。
- [ ] PDF、Word、Markdown/TXT 产生结构化 Element 和 Parent-Child Chunk。
- [ ] Parent 不进入向量/BM25，Child 带 section path 和稳定 ID。
- [ ] 检索使用 Vector Top30、BM25 Top30、RRF 和宽候选 Rerank。
- [ ] 模型与关键词回退阈值独立。
- [ ] ContextAssembler 执行文档配额、Parent/相邻补全和 Token 预算。
- [ ] 现有聊天响应和 SSE 协议保持兼容。
- [ ] 旧 collection 可安全迁移，失败不影响当前 active collection。
- [ ] 全部测试、编译、导入和 diff 检查通过。
