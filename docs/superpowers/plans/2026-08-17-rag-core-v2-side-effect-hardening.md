# RAG Core v2 Side-Effect Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 修复 RAG 核心流程 v2 合并前确认的迁移、恢复、重索引、元数据、来源契约和降级拒答副作用。

**Architecture:** 保留现有 MySQL、ChromaDB、内存 BM25 和后台 Worker。用 MySQL 任务租约串行化多进程恢复，用独立 artifact 名隔离重索引；ContextAssembler 在兼容旧列表接口的同时返回最终直接命中和上下文单元，RAGEngine 统一用该结果生成回答和 sources。

**Tech Stack:** Python 3.10+、FastAPI、PyMySQL、ChromaDB、pytest、Pydantic、tiktoken

---

## 文件结构与职责

- `app/core/metadata_store.py`：schema 默认值、旧状态/JSON 迁移、恢复租约事务、active chunk_count 语义。
- `app/core/indexing_worker.py`：先 claim 恢复租约，再清理并重新排队。
- `app/core/parser.py`：受限 artifact 名称和原子持久化。
- `app/core/rag_engine.py`：传递重索引 artifact 名；统一使用组装结果构造上下文、sources 和幻觉检测输入。
- `app/core/reindex_service.py`：为每个 reindex task 使用独立 artifact。
- `app/core/context_assembler.py`：输出最终直接命中及预算后的上下文单元。
- `app/api/documents.py`：更新响应保留 active chunk_count。
- `config.py`、`.env.example`、`README.md`：关键词降级默认阈值和运行语义。
- 对应 `tests/` 文件：每个行为先写失败测试，再做最小实现。

---

### Task 1: 修正旧 schema 与 JSON 首次迁移

**Files:**
- Modify: `app/core/metadata_store.py:20-320`
- Modify: `app/core/metadata_store.py:1794-1870`
- Test: `tests/test_metadata_store.py:71-260`
- Test: `tests/test_metadata_store.py:1316-1400`
- Test: `tests/test_async_indexing.py:1340-1390`

- [ ] **Step 1: 写旧文档状态分类的失败测试**

在 `tests/test_metadata_store.py` 扩展 schema/迁移夹具，使其保存 documents、tasks 和 document_versions；新增以下行为测试：

```python
def test_legacy_state_backfill_only_activates_ready_documents():
    connection = LegacyStateConnection(
        documents={
            "ready": {"status": "ready", "current_version": 0},
            "queued": {"status": "queued", "current_version": 0, "task_id": "t1"},
            "failed": {"status": "failed", "current_version": 0},
        },
        tasks={
            "t1": {
                "task_id": "t1", "doc_id": "queued", "status": "queued",
                "document_version": 1, "index_version": 1,
            }
        },
    )
    store = object.__new__(MySQLMetadataStore)
    store._conn = connection

    store._backfill_legacy_document_states()
    store._backfill_active_document_versions()

    assert connection.documents["ready"]["current_version"] == 1
    assert connection.document_versions[("ready", 1)]["status"] == "active"
    assert connection.documents["queued"]["current_version"] == 0
    assert connection.document_versions[("queued", 1)]["status"] == "staging"
    assert not any(doc_id == "failed" for doc_id, _ in connection.document_versions)


def test_legacy_state_backfill_only_runs_when_current_version_is_added():
    existing = SchemaConnection({("documents", "current_version")})
    store = object.__new__(MySQLMetadataStore)
    store._conn = existing

    store._ensure_schema()

    assert not existing.legacy_state_backfill_executed
```

同时更新 schema 断言：

```python
assert "ADD COLUMN current_version INT NOT NULL DEFAULT 0" in sql
assert "recovery_owner VARCHAR(128) NULL" not in sql  # Task 2 才加入
```

- [ ] **Step 2: 运行迁移测试并确认 RED**

Run:

```powershell
$env:PYTHONPATH=(Resolve-Path '..').Path
pytest tests/test_metadata_store.py::test_legacy_state_backfill_only_activates_ready_documents tests/test_metadata_store.py::test_ensure_schema_creates_version_and_chunk_tables_and_new_columns -q -p no:cacheprovider
```

Expected: FAIL，原因分别是 `_backfill_legacy_document_states` 不存在，以及 current_version 仍使用默认 1。

- [ ] **Step 3: 仅在字段本次新增时执行旧状态分类**

在 `_COLUMN_MIGRATIONS` 和 CREATE TABLE 中把 current_version 默认值改为 0：

```python
("documents", "current_version"): "INT NOT NULL DEFAULT 0",
```

让 `_ensure_column` 返回布尔值：只有当前进程成功执行对应 `ALTER TABLE ... ADD COLUMN` 时返回 `True`；字段原本存在或遇到 duplicate-column 并发竞争时返回 `False`。`_ensure_schema` 保存 `current_version_added`，只在该值为 `True` 时调用以下分类方法，再执行 `_backfill_active_document_versions()`：

```python
current_version_added = False
for (table, column), ddl in _COLUMN_MIGRATIONS.items():
    added = self._ensure_column(table, column, ddl)
    if (table, column) == ("documents", "current_version"):
        current_version_added = added
if current_version_added:
    self._backfill_legacy_document_states()
self._backfill_active_document_versions()
```

分类方法本身保持事务幂等：

```python
def _backfill_legacy_document_states(self) -> None:
    cursor = self._conn.cursor()
    try:
        cursor.execute(
            "UPDATE documents SET current_version=1, active_index_version=1 "
            "WHERE status='ready' AND current_version=0"
        )
        cursor.execute(
            "UPDATE documents SET active_index_version=0 "
            "WHERE current_version=0 AND status IN "
            "('queued','parsing','chunking','indexing')"
        )
        cursor.execute(
            """
            INSERT IGNORE INTO document_versions
                (version_id, doc_id, version, content_hash, filename, file_type,
                 storage_path, parser_version, parsed_artifact_path, status, created_at)
            SELECT SHA2(CONCAT('legacy-staging:', d.doc_id, ':', dit.document_version), 256),
                   d.doc_id, dit.document_version, d.content_hash, d.filename,
                   d.file_type, d.storage_path, d.parser_version,
                   d.parsed_artifact_path, 'staging', COALESCE(d.created_at, NOW(3))
            FROM documents d
            INNER JOIN document_index_tasks dit
                    ON dit.doc_id=d.doc_id AND dit.task_id=d.task_id
            LEFT JOIN document_versions dv
                   ON dv.doc_id=d.doc_id AND dv.version=dit.document_version
            WHERE d.current_version=0
              AND d.status IN ('queued','parsing','chunking','indexing')
              AND dit.status IN ('queued','parsing','chunking','indexing')
              AND dv.doc_id IS NULL
            """
        )
        self._conn.commit()
    except Exception:
        self._conn.rollback()
        raise
    finally:
        cursor.close()
```

保持 active 回填只处理 `current_version > 0`，使正在更新且已有 active version 的 v2 文档不被重置。

- [ ] **Step 4: 验证旧状态迁移 GREEN**

Run:

```powershell
$env:PYTHONPATH=(Resolve-Path '..').Path
pytest tests/test_metadata_store.py -q -p no:cacheprovider
```

Expected: PASS。

- [ ] **Step 5: 写 JSON 首次导入版本行的失败测试**

扩展 `test_legacy_json_metadata_migration_imports_kbs_and_documents`：

```python
version_inserts = [
    params for sql, params in executed
    if "INSERT IGNORE INTO document_versions" in " ".join(sql.split())
]
assert any(params[1:4] == ("doc1", 1, None) for params in version_inserts)
document_insert = next(
    params for sql, params in executed
    if "INSERT IGNORE INTO documents" in " ".join(sql.split())
)
assert document_insert[9:11] == (1, 1)
```

增加失败场景：版本插入抛错时原 JSON 文件不能归档。

- [ ] **Step 6: 运行 JSON 迁移测试并确认 RED**

Run:

```powershell
$env:PYTHONPATH=(Resolve-Path '..').Path
pytest tests/test_async_indexing.py::test_legacy_json_metadata_migration_imports_kbs_and_documents tests/test_async_indexing.py::test_legacy_json_migration_keeps_files_when_version_insert_fails -q -p no:cacheprovider
```

Expected: FAIL，当前导入不写 document_versions，且缺少失败测试所需行为。

- [ ] **Step 7: 在 JSON 导入事务中创建 active version**

文档 INSERT 显式增加 `current_version=1, active_index_version=1`。每个 ready 文档紧接着执行：

```python
cur.execute(
    """
    INSERT IGNORE INTO document_versions
        (version_id, doc_id, version, content_hash, filename, file_type,
         storage_path, parser_version, parsed_artifact_path, status, created_at)
    VALUES (%s, %s, 1, NULL, %s, %s, NULL, NULL, NULL, 'active', %s)
    """,
    (
        hashlib.sha256(f"legacy-json:{doc_id}:1".encode()).hexdigest(),
        doc_id,
        filename,
        os.path.splitext(filename)[1],
        self._mysql_dt(created_at),
    ),
)
```

非 ready JSON 文档保持 `current_version=0`，不创建 active version。只有整个事务 commit 成功后才调用 `_archive_legacy_json`。

- [ ] **Step 8: 验证 Task 1 并提交**

Run:

```powershell
$env:PYTHONPATH=(Resolve-Path '..').Path
pytest tests/test_metadata_store.py tests/test_async_indexing.py -q -p no:cacheprovider
git diff --check
```

Expected: PASS，且无空白错误。

Commit:

```powershell
git add app/core/metadata_store.py tests/test_metadata_store.py tests/test_async_indexing.py
git commit -m "fix: preserve legacy indexing states during migration"
```

---

### Task 2: 增加多进程恢复租约

**Files:**
- Modify: `app/core/metadata_store.py:20-70`
- Modify: `app/core/metadata_store.py:1550-1616`
- Modify: `app/core/metadata_store.py:1789-1800`
- Modify: `app/core/indexing_worker.py:18-110`
- Test: `tests/test_metadata_store.py:1580-1670`
- Test: `tests/test_async_indexing.py:1060-1175`

- [ ] **Step 1: 写恢复租约事务的失败测试**

扩展 `LifecycleLockOrderConnection` 保存 owner 和 lease，新增：

```python
def test_recovery_lease_only_allows_one_owner_and_locks_in_order():
    connection = RecoveryLeaseConnection(task_status="indexing")
    store = object.__new__(MySQLMetadataStore)
    store.create_connection = lambda: connection
    now = datetime(2026, 8, 17, 10, 0, 0)

    assert store.claim_task_recovery("task1", "worker-a", now, now + timedelta(minutes=5)) is True
    assert store.claim_task_recovery("task1", "worker-b", now, now + timedelta(minutes=5)) is False

    statements = [sql for sql, _ in connection.executed]
    assert next(i for i, sql in enumerate(statements) if "FROM documents" in sql) < next(
        i for i, sql in enumerate(statements) if "FROM document_index_tasks" in sql and "FOR UPDATE" in sql
    )


def test_expired_recovery_lease_can_be_reclaimed():
    connection = RecoveryLeaseConnection(
        task_status="indexing",
        recovery_owner="worker-a",
        recovery_lease_until=datetime(2026, 8, 17, 9, 59, 0),
    )
    store = object.__new__(MySQLMetadataStore)
    store.create_connection = lambda: connection
    now = datetime(2026, 8, 17, 10, 0, 0)

    assert store.claim_task_recovery("task1", "worker-b", now, now + timedelta(minutes=5)) is True
```

再增加 `complete_task_recovery` owner 不匹配返回 False、匹配时 task/document 变 queued 并清空租约的测试。

- [ ] **Step 2: 运行租约测试并确认 RED**

Run:

```powershell
$env:PYTHONPATH=(Resolve-Path '..').Path
pytest tests/test_metadata_store.py -k "recovery_lease or complete_task_recovery" -q -p no:cacheprovider
```

Expected: FAIL，两个租约方法尚不存在。

- [ ] **Step 3: 增加 schema 字段和序列化字段**

在列迁移和 CREATE TABLE 中增加：

```python
("document_index_tasks", "recovery_owner"): "VARCHAR(128) NULL",
("document_index_tasks", "recovery_lease_until"): "DATETIME(3) NULL",
```

`_task_from_row` 增加：

```python
"recovery_owner": row.get("recovery_owner"),
"recovery_lease_until": row.get("recovery_lease_until"),
```

- [ ] **Step 4: 实现 claim 与 complete 事务**

新增方法，锁序保持 document → task → version：

```python
def claim_task_recovery(self, task_id, owner, now, lease_until) -> bool:
    conn = self.create_connection()
    try:
        cur = conn.cursor()
        cur.execute("SELECT doc_id, kb_id FROM document_index_tasks WHERE task_id=%s", (task_id,))
        task_owner = cur.fetchone()
        if not task_owner:
            conn.commit()
            return False
        cur.execute(
            "SELECT * FROM documents WHERE doc_id=%s AND kb_id=%s FOR UPDATE",
            (task_owner["doc_id"], task_owner["kb_id"]),
        )
        document = cur.fetchone()
        cur.execute("SELECT * FROM document_index_tasks WHERE task_id=%s FOR UPDATE", (task_id,))
        task = cur.fetchone()
        if (
            not task
            or task["status"] not in {"queued", "parsing", "chunking", "indexing"}
            or not document
            or document.get("task_id") != task_id
            or document.get("status") == "deleting"
            or (
                task.get("recovery_owner")
                and task.get("recovery_lease_until")
                and task["recovery_lease_until"] > now
            )
        ):
            conn.commit()
            return False
        cur.execute(
            "SELECT version FROM document_versions "
            "WHERE doc_id=%s AND version=%s AND status='staging' FOR UPDATE",
            (task["doc_id"], task["document_version"]),
        )
        if not cur.fetchone():
            conn.commit()
            return False
        cur.execute(
            "UPDATE document_index_tasks SET recovery_owner=%s, recovery_lease_until=%s, updated_at=%s "
            "WHERE task_id=%s",
            (owner, lease_until, now, task_id),
        )
        conn.commit()
        return cur.rowcount == 1
    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()
        conn.close()
```

`complete_task_recovery` 重复相同锁序和身份校验，只允许 owner 匹配时把 task/document 置 queued，并清空 recovery 字段。不要复用旧 `requeue_indexing_task`，避免先 requeue 后清理。

`list_recoverable_tasks` 增加：

```sql
AND (recovery_owner IS NULL OR recovery_lease_until IS NULL OR recovery_lease_until <= NOW(3))
```

- [ ] **Step 5: 验证 metadata 租约 GREEN**

Run:

```powershell
$env:PYTHONPATH=(Resolve-Path '..').Path
pytest tests/test_metadata_store.py -q -p no:cacheprovider
```

Expected: PASS。

- [ ] **Step 6: 写 Worker 双恢复和清理失败测试**

在 `tests/test_async_indexing.py` 的 `InMemoryStore` 增加相同租约 API，再新增：

```python
def test_two_workers_only_lease_and_cleanup_task_once(tmp_path):
    store, repository, engine, task_id = recoverable_fixture(tmp_path)
    worker_a = IndexingWorker(
        store=store, engine_factory=lambda: engine, autostart=False,
        chunk_repository=repository, recovery_owner="worker-a",
    )
    worker_b = IndexingWorker(
        store=store, engine_factory=lambda: engine, autostart=False,
        chunk_repository=repository, recovery_owner="worker-b",
    )

    worker_a.recover_unfinished_tasks()
    worker_b.recover_unfinished_tasks()

    assert engine.remove_calls == 1
    assert worker_a._queue.qsize() + worker_b._queue.qsize() == 1


def test_recovery_cleanup_failure_keeps_lease_and_does_not_enqueue(tmp_path):
    store, repository, engine, task_id = recoverable_fixture(tmp_path)
    engine.remove_error = RuntimeError("chroma unavailable")
    worker = IndexingWorker(
        store=store, engine_factory=lambda: engine, autostart=False,
        chunk_repository=repository, recovery_owner="worker-a",
    )

    worker.recover_unfinished_tasks()

    assert worker._queue.empty()
    assert store.tasks[task_id]["recovery_owner"] == "worker-a"
    assert store.tasks[task_id]["status"] == "indexing"
```

- [ ] **Step 7: 运行 Worker 测试并确认 RED**

Run:

```powershell
$env:PYTHONPATH=(Resolve-Path '..').Path
pytest tests/test_async_indexing.py -k "two_workers_only or recovery_cleanup_failure" -q -p no:cacheprovider
```

Expected: FAIL，当前 Worker 构造函数没有 recovery_owner，且先 requeue 后清理。

- [ ] **Step 8: 改为 claim → cleanup → complete → enqueue**

构造函数增加：

```python
def __init__(..., recovery_owner: str | None = None, recovery_lease_seconds: int = 300):
    self.recovery_owner = recovery_owner or uuid.uuid4().hex
    self.recovery_lease_seconds = recovery_lease_seconds
```

`recover_unfinished_tasks` 对每项执行：

```python
now = datetime.now()
lease_until = now + timedelta(seconds=self.recovery_lease_seconds)
if not self.store.claim_task_recovery(
    task_id, self.recovery_owner, now, lease_until
):
    continue
try:
    self._cleanup_recoverable_task(task, doc, version)
except Exception as exc:
    logger.warning("恢复任务清理失败，保留租约等待接管: task=%s error=%s", task_id, exc)
    continue
if self.store.complete_task_recovery(task_id, self.recovery_owner):
    self.enqueue(task_id)
```

`_cleanup_recoverable_task` 不吞掉 Chroma、repository 或 artifact 删除异常；只有全部成功才允许 complete。删除操作保持幂等。

- [ ] **Step 9: 验证 Task 2 并提交**

Run:

```powershell
$env:PYTHONPATH=(Resolve-Path '..').Path
pytest tests/test_metadata_store.py tests/test_async_indexing.py tests/test_chunk_repository.py -q -p no:cacheprovider
git diff --check
```

Expected: PASS。

Commit:

```powershell
git add app/core/metadata_store.py app/core/indexing_worker.py tests/test_metadata_store.py tests/test_async_indexing.py
git commit -m "fix: lease indexing recovery across processes"
```

---

### Task 3: 隔离知识库重索引 artifact

**Files:**
- Modify: `app/core/parser.py:117-205`
- Modify: `app/core/rag_engine.py:240-285`
- Modify: `app/core/reindex_service.py:45-90`
- Test: `tests/test_structured_parsers.py`
- Test: `tests/test_rag_engine.py:190-235`
- Test: `tests/test_reindex_service.py`

- [ ] **Step 1: 写安全 artifact 名和重索引隔离测试**

新增 parser 测试：

```python
def test_save_artifact_supports_safe_reindex_name_without_overwriting_active(tmp_path):
    parser = DocumentParser(artifact_dir=tmp_path)
    parsed = parsed_document(version=2)
    active = parser.save_artifact(parsed)
    reindexed = parser.save_artifact(parsed, artifact_name="reindex-task-1")

    assert active.name == "v2.json"
    assert reindexed.name == "reindex-task-1.json"
    assert active != reindexed
    assert active.exists() and reindexed.exists()


@pytest.mark.parametrize("name", ["../escape", "a/b", "", ".hidden"])
def test_save_artifact_rejects_unsafe_artifact_name(tmp_path, name):
    parser = DocumentParser(artifact_dir=tmp_path)
    with pytest.raises(ValueError, match="artifact name"):
        parser.save_artifact(parsed_document(), artifact_name=name)
```

扩展 reindex fake engine，记录 `prepare_document_index` kwargs，并断言每个文档收到 `artifact_name=f"reindex-{task_id}"`。

- [ ] **Step 2: 运行测试并确认 RED**

Run:

```powershell
$env:PYTHONPATH=(Resolve-Path '..').Path
pytest tests/test_structured_parsers.py -k "artifact_name or safe_reindex" tests/test_reindex_service.py -q -p no:cacheprovider
```

Expected: FAIL，save_artifact 和 prepare_document_index 尚不接受 artifact_name。

- [ ] **Step 3: 实现受限 artifact 名并贯通调用**

在 parser 中增加：

```python
_ARTIFACT_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")

def save_artifact(
    self, parsed: ParsedDocument, *, artifact_name: str | None = None
) -> Path:
    stem = artifact_name or f"v{parsed.version}"
    if not self._ARTIFACT_NAME_RE.fullmatch(stem):
        raise ValueError("artifact name is invalid")
    target = self._resolve_artifact_path(target_dir / f"{stem}.json")
    # 保留现有 NamedTemporaryFile、fsync 和 os.replace 原子写逻辑
```

`RAGEngine.prepare_document_index` 和 `parse_document_for_index` 增加关键字参数 `artifact_name: str | None = None`，并传给 `save_artifact`。

ReindexService 在 task 创建后调用：

```python
prepared = self.engine.prepare_document_index(
    snapshot["storage_path"],
    kb_id,
    document_id=document["id"],
    document_version=version,
    index_version=self.index_version,
    display_name=snapshot["filename"],
    artifact_name=f"reindex-{task_id}",
)
```

- [ ] **Step 4: 写真实文件失败回归测试**

使用临时 artifact 目录和第二文档失败的 Engine：

```python
def test_reindex_failure_only_deletes_task_artifact_and_keeps_active(tmp_path):
    active = tmp_path / "kb1" / "doc-1" / "v1.json"
    active.parent.mkdir(parents=True)
    active.write_text("active", encoding="utf-8")
    service, _, engine, _ = make_file_backed_service(tmp_path, fail_doc="doc-2")

    with pytest.raises(RuntimeError, match="parse failed"):
        service.reindex("kb1")

    assert active.read_text(encoding="utf-8") == "active"
    assert not list(tmp_path.rglob("reindex-*.json"))
```

- [ ] **Step 5: 验证 Task 3 并提交**

Run:

```powershell
$env:PYTHONPATH=(Resolve-Path '..').Path
pytest tests/test_structured_parsers.py tests/test_rag_engine.py tests/test_reindex_service.py -q -p no:cacheprovider
git diff --check
```

Expected: PASS。

Commit:

```powershell
git add app/core/parser.py app/core/rag_engine.py app/core/reindex_service.py tests/test_structured_parsers.py tests/test_rag_engine.py tests/test_reindex_service.py
git commit -m "fix: isolate reindex parsed artifacts"
```

---

### Task 4: 保留 active chunk_count

**Files:**
- Modify: `app/core/metadata_store.py:1008-1085`
- Modify: `app/api/documents.py:279-380`
- Test: `tests/test_metadata_store.py:798-970`
- Test: `tests/test_async_indexing.py`
- Test: `tests/test_documents_api.py:210-330`

- [ ] **Step 1: 写数据库和 API 失败测试**

在 metadata transaction 测试中断言 update SQL 不包含 chunk_count：

```python
def test_create_version_keeps_active_chunk_count_while_queued():
    store, connections = _store_with_connection_factory(max_version=1)
    store.create_document_version_and_task(
        "doc1", "kb1", "new-hash", "/uploads/task2/new.txt",
        "new.txt", ".txt", "task2", "2026-08-17T10:00:00",
    )

    update_sql, update_params = next(
        item for item in connections[0].executed
        if item[0].startswith("UPDATE documents")
    )
    assert "chunk_count" not in update_sql
    assert connections[0].document["chunk_count"] == 4
```

API 测试断言 queued 更新响应返回原 active count：

```python
response = asyncio.run(
    documents.update_document("kb1", "doc1", FakeUploadFile("new.txt", b"new"))
)
assert response.status == DocStatus.QUEUED
assert response.chunk_count == 2
```

Worker 失败测试增加：

```python
assert store.documents["doc1"]["status"] == "ready"
assert store.documents["doc1"]["chunk_count"] == 4
```

- [ ] **Step 2: 运行测试并确认 RED**

Run:

```powershell
$env:PYTHONPATH=(Resolve-Path '..').Path
pytest tests/test_metadata_store.py -k "keeps_active_chunk_count" tests/test_documents_api.py -k "update" tests/test_async_indexing.py -k "structured_index_failure" -q -p no:cacheprovider
```

Expected: FAIL，当前 SQL 和响应固定清零。

- [ ] **Step 3: 最小修改 active count 语义**

把更新入队 SQL 改为：

```sql
UPDATE documents
SET status=%s, task_id=%s, error_message=%s, updated_at=%s
WHERE doc_id=%s AND kb_id=%s
```

同步调整参数和测试 fake cursor。API 成功响应改为：

```python
chunk_count=current.get("chunk_count", 0),
```

不要修改初次上传的 count=0，也不要修改激活事务写入新 Child 数量的逻辑。

- [ ] **Step 4: 验证 Task 4 并提交**

Run:

```powershell
$env:PYTHONPATH=(Resolve-Path '..').Path
pytest tests/test_metadata_store.py tests/test_documents_api.py tests/test_async_indexing.py -q -p no:cacheprovider
git diff --check
```

Expected: PASS。

Commit:

```powershell
git add app/core/metadata_store.py app/api/documents.py tests/test_metadata_store.py tests/test_documents_api.py tests/test_async_indexing.py
git commit -m "fix: preserve active chunk count during updates"
```

---

### Task 5: 让 sources 对应最终生成命中

**Files:**
- Modify: `app/core/context_assembler.py`
- Modify: `app/core/rag_engine.py:390-570`
- Modify: `app/core/rag_engine.py:708-730`
- Test: `tests/test_context_assembler.py`
- Test: `tests/test_rag_engine.py:45-165`
- Test: `tests/test_chat_stream.py`

- [ ] **Step 1: 写 AssembledContext 的失败测试**

新增结果类型导入和测试：

```python
def test_assemble_result_returns_only_hits_that_survive_quota_and_budget():
    candidates = [
        candidate("a-1", doc_id="a", content="one two"),
        candidate("a-2", doc_id="a", content="three four"),
        candidate("b-1", doc_id="b", content="five six"),
    ]

    result = assembler(
        token_budget=4, per_document_limit=1
    ).assemble_result(candidates, final_k=2)

    assert [item.chunk_id for item in result.selected_candidates] == ["a-1", "b-1"]
    assert [unit.chunk_id for unit in result.units if unit.context_role == "hit"] == [
        "a-1", "b-1"
    ]
    assert assembler().assemble([candidate("legacy")], final_k=1)[0].chunk_id == "legacy"
```

另加预算只能容纳第一个 hit 时，selected_candidates 也只包含第一个的测试。

- [ ] **Step 2: 运行 assembler 测试并确认 RED**

Run:

```powershell
$env:PYTHONPATH=(Resolve-Path '..').Path
pytest tests/test_context_assembler.py -k "assemble_result" -q -p no:cacheprovider
```

Expected: FAIL，`AssembledContext` 和 `assemble_result` 不存在。

- [ ] **Step 3: 实现兼容的组装结果**

新增：

```python
@dataclass(frozen=True)
class AssembledContext:
    selected_candidates: tuple[RetrievalCandidate, ...]
    units: tuple[ContextUnit, ...]


def assemble_result(
    self, candidates: list[RetrievalCandidate], final_k: int
) -> AssembledContext:
    if final_k <= 0 or not candidates:
        return AssembledContext((), ())
    selected = self._apply_document_quota(candidates)[:final_k]
    hits = [self._hit_unit(candidate) for candidate in selected]
    supplements = self._expand_context(selected, {unit.chunk_id for unit in hits})
    units = tuple(self._fit_token_budget(hits + supplements))
    surviving_hit_ids = {
        unit.chunk_id for unit in units if unit.context_role == "hit"
    }
    surviving = tuple(
        candidate for candidate in selected if candidate.chunk_id in surviving_hit_ids
    )
    return AssembledContext(surviving, units)


def assemble(self, candidates, final_k):
    return list(self.assemble_result(candidates, final_k).units)
```

- [ ] **Step 4: 写同步与流式 sources 失败测试**

创建 5 个宽候选，让 fake assembler 返回其中 2 个 selected candidates、Parent 和邻居 units：

```python
def test_chat_sources_match_selected_hits_and_respect_top_k():
    hits = [make_hit(f"hit-{index}") for index in range(5)]
    engine = engine_with_assembled_result(
        hits,
        selected=hits[1:3],
        units=[hit_unit(hits[1]), parent_unit(hits[1]), hit_unit(hits[2])],
    )

    result = engine.chat("question", "kb1", top_k=2, enable_expansion=False)

    assert [source["content"] for source in result["sources"]] == [
        hits[1].content, hits[2].content,
    ]
    assert "parent context" in engine.generated_context
```

对 `chat_stream` 断言最后 `[SOURCES]` 事件同样只有两个直接命中。对 hallucination detector 注入记录函数，断言只接收 selected candidates。

- [ ] **Step 5: 运行 RAG 测试并确认 RED**

Run:

```powershell
$env:PYTHONPATH=(Resolve-Path '..').Path
pytest tests/test_rag_engine.py tests/test_chat_stream.py -k "selected_hits or respect_top_k" -q -p no:cacheprovider
```

Expected: FAIL，当前 chat/chat_stream 遍历全部 Retriever results。

- [ ] **Step 6: RAGEngine 统一使用 AssembledContext**

把 `_assemble_retrieval_context` 改为返回 `AssembledContext`。对 dict legacy result 继续转换为 RetrievalCandidate。同步和流式流程统一使用：

```python
assembled = self._assemble_retrieval_context(results, top_k)
if not assembled.units or not assembled.selected_candidates:
    return existing_refusal_path

context_units = list(assembled.units)
source_candidates = list(assembled.selected_candidates)
```

生成读取 `context_units`；sources、memory sources 和 `_detect_hallucinations` 读取 `source_candidates`。保留拒答后 sources 置空和 SSE 格式。

- [ ] **Step 7: 验证 Task 5 并提交**

Run:

```powershell
$env:PYTHONPATH=(Resolve-Path '..').Path
pytest tests/test_context_assembler.py tests/test_rag_engine.py tests/test_chat_stream.py -q -p no:cacheprovider
git diff --check
```

Expected: PASS。

Commit:

```powershell
git add app/core/context_assembler.py app/core/rag_engine.py tests/test_context_assembler.py tests/test_rag_engine.py tests/test_chat_stream.py
git commit -m "fix: align rag sources with assembled context"
```

---

### Task 6: 收紧关键词降级默认阈值

**Files:**
- Modify: `config.py:55-70`
- Modify: `.env.example`
- Modify: `README.md`
- Modify: `tests/test_retriever.py:300-390`

- [ ] **Step 1: 写默认拒绝零关键词候选的失败测试**

保留“请求模型阈值不套用关键词评分”的既有测试，但把显式关键词阈值设为 0；新增默认值测试：

```python
def test_keyword_fallback_default_threshold_rejects_zero_signal(monkeypatch):
    retriever = Retriever.__new__(Retriever)
    retriever.top_k = 5
    retriever.rerank_candidate_k = 30
    retriever.model_rerank_threshold = 0.35
    retriever.keyword_rerank_threshold = 0.01
    retriever._is_complex = lambda query: False
    retriever._hybrid_search = lambda *args, **kwargs: [
        make_candidate("zero", rrf_score=0.1)
    ]
    retriever._rerank = lambda query, candidates, method: RerankResult(
        (replace(candidates[0], rerank_score=0.0),), "keyword", False
    )

    assert retriever.retrieve(
        "query", "kb1", threshold=0.99,
        enable_expansion=False, enable_decomposition=False,
    ) == []
```

增加配置断言：`KEYWORD_RERANK_THRESHOLD == 0.01`。

- [ ] **Step 2: 运行测试并确认 RED**

Run:

```powershell
$env:PYTHONPATH=(Resolve-Path '..').Path
pytest tests/test_retriever.py -k "keyword_fallback_default_threshold" -q -p no:cacheprovider
```

Expected: FAIL，当前默认配置为 0。

- [ ] **Step 3: 修改默认配置和说明**

`config.py`：

```python
KEYWORD_RERANK_THRESHOLD = float(os.getenv("KEYWORD_RERANK_THRESHOLD", "0.01"))
```

`.env.example`：

```env
KEYWORD_RERANK_THRESHOLD=0.01
```

README 环境变量表把默认值改为 0.01，并说明：模型 rerank 不可用时使用关键词阈值；显式设 0 会允许零关键词候选，降低拒答严格度。

- [ ] **Step 4: 验证 Task 6 并提交**

Run:

```powershell
$env:PYTHONPATH=(Resolve-Path '..').Path
pytest tests/test_retriever.py tests/test_rag_engine.py -q -p no:cacheprovider
git diff --check
```

Expected: PASS。

Commit:

```powershell
git add config.py .env.example README.md tests/test_retriever.py
git commit -m "fix: reject zero-signal keyword fallback results"
```

---

### Task 7: 最终一致性验证与迁移演练

**Files:**
- Verify: `app/core/metadata_store.py`
- Verify: `app/core/indexing_worker.py`
- Verify: `app/core/parser.py`
- Verify: `app/core/reindex_service.py`
- Verify: `app/core/context_assembler.py`
- Verify: `app/core/rag_engine.py`
- Verify: `app/api/documents.py`
- Verify: `config.py`
- Verify: `.env.example`
- Verify: `README.md`
- Verify: all modified tests

- [ ] **Step 1: 运行合并前专项测试**

Run:

```powershell
$env:TEMP='D:\codex-kbzhy-temp'
$env:TMP=$env:TEMP
$env:PYTHONPATH=(Resolve-Path '..').Path
pytest tests/test_metadata_store.py tests/test_async_indexing.py tests/test_chunk_repository.py tests/test_reindex_service.py tests/test_structured_parsers.py tests/test_documents_api.py tests/test_retriever.py tests/test_context_assembler.py tests/test_rag_engine.py tests/test_chat_stream.py -q -p no:cacheprovider --basetemp "$env:TEMP\hardening-targeted"
```

Expected: 全部 PASS；只允许已有的 jieba/setuptools 与 PyMuPDF SWIG 弃用 warning。

- [ ] **Step 2: 执行双恢复者确定性演练**

Run:

```powershell
$env:PYTHONPATH=(Resolve-Path '..').Path
pytest tests/test_async_indexing.py -k "two_workers_only or expired_recovery_lease or recovery_cleanup_failure" -vv -p no:cacheprovider
```

Expected: 同一有效租约只清理/入队一次；过期租约可接管；清理失败不 enqueue。

- [ ] **Step 3: 执行旧库与 JSON 迁移演练**

Run:

```powershell
$env:PYTHONPATH=(Resolve-Path '..').Path
pytest tests/test_metadata_store.py tests/test_async_indexing.py -k "legacy_state_backfill or legacy_json" -vv -p no:cacheprovider
```

Expected: ready→active，recoverable→staging，failed 不伪造 active，JSON 首次导入即有 active version。

- [ ] **Step 4: 运行完整测试**

Run:

```powershell
$env:TEMP='D:\codex-kbzhy-temp'
$env:TMP=$env:TEMP
$env:PYTHONPATH=(Resolve-Path '..').Path
pytest -q -p no:cacheprovider --basetemp "$env:TEMP\hardening-full"
```

Expected: 全部 PASS，无新增 warning。

- [ ] **Step 5: 编译、导入和 diff 检查**

Run:

```powershell
$env:PYTHONPATH=(Resolve-Path '..').Path
python -m compileall -q app tests
python -c "from KBzhy.main import app; print(app.title)"
git diff --check
git status --short
```

Expected: 编译退出码 0；输出 `KBzhy RAG Knowledge Base QA System`；工作区只包含计划内变更或为空。

- [ ] **Step 6: 审查提交范围**

Run:

```powershell
git log --oneline 26457a5..HEAD
git diff --stat 26457a5..HEAD
```

Expected: 只有本计划六个修复提交；不包含权限、评测、BM25 持久化、GC 或 migration runner 改动。

若最终验证产生必要的小修复，先补失败测试并单独提交：

```powershell
git add <only-the-fix-files>
git commit -m "fix: close rag hardening regression"
```

---

## 最终验收清单

- [ ] 旧 ready 文档升级为 active version，旧在途文档保留 staging 语义。
- [ ] 旧 JSON 首次迁移即创建 active document version。
- [ ] 同一任务的有效恢复租约只能由一个进程持有。
- [ ] 清理失败不会 requeue/enqueue；租约过期后可接管。
- [ ] reindex 失败不覆盖或删除 active artifact。
- [ ] 更新处理中和失败后保留 active chunk_count。
- [ ] sources 不超过 top_k，且只包含生成使用的直接命中。
- [ ] Parent/邻居仍可进入上下文，但不进入 sources。
- [ ] 默认关键词降级过滤零分候选；显式阈值 0 仍可配置。
- [ ] REST/SSE 字段格式保持兼容。
- [ ] 全量测试、编译、导入和 diff 检查通过。
