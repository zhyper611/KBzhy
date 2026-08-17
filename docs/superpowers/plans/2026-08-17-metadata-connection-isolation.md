# Metadata Connection Isolation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让 API 请求和后台索引 Worker 立即看到已提交的文档与索引任务，并消除 PyMySQL 连接的跨线程共享。

**Architecture:** `MySQLMetadataStore` 使用 `threading.local()` 为每个线程保存普通操作连接，该连接开启 autocommit，使每条读语句都看到最新已提交数据。现有 `create_connection()` 继续返回关闭 autocommit 的独立连接，保留文档、版本和任务的原子事务边界。

**Tech Stack:** Python 3.11+、PyMySQL、MySQL、pytest、FastAPI 后台线程 Worker

---

### Task 1: 固化连接隔离和 autocommit 契约

**Files:**
- Modify: `tests/test_metadata_store.py`

- [ ] **Step 1: 编写线程隔离失败测试**

构造记录建连参数的假 PyMySQL 对象，在主线程和 Worker 线程访问 `store._conn`，断言两个线程获得不同连接，且同一线程重复访问会复用连接。

```python
def test_runtime_connections_are_isolated_per_thread():
    store = _build_connection_lifecycle_store()
    main_connection = store._conn
    worker_connections = []

    thread = threading.Thread(
        target=lambda: worker_connections.extend([store._conn, store._conn])
    )
    thread.start()
    thread.join()

    assert worker_connections[0] is worker_connections[1]
    assert worker_connections[0] is not main_connection
```

- [ ] **Step 2: 编写 autocommit 边界失败测试**

断言线程本地普通连接的建连参数是 `autocommit=True`，而 `create_connection()` 创建的事务连接仍是 `autocommit=False`。

```python
def test_runtime_connection_uses_autocommit_but_transaction_connection_does_not():
    store = _build_connection_lifecycle_store()

    runtime_connection = store._conn
    transaction_connection = store.create_connection()

    assert runtime_connection.autocommit_enabled is True
    assert transaction_connection.autocommit_enabled is False
```

- [ ] **Step 3: 运行新测试确认 RED**

Run:

```powershell
pytest tests/test_metadata_store.py -k "runtime_connections_are_isolated or runtime_connection_uses_autocommit" -vv
```

Expected: FAIL，现有实现在两个线程中返回同一个 `_conn`，并且普通连接的 `autocommit` 是 `False`。

### Task 2: 实现线程本地连接

**Files:**
- Modify: `app/core/metadata_store.py`
- Test: `tests/test_metadata_store.py`

- [ ] **Step 1: 引入线程本地状态**

```python
import threading

self._connection_state = threading.local()
self._conn = self._connect(autocommit=True)
```

- [ ] **Step 2: 将 `_conn` 改为线程本地属性**

```python
@property
def _conn(self):
    state = self._connection_state
    connection = getattr(state, "connection", None)
    if connection is None:
        connection = self._connect(autocommit=True)
        state.connection = connection
    return connection

@_conn.setter
def _conn(self, connection):
    if not hasattr(self, "_connection_state"):
        self._connection_state = threading.local()
    self._connection_state.connection = connection
```

- [ ] **Step 3: 区分普通连接与事务连接**

```python
def _connect(self, *, autocommit: bool = False):
    return self._pymysql.connect(
        **{**self._connect_kwargs, "autocommit": autocommit}
    )

def create_connection(self):
    return self._connect(autocommit=False)

def _reconnect(self):
    try:
        self._conn.close()
    except Exception:
        pass
    self._conn = self._connect(autocommit=True)
```

构造阶段的 schema 迁移仍在事务连接上执行：先用 `autocommit=False` 连接完成 `_ensure_schema()` 和旧数据迁移，然后关闭该初始化连接，再将当前线程切换到 `autocommit=True` 运行连接。

- [ ] **Step 4: 运行新测试确认 GREEN**

Run:

```powershell
pytest tests/test_metadata_store.py -k "runtime_connections_are_isolated or runtime_connection_uses_autocommit" -vv
```

Expected: 2 passed.

- [ ] **Step 5: 运行相关回归**

Run:

```powershell
pytest tests/test_metadata_store.py tests/test_async_indexing.py tests/test_documents_api.py -q
```

Expected: 全部通过。

### Task 3: 遵守 Embedding 供应商单批上限

**Files:**
- Modify: `app/core/retriever.py`
- Test: `tests/test_retriever.py`

- [ ] **Step 1: 编写 12 条文本的失败测试**

调用 `_BailianEmbeddings.embed_documents()` 处理 12 条文本，断言供应商请求批次是 `[10, 2]`，结果顺序与输入一致。

```powershell
pytest tests/test_retriever.py::test_bailian_document_embeddings_are_batched_by_provider_limit -vv
```

Expected: FAIL，现有请求批次为 `[12]`。

- [ ] **Step 2: 按 `_EMBEDDING_BATCH_SIZE` 分批**

```python
embeddings = []
for start in range(0, len(texts), _EMBEDDING_BATCH_SIZE):
    batch = texts[start:start + _EMBEDDING_BATCH_SIZE]
    result = self._client.embeddings.create(model=self.model, input=batch)
    embeddings.extend(item.embedding for item in result.data)
return embeddings
```

- [ ] **Step 3: 运行单测与检索/索引回归**

```powershell
pytest tests/test_retriever.py::test_bailian_document_embeddings_are_batched_by_provider_limit -vv
pytest tests/test_retriever.py tests/test_async_indexing.py -q
```

Expected: 全部通过。

### Task 4: 恢复现有排队任务并完成验证

**Files:**
- No production file changes expected

- [ ] **Step 1: 运行全量验证**

```powershell
pytest -q -p no:cacheprovider
python -m compileall -q app tests
git diff --check
```

Expected: pytest 零失败，编译与 diff 检查退出码为 0。

- [ ] **Step 2: 重启后端触发恢复扫描**

停止当前 Uvicorn 会话，使用原启动命令重启。启动日志必须出现 Worker recovery completed，且不再出现当前任务的“索引任务不存在”。

- [ ] **Step 3: 轮询当前 PDF 状态**

```powershell
Invoke-RestMethod "http://127.0.0.1:8000/api/knowledge-bases/ac0e60ba116d/documents?page_size=100"
```

Expected: `中二知识笔记.pdf` 最终为 `ready`，`chunk_count > 0`。

- [ ] **Step 4: 浏览器复核**

刷新文档详情页，确认文件行稳定存在、状态为“就绪”、分块数大于 0；打开分块抽屉确认内容可读。

- [ ] **Step 5: 提交门禁**

本计划不自动提交。只有用户明确授权后，才暂存本轮连接隔离修复、测试和设计/计划文档并执行 `git commit`。
