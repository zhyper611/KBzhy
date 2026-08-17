# RAG 核心流程 v2 副作用加固设计

## 背景

`codex/rag-core-pipeline-v2` 已完成结构化解析、Parent/Child 切块、版本化索引、RRF、宽候选重排和上下文组装。合并前审查确认，现有实现仍存在旧库迁移、多进程恢复、重索引产物隔离、更新失败元数据、最终来源契约和关键词降级拒答六类生产边界问题。

本轮只修复这些已确认问题，并补齐与旧 JSON 元数据迁移相关的版本回填。保留现有 MySQL、ChromaDB、内存 BM25、FastAPI API 和索引 Worker 架构，不引入独立迁移服务、消息队列、权限或评测系统。

## 目标

- 旧库升级后，ready 文档继续代表已有 active 版本，在途文档继续作为 staging 任务恢复，不产生“未完成却 ready”的状态。
- 多个应用进程同时启动时，同一索引任务只能由一个恢复者清理和重新排队；恢复进程崩溃后任务可以被接管。
- 知识库重索引失败不得覆盖或删除当前 active 解析产物。
- 文档更新失败时，上一 active 版本的 `chunk_count`、向量和状态保持一致。
- `top_k` 限制用户可见来源数量，来源必须对应实际进入生成上下文的直接命中。
- 模型 rerank 降级为关键词评分时，默认拒绝零关键词命中的候选。
- 旧 JSON 数据首次导入完成后立即拥有完整的 active version 元数据。

## 非目标

- 不把启动时 schema 调整改造成独立 migration runner。
- 不增加历史版本、inactive chunk、旧 collection 或旧 artifact 的自动 GC。
- 不持久化 BM25，也不改为外部搜索引擎。
- 不实现正式证据充分性判定器、权限系统或离线评测。
- 不调整 Parent/Child 默认大小、RRF 宽度和上下文 token 总预算。

## 方案选择

采用定向修复方案。仅用停写、单 Worker 等部署约束规避问题，无法长期保证正确性；全面重做迁移和调度系统又超出本轮范围。定向方案通过现有 MySQL 事务扩展任务状态，在不替换存储的前提下关闭竞态和契约缺口。

## 旧库状态迁移

### 新安装

`documents.current_version` 的数据库默认值改为 `0`。新上传文档显式以 `current_version=0` 和 staging version 1 创建，首次索引激活后再切换为版本 1。

### 从旧 schema 升级

schema 检查必须知道 `current_version` 是否是本次新加字段，不能在每次启动时重复重写文档状态。

仅当字段本次新增时执行旧文档分类：

- `documents.status='ready'`：设 `current_version=1`、`active_index_version=1`，创建或补全 active `document_versions` 版本 1。
- `documents.status IN ('queued','parsing','chunking','indexing')` 且存在对应任务：保持 `current_version=0`，为任务创建 staging `document_versions` 版本 1；文件名、类型和存储路径取自旧 document。
- `failed`、`stale`、`deleting`：保持 `current_version=0`，不伪造 active 版本。

旧任务新增的 `document_version` 和 `index_version` 默认值仍为 1，因此恢复器可以关联上述 staging 版本。

如果旧在途任务缺少文件路径，恢复时按现有失败路径进入 `failed`，不得改为 `ready`。

### 旧 JSON 首次导入

JSON 导入的 ready 文档显式写入 `current_version=1`。导入事务提交后立即执行 active version 回填并验证，不再依赖第二次进程启动。JSON 文件只有在文档和版本数据均提交成功后才改名为 `.migrated`。

## 多进程恢复租约

### 数据字段

`document_index_tasks` 增加：

- `recovery_owner VARCHAR(128) NULL`
- `recovery_lease_until DATETIME(3) NULL`

恢复者 owner 使用进程级随机 ID。租约默认 5 分钟，只约束启动恢复阶段，不替代正常索引任务的 `claim_task`。

### 原子流程

1. 各进程可以读取同一批 recoverable task。
2. 清理前调用 `claim_task_recovery(task_id, owner, now, lease_until)`。
3. MySQL 事务按 document → task → version 顺序加锁，并确认任务仍拥有文档、版本仍是 staging。
4. 仅当任务尚未被租用，或旧租约已经过期时，写入 owner 和 lease；其他进程返回 `False`。
5. 租约持有者清理本任务的 Chroma children、MySQL staging chunks 和临时 artifact。
6. 调用 `complete_task_recovery(task_id, owner)`，仅匹配 owner 时清空租约并把 task/document 置为 `queued`。
7. 完成重新排队后才放入当前进程的内存队列。

如果进程在清理或重新排队前崩溃，任务保持原索引阶段和租约信息；租约过期后其他进程可以重新执行幂等清理。正常 Worker 不处理尚有有效恢复租约的任务。

`list_recoverable_tasks` 包含原有索引阶段以及租约已过期的恢复任务，不返回仍被有效租用的任务。

## 重索引 artifact 隔离

普通文档索引继续写：

```text
data/parsed/{kb_id}/{doc_id}/v{document_version}.json
```

知识库重索引写独立路径：

```text
data/parsed/{kb_id}/{doc_id}/reindex-{task_id}.json
```

`DocumentParser.save_artifact` 接受受校验的 artifact 名称或 scope，最终路径仍必须位于配置的 artifact 根目录内。普通索引不传 scope，保持现有路径兼容。

重索引失败只删除本次 task 对应的 artifact。重索引成功后，`activate_reindex` 把 document/version 的 `parsed_artifact_path` 更新为新的独立文件。旧 active artifact 暂时保留，后续 GC 不在本轮范围。

## 更新失败时保留 active 计数

`create_document_version_and_task` 创建 staging version 和 queued task 时，不再把 `documents.chunk_count` 改为 0。`documents.chunk_count` 始终描述当前 active 版本：

- 初次上传仍为 0。
- 更新处理中保留旧 active 数量。
- 新版本激活时原子替换为新 Child 数量。
- 更新失败时只恢复 document 状态并记录错误，数量无需额外回写。

HTTP 更新响应中的 `chunk_count` 返回当前 active 数量，而不是固定 0。前端仍可通过 `status=queued` 判断新版本正在处理。

## 最终候选、上下文与来源

Retriever 继续返回 rerank 后的宽候选，供文档配额在更大范围内选出多样来源。`top_k` 的最终截取由 ContextAssembler 执行。

ContextAssembler 增加一个明确的组装结果：

```python
@dataclass(frozen=True)
class AssembledContext:
    selected_candidates: tuple[RetrievalCandidate, ...]
    units: tuple[ContextUnit, ...]
```

`selected_candidates` 是执行去重、单文档配额并截取 `top_k` 后的直接命中；`units` 在这些直接命中基础上补充 Parent 和邻居，并应用 token 预算。

同步和流式问答统一使用：

- `units` 构造 LLM 上下文。
- `selected_candidates` 构造 sources、会话来源和轻量幻觉检测输入。

Parent 与邻居仍不伪装为直接来源。sources 数量不超过 `top_k`，且每条 source 都对应本次生成使用的一个直接命中。

## 关键词 rerank 降级边界

保留模型和关键词两套独立阈值，因为两种评分不可直接比较。

- `KEYWORD_RERANK_THRESHOLD` 默认值由 `0` 改为 `0.01`。
- 使用默认配置时，关键词分数为 0 的候选被过滤并触发既有拒答路径。
- 运维人员仍可显式配置为 0，以选择可用性优先行为。
- 请求中的 `similarity_threshold` 继续只覆盖模型/LLM rerank 阈值，不直接套用到关键词评分。

README 和 `.env.example` 必须明确这一降级语义。

## 错误处理

- 获取恢复租约失败：当前进程跳过任务，不执行任何清理。
- 租约持有者清理部分失败：保留租约直到过期，不把任务放入内存队列；下次接管重复幂等清理。
- 完成恢复时 owner 不匹配：返回 `False`，当前进程不得 enqueue。
- 重索引失败：删除临时 collection、reindex staging rows 和本次独立 artifact，旧 active 数据不变。
- ContextAssembler 无法读取 Parent/邻居：仍保留直接命中；若最终没有直接命中，进入既有拒答。

## 兼容性

- REST 响应字段和 SSE `[SOURCES]` 事件格式不变。
- `sources` 数量重新符合 `top_k`，属于对现有语义的修正。
- 普通 artifact 路径不变；只有 reindex 使用新路径。
- 新增 MySQL 字段允许 NULL，不影响现有任务记录。
- 现有 active Chroma collection 和旧 versionless Chroma 元数据兼容逻辑不变。
- 同内容上传 `409`、严格文件名校验等已确认产品行为不在本轮回退。

## 测试策略

所有修复按 TDD 实施，先验证测试在旧实现上因目标问题失败。

### 迁移测试

- 旧 ready 文档升级后得到 active version 1。
- 旧 queued/parsing/chunking/indexing 文档得到 staging version 1 且 `current_version=0`。
- 旧失败文档不会被伪造为 active。
- JSON ready 文档第一次导入后立即有 active version。

### 恢复测试

- 两个 owner 同时 claim，只有一个成功。
- 有效租约不能被抢占。
- 过期租约可以接管。
- 非 owner 不能 complete recovery 或 enqueue。
- 清理失败后任务不进入队列；接管后可幂等恢复。

### 重索引测试

- reindex artifact 路径和 active artifact 不同。
- 第二个文档失败时，第一个文档的原 active artifact 仍存在。
- 成功激活后数据库指向新的 reindex artifact。

### API 与检索测试

- 更新处理中及失败后保留旧 `chunk_count`。
- `top_k=2` 时同步和流式 sources 都不超过 2 条。
- sources 只包含 ContextAssembler 选中的直接命中。
- Parent/邻居仍进入生成上下文但不进入 sources。
- 模型 rerank 降级后，默认过滤零关键词候选；显式阈值 0 仍可放行。

### 回归验证

- 运行受影响模块测试。
- 运行完整 pytest。
- 运行 `compileall` 和 FastAPI 导入检查。
- 执行 `git diff --check`，确保只包含计划内文件。

## 上线约束

即使完成本轮修复，首次 schema 升级仍会执行 DDL。上线前必须备份 MySQL，在生产数据副本上测量 ALTER/索引耗时，并通过单独维护窗口或单副本先行方式完成迁移。自动 GC、BM25 持久化、解析峰值内存优化和独立 migration runner 作为后续迭代处理。
