# KBzhy RAG 核心流程 v2 设计

## 1. 目标

在保留现有 MySQL、ChromaDB、内存 BM25、本地文件存储和 FastAPI/React 技术栈的前提下，完成以下三项改造：

1. 数据治理与增量索引。
2. PDF、Word、Markdown/TXT 的结构化解析与 Parent-Child 切块。
3. 混合召回、RRF 融合、宽候选重排和上下文选择重构。

本轮优先提高检索证据的正确性与上下文完整性，不修改聊天响应协议，不重构生成与引用流程。

## 2. 范围

### 2.1 本轮包含

- 文档 Hash、版本、解析器版本和索引版本。
- 稳定的 Chunk ID、Chunk Hash 和幂等索引。
- 结构化解析中间模型及本地持久化产物。
- PDF、Word、Markdown/TXT 的结构恢复。
- Parent-Child 切块、标题路径和相邻块关系。
- BM25 TopN 与向量 TopN 的 RRF 融合。
- 对 20～50 个候选执行 Rerank，再选择最终上下文。
- Rerank 后阈值、文档配额、去重、相邻块或 Parent 补全。
- 对应的单元测试、集成测试和既有测试迁移。

### 2.2 本轮不包含

- 身份认证、租户、RBAC、ACL。
- 正式离线评测平台及 CI 质量门禁。
- 结论级引用、证据充分性 Judge、生成后事实校验。
- 聊天响应 Schema 和 SSE 协议调整。
- PostgreSQL、pgvector、Elasticsearch 迁移。
- MinIO/S3、Celery、RabbitMQ、Kafka。
- Excel、PPT、图片 OCR 的深度结构化改造。
- GraphRAG、Agentic RAG、知识图谱。

## 3. 方案选择

### 3.1 方案 A：在现有模块上分层重构（采用）

保留现有 API 和存储，在核心层增加明确的数据模型、仓储和上下文组装模块，逐步替换旧逻辑。

优点：

- 不需要维护两套完整流水线。
- 能复用现有上传、任务、会话和前端能力。
- 改造可以按索引、解析、检索三个阶段分别验证。

代价：

- `rag_engine.py`、`retriever.py` 和索引 Worker 会发生接口联动。
- 现有 Chroma 数据需要重新索引才能获得新元数据。

### 3.2 方案 B：保留旧流程，另建完整 v2 流程

优点是回滚简单，缺点是解析、索引、检索和测试长期重复。当前项目规模不值得承担双实现成本，因此不采用。

### 3.3 方案 C：仅调整检索顺序

只修改 RRF、Rerank 和 MMR，开发最快，但无法解决 Chunk ID 不稳定、结构丢失、旧版本残留和 Parent 补全问题，因此不采用。

## 4. 目标架构

```text
离线入库
原文件
  → content_hash / 文档版本
  → StructuredParser
  → ParsedDocument + DocumentElement[]
  → 解析产物 JSON
  → StructuralChunker
  → ParentChunk[] + ChildChunk[]
  → MySQL Chunk Repository
  → Child Embedding
  → ChromaDB + BM25

在线检索
原始问题
  → 必要时会话改写
  → Vector Top30 || BM25 Top30
  → RRF Fusion
  → chunk_id 去重
  → Rerank Top20～50
  → Rerank 阈值
  → ContextAssembler
  → 文档配额、相邻块/Parent 补全、Token 预算
  → 最终上下文
  → 保持现有生成流程
```

## 5. 核心数据模型

### 5.1 结构化解析模型

新增独立模型模块，避免解析器、切块器和检索器通过松散字典传递数据。

```python
class DocumentElement:
    element_id: str
    element_type: str
    text: str
    order: int
    page: int | None
    section_path: list[str]
    bounding_box: dict | None
    metadata: dict

class ParsedDocument:
    document_id: str
    version: int
    title: str
    language: str
    elements: list[DocumentElement]
    metadata: dict
```

`element_type` 首期支持：`heading`、`paragraph`、`list`、`table`、`code`。

### 5.2 Chunk 模型

```python
class KnowledgeChunk:
    chunk_id: str
    document_id: str
    document_version: int
    parent_chunk_id: str | None
    chunk_type: str
    content: str
    retrieval_text: str
    content_hash: str
    section_path: list[str]
    page_start: int | None
    page_end: int | None
    position: int
    token_count: int
    index_version: int
    metadata: dict
```

- `content` 保存原始语义内容。
- `retrieval_text` 由标题路径和内容组成，用于 BM25、Embedding 和 Rerank。
- Parent 仅保存在 MySQL，不进入向量和 BM25 索引。
- Child 进入 ChromaDB 和 BM25。
- `chunk_id` 由文档 ID、文档版本、位置和内容 Hash 确定，重复执行同一任务得到相同 ID。

## 6. MySQL 数据设计

### 6.1 扩展 knowledge_bases

新增 `active_collection_name`，记录当前知识库实际使用的 Chroma collection。正常文档更新仍在该 collection 内完成；该字段只用于现有数据首次迁移时安全切换到 v2 collection，不实现持续的蓝绿索引发布。

### 6.2 扩展 documents

新增字段：

```text
content_hash
current_version
parser_version
active_index_version
parsed_artifact_path
```

### 6.3 新增 document_versions

保存每次文档更新的状态和解析产物：

```text
version_id
doc_id
version
content_hash
parser_version
parsed_artifact_path
status
created_at
```

同一文档的 `doc_id + version` 唯一，同一知识库内可通过 `content_hash` 检测重复文件。

### 6.4 新增 document_chunks

MySQL 作为 Chunk 元数据、Parent 内容和相邻关系的事实来源：

```text
chunk_id
doc_id
document_version
parent_chunk_id
chunk_type
content
retrieval_text
content_hash
section_path_json
page_start
page_end
position
token_count
index_version
status
```

索引至少覆盖：

- `doc_id + document_version + position`
- `parent_chunk_id`
- `status + index_version`

### 6.5 扩展索引任务

`document_index_tasks` 增加：

```text
document_version
index_version
attempt_count
```

现有 `task_id` 继续作为本次索引写入的批次标识。

## 7. 增量索引与一致性

### 7.1 上传与更新

```text
计算文件 SHA-256
  ├─ 更新操作且 Hash 未变化：返回 no-op，不重新解析和 Embedding
  ├─ 同知识库已有相同 Hash：返回 409 和已有 doc_id，不创建重复记录
  └─ 内容变化：创建新 document_version 和索引任务
```

跨知识库即使内容相同也分别建立索引，避免引入跨知识库共享和生命周期耦合。

### 7.2 索引任务

1. 使用 `doc_id + document_version` 获取单文档任务锁。
2. 解析原文件并写入临时解析产物。
3. 生成 Parent 和 Child，写入 MySQL，状态为 `staging`。
4. 生成 Child Embedding。
5. 将新 Child 以新 ID 写入 Chroma/BM25，旧 Child 暂时保留。
6. 验证新索引完整后，将 MySQL 当前版本原子切换到新版本。
7. 检索层只接受与 MySQL 当前版本一致的候选，因此切换后旧 Child 即使尚未清理也不会进入 RRF。
8. 清理旧 Chroma/BM25 Child，并提交解析产物路径。

### 7.3 失败处理

- 解析、切块或 Embedding 失败：删除本任务 `staging` Chunk 和已写入的新 Child，旧版本保持 active。
- 新 Child 写入失败：旧索引从未删除，无需重建；清理本任务已经写入的部分新 ID。
- MySQL 激活成功但旧 Child 清理失败：查询仍通过 active version 过滤旧候选，后台恢复任务继续清理。
- 相同任务重复执行：稳定 Chunk ID 与 upsert 语义保证幂等。
- 服务启动时扫描未完成任务，并根据任务阶段决定继续、回滚或标记失败。

本轮不实现跨 MySQL/Chroma 的严格分布式事务，使用单文档锁、MySQL 事实数据和补偿恢复保证最终一致性。

## 8. 结构化解析

### 8.1 PDF

- 使用 PyMuPDF block/dict 输出，而不是只调用纯文本提取。
- 保存页码、块顺序和 bounding box。
- 根据字号、字体、位置和短文本特征识别基础标题。
- 统计重复出现的页首、页尾文本，移除高频页眉页脚。
- 合并同一页面内被错误断开的正文行。
- 不在本轮实现扫描 PDF OCR；无文本 PDF 明确标记解析失败，不索引占位错误信息。

### 8.2 Word

- 根据 `Heading 1/2/3` 等样式维护 `section_path`。
- 段落、列表和表格分别生成 Element。
- 表格规范化为带表头的 Markdown 文本，避免简单拼接后失去行列含义。

### 8.3 Markdown/TXT

- Markdown 识别 Heading、列表、表格和代码围栏。
- 代码块作为原子 Element，在超过 Token 上限时再按行兜底。
- TXT 使用标题启发式和段落结构；无法识别标题时保持空 `section_path`。

### 8.4 其他格式

Excel、PPT 和图片继续适配旧解析结果，通过兼容转换器生成基础 Paragraph/Table Element，不在本轮提升解析精度。

## 9. Parent-Child 切块

### 9.1 Parent

- 以章节为优先边界。
- 建议目标 1500～2500 Token。
- 超长章节按子标题、段落顺序拆成多个 Parent。
- Parent 不参与检索，只用于补全上下文。

### 9.2 Child

- 以段落、条款、列表项、表格分组为优先边界。
- 建议目标 200～500 Token。
- 不跨 Parent 合并。
- 过短 Child 优先和同一 Parent 内相邻内容合并。
- `retrieval_text` 包含标题面包屑，例如：

```text
教育基础知识 > 教育概述 > 教育与社会的关系

政治经济制度决定教育的性质和目的……
```

### 9.3 Token 计数

引入 `TokenCounter` 抽象。首期使用可本地运行的 tokenizer 近似统计，并将实现封装，避免未来更换生成模型时修改切块器。

## 10. 检索链路

### 10.1 Candidate 模型

```python
class RetrievalCandidate:
    chunk_id: str
    document_id: str
    content: str
    metadata: dict
    vector_rank: int | None
    bm25_rank: int | None
    vector_score: float | None
    bm25_score: float | None
    rrf_score: float
    rerank_score: float | None
```

各阶段不再反复覆盖同一个 `score` 字段。

### 10.2 宽召回

- 向量检索默认 Top30。
- BM25 默认 Top30。
- 查询扩展默认关闭，保留原始问题。
- 多轮指代明显时继续使用现有查询改写。
- 复杂问题拆解改为显式配置，不能无条件改变原问题。

### 10.3 RRF 融合

```text
RRF(d) = Σ 1 / (k + rank_i(d))
```

- 默认 `k=60`，通过配置项管理。
- 基于 `chunk_id` 合并候选，不使用内容前缀。
- 原始 BM25 和向量分数仅用于 Trace，不直接跨尺度相加。
- RRF 后保留默认 Top40 进入后续阶段。

### 10.4 Rerank

- 专用模型默认处理 RRF Top30，可配置 20～50。
- 根据 rerank 分数排序后再应用模型专属阈值。
- 模型失败时回退关键词重排，但使用独立阈值或只排序不拒答。
- 不把关键词覆盖率伪装成模型相关性分数。

### 10.5 MMR 与多样性

- MMR 不再位于 Rerank 之前。
- 默认先使用文档配额和 Parent 去重控制重复。
- 只有候选高度同质时才在 Rerank 后启用 MMR。
- MMR 失败时保持 Rerank 顺序，不重新使用融合分数排序。

## 11. ContextAssembler

从 `Retriever` 中拆出独立的上下文组装职责：

1. 接收 Rerank 后的 Child。
2. 按 `chunk_id`、`parent_chunk_id` 去重。
3. 默认每个文档最多保留 3 个检索命中。
4. 对高分命中按配置补充同 Parent 相邻 Child。
5. 当 Parent 未超出单来源预算时，可直接补充 Parent。
6. 按 Token 预算选择最终上下文，默认输出 5～12 个上下文单元。
7. 保留原命中 Child 的 rerank 分数和来源信息。

Parent 或相邻块只用于补充语境，不继承命中分数，也不显示为独立检索命中。

## 12. 模块边界

建议新增或调整：

```text
app/core/document_models.py       结构化文档和 Chunk 模型
app/core/parser.py                格式入口和兼容调度
app/core/parsers/                 PDF、Word、Markdown/TXT 解析器
app/core/splitter.py              StructuralChunker
app/core/chunk_repository.py      MySQL Chunk/版本访问
app/core/indexing_worker.py       幂等索引和补偿切换
app/core/retriever.py             召回、RRF、Rerank
app/core/context_assembler.py     Parent/相邻块和 Token 预算
app/core/rag_engine.py            仅负责流程编排
```

`rag_engine.py` 不再直接处理解析细节、Chunk 元数据拼装或上下文选择。

## 13. 兼容与迁移

- 数据库变更采用幂等 schema migration，不能通过删除重建表完成。
- 现有文档记录补默认 `current_version=1`、`index_version=1`。
- 旧 Chroma Chunk 缺少稳定 ID、Parent 和 section path，不能直接兼容 v2 上下文组装。
- 提供一次性“重新索引现有 ready 文档”操作。
- 一次性迁移为每个知识库创建临时 v2 collection；全部文档成功后更新 `active_collection_name`，失败时继续使用旧 collection。
- 迁移切换稳定后再由显式清理操作删除旧 collection；普通文档更新不创建新 collection。
- API 请求和聊天响应保持兼容，前端不要求同步改造。

## 14. 错误处理

- 空文档、扫描 PDF、低质量解析结果不进入索引，文档状态明确失败原因。
- Parser 错误、Embedding 错误和 Chroma 写入错误分阶段记录。
- Rerank 失败允许回退，但返回结果应记录实际使用的方法。
- Parent/相邻块读取失败时退化为仅使用命中 Child，不中断问答。
- Token 预算异常时保留最高分 Child，禁止返回空上下文后继续生成。

## 15. 测试策略

这部分是代码回归保障，不建设正式评测平台。

### 15.1 数据与索引

- 相同文件重复上传不重复 Embedding。
- 相同任务重复执行不会产生重复 Chunk。
- 文档内容未变化时更新为 no-op。
- 新版本索引失败后旧版本仍可检索。
- 删除和更新后旧 Chunk 不再出现在 Chroma、BM25 和 MySQL active 查询中。

### 15.2 解析与切块

- PDF 页码、标题路径和段落顺序。
- Word Heading 层级、列表和表格。
- Markdown Heading、代码块和表格。
- Parent-Child 关系、稳定 Chunk ID、Token 上限和内容完整性。
- 表格/代码内容中不残留内部特殊标记。

### 15.3 检索与上下文

- RRF 排名计算确定且不依赖原始分数尺度。
- 同一 Chunk 在双路召回中正确合并。
- Rerank 接收宽候选集，阈值只在 Rerank 后应用。
- 关键词回退不复用模型阈值。
- 文档配额、Parent 去重、相邻块补全和 Token 预算。
- Rerank、Parent 读取失败时的降级路径。

### 15.4 现有回归

- 保持当前 API、异步索引、文档更新、会话和 SSE 测试通过。
- 针对当前仓库已有未提交修改，实施前逐文件比对，不覆盖用户改动。

## 16. 实施顺序

1. 数据模型、Schema migration、Chunk Repository。
2. Hash、版本和幂等索引机制。
3. Structured Parser 与解析产物持久化。
4. Parent-Child StructuralChunker。
5. Child 写入 Chroma/BM25 和旧数据重索引支持。
6. Candidate 模型、宽召回与 RRF。
7. 宽候选 Rerank、独立阈值和降级策略。
8. ContextAssembler。
9. RAGEngine 接线与兼容回归。

## 17. 完成标准

- 相同文档和相同索引任务不会产生重复 Chunk 或重复 Embedding。
- 文档更新失败不会丢失上一可用版本。
- PDF、Word、Markdown/TXT 的 Child 均带稳定 ID、Parent、section path 和页码或顺序信息。
- 检索使用 BM25/Vector 独立 TopN、RRF、宽候选 Rerank 和 Rerank 后阈值。
- 上下文能够补充 Parent 或相邻块，并遵守文档配额和 Token 预算。
- 现有 API 与 SSE 协议保持兼容。
- 新增核心测试及全部现有测试通过。
