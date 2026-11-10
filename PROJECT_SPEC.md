# PROJECT_SPEC — 规格索引

**本文件不是原始规格书的副本。** 原始 `PROJECT_SPEC.md` 未随仓库发布，但源码中有 21 个不同的章节号被引用（共 100+ 处）。本文件的作用是：对每一个被引用的章节号，指出它在**当前代码中的真实落点**和**实际生效的契约**。

用途：当你读到 `PROJECT_SPEC §13.1(e)` 这类注释时，来这里查它到底约束了什么、由哪里实现、由哪些测试守住。

> 维护规则：新增或修改一个被 `§` 引用的契约时，同步更新本文件对应条目。不要把本文件变成代码的第二份副本——只写**契约**和**落点**。

---

## §2 架构边界与装配

### §2.1 分层与依赖方向

| 层 | 目录 | 允许依赖 |
|---|---|---|
| domain | `src/evidenceflow/domain/` | 仅 pydantic，无副作用 |
| 适配层 | `embedding/` `storage/` `llm/` `ingestion/` | domain + 第三方 |
| 能力层 | `retrieval/` `context/` `generation/` `citation/` | domain + 适配层，**互不 import** |
| 编排层 | `rag/service.py` `agent/service.py` | 能力层 |
| 装配 | `runtime.py` | 全部（唯一装配点） |
| 入口 | `__main__.py` | 编排层 + runtime |

**守住它的测试**：`tests/test_architecture_boundaries.py`（AST 静态扫描）。

### §2.3 编排层不得 import 评测层

`runtime.py` / `retrieval/` / `generation/` / `tracing.py` 不得静态 import `evidenceflow.eval`。反向依赖会成环，并把评测逻辑漏进 tracing。

**落点**：`tests/test_architecture_boundaries.py::test_runtime_modules_do_not_import_eval`。

### §2.4 已退役模块不得复活

`evidenceflow.retrieval.bm25` / `fusion` / `hybrid_fusion` / `multilingual` / `translation` 被永久禁止。

**为什么**：全查询 BM25 + RRF 融合曾经存在并被证伪；现役的 BM25 只服务于 clause 池（`retrieval/clause_bm25.py`），从不排整个查询、从不融合分数。

**落点**：`test_runtime_modules_do_not_import_retired_retrieval`。

### §2.5 / §2.6 单一装配入口

映射 Settings → 生产依赖只能有一个地方：`runtime.py`。其他模块不得定义 `from_settings`。

**落点**：`test_production_components_do_not_define_secondary_settings_factories`。

---

## §3 领域模型

`src/evidenceflow/domain/` 中的 pydantic 模型均为 **frozen**，且不持有任何资源（无客户端、无文件句柄）。

关键模型：`Document` / `DocumentSegment` / `DocumentChunk` / `RetrievedChunk` / `RerankedRetrievedChunk` / `BuiltContext` / `ContextEntry` / `GeneratedAnswer` / `RAGAnswer` / `SourceReference` / `RunTrace` / `TraceEvent`。

**落点**：`tests/test_domain_models.py`。

### §3.1 身份生成

`document_id` = hash(NFC 规范化后的 POSIX 相对路径)；`content_hash` = sha256(**原始文件字节**，不是抽取出的文本)。

`chunk_id` 的身份载荷 = `{document_id, chunk_size, chunk_overlap, chunk_index, content_hash}` —— **不含 `heading_path`**。这一条有实际后果：补全章节归属不需要重算 embedding（见 §9.3）。

**落点**：`src/evidenceflow/domain/ids.py`，`tests/test_domain_models.py`。

---

## §6 运行配置

`src/evidenceflow/config.py::Settings` 是运行时参数的唯一来源。`.env` 可覆盖；无真实凭据入库。

---

## §9 切块策略

### §9.3 hybrid 切块

`chunk_strategy="hybrid"` + PDF 时：
1. `parse_structured_document` 重新打开 PDF、自建 canonical block 视图（**不消费 Cleaner 输出**）；
2. standalone 类型（table / table_caption / figure_caption / equation）独立成块；
3. 正文按 `section_path` 连续分组，组边界强制断开；
4. 贪心打包 ≤ `chunk_size` token，**overlap 冻结为 0**；
5. 每块写入 `heading_path = " > ".join(section_path)` 与 `element_ids` 溯源。

`chunk_strategy="fixed"`（回滚通道）与 Markdown/TXT 走固定滑窗，**该路径消费 Cleaner 输出**。

> 2026-11-10 修正：`IngestionService` 此前在 hybrid 路径上**无用地**调用了 `clean_document`，既白算一遍，又让 `DocumentCleaningError` 能失败一条本可成功的路径。现已改为仅在非 hybrid 路径调用。

**落点**：`src/evidenceflow/ingestion/{structure,chunker,service}.py`；
`tests/test_ingestion_service.py`（三个测试锁定：hybrid 在 cleaner 失败时仍成功、hybrid 不调用 cleaner、fixed 仍调用 cleaner）。

### §9.4 索引身份隔离

`chunk_strategy="hybrid"` 不得复用 `FIXED_IDENTITY_COLLECTIONS` 中的 collection 名。混用不同切块语义的向量会让检索语义漂移。

**落点**：`config.py::Settings.validate_chunk_configuration`，`tests/test_config.py`。

---

## §10 Chunk 身份

`build_chunk_id` 的载荷见 §3.1。强制要求：`chunk_index` 在文档内连续、`page_start/page_end` 取自所属元素的真实页码区间（不重编号）。

---

## §12.5 RunTrace

**契约**：tracing 只观察，永不改变答案或异常传播。

| 规则 | 实现 |
|---|---|
| 未传 recorder 时不构造任何 payload，开销与无 trace 相同 | 各组件 `trace_recorder` 默认 `None` |
| recorder 自身抛错只 warn，不改变结果 | `tracing.safe_record` |
| 并发请求各自隔离 | `tracing.trace_scope`（ContextVar）+ `resolve_recorder` |
| 事件名以 `.started` / `.completed` 结尾成对出现 | `RunTrace` 的 `finish()` 校验 |
| payload 只含 ID、计数、分数、耗时、token 用量 | `domain/trace.py` 白名单 |

`RAGService` 记录 `run.*` / `context.*` / `citation_validation.completed`；检索与生成的内部事件由各自组件记录。

**落点**：`src/evidenceflow/tracing.py`、`src/evidenceflow/domain/trace.py`、`tests/test_tracing.py`。

---

## §13.1 端到端评测（rag_e2e_v1 合同）

这是被引用最多的章节。它是**冻结评测集**的完整合同。

### 数据集与身份

- 数据目录：`evals/rag_e2e_v1.5/`（当前冻结版本；v1.4 保留为历史）。
- **不要原地修改**：`evals/` 下全部是冻结产物，改题集只能新开版本目录。
- 数据集 `fingerprint` 与 `dataset.sha256` 必须可复现。

### 分项合同

| 章节 | 内容 | 落点 |
|---|---|---|
| §13.1(a) | 数据集 schema、`required_claims`、`required_evidence_groups`、`case_id` 命名 | `eval/e2e/dataset.py`、`scripts/e2e_dataset.py` |
| §13.1(b) | 证据映射：`document_id + page_range + evidence_anchor`，与 chunk_id 解耦 | `eval/e2e/mapping.py` |
| §13.1(c) | 观测记录 `E2EObservation`（检索阶段、最终集合、token 统计） | `eval/e2e/diagnosis.py` |
| §13.1(c′) | 冻结 fixture 与 `mapping_sha256` 绑定 | `evals/*/evidence/fixtures/` |
| §13.1(d) | `key_match`：答案键匹配（NFKC、行尾连字符折叠、千分位归一） | `eval/e2e/key_match.py` |
| §13.1(e) | 失败归因：检索层 / 生成层 / 引用层分层归因 | `eval/e2e/diagnosis.py`、`scripts/e2e_diagnose.py` |
| §13.1(f) | 门禁判定与预注册 | `evals/*/evidence/round*/PREREG*` |
| §13.1(g) | 指标口径（Precision / Recall / MRR / MAP / nDCG，TREC-BEIR 语义） | `eval/ir_metrics.py`、`eval/metrics.py` |

### 索引身份强制（2026-11-10 起）

`E2EObservation.index_signature` 必须与 mapping 的 `index_signature` 一致，否则 `check_identity` 拒绝。

**为什么需要三条**：`collection_name` 与 `chunk_count` 的组合**不能唯一标识一个索引**——重建后同名同数、但证据已经移动是可能的。2026-11-09 激活 staging 后就发生过这种情况（同名同数、483 处 heading 差异）。

**落点**：`eval/e2e/diagnosis.py::check_identity`、`scripts/e2e_dataset.py::verify_manifest_identity`，
`tests/test_rag_e2e_v14_frozen.py`（含 `test_mapping_is_bound_to_one_index_identity`）。

### 已知的指标口径限制（重要）

1. **指标是集合型，不是排序型。** 每个 case 的 `included_count == len(rows) == 6` 且 `skipped_count == 0`（实测 207/207）——`max_context_tokens=4096` 从不触底。因此**只改变排序、不改变 top-6 集合的处理，在指标上必然恒等**。
2. `near_miss` 的实体×属性共现判定使用 200 字符邻近窗口，属冻结口径，不得修改。
3. 两类不可答题不参与证据映射，不进 `context_all_required` / `context_claim_recall` 的分母。
4. `out_of_corpus` 题复用 `GEN-U01..05`；语料变更后必须重新核对 absence_terms。
5. **语料一旦真正变化**（新增 PDF、重新切分），证据映射必须重做。

### 检索变体的验收标准（2026-11-10 起）

单点帕累托判据（`groups >= baseline+3` 且 `0 single-doc lost`）已废弃——中间那条在结构上不可通过，任何重排序改动都必然有得有失。

现行标准见 `work/agent_design/next/GATE_UPGRADE_2026-11-10.md`：

```
主判据：配对 bootstrap 95% CI 下界 > 0（重采样单位为 case，不是 group）
辅判据：single_doc 切片 CI 下界 >= 0
参考项：context drop、pairs、p95 latency
```

工具：`work/agent_design/next/paired_gate.py`（seed 固定，可复现）。

---

## §17.3.1 并发安全

共享组件（`Generator` / `LearnedReranker` / `LLMClient`）的每调用诊断量必须按线程隔离；模型与 tokenizer 调用必须串行化（fast tokenizer 并发会 "Already borrowed"）。

**落点**：`src/evidenceflow/concurrency.py::PerThread`、`tests/test_concurrency.py`。

---

## 附：修改本文件时

- 只写**契约**与**落点**，不复制实现细节。
- 每加一条契约，如果它能被测试守住，就加一个指向该测试的链接。
- 如果某条契约已被撤销或证伪，**保留条目并注明撤销原因与日期**，不要删除——历史判断的痕迹比干净的目录更有价值。
