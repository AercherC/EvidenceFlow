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
| 编排层 | `agent/adaptive.py` `agent/service.py` `rag/service.py` | 能力层 |
| 装配 | `runtime.py` | 全部（唯一装配点） |
| 入口 | `__main__.py` | 编排层 + runtime |

**守住它的测试**：`tests/test_architecture_boundaries.py`（AST 静态扫描）。

### §2.3 编排层不得 import 评测层

`runtime.py` / `retrieval/` / `generation/` / `tracing.py` 不得静态 import `evidenceflow.eval`。反向依赖会成环，并把评测逻辑漏进 tracing。

**落点**：`tests/test_architecture_boundaries.py::test_runtime_modules_do_not_import_eval`。

### §2.4 已退役模块不得复活

`evidenceflow.retrieval.bm25` / `fusion` / `hybrid_fusion` / `multilingual` / `translation` 被永久禁止。

**为什么**：全查询 BM25 + RRF 融合曾经存在并被证伪。现役 `retrieval/clause_bm25.py` 只向重排器提供少量词项候选，最终排序由重排器决定。

**落点**：`test_runtime_modules_do_not_import_retired_retrieval`。

### §2.5 / §2.6 单一装配入口

映射 Settings → 生产依赖只能有一个地方：`runtime.py`。其他模块不得定义 `from_settings`。

**落点**：`test_production_components_do_not_define_secondary_settings_factories`。

---

## §3 领域模型

`src/evidenceflow/domain/` 中的 pydantic 模型均为 **frozen**，且不持有任何资源（无客户端、无文件句柄）。

关键模型：`Document` / `DocumentSegment` / `DocumentChunk` / `RetrievedChunk` / `RerankedRetrievedChunk` / `BuiltContext` / `ContextEntry` / `GeneratedAnswer` / `RAGAnswer` / `SourceReference` / `RunTrace` / `TraceEvent` / `EvidenceIndexIdentity` / `EvidenceNeighborhood`。

**落点**：`tests/test_domain_models.py`。

### §3.1 身份生成

`document_id` = hash(NFC 规范化后的 POSIX 相对路径)；`content_hash` = sha256(**原始文件字节**，不是抽取出的文本)。

`chunk_id` 的身份载荷 = `{document_id, chunk_size, chunk_overlap, chunk_index, content_hash}` —— **不含 `heading_path`**。这一条有实际后果：补全章节归属不需要重算 embedding（见 §9.3）。

**落点**：`src/evidenceflow/domain/ids.py`，`tests/test_domain_models.py`。

### §3.2 只读评测索引身份

`EvidenceIndexIdentity.governance` 有三个取值：`lifecycle_active` / `lifecycle_staging` /
`settings_fallback`。新增 `lifecycle_staging` 只服务于只读评测：它表示这份快照来自 staging 指针，
**不代表** staging 被激活，也不把任何 Collection 的 `lifecycle` flag 改成 ACTIVE。

`runtime.build_evaluation_index(settings, target="active"|"staging")` 是唯一入口：它复用线上
`Settings`（Embedding / Reranker / 候选深度 / Context 预算完全一致），只以
`create_if_missing=False` 打开目标 Collection，读取前后各读一次目标指针并要求完全相等；
staging 集合必须带 `lifecycle=STAGING`、active 集合必须带 `lifecycle=ACTIVE`，否则
`IndexEvaluationError`。`describe_evaluation_index` 只读指针、不打开 Collection、不加载模型。

`scripts/build_staging_index.py` 在结构校验通过后把**实测** document/chunk 数与 staging manifest
写回 staging 指针（`IndexLifecycle.record_staging_measurement`），这样只读评测才能拿到真实
`chunk_count`；active/previous 指针与 Collection flag 均不动。

**落点**：`domain/evidence.py`、`runtime.py`、`knowledge/lifecycle.py`；
`tests/test_evaluation_index.py`。

---

## §6 运行配置

`src/evidenceflow/config.py::Settings` 是运行时参数的唯一来源。`.env` 可覆盖；无真实凭据入库。

---

## §9 切块策略

### §9.3 hybrid 切块

`chunk_strategy="hybrid"` + PDF 时：
1. `parse_structured_document` 重新打开 PDF、自建 canonical block 视图（**不消费 Cleaner 输出**）；
2. standalone 类型（table / table_caption / figure_caption / equation）独立成块。所有块按首个来源元素的阅读位置编号，同一元素的拆分窗口保持先后顺序；
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

### §10.1 只读证据导航

`EvidenceNavigator.neighborhood(chunk_id, before, after)` 在一个已加载索引中返回锚点及同一 `document_id` 下按 `chunk_index` 精确相邻的 Chunk；遇到缺失序号即停止，不跨文档。缺失的 `chunk_id` 明确报错，邻域半径限为 0–3。结果保留真实页码和索引身份。`EvidenceNavigator.chunk(chunk_id)` 是同一快照上的身份探针：命中返回深拷贝，缺失同样明确报错。

`runtime.build_evidence_navigator` 仅从当前 active collection 只读装配，不创建 collection 或加载模型；有 lifecycle 时结果绑定 active 指针的 `index_signature` 和 `chunk_count`，快照数量不一致即拒绝。无 lifecycle 的环境明确标记为 `settings_fallback`，不伪造签名。

### §10.2 同一次 run 共享一个索引快照（只读工具）

`runtime.build_evidence_tools(settings)` 装配 `agent/tools.py::EvidenceTools`，对外只有两个只读动作：

| 动作 | 契约 |
|---|---|
| `search(query)` | 走现有生产 retriever，返回既有 `RetrievedChunk` / `RerankedRetrievedChunk`；每个结果的 `chunk_id` 必须在绑定快照中再次命中，且 `document_id` / `chunk_index` / `content` / `content_hash` 与快照一致，否则 `EvidenceProvenanceError` |
| `open(chunk_id, before=1, after=1)` | 走同一个 `EvidenceNavigator.neighborhood`，返回既有 `EvidenceNeighborhood`；缺失 ID 抛 `UnknownChunkError`，不跨文档、不跨索引回退 |

**共享快照是硬合同，不是实现细节。** `search` 与 `open` 必须由**同一次** `iter_all_chunks_with_embeddings()` 读取构造：该次读取同时喂给 Exact Dense index、BM25 与 `EvidenceNavigator`。因此同一次 run 内不可能出现“search 命中来自新索引、open 却读旧索引”（或反之）的混合证据。禁止先 `build_retriever` 再 `build_evidence_navigator` 拼装——那是两次独立读取，正是本契约要排除的情形。pointer 文件在**集合读取前后各读一次**，只用于证明这份快照仍属于被声明的索引，不触发第二次集合读取。

生产调用方只能使用 `build_evidence_tools` 装配；直接调用 `EvidenceTools` 构造器不能证明 retriever 与 navigator 同源，仅供注入受控测试替身。

**失败语义**：

1. 读取前后 active 指针不同，或快照数量与指针 `chunk_count` 不一致（读取期间指针移动或 collection 变化）→ `ActiveCollectionError`，不返回任何结果，不把旧快照标成新签名；
2. 无法证明来源的 retriever 结果 → `EvidenceProvenanceError`；`EvidenceTools` 只接受 `EvidenceNavigator` 实例，不接受裸 chunk 列表或外部快照对象（构造期 `TypeError`）；
3. 缺失 `chunk_id` → `UnknownChunkError`；重复身份 / 数量不符 → `EvidenceNavigationError`（由 navigator 构造期拒绝）。

CLI `ask`、WebUI 和 HTTP API 统一通过 `build_agent_service` 使用该工具组。评测对照可在显式指定的只读快照上复用这两种工具。

**落点**：`src/evidenceflow/{domain/evidence,retrieval/evidence_navigation,agent/tools,runtime}.py`；`tests/test_evidence_navigation.py`、`tests/test_evidence_tools.py`、`tests/test_runtime.py`。

> 动作 JSON 合同、单题预算和去重规则见 §10.3。

### §10.3 证据 Agent

生产问答由证据 Agent 使用 §10.2 的两个只读工具。普通 RAG 与固定 Agent 由 §13.3 的评测入口装配。
共享的自适应服务实例必须为每次 `run` 隔离证据、动作、预算和生成 Context，避免并发问题串扰。
请求结束时释放工作线程持有的运行状态，答案只返回其公开结果合同。

#### 公共入口与装配

| 项 | 合同 |
|---|---|
| 装配 | `runtime.build_agent_service` 通过 `build_evidence_tools` 装配共享快照工具，再构造 `AdaptiveAgentService` |
| 入口 | CLI `ask`、WebUI 和 HTTP API 均调用 `build_agent_service`。旧 CLI 参数 `--agent` 兼容同一行为 |
| 模型能力 | 模型输出下述四种结构化动作，代码负责参数校验、去重、预算、截止时间和终止 |
| 回退 | 自适应路径**没有普通 RAG 回退**。预算/截止时间耗尽时不得暗中调用 `RAGService.answer`；固定 Agent 的回退语义不受影响 |

#### HTTP API

FastAPI 在 `lifespan` 装配并绑定一次只读快照。`GET /healthz` 检查进程响应，`GET /readyz` 返回已加载索引身份，
索引不可用时返回 `503`。发布新 active 后重启进程即可切换快照。

`POST /v1/ask` 使用现有 Agent 和引用校验，返回最终回答、实际引用、索引身份、终止原因和 Agent 预算计数。
证据不足作为 `200` 的 `insufficient_context` 返回。上游请求或模型输出失败返回 `502`，Agent 执行超时返回 `504`。
进程限制同时执行和排队的问答数，满载或排队超时返回 `503`。每次问答整体在一个工作线程执行，工作退出后释放执行名额。

#### 动作 JSON 合同

生产模型每次决策返回一个 JSON 对象（`extra="forbid"`），仅含 `action/chunk_id/query/reason`。
需求—证据合同属于本节末尾的实验候选，未启用到生产入口。生产 `action` 为下列之一：

| 动作 | 必填字段 | 代码行为 |
|---|---|---|
| `open` | `chunk_id` | 只允许打开**已由 search 命中**且属于同一 §10.2 快照的 `chunk_id`；调用 `EvidenceTools.open`，半径由代码固定为 1，模型不能指定 |
| `search` | `query`, `reason` | 在剩余检索与重排预算内改写检索；`query` 非空、长度受限、与已执行查询（含原始问题）规范化后不同；模型用 `reason` 描述具体证据缺口 |
| `answer` | — | 用当前收集证据构建上下文并生成答案 |
| `insufficient_evidence` | — | 结束探索并确定性拒答，不调用生成模型 |

普通动作只描述下一步与所需参数。跨文档定向补搜的论文身份由首轮检索计划绑定到快照，模型只提供题项的原文或搜索词。

#### 证据与 Context 纳入

1. 第一次 `search` 使用现有生产 retriever（经 `EvidenceTools`，不另造检索器）。
2. 普通决策最多显示 8 个检索摘要，每个最多 240 字符，并显示打开的邻文。若各题项明确指向不同且唯一的论文，决策在生成 Context 预算内逐项核对原文。摘录必须出现在该题项的论文块中，明确问数值时摘录须包含实值。缺项在原论文范围内补搜，已有支持块和补搜的新首块优先纳入下一轮 Context。

3. `open` 返回的真实邻域（anchor + 同文档相邻 chunk）进入证据集合；相邻 chunk 是原始 `DocumentChunk`，**没有相似度或 rerank 分数**。代码不构造、不推断、不展示任何伪检索分数。
4. `ContextBuilder.build_evidence` 接收 `ContextEvidence`，来源为 `retrieval` 或 `opened_neighborhood`。`retrieval` 条目保留真实 `search_index` 与 `retrieval_rank`，`opened_neighborhood` 条目的两者为 `None`。引用校验与来源映射使用 `ContextEntry` 的真实 `chunk_id`、`document_id`、`file_name` 和页码。
5. 跨 search 命中按块身份去重并保留首次命中来源，检索与打开证据按轮次交错。多子问题先纳入各路前 3 块，再补首轮尾部。`ContextBuilder` 按实际渲染 token 预算纳入证据，生成和引用校验复用同一份 Context。需求驱动的替代选择器暂不接入生产。

6. `RAGAnswer.retrieved_chunks` 只保存真实检索记录（`RetrievedChunk` / `RerankedRetrievedChunk`）；opened 邻居不进入该字段，避免把无分数证据伪装成检索结果。`SourceReference` 仍由 Context 条目生成，现有引用白名单/未知标签校验不变。

#### 预算与截止时间

| 预算 | 默认 | 语义 |
|---|---|---|
| `agent_adaptive_llm_calls_max` | 5 | 本 run 的 LLM 调用总上限，必须覆盖决策模型调用、生成和一次格式修复 |
| 生成预留 | 2 | 固定为生成 + 一次格式修复；决策调用不得侵占。服务在每次决策前要求 `已用 + 1 + 2 <= 总上限` |
| `agent_adaptive_decision_calls_max` | 3 | 决策模型调用上限（含一次无效动作修复） |
| `agent_adaptive_searches_max` | 2 | 含首次 search，支持 1–3；默认可改写一次，设为 3 时可改写两次，仍受重排和决策预算约束 |
| `agent_adaptive_opens_max` | 2 | `open` 工具调用上限 |
| `agent_adaptive_rerank_pairs_max` | 200 | 每次真实 `search` 按生产 retriever 的单次 pair 上限记账 |
| `agent_adaptive_deadline_seconds` | 20.0 | 墙钟截止时间；每个决策/检索/open/生成步骤开始前检查 |

每次真实 `search` 在结果里按生产 retriever 的 pair 上限做上界记账，因此检索成本上界 = 实际 search 次数 × 该上限，并由 `searches_max` 封顶。预算/截止时间一旦不允许下一步，探索立即结束；若已有证据且生成预留仍完整，则用预留合成答案并把终止原因记为 `budget_exhausted`；若无证据，则确定性返回证据不足。

若首次动作无效且已无额度再发修复调用，不额外超支：直接按 `budget_exhausted` 结束（用已收集证据合成，无证据则确定性拒答），而不是把无效动作伪装成证据不足。

生产路径的 `support` 把回答标为 `unassessed`，拒答标为 `none`。预算后的生成仍使用已收集证据。实验候选使用独立支持合同和工具后评估预留，见下文。

**时间限制**：每次 LLM 传输使用客户端超时与剩余时间的较小值。重试和生成修复共用同一截止时间；迟到的回复丢弃。同步网络和本地检索无法被强制抢占，实际端到端耗时单独实测。

#### 失败语义

| 情形 | 行为 |
|---|---|
| 动作 JSON 不合法 / 未知字段 / 字段组合错误 | 不执行该动作；用一次修复请求重问（计入决策与总预算）。修复后仍不合法 → `AdaptiveActionError`，**不**回退普通 RAG |
| `open` 的 `chunk_id` 不在已 search 命中集合或快照中 | 同“无效动作”：拒绝执行、至多一次修复、否则 `AdaptiveActionError` |
| 重复 `open` 同一 chunk / 重复 `search` 同一规范化查询 | 同“无效动作”；重复动作不会再次触发工具调用 |
| 模型请求超出 search/open 次数预算 | 不执行该动作；结束探索并按“预算耗尽”路径处理（有证据则预留合成） |
| 决策/生成所需的 LLM 预算或截止时间已耗尽 | 不启动新调用，不调用普通 RAG；预算耗尽走预留合成/确定性拒答，截止时间耗尽抛 `DeadlineExceeded` |
| LLM 服务/网络失败（`LLMError`） | 原样向上传播；与“证据不足”明确区分，不回退普通 RAG |
| 证据确实不足 | `answer.status = insufficient_context`、`termination_reason = insufficient_evidence`、`support = none`，不调用生成模型 |
| 生成格式错误 | 仍走现有 Generator 的一次格式修复；再失败抛 `GenerationContractError` |

`AdaptiveActionError` 表示模型动作合同失败。`DeadlineExceeded`、`LLMError`、`GenerationContractError` 和 `CitationValidationError` 表示运行或服务失败。

**落点**：`src/evidenceflow/agent/adaptive.py`、`src/evidenceflow/agent/tools.py`、`src/evidenceflow/{runtime,__main__}.py`；`tests/test_ask_cli.py`、`tests/test_webui.py` 和 `tests/test_tracing.py` 覆盖生产装配和公开入口。

#### 按子问题保留候选与前页补读

`AGENT_ADAPTIVE_TARGETED_EVIDENCE` 默认关闭，可在 staging 上单独评测。启用后只为首轮真实拆句分支
保留摘要位，8 条上限及最终 Top-K 沿用配置。若其中一个正文命中以英文小写词起头，且在前两块内
存在同章节的前页正文，可从共享快照补入一块原文。只允许越过同页的一条图表题注；实际新增的块
记为一次 `open`，保留原文和页码，未补到新块时不占用动作预算。单题继续沿用原路径。

#### 未启用的需求—证据候选

2026-11-13 成对验证后未通过生产替换验收：诊断 5/7 → 7/7，回归 9/10 → 8/10，新留出 6/6 → 5/6。`agent.experimental` 保留候选。`runtime` 只在 `build_evaluation_paths` 显式选择 `hybrid_agent` 时装配它。

`hybrid_agent` 复用评测索引快照及正规检索器。先执行一次首轮检索，有命中且题面明确查询单个属性时可用首轮 Context 直接生成，标记 `unassessed`。其他请求由需求—证据循环补证，最终共享生成和引用校验。评测报告保存实际分支和实际搜索词，用于检查直答漏路由和补搜质量。

直答返回证据不足后，候选只在剩余额度可覆盖两次评估和最终生成时尝试恢复。最多补搜一次，检索短语必须逐字来自原问题或带块 ID 的候选原文。子句检索启用 BM25 时，非 CJK 的单句补搜用正分词项候选替换少量 Dense 尾部位置，沿用相同重排器、重排深度与 TopK。新块进入完整支持的 Context 后才重新生成；否则返回原拒答与其对应 Context。评测把该分支记为 `direct_recovery`。

生成接收每个已选要点的一组支持原文及其当前 Context 标签。原文校验与标签绑定可确定执行，语义完整性和归属由逐项答案评审验收。

实验模型每次决策返回一个 JSON 对象（`extra="forbid"`），包含 `action/chunk_id/query/reason/requirement_key/requirements`。
`requirements` 为非空列表；每条含稳定的 `key`、来自原问题的要点 `text`、证据直接给出的具体 `answer`、支持组合 `evidence_sets` 及部分支持时待补的 `missing_detail`。
组合外层为 OR、内层为 AND，每组共同支持已知事实；块引用含 `chunk_id` 与连续原文 `quote`。空列表表示尚无支持。
实验响应的 `cross_document` 标记要点是否需要跨文档比较或综合。单篇论文要点只保留同一文档内的组合；整组证据都被排除时，该要点仍可继续补证。历史评测回放按原记录解释未标记的组合。
程序校验候选身份与原文匹配（允许空白、Unicode 连字和真实行尾断词差异；不删除普通词内连字符或数值符号），拒绝空组合以及后轮删除或改写既有要点。同一块可引用多个片段，逐条验证后按块去重计量。
缺失具体答案时 `answer=null`、`evidence_sets=[]`，可用 `missing_detail` 提供定向补搜的问题。非空答案与非空支持组合必须同时出现。旧记录省略 `missing_detail` 时按原有合同解释。
这些检查不证明语义支持；语义判断仍需成对评测。动作名称沿用生产四种动作。


候选将累计候选全文交给决策模型，校验需求与引用后使用 `EvidenceSelector` 构建同一份生成/引用上下文。
选择器先覆盖完整要点，再优先保留可纳入的部分支持，最后用余量保留完整要点的替代组合。
结果区分 covered、partial、missing、budget_excluded。合规的补证动作仍按预算执行，即使评估暂标为全覆盖。
工具调用须预留下一轮评估；部分回答前可按具体缺口定向补搜。支持评估通过后，动作失败仅修动作；若修复额度已尽，可使用已验证原文的支持进行生成。
补搜以待补属性为查询核心，可用目标论文或方法的名称、已发现的章节号和术语。实验响应可用
`search_query` 单独表达查询；自动补搜优先用它，其次尝试 `missing_detail` 或要点文本。
显式补搜通过 `requirement_key` 指向待补要点，优先使用该要点未执行过的专用查询，
其次使用动作的 `query`。旧响应在缺口唯一时可省略标识。动作修复遵循同一合同，检索
仍受现有次数、重排与评估预算约束。已有合格首轮证据与补搜结果共用选择器。
`support` 独立于生成状态，覆盖提示不作为事实来源。需求与证据状态按线程隔离。

离线行为测试见 `tests/test_agent_experimental.py`、`tests/test_evidence_assessment.py`、`tests/test_evidence_selection.py`。
历史成对报告见 `artifacts/agent_diagnosis_2026_11_13/assessed-agent-final.json`。定点复测见同目录的
`focused-current-diagnostics.json`、`focused-source-scope.json` 和 `focused-verified-context-generation.json`。
M216 验收需要明确新增带噪查询与自适应查询一起输入 decoder，以及 PF-Net 对抗训练目标。
M221 验收需要核对 CMLP 确实属于 PF-Net 编码器，并在答案写出论文名时核对 SVDFormer 的名称。
两题的关键词分数用于快速回归。逐项验收记录保存在本地开发目录的 `evals/rag_e2e_v1.5/AGENT_ANSWER_REVIEW.md`。生产替换另需独立留出集验证。

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

证据 Agent 记录 `run.*`、生成使用的 `context.*`、`citation_validation.completed` 和 `agent.adaptive.*`。`agent.search.started/completed` 按 `search_index` 区分轮次，完成时记录 `hit_count`；`agent.open.started/completed` 按 `open_index` 区分次数，分别记录 `chunk_id` 与 `hit_count`；`agent.decision.started/completed` 按 `decision_index` 区分模型动作，完成时记录通过校验的动作名。实验候选的 `agent.assessment.completed` 记录 `required_count/covered_count/partial_count/missing_count/budget_excluded_count/context_tokens`。检索与生成的内部事件由各自组件记录。

WebUI 通过 SSE 用同一请求的公开 Trace 更新检索时间线。每条回答可重新打开对应的步骤详情，查看实际检索轮次、候选分数、耗时和引用数量。左侧导航切换浏览器内保存的对话，行内引文按回答和标签定位到同版本索引中的真实引用块。文档库从同一只读快照列出文档及可阅读片段。查询全文、自由文本决策原因和引文正文留在 Trace 之外。

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

`e2e_diagnose.py observe` 通过 `build_evaluation_index` 读取 active 快照，输出该快照的集合名、块数和签名。
新观测要求已登记 active 指针。`check_identity` 拒绝与 mapping 签名冲突的观测，并继续读取历史无签名记录。

**为什么需要三条**：`collection_name` 与 `chunk_count` 的组合**不能唯一标识一个索引**——重建后同名同数、但证据已经移动是可能的。2026-11-09 激活 staging 后就发生过这种情况（同名同数、483 处 heading 差异）。

**落点**：`eval/e2e/diagnosis.py::check_identity`、`scripts/e2e_dataset.py::verify_manifest_identity`，
`tests/test_rag_e2e_v14_frozen.py`（含 `test_mapping_is_bound_to_one_index_identity`）。

### §13.2 active/staging 质量报告与激活门禁

**评测只读 staging，不切 active。** `scripts/e2e_index_quality.py compare` 通过
`runtime.build_evaluation_index(target="staging")` 只读打开 staging，active 指针全程不变；
两侧用同一份题集、同一套模型与配置，gold 映射各由 `scripts/e2e_dataset.py map --target
active|staging` 在对应索引上生成。

**报告合同** `rag-e2e-index-quality-v1`（`eval/e2e/index_quality.py`）：

- 绑定两个真实身份：`active` / `staging` 各自的 `collection_name` + `index_signature` +
  `chunk_count`；mapping 的这三项必须与对应指针一致，否则拒绝出报告。
  比对脚本还会用已读快照核对 mapping 已记录的 `chunk_id_hash`，防止同签名同块数的重建混用映射。
- 复用既有口径，不新造指标：候选证据覆盖与最终 Context 覆盖直接来自
  `diagnosis` 的 group stage（`CANDIDATE_COVERED_STAGES` / `CONTEXT_COVERED_STAGES`）。
- 新观测读取同一次检索 trace 的分支排名，按实际参与重排的候选并集归因。Context 覆盖按实际纳入的块判定。
  已冻结的旧观测缺少分支候选时，沿用全局 Dense 候选和已返回分支片段的估算口径。
- 输出逐题变化（`CaseChange`）与受影响切片（`SliceChange`，含 `multi_doc` / `single_doc` /
  `unanswerable` / `all`）。

**三种结论**（代码不发明普适阈值）：

| 结论 | 条件 |
|---|---|
| `passed` | 无 blocker、旧题没有最终 Context 覆盖丢失，且新增 PDF 的人工批准证据组题全部进入最终 Context |
| `failed` | 旧题丢失最终 Context 覆盖，或新增 PDF 的人工批准证据组题在 staging 漏检 |
| `unverifiable` | 有 blocker：新增文档缺人工批准证据组 gold、gold 在 staging 上无法定位、两侧题集不一致、或只观测了部分题集（`--limit` 冒烟） |

**CLI 激活门禁。** CLI 的 `--activate`、`--activate-staging` 在结构校验
`validate_collection` 通过后，必须再复核一份 `--quality-report`：两个身份与当前指针完全一致、
`status == "passed"`、无 blocker、无 `new_documents_without_human_gold`、无 `context_groups_lost`、
无 `new_document_context_misses`。
未提供报告、报告不可读、身份不符或结论不达标一律拒绝激活（退出码 4），staging 与指针保留、
active 不动。结构校验（可读性 / 数量 / 维度 / smoke）仍**独立**存在，它不构成质量门禁。

**WebUI 增量入库。** 上传一份 PDF 后复用原索引的片段和向量，只解析、清洗、切块和向量化新增文档。
后台确认索引写入完整后原子切换 active 指针，旧索引可回滚。增量清单记录复用的索引签名。
质量评测可单独运行。「就此文档提问」按文档 ID 限定整轮检索和补搜。

**分阶段检索。** 显式文档范围在 Dense 取 Top-K 前限定向量行，BM25 在相同文档内筛选。
多个明确引用编号可为单题提供跨语言词项候选；词项候选使用原重排预算中的尾部位置。
双题项策略从整题分支借出配额给各题项，保持总重排上限。能唯一识别论文的题项可独立定向，
另一题项仍按全库检索。两项来源都不明确的跨语言查询沿用原整题候选深度。
默认 Agent 优先把已召回的各题项首块放入上下文，沿用原 token
预算和引用身份。评测分别观察送重排的候选、融合后名次、最终上下文和答案。

**扩容路径。** 当前 Chroma 负责持久化，进程内 Exact Dense 提供质量基线。规模扩大时，
先压测分支检索、索引载入和单文档入库成本。服务化应将权限、文档身份与版本过滤下推到召回层，
并按变更文档更新全文与向量索引。发布时两路检索和引用使用同一可回滚版本。

**工作台取证与会话。** SSE 单独发布已执行的搜索、阅读、证据复用与上下文条目。面板可以按同一快照
读取原文，通用 RunTrace 继续脱敏。LangGraph 按 `thread_id` 将近四次有引文的回答存入本地 SQLite。
对话文本用于解析指代，回答证据按索引版本、文档与片段身份校验，证据仍使用既有 token 预算。
失败和取消的操作不写入可复用证据。每轮的参数覆盖在提交时固定。

**新 PDF 必须有可核验的题。** `unstested_new_documents` 要求每篇新增文档至少被一道人为批准的
证据组题引用；缺失则报告为 `unverifiable`。已批准的证据组在 staging 上无法映射同样为
`unverifiable`，可映射但未进入最终 Context 则为 `failed`。不得用"检索跑通了"代替人工 gold。

**落点**：`domain/index_quality.py`（报告合同与门禁）、`eval/e2e/index_quality.py`、`eval/e2e/observe.py`、`scripts/e2e_index_quality.py`、
`knowledge/staging.py`、`webui/imports.py`；`tests/test_e2e_index_quality.py`、
`tests/test_staging_quality_gate.py`、`tests/test_e2e_path_compare.py`、`tests/test_webui_imports.py`。

WebUI 从 active 的语料清单追加一份 PDF 并构建候选，阶段来自现有 BuildJob。发布成功后，新问答和文档库按 active 指针加载新快照，在途问答绑定其起始版本。

### §13.3 同题三路径对照（普通 RAG / 固定 Agent / 自适应 Agent）

`domain/question.py` 提供检索、决策和生成共用的题面目标。命名题以题面描述补入正分词项，
替换 Dense 尾部。并列机制题取各自的描述，点名的训练任务可在快照中唯一对应的论文内召回，
以所问细节重排。若不同题项分别明确限定不同且唯一的论文，先在各论文内取 Dense 与正分词项候选，
保持真实的全库 Dense 名次。多目标候选分配完后，整题分支用剩余重排配额搜索全库。

`CLAUSE_BRANCH_FUSION_ORDER` 默认 `rank_aware`。整题分支每轮先选一次；各子问题每轮至多一次，
按下一块未入选证据在子问题和整题中的逆名次之和分配取块顺序，平局按题序。轮转顺序与要求的
输出深度无关，评测可以用同次完整融合的前缀复现线上 Top-K。`clause_first` 和 `original_first`
保留历史顺序。各模式共用候选、重排预算和 Top-K，命中的分数、候选名次和 Trace 分支均来自实际
选中该块的分支。行为与 runtime 装配由 `tests/test_clause_branch.py` 覆盖。

命名、显式多个问句和并列机制题使用 `generation/grounded_parts.py` 的逐项合同。每条答项保留题序，
引文摘录须匹配所引块原文，查值还须匹配该摘录。机制与目标按字段分别提交。生成器从校验后的标签
渲染引文，并沿用单次修复额度。至少一个答项通过原文校验时返回 `answered`，正文写明缺项；所有答项为空时
返回 `insufficient_context`。来源匹配验证的是原文位置，语义完整性及事实是否受支持仍按题验收。

`CLAUSE_BRANCH_SINGLE_QUERY_BM25` 为非 CJK、未拆句的常规问题保留 Dense 候选，在每次搜索的
重排上限内追加少量正分 BM25 命中，检索 Trace 记录实际候选数。`RERANKER_DENSE_RRF_WEIGHT` 在基础 Dense
路径中用 Dense 与 Cross-Encoder 名次做加权 RRF。两项默认关闭，沿用原 Top-K，引用继续使用索引原文。

`CLAUSE_BRANCH_PARALLEL_PAPERS` 为未拆句、明确分别询问两篇已识别论文的问题开启两条同文档候选分支，
并保留整题的全库分支。三条分支复用同一次 Dense 排名，总重排对数不超过基础候选深度和整题上限，默认关闭。

`scripts/e2e_path_compare.py` 在同一 `EvaluationIndex` 快照上装配三条路径：普通 RAG 用线上
`multi_doc_strategy` 的检索器；固定 Agent 对照强制 `clause_branch`；证据 Agent 沿用线上配置
策略并使用绑定同一快照的只读 `EvidenceTools`。每条结果记录答案、真实引用（标签 / chunk /
文件 / 页码）、已有确定性 `answer_level.grade_answer` 的 PASS 与 claim 计数、耗时、LLM 调用数、rerank pairs、终止原因；普通 RAG 的检索观测复用
`eval/e2e/observe.py::observe_case`（与 `e2e_diagnose.py observe` 同一实现，不再有第二份）。

自适应结果额外记录实际传给 Generator 的 `context_chunk_ids`，以及其中首轮未命中的
`supplemental_context_chunk_ids`。确定性拒答没有进入生成路径时，这两项为 `null`。
`evidence` 逐块记录本次收集的唯一证据，包括被 Context 预算排除的块。每条包含 `origin`、
`search_index`、`in_generation_context` 和 `cited`，可用来区分首轮命中、补搜和 `open`。
报告保存实际自适应预算；重复使用 `--case-id` 可在原冻结题集上选择同一批题做配对实验。

答案评分和诊断共用 D7 状态规则。允许拒答的可答题可用拒答通过答案评分，但仍受禁止论断约束。
单题单路径失败记异常类型、耗时和未知调用数，保留同题其他路径及后续题的结果。报告写出后，有失败时退出码为 1。

**真实 LLM 必须显式 opt-in**：不给 `--allow-llm` 时只做 retrieval-only（默认仅 `rag` 路径，
不构造 LLM 客户端）；Agent 路径在 retrieval-only 下直接拒绝，避免"看起来跑了"。

**落点**：`scripts/e2e_path_compare.py`、`runtime.build_evaluation_paths`；
`tests/test_e2e_path_compare.py`。

### 已知的指标口径限制（重要）

1. **指标是集合型，不是排序型。** 每个 case 的 `included_count == len(rows)` 且 `skipped_count == 0`——`max_context_tokens=4096` 从不触底。因此**只改变排序、不改变最终集合的处理，在指标上必然恒等**。

   > 2026-11-10 复核并更新：`reranker_final_k` 已由 6 提到 8，实测 207 题中 **206 题为 8 条、1 题为 7 条**，`skipped_count` 仍全为 0。本地实验记录在未纳入仓库的 `work/analysis/prod_reach.json`。论证链不变，但**常量必须写成随 k 变化的表述**，否则后来者会用 6 这个过时值去复核。
2. `near_miss` 的实体×属性共现判定使用 200 字符邻近窗口，属冻结口径，不得修改。
3. 两类不可答题不参与证据映射，不进 `context_all_required` / `context_claim_recall` 的分母。
4. `out_of_corpus` 题复用 `GEN-U01..05`；语料变更后必须重新核对 absence_terms。
5. **语料一旦真正变化**（新增 PDF、重新切分），证据映射必须重做。

### 检索变体的验收标准（2026-11-10 起）

单点帕累托判据（`groups >= baseline+3` 且 `0 single-doc lost`）已废弃——中间那条在结构上不可通过，任何重排序改动都必然有得有失。

现行标准如下；原始分析记录 `work/agent_design/next/GATE_UPGRADE_2026-11-10.md` 是未纳入仓库的本地文件，不作为本规格的外部依赖：

```
主判据：配对 bootstrap 95% CI 下界 > 0（重采样单位为 case，不是 group）
辅判据：single_doc 切片 CI 下界 >= 0
参考项：context drop、pairs、p95 latency
```

本地分析脚本 `work/agent_design/next/paired_gate.py` 使用固定 seed；该脚本未纳入仓库，不是发布工具。需要在干净检出中重跑门禁时，须先将其作为独立任务整理入库。

---

## §17.3.1 并发安全

共享组件（`Generator` / `LearnedReranker` / `LLMClient`）的每调用诊断量必须按线程隔离；模型与 tokenizer 调用必须串行化（fast tokenizer 并发会 "Already borrowed"）。

**落点**：`src/evidenceflow/concurrency.py::PerThread`、`tests/test_concurrency.py`。

---

## 附：修改本文件时

- 只写**契约**与**落点**，不复制实现细节。
- 每加一条契约，如果它能被测试守住，就加一个指向该测试的链接。
- 如果某条契约已被撤销或证伪，**保留条目并注明撤销原因与日期**，不要删除——历史判断的痕迹比干净的目录更有价值。
