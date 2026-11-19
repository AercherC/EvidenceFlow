# EvidenceFlow 技术说明

A research agent that reads local scientific PDFs and answers with verifiable citations.
The repository includes the implementation and selected evaluation fixtures. Tests and standalone evaluation scripts referenced below are maintained in the local development checkout.
See [快速开始](README.md) for installation and the first local run.

## Capability Layers

### 1. Scientific PDF Ingestion

```text
Documents -> Clean -> Structured ScientificDocument -> Hybrid Chunk (BGE-M3, <=512 tokens)
```

PDF parsing and cleaning use a fused strategy (selected from a parser benchmark). Table
structure is linearized, not preserved.

### 2. Indexing & Retrieval

```text
BGE-M3 Embedding -> Chroma
Query -> BGE-M3 Dense -> Exact Dense Top50 (in-process ExactDenseIndex, numpy.exact)
```

The production collection is resolved through the lifecycle active pointer, never
hard-coded. Chroma is the sole persistence source; the exact dense path is reproducible
across processes. The ACTIVE index is never ingested into directly — rebuilds go through
`scripts/build_staging_index.py` into a staging collection first.

### 3. Learned Reranking

```text
Exact Dense Top50 -> BGE Reranker -> Top8
```

`runtime.py` is the composition root for ingestion and question answering.
The CLI, WebUI, and HTTP API share `AdaptiveAgentService`. The Agent searches the local index,
then chooses whether to search again, open a neighboring passage, answer with citations,
or report insufficient evidence. Both tools read from the same index snapshot.

The question plan supplies the same ordered targets to retrieval, the Agent, and generation.
An explicit name lookup uses its description to replace a few Dense tail candidates with positive
BM25 matches. For a question that asks what two additions consist of, retrieval searches the two
descriptions independently. When an additional training task names a unique indexed paper, its
branch recalls within that paper and reranks against the requested training detail. The whole
question uses the remaining pair budget.
When separate subquestions explicitly name different indexed PDFs, each subquestion draws
its candidates from the corresponding paper before reranking. The full question still searches
the whole collection within the shared pair budget.

To try paper-aware retrieval with the bundled research corpus, set
`CLAUSE_BRANCH_PAPER_MANIFEST=evals/research_ingestion/academic_corpus_manifest_v1.json`.
For a query or subquestion naming one indexed paper, up to five Dense tail slots can take candidates
from that paper. Single-subquestion searches preserve the base ranking depth and score the pool once.
Clause reranking uses the paper title and section path, and explicitly
named English clauses use `cross-encoder/ms-marco-MiniLM-L6-v2`. Cache that model alongside
the existing BGE models before running offline. Citations and generated context use the
original indexed text.

The default `rank_aware` fusion reserves one whole-question hit per round. Subquestions
take one hit each, prioritizing their next unseen chunk by reciprocal rank in the subquestion
and whole-question lists. This matters when Top-K ends in the middle of a round. Set
`CLAUSE_BRANCH_FUSION_ORDER=clause_first` or `original_first` to replay a historical order.
Each selected hit keeps its selecting branch's scores and candidate rank.

For explicitly scoped subquestions, the decision call supplies cited excerpts and a possible
rewrite per part. The Agent checks the excerpt and paper identity, checks for a literal quantity
on numeric requests, and spends remaining search budget on the first uncovered part. Previously
supported passages and the best new hit get priority when building the answer context.
Other questions use the bounded retrieval summaries and opened passages to choose an action.

For staging comparisons, `AGENT_ADAPTIVE_TARGETED_EVIDENCE=true` reserves summary slots for
the initial clause branches. When a hit appears to continue the previous PDF page, the
Agent can add one preceding block from the same section within the existing open budget.
The candidate is off by default.

Clause candidates must fit `CLAUSE_BRANCH_PAIR_BUDGET`; an undersized budget is rejected
before reranking instead of silently dropping clause candidates. Repeated clauses count
each scoring call toward the reported pair cost.

For staging comparisons, `CLAUSE_BRANCH_SINGLE_QUERY_BM25=true` preserves the Dense pool for
unsplit non-CJK queries and appends positive BM25 matches that are outside it. It can score up
to `CLAUSE_BRANCH_BM25_K` more pairs per applicable query, within the existing per-search
limit and with the same final Top-K. The candidate is off by default.
On the base Dense path, `RERANKER_DENSE_RRF_WEIGHT=0.4` blends the Dense and cross-encoder
rankings before Top-K. It uses the same candidates and retains their original scores.
The default `0` uses the cross-encoder order alone.

`CLAUSE_BRANCH_PARALLEL_PAPERS=true` tests a document-scoped route when an unsplit question
explicitly asks for separate facts about two papers recognized by the configured aliases.
Both paper branches share one Dense lookup with the whole-query branch. Their combined
reranking fits `RERANKER_INPUT_K`, and the final Top-K stays at the base depth.

The assembled components can be shared across concurrent requests (e.g. behind a web
server). Per-call diagnostics (`Generator.last_diagnostics`, `last_usage`, reranker
timings) are kept per thread, model and tokenizer calls are serialized, and a per-request
trace recorder is supplied with `evidenceflow.tracing.trace_scope(recorder)` instead of
mutating the shared components.
The Agent releases its working evidence and generated Context when the request exits.

### 4. Evidence Context

```text
Retrieved / opened evidence -> bounded ContextBuilder -> BuiltContext
```

### 5. Answer & Citation

```text
Context Builder -> LLM (JSON output) -> Citation Validation -> RAGAnswer(status)
```

- Name lookups, questions with multiple explicit requests, and coordinated explanations use a
  per-part response. The generator verifies excerpts against the cited passages and writes the
  citations. Lookup values must occur in their excerpts. Invalid output gets one repair attempt.
- `answered` means the answer provides at least one requested detail with a valid citation. The
  answer body identifies gaps. `insufficient_context` means the context supports none of the
  requested information.
- The default LLM is DeepSeek. Switching providers requires overriding `LLM_BASE_URL`,
  `LLM_MODEL`, and usually `LLM_THINKING`.
- The LLM HTTP client ignores `HTTP(S)_PROXY` by default; set `LLM_TRUST_ENV=true` on
  networks that require a proxy.
- When retrieval yields no usable context, the deterministic abstention message follows
  the question language (Chinese if the question contains Han characters, else English).
- Endpoints, model names, and API keys come from `.env` or environment variables. No real
  credentials are stored in this repository.

### 6. Evaluation

Retrieval quality is measured with standard IR metrics (Precision, Recall, MRR, MAP, nDCG)
implemented canonically in `src/evidenceflow/eval/ir_metrics.py` following TREC / BEIR
semantics. Gold labels use `document_id + page_range + evidence_anchor`, so they do not
depend on chunk IDs and stay stable across re-chunking.

## Usage

```powershell
conda activate agent-gpu

# Optional: reuse a local model cache and stay offline
$env:HF_HOME = (Resolve-Path .cache/huggingface)
$env:TRANSFORMERS_OFFLINE = '1'

python -m evidenceflow ask "How does PoinTr represent point clouds for a Transformer?"
```

`python -m evidenceflow ingest` is only for a fresh environment (no lifecycle state file,
target is not the ACTIVE collection). Running it against the ACTIVE index is refused.

Copy `.env.example` to `.env` and fill in your own credentials before the first run.

### Local WebUI (optional)

The EvidenceFlow workspace uses the same evidence Agent as `ask`. Its three-pane interaction
uses SuperMew's chat, conversation history, and knowledge panel as the design template.
[SuperMew on GitHub](https://github.com/icey1287/SuperMew)

FastAPI serves the browser assets and streams actual search, follow-up retrieval, generation,
and citation validation steps. The UI offers searchable local history, conversation renaming
and Markdown export, inline citations with neighbouring passages, and a searchable document
library with an indexed passage reader. On smaller screens, history and evidence open as drawers.

```powershell
conda activate agent-gpu
python -m pip install -e ".[web]"

python -m evidenceflow.webui --host 127.0.0.1 --port 7860
```

Open <http://127.0.0.1:7860>. Enter sends the question, and Shift+Enter inserts a line
break. The browser assets are included in the Python package.

从项目根目录运行 WebUI 时，默认从 `.cache/huggingface` 离线加载 Hugging Face 模型。
可用环境变量 `HF_HOME` 指定其他缓存，或设置 `HF_HUB_OFFLINE=0` 允许 Hub 联网。
离线设置只作用于 Hugging Face 模型加载，问答仍会调用 `.env` 配置的 LLM 服务。

「取证记录」显示实际搜索词、文档范围、分支查询和原文位置，展开片段可回读原文。
「设置」可为当前对话调整检索轮数、相邻原文阅读次数、截止时间和会话证据复用。
每轮保留当时的生效预算。已有环境升级后用 `python -m pip install -e ".[web]"` 安装会话依赖。

浏览器保留完整对话，LangGraph 把最近四个有引文的回答及证据身份写入索引状态文件旁的
`*.webui/conversations.sqlite3`。重复提问可在核对同版本原文后直接复用，常见指代式追问
会尝试使用上一轮文档范围和证据。「本轮重新检索」可跳过证据复用。

#### 在文档库添加 PDF

1. 打开「文档库 → 添加 PDF」，选择一份带文字层的本地 PDF（上限 100 MiB）。
2. 页面显示解析、清洗切块、向量化和索引写入进度。系统只处理新 PDF，沿用已发布的旧片段和向量。构建期间问答继续使用原索引。
3. 写入完成后自动发布和刷新文档库。选择新文档中的「就此文档提问」可将该轮所有检索限定到这份 PDF。

入库要求已配置带语料清单的生命周期索引。候选位同一时间处理一项任务。由 CLI 创建的候选继续通过 CLI 发布或清理。工作台任务可在刷新页面后继续查看，构建失败时可在页面放弃。检索质量报告仍可通过 `scripts/e2e_index_quality.py compare` 独立测量。

### HTTP API (optional)

```powershell
conda activate agent-gpu
python -m pip install -e ".[api]"
python -m uvicorn evidenceflow.api.app:create_app --factory --host 127.0.0.1 --port 8000 --workers 1
```

```powershell
Invoke-RestMethod http://127.0.0.1:8000/readyz
Invoke-RestMethod http://127.0.0.1:8000/v1/ask -Method Post -ContentType 'application/json' `
  -Body (@{query = 'How does PoinTr represent point clouds?'} | ConvertTo-Json -Compress)
```

- `GET /healthz` reports HTTP liveness. `GET /readyz` reports the loaded index identity,
  or returns `503` when the active index is unavailable. The process binds one read-only
  snapshot at startup. Restart it to load a newly published active index.
- `POST /v1/ask` accepts `{"query":"..."}` and returns the final answer, its exact citations,
  the index identity, termination reason, and Agent usage counters. `insufficient_context`
  is an HTTP `200` answer status. Model request or output failures return `502` and the Agent
  deadline returns `504`. Each submission starts a separate run.
- Use one worker per model instance initially. `API_MAX_CONCURRENT_REQUESTS` defaults to `1`,
  `API_MAX_QUEUED_REQUESTS` to `4`, and `API_QUEUE_TIMEOUT_SECONDS` to `5`. A full queue or
  queue timeout returns `503`. The Agent execution deadline starts separately when work begins.

Interactive OpenAPI documentation is available at <http://127.0.0.1:8000/docs>.

### Adaptive retrieval budget

The default allows two searches per question, including the initial search. To try a third
search with the default reranker limits, set the budgets before running `ask`
or launching the WebUI:

```powershell
$env:AGENT_ADAPTIVE_SEARCHES_MAX = '3'
$env:AGENT_ADAPTIVE_DECISION_CALLS_MAX = '5'
$env:AGENT_ADAPTIVE_LLM_CALLS_MAX = '7'
$env:AGENT_ADAPTIVE_RERANK_PAIRS_MAX = '300'
```

For a paired experiment, keep the budgets above and choose the same frozen case IDs for both
runs. The first report includes the ordinary RAG baseline and the two-search agent. The second
report adds the three-search agent. The example selects three cases:

```powershell
$common = @('--dataset', 'evals/rag_e2e_v1.5/dataset.json', '--target', 'active', '--allow-llm',
            '--case-id', 'ARR-001', '--case-id', 'E2E-M213', '--case-id', 'E2E-U901')
$env:AGENT_ADAPTIVE_SEARCHES_MAX = '2'
python scripts/e2e_path_compare.py @common --paths 'rag,adaptive_agent' --out artifacts/eval/paths-k2.json
$env:AGENT_ADAPTIVE_SEARCHES_MAX = '3'
python scripts/e2e_path_compare.py @common --paths adaptive_agent --out artifacts/eval/paths-k3.json
```

Check the index identity and `adaptive_budget` in both reports. Each adaptive result lists every
collected chunk's source, search round, inclusion in the generation Context, and citation status.
Failed case/path results stay in the report. The script continues with the remaining cases and
returns exit code 1 after writing the report if any case/path failed.

### 需求驱动的证据选择实验与离线回放

**2026-11-13 验收结论：未替换生产路径。** 新候选在诊断题上改善，但新留出题从 6/6 降为 5/6，
10 题回归集从 9/10 降为 8/10。候选隔离在 `agent.experimental`，未从公共入口导出。
下述需求合同仅用于该实验与回放；现有 `agent.adaptive` 保持原有行为。

`agent.selection.EvidenceSelector` 接受回答需求及其支持组合，在 Context 预算内
选择证据，直接返回可交给生成器的 `BuiltContext`。同一块支持多个需求时只纳入一次；
多块共同支持的组合必须完整纳入。结果区分 `covered`、`partial`（纳入了部分事实的证据）、
`missing`（候选中尚无支持）和 `budget_excluded`（有支持组合但本次预算内未选入）。
Token 数复用正式 ContextBuilder 的完整渲染计量，包含来源头、标签和分隔符。

当前策略先以确定性贪心覆盖：优先新增覆盖与 token 代价之比，再比较覆盖数与总长度。
覆盖后在不丢失其他要点且预算允许时，改用模型排在前面、更具体的支持组合。
剩余容量优先保留其他要点的部分支持，其次纳入完整要点的替代组合。
它不承诺全局最优，`budget_excluded` 也不等于证明所有组合都无法放下。
未关联需求的候选不会为了填满预算而入选；没有需求时返回空上下文。

实验候选每轮决策同时输出需求、证据直接给出的具体答案、支持原文和动作。
已有部分事实时保留具体答案及支持组合，用 `missing_detail` 标记待补信息。没有任何支持时，
具体答案为 null，支持组合为空。每个支持组合先经过候选身份与
原文匹配校验（允许 PDF 排版空白、连字和行尾断词差异），再交给选择器；
同一块可支持多个需求，后续轮次不能删掉既有需求。
合规的补证动作按预算执行，即使当前模型评估显示全覆盖。部分回答前若仍有搜索与后续评估额度，
先按尚未查询过的具体缺口定向补搜。动作无效时保留同轮已通过校验的支持，只修动作字段。
工具调用必须留出下一轮评估额度，额度耗尽时以已评估支持为限。

`support` 描述实际上下文的评估覆盖，独立于答案状态；`evidence_assessed` 区分尚未评估。
`requirements` 与四个 `*_requirement_keys` 字段记录支持关系及去向，跨轮命中由
`evidence.search_indices` 保留。旧固定 Agent 仍作评测基线，共用结果类型位于 `agent.result`。

回放使用评测映射提供已知支持关系，隔离检查“已经找到的证据能否保留”。它不是答案正确率，
也不验证模型能否自行识别支持关系；生产路径不读取该映射。

用现有报告回放（需要同一索引与题集的 mapping；输出文件必须尚不存在）：

```powershell
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
# 对当前 active 索引生成相应题集的映射；已有匹配文件时复用它。
python scripts/e2e_dataset.py map `
  --dataset evals/rag_e2e_v1.4/dataset.json `
  --out artifacts/agent_diagnosis_2026_11_13/selection-replay-mapping.json
python scripts/e2e_selection_replay.py `
  --report artifacts/agent_diagnosis_2026_11_13/webui-readiness-paired.json `
  --mapping artifacts/agent_diagnosis_2026_11_13/selection-replay-mapping.json `
  --max-context-tokens 4096 `
  --out artifacts/agent_diagnosis_2026_11_13/selection-replay-stage1.json
```

输入报告记录累计候选及实际生成上下文；回放只读打开这些候选的完整原文，不新增候选，
不调用检索、重排或 LLM。没有进入生成的旧运行与生成了空上下文的旧运行分别记录。
选择合同见 `tests/test_evidence_selection.py`，回放口径见 `tests/test_e2e_selection_replay.py`。

2026-11-13 的 7 题诊断回放（4096 token，固定已知支持关系）：10 个必需证据组中，
累计候选覆盖 9 组，旧上下文保留 7 组，新选择器保留 9 组。M202、M225 从 1/2 组
变为 2/2 组；M206 仍缺 G2，其他题的覆盖未下降。该诊断集不是留出评测；
无答案题没有必需证据组，不能用这个回放结果判断其拒答是否正确。

最后一轮在相同索引、模型和预算下比较，未发现运行或引用校验错误：

| 样本 | 旧 Agent | 实验候选 |
|---|---:|---:|
| 诊断题（7） | 5/7 | 7/7 |
| 回归题（10） | 9/10 | 8/10 |
| 预选新留出题（6） | 6/6 | 5/6 |

新留出题平均耗时由 4.29 秒增至 5.39 秒，输入 token 总数由 36,924 增至 46,092。
M202、M225 得到修复；M221 将判别器的相似结构误当成编码器内部事实，M216 丢失
AdaPoinTr 训练任务信息。两例说明原文存在与语义支持是两回事，诊断收益不能代替留出收益。
最后一轮报告：`artifacts/agent_diagnosis_2026_11_13/assessed-agent-final.json`，内含验收结论。
诊断/回归题参与修正；仅 6 道新题为留出题，且没有剩余未用的无答案题，不能据此推断总体准确率。

### WebUI service behavior

The first library request or question loads the evidence Agent. After publication the next request
loads the new active snapshot, while a running question finishes on the snapshot it started with.
One question runs at a time and Stop ends it
at the next step boundary. The evidence panel shows public RunTrace steps as they happen. Completed
answers retain their retrieval steps, candidate scores, and budget usage.

History is saved in the browser for the current origin. Labels such as `[S1]` belong to one answer
and open its exact indexed passage. Versioned citations can be reopened after a restart serving
the same snapshot. For an unmanaged index, resubmit the question after restarting to locate the
source again. The document library uses the snapshot loaded for question answering.

「你是什么模型」读取工作台的模型配置。若限定文档时显示 HTTP 422 和版本提示，请重启 WebUI，
让后端加载当前请求结构。重试同一问题会保留所选文档。

双题项检索可从整题分支分出部分候选名额，已明确点名的题项可单独限定到对应论文。
跨语言查询的两项来源均不明确时保留整题候选深度。
含多个引用编号的单题会合并 BM25 候选。阅读「检索过程」可区分候选未召回、重排后未入选和上下文缺失。

要比较普通 RAG、生产 Agent 和共享首轮检索候选，可显式执行以下命令。逐项答案标准见
`evals/rag_e2e_v1.5/AGENT_ANSWER_REVIEW.md`。

```powershell
python scripts/e2e_path_compare.py --dataset evals/rag_e2e_v1.5/dataset.json --target active --paths rag,adaptive_agent,hybrid_agent --case-id E2E-M216 --case-id E2E-M221 --allow-llm --out artifacts/eval/hybrid.json
```

The workspace binds to loopback for single-user use. Trace details show the public RunTrace
actions, counters, scores, and durations.

## Repository layout

```text
src/evidenceflow/     implementation
  config.py           Settings (single source of run-time parameters)
  runtime.py          composition root for ingest / ask
  domain/             domain models
  ingestion/          PDF loading, cleaning, structure parsing, chunking
  retrieval/          dense retrieval and reranking
  eval/               IR metrics, e2e evaluation, gates
  storage/            Chroma persistence
  webui/              local FastAPI workspace, static frontend and SSE
  api/                optional FastAPI service over the same evidence Agent
scripts/              CLI entry points and maintenance scripts
tests/                offline test suite (Fake/Stub, no network)
evals/research_*/     research question sets and dense baseline used by tests
```

## 本地开发测试

```powershell
python -m pytest -q
```

The suite is designed to run offline against fixed inputs. Model-backed paths are
exercised through fakes and stubs rather than real API calls. Tests that asserted on
unpublished historical artifacts, and one-off cleanup/phase scripts, were removed; they are
recoverable from git history at commit 50b3323.

The WebUI suite exercises the HTTP and SSE routes with fake services. Install the web extra
to include these route checks.

```powershell
python -m pytest -q tests/test_webui.py
```

Tests marked `requires_model` load the real BGE-M3 model. They run when the model is in
the local Hugging Face cache (honours `HF_HOME`) and are skipped otherwise; set
`EVIDENCEFLOW_RUN_MODEL_TESTS=1` to force them (this may download the model).

## Known limitations

- Academic PDFs hyphenate across line breaks, which complicates anchoring.
- Tables are linearized; row/column structure is not preserved.
- References and supplementary sections have no structured metadata yet.
- Multi-document questions can only reliably retrieve a subset of their sources.

The system is currently text-centric. There is no evidence yet that multimodal parsing
would pay off.
