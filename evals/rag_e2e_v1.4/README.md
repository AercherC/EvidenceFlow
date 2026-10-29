# rag_e2e_v1.4

- 合同：`PROJECT_SPEC.md` §13.1（rag_e2e_v1）
- 规模：207 题
- dataset fingerprint：`464fda056cd9d380f05704141269a81506167ac149eba106777b347516df8254`
- dataset.sha256（LF 字节）：`14611544333b525950aeae7c67327475749e0ab7fc22c3a0d92f9ac6b57a0c00`
- corpus_version：`bb184726b7f98756bd519d70e1a2a1f1224b5108e689680b2c328023ba1f4332`

## 分布

| 维度 | 取值 |
|---|---|
| expected_status | {<AnswerStatus.ANSWERED: 'answered'>: 202, <AnswerStatus.INSUFFICIENT_CONTEXT: 'insufficient_context'>: 5} |
| unanswerable_kind | {'None': 202, 'near_miss': 4, 'out_of_corpus': 1} |
| question_type | {'ablation': 22, 'conclusion_limitation': 7, 'experiment_setup': 24, 'implementation': 24, 'method': 27, 'motivation_concept': 25, 'multi_document': 52, 'out_of_corpus': 1, 'quantitative_results': 25} |
| query_language | {'en': 105, 'zh': 102} |
| provenance.source | {'canonical40': 40, 'derived_t1': 95, 'derived_t2': 15, 'derived_t3': 5, 'derived_t5': 4, 'derived_t6': 4, 'multidoc_v12': 43, 'phase12_unanswerable': 1} |

`near_miss` 占两类不可答题：4/5 = 0.80（Gate 要求 ≥0.50）

## 校验

冻结前 `validate_dataset(require_approved=True, allow_machine=True, enforce_grounding=True)`：issues = 0，case_count = 207。

## 已知局限

1. `near_miss` 的实体×属性共现判定使用 **200 字符邻近窗口**；换窗口会改变候选集合，
   因此该阈值属于冻结口径的一部分，不得在后续阶段修改。
2. 两类不可答题不参与证据映射，因此不进入 `context_all_required` / `context_claim_recall` 的分母。
3. `out_of_corpus` 题复用 Phase 12 的 `GEN-U01..05`，其 absence_terms 零命中已在
   当前 817-chunk 语料上复核；语料若再变更，必须先重新复核。
4. 难度标签来自候选 query 的 Dense Top20 排名，只作分片标签，不作为淘汰依据。

## 不要修改本目录

`evals/` 下全部为冻结产物。要改题集只能新开版本目录，并同步更新 `PROJECT_SPEC.md`。
