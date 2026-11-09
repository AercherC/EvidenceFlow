# rag_e2e_v1.5

**与 v1.4 的唯一差别是索引身份字段**。题集、证据映射、claim anchors、检索 fixture 与全部期望指标
与 v1.4 逐字节相同；改变的只有 `mapping.json` 的 `collection_name` / `index_signature`，以及随之重算的
`evidence/fixtures/retrieval_v14.json` 的 `mapping_sha256`。

## 为什么需要 v1.5

v1.4 冻结时绑定的索引是 `ef_research_bge_m3_hybrid_c512_v2`。2026-11-09 激活 staging 之后，
生产 active 指针指向 `ef_research_bge_m3_hybri-01a1d521__staging__4c7c0173a18adc3f`，
两者 `index_signature` 不同（`b4e9432c…` → `4c7c0173…`），于是 `check_identity` 会拒绝任何新观测：

```
observation E2E-M001 is from collection '…4c7c0173…', mapping is for 'ef_research_bge_m3_hybrid_c512_v2'
```

## 为什么证据仍然有效

两个索引经实测是同一索引的元数据修复版本，而不是不同的语料：

| 对比项 | 结果 |
|---|---|
| chunk 集合 | 817/817 完全相同 |
| chunk 内容 / token_count | **0 处差异** |
| 元数据差异 | **483 处，全部是 `heading_path`**（chunker 缺陷修复） |
| embedding 向量 | 0/817 逐位相同（cos 0.9951–1.0000，浮点重算漂移） |
| 映射引用的 188 个 chunk_id | 全部存在于新索引 |
| **在当前 active 上实测 P0 召回** | **152/233，与 v1.4 基线一致** |

## 身份字段

- dataset fingerprint：`464fda056cd9d380f05704141269a81506167ac149eba106777b347516df8254`（同 v1.4）
- dataset.sha256：`14611544333b525950aeae7c67327475749e0ab7fc22c3a0d92f9ac6b57a0c00`（同 v1.4）
- `collection_name`：`ef_research_bge_m3_hybri-01a1d521__staging__4c7c0173a18adc3f`
- `index_signature`：`4c7c0173a18adc3f40406ae38146bb12450a9d1bca3e2ceaa200d9b12cbc2b94`
- `chunk_id_hash`：`e87a613ebe65b10262996e83974b55839e0b6c27fbe22053ee7b8826dd1f153d`（同 v1.4）

## 已知局限

1. `near_miss` 的实体×属性共现判定使用 **200 字符邻近窗口**；换窗口会改变候选集合，
   因此该阈值属于冻结口径的一部分，不得在后续阶段修改。
2. 两类不可答题不参与证据映射，因此不进入 `context_all_required` / `context_claim_recall` 的分母。
3. `out_of_corpus` 题复用 Phase 12 的 `GEN-U01..05`，其 absence_terms 零命中已在
   当前 817-chunk 语料上复核；语料若再变更，必须先重新复核。
4. 难度标签来自候选 query 的 Dense Top20 排名，只作分片标签，不作为淘汰依据。
5. **语料一旦真正变化**（新增 PDF、重新切分），本目录的证据映射必须重做，不能沿用。
   与 v1.4 的差别仅在于 v1.5 现在能通过 `index_signature` 把这种变化检测出来，
   而 v1.4 只能比较集合名与 chunk 总数，检测不到。

## 不要修改本目录

`evals/` 下全部为冻结产物。要改题集只能新开版本目录，并同步更新 `PROJECT_SPEC.md`。
