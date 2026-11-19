# Agent 答案逐项验收

这两题用于诊断回归。阅卷时给每个必答点记录「通过 / 部分 / 缺失 / 错误」、答案片段及对应引用。核对引用时，从实际生成 Context 解析回答所用的 `[S#]`，只用该陈述引用的原文判断支持关系。所有必答点通过，且回答额外写出的实体归属有据，才记为通过。关键词分数继续作为快速回归信号。

| 题目 | 必答点 | 应判失败的例子 |
|---|---|---|
| E2E-M216 | 明确 AdaPoinTr 的辅助去噪任务；说明在真值中心点上加噪得到带噪查询，并把带噪查询和自适应查询一起输入 decoder；说明 PF-Net 用分类网络作判别器，并以对抗训练区分预测与真实缺失区域。 | 只说「辅助去噪」和「对抗损失」；只列判别器层结构，省略训练目标。 |
| E2E-M221 | 明确 CMLP 是 PF-Net 编码器的特征提取器；明确 ResNet-18 是所问方法的 2D 分支骨干；如果主动给出论文名，核对 SVDFormer 与 SVFNet 模块的归属。 | 用「判别器结构类似 CMLP」作为编码器特征提取器的唯一引用；把 SVFNet 当成 SVDFormer 论文名。 |

## 已有对照的人工复核

| 保存的对照 | E2E-M216 | E2E-M221 |
|---|---|---|
| `requirement-bound-live-recheck.json` | 关键词 2/2；缺 decoder 的去噪机制，未通过。 | 关键词 2/2；CMLP 和 ResNet-18 正确，论文归属错误，未通过。 |
| `requirement-generation-fixed-context.json` | 任务名称、带噪查询的构造与使用、判别器及对抗目标均写出，通过。 | 组件答案正确，仍把 SVFNet 当成论文名，未通过。 |

两份历史输出位于 `artifacts/agent_diagnosis_2026_11_13/`。生产替换另用从未参与调试的单要点、多要点和不可答题；按题比较语义完整性、引文支持、错误作答、超时和延迟。

## 本次候选对照

`hybrid-stage-validation.json` 中，M216 一次定向补搜找回了 AdaPoinTr 第 6 页机制原文，回答写清了带噪查询及 decoder 的用法。模型给 PF-Net 要点选中的同一块原文涵盖对抗训练，最后的要点评估和回答却都只写了分类网络结构，判为未通过。M221 找回了 PF-Net 编码器的直接原文并答出所问骨干，仍把 SVFNet 写作方法名，判为未通过。两题均不计入独立留出成绩。

## 原文线索与按要点交接

- `hybrid-grounded-recovery-final.json` 中，M027 的实际补搜词来自首轮原文，补回并引用 Hunyuan3D-Omni 的参考文献条目。U001 的模型仍建议包含 Adam 的查询，程序未执行该补搜，保留首轮拒答。
- `grounded-handoff-fixed-context.json` 沿用历史定点对照的索引和有序原文块，按人工划分的要点传入原文摘录与对应标签。M216 明确解释去噪机制、PF-Net 判别器结构以及与真实缺失区域的对抗训练，通过固定输入验收。M221 答对两项属性，仍把 SVFNet 写作方法名，未通过。
- `hybrid-grounded-handoff.json` 的早期端到端对照中，M216 发生动作合同错误，M221 的自动评估误把仅介绍整体架构的原文标为可支持编码器特征提取器。固定输入的结果仅衡量给定正确原文与要点时的生成。

## 清理后的端到端验收

- **M216 未通过。** 在 `acceptance-complex-assessment.json` 中，移除复杂题不使用的 `search_clue` 后，路径从动作合同错误恢复到可返回部分答案。补搜 `AdaPoinTr decoder extra training task geometry-aware self-supervised objective` 未把关键机制块带入候选，缺辅助去噪任务和带噪查询的构造与使用。PF-Net 第 5 页原文同时支持分类网络和区分真实与预测缺失区域，最终只写了分类网络，遗漏判别目标。此前同索引对照中，含 `auxiliary denoising task` 与 `Section 3.6` 的专用查询能将机制块召至 Top8，题面和候选如何产生这样的查询仍待验证。
- **M221 通过。** 同一输出中的 CMLP 由 PF-Net 第 2 页及第 3–4 页的编码器原文支撑，ResNet-18 由 SVDFormer 第 4 页的 2D 骨干原文支撑。回答按题序直接给属性和值。
- **M027 未稳定通过。** `acceptance-grounded-search.json` 中用题面短语补回 Hunyuan3D-Omni 的参考文献条目并作答。`acceptance-direct-recovery.json` 中报动作合同错误。可选补搜在修复额度耗尽时改为保留首轮拒答，随后 `acceptance-recovery-error-fallback.json` 的单题验收正常结束，实际未发起补搜，仍漏答。
- **U001 通过。** 在 `acceptance-direct-recovery.json` 中保留有引用的拒答，只执行首轮检索，共调用模型两次。

以上是同索引上的诊断题，生产替换仍需独立留出题的语义与引文验证。
