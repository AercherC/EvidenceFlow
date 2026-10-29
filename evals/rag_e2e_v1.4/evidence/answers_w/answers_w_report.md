# W (R2_c25_5en_rr_k8) answer-level check on frozen rag_e2e_v1.4

Caveat: the 30 new-case baselines were generated on 2026-10-27, the candidate runs and the 8 other baselines now; the LLM endpoint may have drifted.

Budget: 92/100 HTTP attempts, 271218/400000 tokens.

## PASS (38 triggered cases)

| slice | n | base1 | base2 | cand1 | cand2 |
|---|---|---|---|---|---|
| all38 | 38 | 8 | 8 | 14 | 14 |
| new30 | 30 | 5 | 5 | 10 | 10 |
| primary_B1 | 2 | 0 | 0 | 1 | 1 |
| primary_B2 | 6 | 1 | 1 | 1 | 1 |
| primary_B3 | 6 | 0 | 0 | 0 | 0 |
| primary_B4 | 6 | 1 | 1 | 2 | 2 |
| primary_B5 | 6 | 3 | 3 | 4 | 4 |
| spare | 4 | 0 | 0 | 2 | 2 |
| legacy_joined | 4 | 1 | 1 | 2 | 2 |
| single_doc | 4 | 2 | 2 | 2 | 2 |

Stable gains: ['E2E-M115', 'E2E-M201', 'E2E-M218', 'E2E-M220', 'E2E-M229', 'E2E-M236']
Stable losses: none
Within-pair flips: none
Net 6; bar max(3, 2 x 0) = 3; met: True; McNemar exact p = 0.03125; stable single-doc losses: none

## Per run

| run | PASS | abst (answerable) | citation errs | errors | prompt/case | completion/case | CITED | IN_CONTEXT_NOT_CITED | RERANK_MISS | CANDIDATE_MISS |
|---|---|---|---|---|---|---|---|---|---|---|
| base1 | 8 | 19 | 0 | {'E2E-M222': 'LLMRequestError'} | 2358.1 | 116.6 | 35 | 2 | 13 | 22 |
| base2 | 8 | 21 | 0 | - | 2423.1 | 113.8 | 36 | 1 | 13 | 22 |
| cand1 | 14 | 9 | 0 | - | 2871.4 | 120.5 | 45 | 1 | 14 | 12 |
| cand2 | 14 | 10 | 0 | - | 2871.4 | 117.4 | 45 | 1 | 14 | 12 |

## Per case

| case | slice | base src | base1 | base2 | cand1 | cand2 |
|---|---|---|---|---|---|---|
| E2E-M113 | legacy_joined | now | CANDIDATE_MISS | CANDIDATE_MISS | CANDIDATE_MISS | CANDIDATE_MISS |
| E2E-M115 | legacy_joined | now | CANDIDATE_MISS | CANDIDATE_MISS | PASS | PASS |
| E2E-M116 | legacy_joined | now | PASS | PASS | PASS | PASS |
| E2E-M118 | legacy_joined | now | CANDIDATE_MISS | CANDIDATE_MISS | CANDIDATE_MISS | CANDIDATE_MISS |
| E2E-M119 | single_doc | now | PASS | PASS | PASS | PASS |
| E2E-M120 | single_doc | now | PASS | PASS | PASS | PASS |
| E2E-M121 | single_doc | now | RERANK_MISS | RERANK_MISS | RERANK_MISS | RERANK_MISS |
| E2E-M122 | single_doc | now | UNATTRIBUTABLE | UNATTRIBUTABLE | UNATTRIBUTABLE | UNATTRIBUTABLE |
| E2E-M201 | primary_B1 | v12_stored_2026-10-27 | RERANK_MISS | RERANK_MISS | PASS | PASS |
| E2E-M203 | primary_B1 | v12_stored_2026-10-27 | RERANK_MISS | RERANK_MISS | CANDIDATE_MISS | CANDIDATE_MISS |
| E2E-M211 | primary_B2 | v12_stored_2026-10-27 | RERANK_MISS | RERANK_MISS | RERANK_MISS | RERANK_MISS |
| E2E-M212 | primary_B2 | v12_stored_2026-10-27 | CANDIDATE_MISS | CANDIDATE_MISS | CANDIDATE_MISS | CANDIDATE_MISS |
| E2E-M213 | primary_B2 | v12_stored_2026-10-27 | CANDIDATE_MISS | CANDIDATE_MISS | CANDIDATE_MISS | CANDIDATE_MISS |
| E2E-M214 | primary_B2 | v12_stored_2026-10-27 | CANDIDATE_MISS | CANDIDATE_MISS | RERANK_MISS | RERANK_MISS |
| E2E-M216 | primary_B2 | v12_stored_2026-10-27 | CANDIDATE_MISS | CANDIDATE_MISS | CANDIDATE_MISS | CANDIDATE_MISS |
| E2E-M217 | primary_B2 | v12_stored_2026-10-27 | PASS | PASS | PASS | PASS |
| E2E-M218 | spare | v12_stored_2026-10-27 | CANDIDATE_MISS | CANDIDATE_MISS | PASS | PASS |
| E2E-M219 | primary_B3 | v12_stored_2026-10-27 | CANDIDATE_MISS | CANDIDATE_MISS | CANDIDATE_MISS | CANDIDATE_MISS |
| E2E-M220 | spare | v12_stored_2026-10-27 | RERANK_MISS | RERANK_MISS | PASS | PASS |
| E2E-M221 | primary_B3 | v12_stored_2026-10-27 | CANDIDATE_MISS | CANDIDATE_MISS | RERANK_MISS | RERANK_MISS |
| E2E-M222 | primary_B3 | v12_stored_2026-10-27 | ERROR | CANDIDATE_MISS | GENERATION | GENERATION |
| E2E-M223 | primary_B3 | v12_stored_2026-10-27 | CANDIDATE_MISS | CANDIDATE_MISS | CANDIDATE_MISS | CANDIDATE_MISS |
| E2E-M224 | primary_B3 | v12_stored_2026-10-27 | CANDIDATE_MISS | CANDIDATE_MISS | RERANK_MISS | RERANK_MISS |
| E2E-M225 | spare | v12_stored_2026-10-27 | CANDIDATE_MISS | CANDIDATE_MISS | RERANK_MISS | RERANK_MISS |
| E2E-M226 | primary_B3 | v12_stored_2026-10-27 | CANDIDATE_MISS | CANDIDATE_MISS | CANDIDATE_MISS | CANDIDATE_MISS |
| E2E-M227 | primary_B4 | v12_stored_2026-10-27 | GENERATION | GENERATION | GENERATION | GENERATION |
| E2E-M228 | primary_B4 | v12_stored_2026-10-27 | RERANK_MISS | RERANK_MISS | RERANK_MISS | RERANK_MISS |
| E2E-M229 | primary_B4 | v12_stored_2026-10-27 | CANDIDATE_MISS | CANDIDATE_MISS | PASS | PASS |
| E2E-M231 | primary_B4 | v12_stored_2026-10-27 | PASS | PASS | PASS | PASS |
| E2E-M232 | primary_B4 | v12_stored_2026-10-27 | CANDIDATE_MISS | CANDIDATE_MISS | CANDIDATE_MISS | CANDIDATE_MISS |
| E2E-M233 | primary_B4 | v12_stored_2026-10-27 | RERANK_MISS | RERANK_MISS | RERANK_MISS | RERANK_MISS |
| E2E-M234 | primary_B5 | v12_stored_2026-10-27 | RERANK_MISS | RERANK_MISS | RERANK_MISS | RERANK_MISS |
| E2E-M235 | primary_B5 | v12_stored_2026-10-27 | RERANK_MISS | RERANK_MISS | RERANK_MISS | RERANK_MISS |
| E2E-M236 | primary_B5 | v12_stored_2026-10-27 | RERANK_MISS | RERANK_MISS | PASS | PASS |
| E2E-M237 | spare | v12_stored_2026-10-27 | CANDIDATE_MISS | CANDIDATE_MISS | RERANK_MISS | RERANK_MISS |
| E2E-M238 | primary_B5 | v12_stored_2026-10-27 | PASS | PASS | PASS | PASS |
| E2E-M239 | primary_B5 | v12_stored_2026-10-27 | PASS | PASS | PASS | PASS |
| E2E-M240 | primary_B5 | v12_stored_2026-10-27 | PASS | PASS | PASS | PASS |

## Answers for gains / losses / flips

### E2E-M115 (legacy_joined)

Query: What is the name of the unified framework for controllable 3D asset generation? At the same time, Which method, referenced as [57], employs a two-stage framework that first generates a coarse low-resolution point cloud and then uses a lifting module to increase density?

Keys: C1 (req=True): [['Hunyuan3D-Omni', 'Hunyuan3DOmni', 'Hunyuan3D Omni']]; C2 (req=True): [['PCN']]

- **base1** CANDIDATE_MISS [CANDIDATE_MISS,CITED]: The knowledge base does not provide enough information to answer the first question: no source names a unified framework for controllable 3D asset generation. Regarding the second question, [S1] states that PCN [57] and NSFA [59] adopt a two-stage generation framework in which a coarse, low-resolution point cloud is first generated by the decoder and then a lifting module increases point cloud density, so the method referenced as [57] is PCN.
- **base2** CANDIDATE_MISS [CANDIDATE_MISS,CITED]: The knowledge base does not provide enough information to answer the first question: no source names a unified framework for controllable 3D asset generation. Regarding the second question, [S1] states that PCN [57] adopted a two-stage generation framework in which a coarse, low-resolution point cloud is first generated by the decoder and then a lifting module increases point cloud density.
- **cand1** PASS [CITED,CITED]: The unified framework for controllable 3D asset generation is Hunyuan3D-Omni [S1]. The method referenced as [57] that employs a two-stage framework—first generating a coarse low-resolution point cloud and then using a lifting module to increase density—is PCN [S2].
- **cand2** PASS [CITED,CITED]: The unified framework for controllable 3D asset generation is Hunyuan3D-Omni, described as "a unified framework for controllable generation of 3D assets" [S1]. The method referenced as [57] that uses a two-stage framework—first generating a coarse, low-resolution point cloud and then applying a lifting module to increase density—is PCN [S2].

### E2E-M201 (primary_B1)

Query: What does the decoder of PCN produce, and what does PF-Net's network output instead?

Keys: C1 (req=True): [['coarse output'], ['detailed output']]; C2 (req=True): [['missing part', 'missing region', 'missing point cloud']]

- **base1** RERANK_MISS [CITED,RERANK_MISS]: PCN's decoder takes the encoder's feature vector and produces a coarse output point cloud Ycoarse followed by a detailed output point cloud Ydetail [S1][S4]. By contrast, PF-Net estimates the missing point cloud hierarchically using a feature-points-based multi-scale generating network, preserving the spatial arrangements of the incomplete input and predicting the detailed geometrical structure of the missing region(s) [S3].
- **base2** RERANK_MISS [CITED,RERANK_MISS]: PCN's decoder takes the encoder's feature vector and produces a coarse output point cloud Ycoarse followed by a detailed output point cloud Ydetail [S1][S4]. By contrast, PF-Net estimates the missing point cloud hierarchically using a feature-points-based multi-scale generating network, preserving the spatial arrangements of the incomplete input and predicting the detailed geometrical structure of the missing region(s) [S3].
- **cand1** PASS [CITED,CITED]: PCN's decoder takes the encoder's feature vector and produces a coarse output point cloud Ycoarse followed by a detailed output point cloud Ydetail [S1][S6]. In contrast, PF-Net takes the partial point cloud as input and outputs only the missing part of the point cloud rather than the whole object, which preserves the spatial arrangements of the original part and helps the network focus on the location and structure of missing regions [S8].
- **cand2** PASS [CITED,CITED]: PCN's decoder takes the encoder's feature vector and produces a coarse output point cloud Ycoarse followed by a detailed output point cloud Ydetail [S1][S6]. In contrast, PF-Net takes the partial point cloud as input and outputs only the missing part of the point cloud rather than the whole object, which preserves the spatial arrangements of the original part and helps the network focus on the location and structure of missing regions [S8].

### E2E-M218 (spare)

Query: How many epochs is PointAttN trained for, and what extra paired input does the method that self-projects the partial cloud into depth maps manage without?

Keys: C1 (req=True): [['400 epochs']]; C2 (req=True): [['color images']]

- **base1** CANDIDATE_MISS [CITED,CANDIDATE_MISS]: PointAttN is trained for a total of 400 epochs, with the learning rate initialized to 1E-4 and decayed by 0.7 every 40 epochs [S1]. The knowledge base does not provide enough information to answer the second part of your question about a method that self-projects the partial cloud into depth maps and what extra paired input it manages without.
- **base2** CANDIDATE_MISS [CITED,CANDIDATE_MISS]: PointAttN is trained for a total of 400 epochs, with the learning rate initialized to 1E-4 and decayed by 0.7 every 40 epochs [S1]. The knowledge base does not provide enough information to answer the second part of your question about a method that self-projects the partial cloud into depth maps and what extra paired input it manages without.
- **cand1** PASS [CITED,CITED]: PointAttN is trained for a total of 400 epochs, with the learning rate initialized to 1E-4 and decayed by 0.7 every 40 epochs [S1]. The method that self-projects the partial point cloud into depth maps (SVDFormer) makes full use of self-structures and does not require any additional paired information such as color images with precisely calibrated camera intrinsic parameters [S2].
- **cand2** PASS [CITED,CITED]: PointAttN is trained for a total of 400 epochs, with the learning rate initialized to 1E-4 and decayed by 0.7 every 40 epochs [S1]. The method that self-projects the partial point cloud into depth maps (SVDFormer) makes full use of self-structures and does not require any additional paired information such as color images with precisely calibrated camera intrinsic parameters [S2].

### E2E-M220 (spare)

Query: One completion network keeps regional features in a small set of seed points; how many patches does its encoder extract from the partial input? Another aligns predicted missing-region proxies with ones built from the true missing region; what kind of loss does it use for that alignment?

Keys: C1 (req=True): [['Np = 128', '128 patches']]; C2 (req=True): [['MSE', 'mean squared error']]

- **base1** RERANK_MISS [CITED,RERANK_MISS]: The knowledge base states that SeedFormer's seed generator takes Np = 128 patches extracted from the input point cloud and produces Ns = 256 seed points, with regional features stored in local seeds [S1][S2]. However, the context does not specify what kind of loss ProxyFormer uses for its Proxy Alignment between predicted missing-region proxies and proxies built from the true missing region; it only says Proxy Alignment is introduced into training to enhance prior knowledge and refine predicted missing proxies [S4].
- **base2** RERANK_MISS [CITED,RERANK_MISS]: The knowledge base answers the first part: SeedFormer's seed generator takes Np = 128 patches extracted from the input point cloud and produces Ns = 256 seed points, with regional features stored in local seeds [S1][S2]. However, for the second part, the context only states that ProxyFormer introduces "Proxy Alignment" in training, converting the true missing part into proxies to enhance prior knowledge and refine the predicted missing proxies [S4]. The knowledge base does not specify what kind of loss is used for this alignment, so there is not enough information to answer that part of the question.
- **cand1** PASS [CITED,CITED]: The first network (SeedFormer) extracts Np = 128 patches from the incomplete point cloud using its encoder [S4][S1]. The second network (ProxyFormer) uses an MSE loss between the predicted missing proxies (pre-MP) and the true missing proxies (true-MP) for proxy alignment [S5].
- **cand2** PASS [CITED,CITED]: The SeedFormer encoder extracts Np = 128 patches from the incomplete point cloud [S1][S4]. The proxy-based network (ProxyFormer) uses an MSE loss between the predicted missing proxies (pre-MP) and the true missing proxies (true-MP) for proxy alignment [S5].

### E2E-M229 (primary_B4)

Query: PF-Net names an earlier network as the first learning-based architecture focused on shape completion. How many models did that network hold out for validation, and how many for testing?

Keys: C1 (req=True): [['PCN', 'Point Completion Network']]; C2 (req=True): [['100 models'], ['150 models']]

- **base1** CANDIDATE_MISS [CITED,CANDIDATE_MISS]: The knowledge base does not provide enough information to answer this. The context identifies PCN as the first learning-based architecture focused on shape completion [S1], but the only dataset split details given are for Completion3D (28,974 training, 800 validation, 1,184 testing models) [S2], which is not stated to be PCN's split. No validation or test counts for PCN itself are provided.
- **base2** CANDIDATE_MISS [CITED,CANDIDATE_MISS]: The knowledge base does not provide enough information to answer this. The context identifies PCN as the first learning-based architecture focused on shape completion [S1], but the only dataset split details given are for Completion3D (28,974 training, 800 validation, 1,184 testing models) [S2], which is not stated to be PCN's own split. No validation or testing counts for PCN itself are provided.
- **cand1** PASS [CITED,CITED]: PF-Net identifies PCN as the first learning-based architecture focusing on shape completion [S1]. According to the PCN paper, 100 models were reserved for validation and 150 models for testing [S2].
- **cand2** PASS [CITED,CITED]: PF-Net identifies PCN as the first learning-based architecture focusing on shape completion [S1]. According to the PCN paper, 100 models were reserved for validation and 150 models for testing [S2].

### E2E-M236 (primary_B5)

Query: When AdaPoinTr joins the known and the predicted parts, what does it concatenate instead of raw point sets, and how many seeds does SeedFormer's seed generator produce?

Keys: C1 (req=True): [['point proxies']]; C2 (req=True): [['Ns = 256', '256 seeds', '256 seed points']]

- **base1** RERANK_MISS [RERANK_MISS,CITED]: The knowledge base does not provide enough information to answer the first part of your question: the only mention of AdaPoinTr is a reference title [S5], which does not describe how it joins known and predicted parts or what it concatenates instead of raw point sets. Regarding the second part, SeedFormer's seed generator produces Ns = 256 seed points (seed features F ∈ R^(Ns×Cs) with Ns = 256) [S1][S2].
- **base2** RERANK_MISS [RERANK_MISS,CITED]: The knowledge base does not provide enough information to answer the first part of the question: the supplied context only lists AdaPoinTr as a reference [S5] and does not describe how it joins known and predicted parts or what it concatenates instead of raw point sets. Regarding the second part, SeedFormer's seed generator produces Ns = 256 seed points (seed features F ∈ R^(Ns×Cs) with Ns = 256) [S1][S2].
- **cand1** PASS [CITED,CITED]: AdaPoinTr concatenates two sets of point proxies (corresponding queries in the embedding space R^Q) rather than concatenating two partial point clouds in R^3 space [S1][S7]. SeedFormer's seed generator produces Ns = 256 seed points (seed features F ∈ R^{Ns×Cs} with Ns = 256 and Cs = 128) [S3][S5].
- **cand2** PASS [CITED,CITED]: AdaPoinTr concatenates two sets of point proxies (corresponding queries in the embedding space R^Q) rather than concatenating two partial point clouds in R^3 space [S1][S7]. SeedFormer's seed generator produces Ns = 256 seed points (seed features F ∈ R^{Ns×Cs} with Ns = 256) [S3][S5].

