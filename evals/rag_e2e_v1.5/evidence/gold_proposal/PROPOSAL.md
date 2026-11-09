# Gold-audit proposal for a v1.4 mapping (DRAFT, NOT FROZEN, needs the user's approval)

Produced offline (no LLM) by `work/agent_design/candidate_miss/round2/partA_*.py` under `round2/PREREG.md` Part A. Nothing under `work/rag_e2e_v1.3/frozen/` was touched.

## What was done

- Scope: all 105 usable multi-doc groups (legacy 9 cases + new 43 cases).
- Candidates: every chunk of the group's gold paper whose text satisfies `claim_hits(answer_keys, text)` for a claim of the group (claims linked to groups by shared anchor strings, as in round 1), plus every chunk round 1 named (`per_group.json` alt_chunks_with_answer_key, chunk-id prefixes in `causes.json`), minus chunks already mapped: 407 rows in 63 groups. Every row was read and judged by hand (`round2/judg/j1.json`, `j2.json`).
- Acceptance rule: accepted only if the chunk's own prose (body or a descriptive caption; not figure-label residue, titles, references, tables, equation-only fragments) states the claim's `text` as a fact of the gold paper's own method, covering every element of the claim at the level the question asks; naming the component/term without its content, or presupposing it in ablation/results text, is rejected.
- Result: **105 accepted alternatives in 41 groups, 302 rejected.**
- Each accepted alternative has a verbatim anchor quote cut from the chunk text, a page locator (the chunk's page range), occurrence, left/right context from the normalized source, and was checked with the repo's own mapper (`stable_mapping.validate_source` + `map_source_to_chunks`, the code `scripts/e2e_dataset.py map` runs, on the uncleaned `load_document` text and the active Chroma chunks): **all 105 validate uniquely and map FULLY_MAPPED onto exactly the proposed chunk** (no extra bundles). The anchors with a context shorter than 20 characters (needed so that the mapper's unique-context proof lies inside the chunk): E2E-M202/G2 2a7b98ee14, E2E-M210/G1 823c5102d3, E2E-M216/G2 f30b476856, E2E-M217/G1 b9bb954958, E2E-M220/G2 b9bb954958, E2E-M221/G1 d5340000ab, E2E-M223/G1 8f2544df98, E2E-M236/G2 63e85bc347, E2E-M243/G1 f30b476856, E2E-M243/G2 b9bb954958.
- `mapping_draft.json` = v1.3 mapping + one single-chunk set per accepted alternative in `groups[].chunk_sets` (105 sets). `claim_anchors` unchanged. Same schema, `dataset_fingerprint` unchanged; sha256 9b3d0249f139ed51b3a6ea18b8c8fd3065da9451d114818d1fa59a91a3d68660 (base ee8c68c2a22528b16beb7180f9db7f003443305db1297dfe551345b450f35d48).

## Effect on production P0 (dense 50 -> rerank -> top 6 -> context)

P0 reaches 140/233 groups under v1.3 and **155/233 under the draft** (+15, none lost): ARR-039/G2, ARR-040/G1, ARR-040/G2, E2E-M201/G2, E2E-M202/G1, E2E-M202/G2, E2E-M203/G1, E2E-M206/G1, E2E-M206/G2, E2E-M210/G1, E2E-M216/G2, E2E-M220/G1, E2E-M221/G1, E2E-M227/G1, E2E-M241/G2. These are groups where an accepted alternative is already in the production context.

## Round-1 suspected gaps

| group | verdict | accepted chunk(s) (dense rank, in P0 context?) | note |
|---|---|---|---|
| E2E-M216/G1 | accepted | f95bdcbf68 (214, no) | the 4 round-1 chunks (ranks 2/3/6/17) and 3 more only name the 'auxiliary/denoising task'; only f95bdcbf68 says what it consists of (noised queries fed with the normal queries to the decoder) |
| E2E-M216/G2 | accepted | f30b476856 (13, yes); f584353605 (16, no); 68e085b77c (59, no) | claim = 'loss consists of completion loss and adversarial loss'; f30b476856 (round-1 pick) states both and the discriminator structure |
| E2E-M220/G1 | accepted | 63e85bc347 (41, yes); 197c316461 (129, no) | 63e85bc347 'seed generator takes Np = 128 patches extracted from the input' |
| E2E-M221/G1 | accepted | f015fb1591 (8, yes); d5340000ab (30, no); aeeaeffb53 (92, no) | Fig. 2 caption 'CMLP ... is our feature extractor' |
| E2E-M243/G1 | accepted | f30b476856 (17, no); 68e085b77c (25, no); f584353605 (66, no) | same PF-Net loss statements as M216/G2 |

## Accepted alternatives

Columns: chunk (first 10 hex), dense rank of the chunk for the full question, whether production P0 context already contains it, page, anchor (verbatim, as it would go into `required_evidence_groups[].alternatives[].evidence_anchor`), reason. Full alternatives (document_id, content_version, locator) are in `proposal.json`.

### ARR-037/G1 (legacy_named; claim C1; v1.3 gold already in P0 context: no)

| chunk | file | rank | in P0 ctx | page | anchor | reason |
|---|---|---|---|---|---|---|
| 02c2a02efa | pointr_geometry_aware_transformers.pdf | 157 | no | 3 | Q = {Q1, Q2, ..., QM} are the dynamic queries for the decoder | 3.1 states PoinTr's decoder takes dynamic queries (q = dynamic queries for the decoder) |

### ARR-037/G2 (legacy_named; claim C2; v1.3 gold already in P0 context: no)

| chunk | file | rank | in P0 ctx | page | anchor | reason |
|---|---|---|---|---|---|---|
| fc633bc453 | adapointr.pdf | 6 | no | 6 | We modify the design of the original Query Generator and generate a dynamic query bank B conditioned on the encoder outputs V and the input point proxies F | AdaPoinTr generates a dynamic query bank conditioned on encoder outputs and input proxies (how queries are adapted) |
| 98a7923e7b | adapointr.pdf | 11 | no | 6 | Proposed adaptive denoising queries, which are selected from a dynamic query bank | Fig. 6 caption: AdaPoinTr's adaptive denoising queries are selected from a dynamic query bank |

### ARR-038/G1 (legacy_named; claim C1; v1.3 gold already in P0 context: no)

| chunk | file | rank | in P0 ctx | page | anchor | reason |
|---|---|---|---|---|---|---|
| 73c30fc951 | pfnet_point_fractal_network.pdf | 18 | no | 3 | The overall architecture of our PF-Net is composed of three fundamental building blocks, named Multi-Resolution Encoder (MRE), Point Pyramid Decoder (PPD), and Discriminator Network | states PF-Net is built from MRE, Point Pyramid Decoder and discriminator (PPD named as its generator) |
| f584353605 | pfnet_point_fractal_network.pdf | 32 | no | 2 | we design a Point Pyramid Decoder (PPD) to generate the missing point cloud hierarchically | Intro: PPD generates the missing cloud hierarchically, primary/secondary/detailed points |
| 75c62aaa01 | pfnet_point_fractal_network.pdf | 119 | no | 4 | Detailed points Ydetail is the ﬁnal prediction of PPD | 3.3 Point Pyramid Decoder body describes the hierarchical primary/secondary/detailed generation |
| 9733ea9474 | pfnet_point_fractal_network.pdf | 268 | no | 4 | Details of Point Pyramid Decoder. PPD gives outputs of three joint layers in ascending scales | Fig. 5 caption: PPD gives outputs of three layers in ascending scales (hierarchical representation) |

### ARR-038/G2 (legacy_named; claim C2; v1.3 gold already in P0 context: no)

| chunk | file | rank | in P0 ctx | page | anchor | reason |
|---|---|---|---|---|---|---|
| b107a5bd4d | seedformer.pdf | 3 | no | 14 | SeedFormer introduce a new shape representation, namely Patch Seeds, in point cloud completion | Conclusion: SeedFormer introduces a new shape representation, Patch Seeds, regional features in local seeds |
| e9d5beb3f1 | seedformer.pdf | 6 | no | 2 | we introduce Patch Seeds as a latent representation in the point cloud completion architecture | Intro: introduces patch seeds as a latent representation characterising the complete shape |
| 2a7b98ee14 | seedformer.pdf | 8 | no | 2-3 | We introduce Patch Seeds as a new representation in the completion architecture to preserve regional information for recovering fine details | Contribution list: Patch Seeds as a new representation preserving regional information |
| 21131bd0bd | seedformer.pdf | 26 | no | 5-6 | Patch Seeds serve as a new shape representation in point cloud completion | 3.2: Patch seeds serve as a new shape representation (seed coordinates + features) |

### ARR-039/G1 (legacy_named; claim C1; v1.3 gold already in P0 context: no)

| chunk | file | rank | in P0 ctx | page | anchor | reason |
|---|---|---|---|---|---|---|
| 5b7750b582 | snowflakenet.pdf | 12 | no | 8 | By further introducing skip-transformer in Snowﬂake Point Deconvolution, SnowﬂakeNet learns to generate locally compact and structured point cloud with highly detailed geometries | Conclusion: SnowflakeNet introduces skip-transformer in SPD |
| 314aa21eae | snowflakenet.pdf | 35 | no | 2 | We introduce a novel skip-transformer to learn splitting patterns in SPD | contribution list: novel skip-transformer learns splitting patterns in SPD (SnowflakeNet's attention unit) |
| f5805b037f | snowflakenet.pdf | 48 | no | 2 | skip-transformer leverages attention mechanism to summarize the splitting patterns used in the previous SPD layer | Intro: skip-transformer leverages attention mechanism to summarise splitting patterns |
| d51039a86a | snowflakenet.pdf | 195 | no | 5 | The skip-transformer is introduced to learn and reﬁne the spatial context between parent points and their child points | 3.3 Skip-Transformer: structure with query/key/value attention vectors |
| a124eb08ea | snowflakenet.pdf | 476 | no | 5 | we propose a novel skip-transformer to serve as the cooperation unit between SPDs | 3.2: skip-transformer serves as the cooperation unit between SPDs (states the component and its role) |

### ARR-039/G2 (legacy_named; claim C2; v1.3 gold already in P0 context: no)

| chunk | file | rank | in P0 ctx | page | anchor | reason |
|---|---|---|---|---|---|---|
| c753377b9d | pointattn.pdf | 7 | yes | 3-4 | designed framework which fully exploits the advantages of self-attention and cross-attention mechanisms | states PointAttN fully exploits self-attention and cross-attention mechanisms |
| 57b27c557f | pointattn.pdf | 10 | no | 10 | we leverage the cross-attention and self-attention mechanisms to perceive the local details and global context of 3D point clouds | Conclusion: leverages cross-attention and self-attention mechanisms |
| ad16fe5389 | pointattn.pdf | 22 | no | 2-3 | By leveraging the cross-attention and self-attention mechanisms, PointAttN can precisely learn both short-range and long-range structural correlations among points | Intro: by leveraging cross-attention and self-attention, PointAttN learns correlations |
| d9db643d73 | pointattn.pdf | 30 | no | 1 | we leverage the cross-attention and self-attention mechanisms to design novel neural network for processing point cloud in a per-point manner to eliminate kNNs | Abstract: leverages cross-attention and self-attention mechanisms |
| 387598920e | pointattn.pdf | 54 | no | 3 | GDP and SFA are drawn on the core idea of Transformer [17], i.e., introducing the multi-head cross-attention and self-attention mechanisms | GDP and SFA introduce multi-head cross-attention and self-attention |

### ARR-040/G1 (legacy_named; claim C1; v1.3 gold already in P0 context: no)

| chunk | file | rank | in P0 ctx | page | anchor | reason |
|---|---|---|---|---|---|---|
| b514a8de50 | svdformer.pdf | 4 | yes | 2 | we design a Self-structure Dual-Generator (SDG) with a pair of parallel refinement units, called Structure Analysis and Similarity Alignment, respectively | Intro: designs a self-structure dual-generator (SDG) with structure analysis and similarity alignment |
| 4cabad13c3 | svdformer.pdf | 6 | no | 9 | we introduce a decoder called Self-structure Dual-generator that breaks down the shape refinement process into two sub-goals | Conclusion: decoder called self-structure dual-generator |
| 997b842875 | svdformer.pdf | 15 | no | 2 | We introduce a Self-structure Dual-Generator (SDG) for refining the coarse completion | contribution: self-structure dual-generator for refining the coarse completion |

### ARR-040/G2 (legacy_named; claim C2; v1.3 gold already in P0 context: no)

| chunk | file | rank | in P0 ctx | page | anchor | reason |
|---|---|---|---|---|---|---|
| 10d277b8e4 | hyper_pcn.pdf | 33 | yes | 8 | we propose Hyper-PCN, a novel point cloud completion framework that leverages hypergraph learning to model high-order correlations within incomplete point clouds | Conclusion: Hyper-PCN leverages hypergraph learning to model high-order correlations |
| 00f342f509 | hyper_pcn.pdf | 41 | yes | 2 | propose Hyper-PCN, a novel point cloud completion method based on hypergraph neural networks | Intro: Hyper-PCN, a completion method based on hypergraph neural networks |
| a4214b15b1 | hyper_pcn.pdf | 42 | yes | 2 | We first model high-order correlations in incomplete point clouds and propose Hyper-PCN, a novel hypergraph-based method for point cloud completion | contributions: hypergraph-based method modelling high-order correlations |
| b95ef5bb3e | hyper_pcn.pdf | 53 | no | 3 | HyperRS constructs distance based hypergraphs and performs progressive refinement to yield hyper features and a coarse shape | 3.1: HyperRS builds distance-based hypergraphs, A-HGNN collaborative hypergraphs |
| 517904ed97 | hyper_pcn.pdf | 215 | no | 5 | enables the construction of a hypergraph that more comprehensively captures global dependencies across the entire shape | 3.5 A-HGNN: constructs a hypergraph from key points and anchors to capture global dependencies |
| a9544137f0 | hyper_pcn.pdf | 404 | no | 4-5 | we stack L layers, each of which constructs a hypergraph based on feature space and performs hypergraph convolution | 3.4 HyperRS: stacks layers that each construct a hypergraph and perform hypergraph convolution |

### E2E-M201/G1 (primary_B1; claim C1; v1.3 gold already in P0 context: yes)

| chunk | file | rank | in P0 ctx | page | anchor | reason |
|---|---|---|---|---|---|---|
| 398ab8bed5 | pcn_point_completion_network.pdf | 16 | yes | 3 | The decoder uses v to ﬁrst generate a coarse output Ycoarse followed by a detailed output Ydetail | Fig. 2 caption: the decoder generates a coarse output Ycoarse followed by a detailed output Ydetail |
| d6ee2aaa20 | pcn_point_completion_network.pdf | 29 | no | 3-4 | In the ﬁrst stage, a coarse output Ycoarse of s points is generated | 4.2: two-stage generation, coarse output Ycoarse then detailed output Ydetail |

### E2E-M201/G2 (primary_B1; claim C2; v1.3 gold already in P0 context: no)

| chunk | file | rank | in P0 ctx | page | anchor | reason |
|---|---|---|---|---|---|---|
| c2a08cc7e0 | pfnet_point_fractal_network.pdf | 4 | no | 1 | Our PF-Net takes the incomplete point cloud as input and directly predicts the missing point cloud | Fig. 1 caption: PF-Net takes the incomplete cloud and directly predicts the missing point cloud |
| 73c30fc951 | pfnet_point_fractal_network.pdf | 5 | no | 3 | our PF-Net, which predicts the missing region of the point cloud from its incomplete known conﬁguration | 3: PF-Net predicts the missing region from its incomplete known configuration |
| f015fb1591 | pfnet_point_fractal_network.pdf | 8 | no | 3 | The input of PF-Net is incomplete point cloud and it predicts the missing point cloud | Fig. 2 caption: input is the incomplete cloud and it predicts the missing point cloud |
| 5f904261f3 | pfnet_point_fractal_network.pdf | 20 | no | 4 | aims to output M × 3 point cloud, which represents the shape of the missing region | decoder outputs an M x 3 cloud representing the shape of the missing region |
| f584353605 | pfnet_point_fractal_network.pdf | 22 | no | 2 | we design a Point Pyramid Decoder (PPD) to generate the missing point cloud hierarchically | PPD generates the missing point cloud hierarchically |
| 68e085b77c | pfnet_point_fractal_network.pdf | 30 | yes | 1 | PF-Net estimates the missing point cloud hierarchically | Abstract: unlike networks that generate the overall shape, PF-Net estimates the missing point cloud |
| 193ee68bc8 | pfnet_point_fractal_network.pdf | 47 | no | 5-6 | Since our method takes the partial shape as input and only output the missing region | states the method takes the partial shape as input and only outputs the missing region |
| f68368f773 | pfnet_point_fractal_network.pdf | 577 | no | 8 | We have presented a new approach to shape completion using a partial point cloud only to generate the missing point cloud | Conclusion: uses a partial point cloud only to generate the missing point cloud |

### E2E-M202/G1 (primary_B1; claim C1; v1.3 gold already in P0 context: no)

| chunk | file | rank | in P0 ctx | page | anchor | reason |
|---|---|---|---|---|---|---|
| cd6dc6f81e | snowflakenet.pdf | 13 | yes | 1-2 | each SPD layer produces child points by splitting their parent point with inheriting shape characteristics captured by the parent point | Intro: each SPD layer produces child points by splitting their parent point, inheriting the parent's shape characteristics |
| 12b6943efe | snowflakenet.pdf | 15 | no | 1 | the child points are progressively generated by splitting their parent points after each SPD | Abstract: child points are progressively generated by splitting their parent points after each SPD |
| 273051be82 | snowflakenet.pdf | 43 | no | 4 | The SPD aims to increase the number of points by splitting each parent point into multiple child points | 3.2 SPD: increases points by splitting each parent into child points via a point-wise splitting operation |
| a124eb08ea | snowflakenet.pdf | 148 | no | 5 | The child point features can be generated by adding the activated shape characteristics | 3.2 SPD: child point features generated by adding the activated parent shape characteristics; point-wise splitting for up-sampling |
| 979b70618b | snowflakenet.pdf | 488 | no | 4 | point-wise splitting operation aims to generate multiple child point features for each | states the point-wise splitting operation generates multiple child point features for each parent |

### E2E-M202/G2 (primary_B1; claim C2; v1.3 gold already in P0 context: no)

| chunk | file | rank | in P0 ctx | page | anchor | reason |
|---|---|---|---|---|---|---|
| 2a7b98ee14 | seedformer.pdf | 19 | yes | 2-3 | the generation of new points as a transformer-style self-attention weighted average of point features in the local field | Intro: generation of new points as a transformer-style self-attention weighted average of point features in the local field |

### E2E-M203/G1 (primary_B1; claim C1; v1.3 gold already in P0 context: no)

| chunk | file | rank | in P0 ctx | page | anchor | reason |
|---|---|---|---|---|---|---|
| 87d86f8018 | pointr_geometry_aware_transformers.pdf | 12 | yes | 2 | We use dynamic queries instead of ﬁxed queirs in the decoder, which are generated by a query generation module that summarizes the features produced by the encoder | Intro: dynamic queries generated by a query generation module that summarises the encoder features |

### E2E-M204/G2 (primary_B1; claim C2; v1.3 gold already in P0 context: yes)

| chunk | file | rank | in P0 ctx | page | anchor | reason |
|---|---|---|---|---|---|---|
| 5a75084066 | proxyformer.pdf | 3 | yes | 3 | ProxyFormer discards the transformer decoder to reduce the number of parameters | Related work: ProxyFormer discards the transformer decoder to reduce parameters |

### E2E-M206/G1 (primary_B1; claim C1; v1.3 gold already in P0 context: no)

| chunk | file | rank | in P0 ctx | page | anchor | reason |
|---|---|---|---|---|---|---|
| b514a8de50 | svdformer.pdf | 17 | yes | 2 | we design a Self-structure Dual-Generator (SDG) with a pair of parallel refinement units, called Structure Analysis and Similarity Alignment, respectively | Intro: SDG with a pair of parallel refinement units, Structure Analysis and Similarity Alignment |
| ed8c66841e | svdformer.pdf | 233 | no | 5 | The architecture of SDG. The upper path represents Structure Analysis and the lower path represents Similarity Alignment | Fig. 4 caption: SDG upper path is structure analysis, lower path is similarity alignment |
| 6f5cb389eb | svdformer.pdf | 240 | no | 5 | The Similarity Alignment unit exploits the potential similar local pattern in Pin for each point in Pl−1 | 3.2 SDG continuation: describes the similarity alignment unit and the structure analysis unit |

### E2E-M206/G2 (primary_B1; claim C2; v1.3 gold already in P0 context: no)

| chunk | file | rank | in P0 ctx | page | anchor | reason |
|---|---|---|---|---|---|---|
| e9d5beb3f1 | seedformer.pdf | 4 | yes | 2 | Based on the designed Patch Seeds and Upsample Transformer, the decoding phase of our method consists of two main steps | Intro: decoding with patch seeds and Upsample Transformer recovers fine details coarse-to-fine |
| 640ca42f20 | seedformer.pdf | 12 | no | 5 | Several upsample layers are used in the coarse-to-fine generation where an Upsample Transformer is applied for producing new points | Fig. 2 caption: coarse-to-fine generation uses an Upsample Transformer to produce new points |
| afd0bdd249 | seedformer.pdf | 14 | no | 1 | we devise an Upsample Transformer by extending the transformer structure into basic operations of point generators | Abstract: devises an Upsample Transformer as the basic operation of point generators |
| b107a5bd4d | seedformer.pdf | 22 | no | 14 | by extending the transformer structure into point generation, we propose a novel Upsample Transformer | Conclusion: proposes a novel upsample transformer for point generation |
| 21131bd0bd | seedformer.pdf | 24 | no | 5-6 | each point in the input point cloud is upsampled into rl points using the Upsample Transformer | each point is upsampled into r points using the upsample transformer |
| 19f24893dd | seedformer.pdf | 49 | no | 6 | shows the design of our Upsample Transformer | Fig. 3 caption: design of the Upsample Transformer and generation of new points |
| b1f21a4ed8 | seedformer.pdf | 79 | no | 6-7 | our Upsample Transformer is designed to incorporate closely-related local information into the generation by aggregating features from neighboring points | 3.3: our Upsample Transformer is designed to incorporate local information into the generation |

### E2E-M207/G2 (primary_B1; claim C2; v1.3 gold already in P0 context: no)

| chunk | file | rank | in P0 ctx | page | anchor | reason |
|---|---|---|---|---|---|---|
| 182a9b221f | pointattn.pdf | 100 | no | 4-5 | The encoder module as Feature Extractor is constructed upon both GDP and SFA blocks | 3.3: the encoder module as feature extractor is constructed upon both GDP and SFA blocks |

### E2E-M208/G1 (spare; claim C1; v1.3 gold already in P0 context: yes)

| chunk | file | rank | in P0 ctx | page | anchor | reason |
|---|---|---|---|---|---|---|
| ec20e8d343 | adapointr.pdf | 45 | no | 13-14 | These two datasets are the popular NYU Depth V2 [47] (denoted as NYUV2 in the following part) and NYUCAD [14] | evaluates on the real-world NYU Depth V2 and NYUCAD |

### E2E-M210/G1 (primary_B1; claim C1; v1.3 gold already in P0 context: no)

| chunk | file | rank | in P0 ctx | page | anchor | reason |
|---|---|---|---|---|---|---|
| ccddf71d9e | pointr_geometry_aware_transformers.pdf | 10 | yes | 3 | Then, we use a lightweight DGCNN [44] to extract the local features around the center points | Fig. caption: a lightweight DGCNN extracts local features around the center points |
| 823c5102d3 | pointr_geometry_aware_transformers.pdf | 77 | no | 9 | We employ a lightweight DGCNN [44] model to extract the point proxy features | Appendix: a lightweight DGCNN with hierarchical downsampling extracts the point proxy features |

### E2E-M216/G1 (primary_B2; claim C1; v1.3 gold already in P0 context: no)

| chunk | file | rank | in P0 ctx | page | anchor | reason |
|---|---|---|---|---|---|---|
| f95bdcbf68 | adapointr.pdf | 214 | no | 6-7 | together with other normal queries Q, which can be expressed as | describes the task: noised queries are fed to the decoder together with the normal queries ([H, H^] = MD([Q, Q^], V)) and rebuilt as local shapes |

### E2E-M216/G2 (primary_B2; claim C2; v1.3 gold already in P0 context: no)

| chunk | file | rank | in P0 ctx | page | anchor | reason |
|---|---|---|---|---|---|---|
| f30b476856 | pfnet_point_fractal_network.pdf | 13 | yes | 5 | The adversarial loss in this work is inspired by Generative Adversarial Network (GAN) [35] | 3.4 continuation: completion loss L_com and the adversarial loss with the discriminator's structure |
| f584353605 | pfnet_point_fractal_network.pdf | 16 | no | 2 | jointly train our PF-Net to minimize both multistage completion loss and adversarial loss | Intro: PF-Net is trained to minimise both multi-stage completion loss and adversarial loss |
| 68e085b77c | pfnet_point_fractal_network.pdf | 59 | no | 1 | we add up multi-stage completion loss and adversarial loss to generate more realistic missing region(s) | Abstract: adds multi-stage completion loss and adversarial loss |

### E2E-M217/G1 (primary_B2; claim C1; v1.3 gold already in P0 context: yes)

| chunk | file | rank | in P0 ctx | page | anchor | reason |
|---|---|---|---|---|---|---|
| b9bb954958 | proxyformer.pdf | 57 | no | 5 | where lp means the alignment loss that we will apply to our training loss | 3.3: lp is the alignment loss (MSE) between pre-MP and true-MP |

### E2E-M218/G2 (spare; claim C2; v1.3 gold already in P0 context: no)

| chunk | file | rank | in P0 ctx | page | anchor | reason |
|---|---|---|---|---|---|---|
| 4ee5db3193 | svdformer.pdf | 66 | no | 3-4 | our method makes full use of self-structures and does not require any additional paired information such as color images | 3: the method does not require additional paired information such as color images |

### E2E-M219/G1 (primary_B3; claim C1; v1.3 gold already in P0 context: no)

| chunk | file | rank | in P0 ctx | page | anchor | reason |
|---|---|---|---|---|---|---|
| ddb4b7a3f5 | pcn_point_completion_network.pdf | 428 | no | 5-6 | For the loss on coarse output, PCN-CD uses CD and PCN-EMD uses EMD | 5.2: for the loss on the coarse output PCN-CD uses CD and PCN-EMD uses EMD; both use CD for the detailed output |

### E2E-M220/G1 (spare; claim C1; v1.3 gold already in P0 context: no)

| chunk | file | rank | in P0 ctx | page | anchor | reason |
|---|---|---|---|---|---|---|
| 63e85bc347 | seedformer.pdf | 41 | yes | 13-14 | The seed generator takes Np = 128 patches (Fig. 6(b)) extracted from the input point cloud | seed generator takes Np = 128 patches extracted from the input point cloud |
| 197c316461 | seedformer.pdf | 129 | no | 15-18 | The outputs are Np = 128 patch centers Pp as well as the corresponding patch features Fp | appendix architecture: outputs are Np = 128 patch centers with patch features |

### E2E-M220/G2 (spare; claim C2; v1.3 gold already in P0 context: no)

| chunk | file | rank | in P0 ctx | page | anchor | reason |
|---|---|---|---|---|---|---|
| b9bb954958 | proxyformer.pdf | 95 | no | 5 | where lp means the alignment loss that we will apply to our training loss (Sec. 3.4) and MSE means the mean squared error | 3.3: alignment loss lp with MSE the mean squared error |

### E2E-M221/G1 (primary_B3; claim C1; v1.3 gold already in P0 context: no)

| chunk | file | rank | in P0 ctx | page | anchor | reason |
|---|---|---|---|---|---|---|
| f015fb1591 | pfnet_point_fractal_network.pdf | 8 | yes | 3 | CMLP (Section 3.2) is our feature extractor | Fig. 2 caption: CMLP is our feature extractor |
| d5340000ab | pfnet_point_fractal_network.pdf | 30 | no | 4 | Comparison of two different feature extractors of 3D point encoders | Fig. 4 caption: compares feature extractors; our CMLP |
| aeeaeffb53 | pfnet_point_fractal_network.pdf | 92 | no | 2 | by utilizing a new feature extractor Combined Multi-Layer Perception (CMLP) | Intro: MRE uses a new feature extractor, Combined Multi-Layer Perception (CMLP) |

### E2E-M223/G1 (primary_B3; claim C1; v1.3 gold already in P0 context: no)

| chunk | file | rank | in P0 ctx | page | anchor | reason |
|---|---|---|---|---|---|---|
| 8f2544df98 | hyper_pcn.pdf | 388 | no | 8 | We therefore adopt L = 6 as the default | ablation concludes: we adopt L = 6 (HyperRS depth) as the default |

### E2E-M224/G1 (primary_B3; claim C1; v1.3 gold already in P0 context: no)

| chunk | file | rank | in P0 ctx | page | anchor | reason |
|---|---|---|---|---|---|---|
| 4ee5db3193 | svdformer.pdf | 72 | no | 3-4 | a refiner equipped with two SDG modules | 3: refiner equipped with two SDG modules; two SDGs refine and upsample |
| a629e4fd0a | svdformer.pdf | 271 | no | 3 | The coarse completion is then upsampled and refined with two SDGs | Fig. 2 caption: refined with two SDGs |

### E2E-M225/G2 (spare; claim C2; v1.3 gold already in P0 context: no)

| chunk | file | rank | in P0 ctx | page | anchor | reason |
|---|---|---|---|---|---|---|
| 14a5e725cc | pcn_point_completion_network.pdf | 254 | no | 14-17 | we use t-SNE [27] to embed the 1024-dimensional global features of the ShapeNet test instances into a 2D space | appendix: the 1024-dimensional global features (states the global feature dimension) |

### E2E-M227/G1 (primary_B4; claim C1; v1.3 gold already in P0 context: no)

| chunk | file | rank | in P0 ctx | page | anchor | reason |
|---|---|---|---|---|---|---|
| 7069cff3b5 | proxyformer.pdf | 1 | yes | 5 | we also train and test on two other datasets, ShapeNet- 55 and ShapeNet-34 proposed by PoinTr [41] | 4: ShapeNet-55 and ShapeNet-34 proposed by PoinTr |

### E2E-M234/G1 (primary_B5; claim C1; v1.3 gold already in P0 context: yes)

| chunk | file | rank | in P0 ctx | page | anchor | reason |
|---|---|---|---|---|---|---|
| 20fdf699e3 | adapointr.pdf | 4 | yes | 3 | We propose ShapeNet-55/34 and Projected-ShapeNet-55/34, which generate incomplete point clouds by cropping and noised back-projecting, respectively | Fig. 3 caption: Projected-ShapeNet-55/34 generate incomplete clouds by noised back-projecting |
| cda3fbd56f | adapointr.pdf | 6 | yes | 2 | we propose noised back-projecting by adding a scaled random noise to the depth map before generating incomplete point clouds | noised back-projecting adds scaled noise to the depth map; yields Projected-ShapeNet-55/34 |
| c61caf9b28 | adapointr.pdf | 20 | yes | 3 | we present Projected-ShapeNet-55/34 dataset by generating incomplete samples with proposed noised back-projecting | Intro: Projected-ShapeNet-55/34 generated with the proposed noised back-projecting |

### E2E-M236/G2 (primary_B5; claim C2; v1.3 gold already in P0 context: yes)

| chunk | file | rank | in P0 ctx | page | anchor | reason |
|---|---|---|---|---|---|---|
| 63e85bc347 | seedformer.pdf | 7 | yes | 13-14 | Ns = 256 seed points (Fig. 6(c)) | seed generator produces a new set of Ns = 256 seed points |

### E2E-M238/G1 (primary_B5; claim C1; v1.3 gold already in P0 context: yes)

| chunk | file | rank | in P0 ctx | page | anchor | reason |
|---|---|---|---|---|---|---|
| 6224e921d9 | pcn_point_completion_network.pdf | 53 | no | 8 | Failure modes: thin structures (top) and disconnected parts (bottom) | Fig. 8 caption: failure modes, thin structures and disconnected parts |

### E2E-M238/G2 (primary_B5; claim C2; v1.3 gold already in P0 context: yes)

| chunk | file | rank | in P0 ctx | page | anchor | reason |
|---|---|---|---|---|---|---|
| 6224e921d9 | pcn_point_completion_network.pdf | 53 | no | 8 | Failure modes: thin structures (top) and disconnected parts (bottom) | Fig. 8 caption: failure modes, thin structures and disconnected parts |

### E2E-M241/G1 (partial_B6; claim C1; v1.3 gold already in P0 context: yes)

| chunk | file | rank | in P0 ctx | page | anchor | reason |
|---|---|---|---|---|---|---|
| b95ef5bb3e | hyper_pcn.pdf | 35 | yes | 3 | HyperRS constructs distance based hypergraphs and performs progressive refinement to yield hyper features and a coarse shape | 3.1: HyperRS builds distance-based hypergraphs, A-HGNN collaborative hypergraphs |
| 00f342f509 | hyper_pcn.pdf | 38 | yes | 2 | propose Hyper-PCN, a novel point cloud completion method based on hypergraph neural networks | Intro: Hyper-PCN, a completion method based on hypergraph neural networks |
| a4214b15b1 | hyper_pcn.pdf | 59 | no | 2 | We first model high-order correlations in incomplete point clouds and propose Hyper-PCN, a novel hypergraph-based method for point cloud completion | contributions: hypergraph-based method |
| 10d277b8e4 | hyper_pcn.pdf | 68 | no | 8 | we propose Hyper-PCN, a novel point cloud completion framework that leverages hypergraph learning to model high-order correlations within incomplete point clouds | Conclusion: leverages hypergraph learning |
| 517904ed97 | hyper_pcn.pdf | 113 | no | 5 | enables the construction of a hypergraph that more comprehensively captures global dependencies across the entire shape | 3.5 A-HGNN builds a hypergraph from key points and anchors |
| a9544137f0 | hyper_pcn.pdf | 253 | no | 4-5 | we stack L layers, each of which constructs a hypergraph based on feature space and performs hypergraph convolution | 3.4 HyperRS: each layer constructs a hypergraph and performs hypergraph convolution |

### E2E-M241/G2 (partial_B6; claim C2; v1.3 gold already in P0 context: no)

| chunk | file | rank | in P0 ctx | page | anchor | reason |
|---|---|---|---|---|---|---|
| b514a8de50 | svdformer.pdf | 1 | yes | 2 | we design a Self-View Fusion Network (SVFNet) that learns an effective descriptor | Intro: designs SVFNet learning a global descriptor from point cloud data and depth maps |
| 4ee5db3193 | svdformer.pdf | 3 | no | 3-4 | SVFNet first leverages multiple self-projected depth maps to produce a globally completed shape | 3: SVFNet leverages multiple self-projected depth maps |
| 997b842875 | svdformer.pdf | 5 | no | 2 | We propose the novel Self-view Fusion Network (SVFNet) equipped with a feature fusion module to enhance the multi-view and cross-modal feature | contribution: SVFNet with a feature fusion module for multi-view cross-modal features |
| 9db0e729a5 | svdformer.pdf | 18 | no | 4 | The SVFNet aims to observe the partial input from different viewpoints | 3.1 SVFNet observes the partial input from different viewpoints |

### E2E-M242/G2 (partial_B6; claim C2; v1.3 gold already in P0 context: yes)

| chunk | file | rank | in P0 ctx | page | anchor | reason |
|---|---|---|---|---|---|---|
| 57b27c557f | pointattn.pdf | 2 | no | 10 | the proposed framework eliminates the explicit local region partition like kNNs | Conclusion: the framework eliminates explicit local region partition like kNNs |
| 4ffa08bdc2 | pointattn.pdf | 6 | no | 2 | By eliminating the concept of explicit local feature region partition like kNNs | Intro: investigates eliminating kNNs; GDP eliminates explicit local region partition |
| d9db643d73 | pointattn.pdf | 10 | no | 1 | to design novel neural network for processing point cloud in a per-point manner to eliminate kNNs | Abstract: processes points per-point to eliminate kNNs |

### E2E-M243/G1 (partial_B6; claim C1; v1.3 gold already in P0 context: no)

| chunk | file | rank | in P0 ctx | page | anchor | reason |
|---|---|---|---|---|---|---|
| f30b476856 | pfnet_point_fractal_network.pdf | 17 | no | 5 | The adversarial loss in this work is inspired by Generative Adversarial Network (GAN) [35] | 3.4 continuation: completion loss plus adversarial loss with the discriminator |
| 68e085b77c | pfnet_point_fractal_network.pdf | 25 | no | 1 | we add up multi-stage completion loss and adversarial loss to generate more realistic missing region(s) | Abstract: adds multi-stage completion loss and adversarial loss |
| f584353605 | pfnet_point_fractal_network.pdf | 66 | no | 2 | jointly train our PF-Net to minimize both multistage completion loss and adversarial loss | Intro: minimise both multi-stage completion loss and adversarial loss |

### E2E-M243/G2 (partial_B6; claim C2; v1.3 gold already in P0 context: no)

| chunk | file | rank | in P0 ctx | page | anchor | reason |
|---|---|---|---|---|---|---|
| b9bb954958 | proxyformer.pdf | 59 | no | 5 | where lp means the alignment loss that we will apply to our training loss | 3.3: lp is the alignment loss between pre-MP and true-MP |

### E2E-M244/G1 (partial_B6; claim C1; v1.3 gold already in P0 context: no)

| chunk | file | rank | in P0 ctx | page | anchor | reason |
|---|---|---|---|---|---|---|
| 4c8d0cfdcf | adapointr.pdf | 211 | no | 7 | As for the auxiliary denoising task, giving the noised queries | 3.7: noised queries with noised centers c^ = n + c_gt (noise added to ground-truth centers) |
| f95bdcbf68 | adapointr.pdf | 269 | no | 6-7 | To avoid knowledge leakage from those noised queries | noised centers c^gt = n + cgt generated together with the queries; attention mask for noised queries |

## Rejected candidates

Per group: chunk (dense rank) - reason. Full rows in `proposal.json`.

- **ARR-037/G1**: 3117d91bb3 (63) - figure-label residue (architecture diagram words), no statement
- **ARR-037/G2**: 7533746d18 (180) - figure-label residue plus a section heading, no statement
- **ARR-038/G1**: 4d75a1295c (62) - ablation/experiment text that only names PPD as the evaluated component; caa680ff9c (175) - loss section; PPD only mentioned as producing three resolutions for the loss terms; f30b476856 (410) - adversarial-loss text; PPD only named in a definition
- **ARR-038/G2**: 044446a647 (5) - related-work contrast; mentions Patch Seeds being propagated but does not present them as the representation; d15eb20be5 (16) - ablation section (Patch Seeds vs global feature); presupposes the representation; 63e85bc347 (27) - visualisation/insight section about seed generation, not the representation statement; 640ca42f20 (45) - Fig. 2 caption describes architecture flow (seed generator obtains patch seeds); does not state the representation; fce6b19570 (48) - figure-label residue; 0ce8117068 (109) - ablation table text; e443ba9d60 (145) - equation fragment; d09a24f1d0 (207) - figure caption of a visualisation; 9f246cedb7 (557) - table caption
- **ARR-039/G1**: e40cf38cc6 (11) - experiment discussion that only credits skip-transformer; 0cae6ab58f (14) - title/author block; a07b8d91e8 (20) - ablation text; 15704770ad (29) - ablation table caption; ffc2115100 (31) - ablation variants; 6de38201e5 (257) - equation fragment; 503564decd (290) - figure legend residue; d675bd7138 (307) - related-work sentence; names skip-transformer without describing the mechanism; 6ca3b4cb25 (390) - table caption; 8add4bfee8 (506) - equation tail; affc5de9e6 (704) - figure caption title only
- **ARR-040/G1**: 216b6bdee0 (7) - title/author block
- **ARR-040/G2**: 167abee5a2 (45) - ablation table text; 59e8906f94 (59) - references; 5636f753af (67) - supplementary visualisation discussion; 8e2e453c4d (77) - title fragment; faa5b96abc (78) - title fragment; 7aaf2f9d41 (107) - references; 6c147b9cb2 (123) - references; e1d8845dbc (221) - feature embedding; hypergraph only named as next step; 6180653896 (226) - ablation discussion of generic hypergraph learning; e8da518585 (239) - references; eb70d41307 (296) - generic related-work definition plus figure-label residue; 572f3d272c (300) - one-line method tail; restatement only; c2fa8d2fc8 (302) - generic HGNN background; 7e99c933c4 (307) - references; ffcc6a08c5 (384) - related work on other tasks; 7d0b8752ac (387) - visualisation section; 993f0ed448 (517) - ablation table; 4d4de9815a (549) - equation fragment (generic hypergraph convolution); 6d19303faa (623) - equation fragment
- **E2E-M113/G2**: cc178cbf80 (574) - results table row, not the AnchorFormer citation statement
- **E2E-M114/G1**: 946ba7e453 (8) - ScanNet reference entry; the claim (reconstructions miss regions) is not stated
- **E2E-M114/G2**: 6b979af259 (217) - appendix sentence listing PFNet as a compared method; no 'fails due to diversity'; e8e48ef263 (488) - table; 82016ae793 (581) - table; a3b8d45bf6 (587) - table; 8cb1adb194 (607) - table; 87c4a742b8 (703) - table; cefca0f2d2 (778) - figure labels
- **E2E-M115/G2**: 5b7750b582 (61) - PCN dataset mention; 99c6c24784 (134) - PCN dataset mention; 57accec754 (200) - reference entry; dd64f321ac (294) - figure caption, PCN dataset; e40cf38cc6 (316) - PCN dataset section; 359ff12526 (363) - table caption; a07b8d91e8 (366) - PCN-baseline ablation, not the two-stage statement; 04503f3833 (428) - PCN dataset mention; 85860061ca (525) - table; 58a4589d12 (539) - benchmark mention; 7a578fc9e4 (540) - table; 25b4e6686e (564) - table
- **E2E-M116/G1**: 6c147b9cb2 (73) - reference entry of SnowflakeNet TPAMI; not the 'subsequent works' statement
- **E2E-M116/G2**: fc1d99ad29 (66) - ablation baseline mention; d85f3b7e45 (70) - ablation comparison; 986b624bc1 (121) - table caption; cee3c3bb36 (142) - results comparison; 02a7fb25a4 (388) - table; e3aacdf74a (414) - table; d815d9d246 (684) - table
- **E2E-M118/G1**: 25b4e6686e (72) - table row 'pcn-baseline' with numbers; no definition
- **E2E-M118/G2**: 23b7e24835 (2) - table caption, not the subsection heading/statement; 193ee68bc8 (10) - continuation of 4.2 without the heading text or the baseline list; a9540366b0 (21) - figure caption; caa680ff9c (22) - loss section; 68e085b77c (44) - abstract; d85056ed60 (51) - table caption; 6b870c9b42 (69) - references; 7745ddefae (76) - data section; dd4406ad38 (84) - caption; 314587afe9 (128) - caption; f68368f773 (151) - conclusion; f0534c19ad (179) - title block; aeeaeffb53 (194) - intro; a165c991d3 (203) - related work; 5f904261f3 (204) - decoder; 73c30fc951 (210) - method intro; 9a0ca9258f (217) - intro; 4d75a1295c (250) - ablation; c2a08cc7e0 (252) - caption; f015fb1591 (272) - caption; f584353605 (273) - intro; ebd820fb9d (281) - references; 75c62aaa01 (282) - PPD; 97959401ab (343) - caption; 5d51676748 (349) - caption; 8fd32d59de (351) - references; 57c6ed3a67 (392) - sampling; c2421a30e4 (403) - table; 0b2f2f5229 (432) - encoder; 72e745149a (516) - references
- **E2E-M201/G1**: ddb4b7a3f5 (156) - experiment settings; presupposes the outputs; d3562bb320 (428) - loss terms; presupposes the outputs
- **E2E-M201/G2**: 6ca19dd4e3 (7) - about Context Encoder (images); 4d75a1295c (10) - about ablation baselines; 273ca6d500 (14) - discriminator role; a9540366b0 (193) - robustness figure caption; 314587afe9 (194) - results caption; c2421a30e4 (204) - table; caa680ff9c (205) - loss section; f30b476856 (214) - adversarial-loss definition; 75c62aaa01 (323) - PPD result sentence; d85056ed60 (546) - table caption
- **E2E-M202/G1**: f5805b037f (14) - about skip-transformer learning splitting patterns, not how new points are produced; a07b8d91e8 (17) - ablation; 5cc65376aa (100) - visualisation caption; c7a47fc58e (107) - visualisation caption; d675bd7138 (114) - related-work comparison; d057efd9a6 (115) - seed generator (not the decoding SPD stage); 058c1cadfa (157) - about other methods; d51039a86a (189) - skip-transformer internals; 503564decd (253) - figure legend residue; a40441e179 (292) - training loss; df1ac22ad5 (441) - figure caption on splitting internals (logits), not the generation statement
- **E2E-M203/G1**: f22212bbb2 (5) - ablation table caption; 3117d91bb3 (18) - figure-label residue; 4b3f9cb849 (57) - ablation; 986d7801d3 (79) - reuses query-generator coordinates; not the query source statement; 54c8a35f55 (252) - table caption
- **E2E-M203/G2**: d08869815a (10) - figure-label residue; abb6d43a6c (66) - defines MP as predicted seed feature + random position encoding but not that MP is the query source
- **E2E-M206/G1**: c7abb05768 (9) - ablation variants; 4e4a5041bb (10) - ablation; d528e7d2db (56) - visual ablation discussion
- **E2E-M206/G2**: 197c316461 (13) - supplementary implementation detail; d15eb20be5 (27) - ablation; 8dcdcc3d92 (32) - supplementary cost discussion; 0c8081f27e (36) - use inside the seed generator; 63e85bc347 (39) - seed visualisation; fce6b19570 (42) - seed generator use plus residue; 0ce8117068 (47) - ablation; 261ba6c5e1 (251) - supplementary alternatives; 4291649945 (537) - table caption
- **E2E-M207/G2**: 387598920e (1) - says the three modules are designed upon GDP and SFA; does not state the encoder's blocks; 4ffa08bdc2 (2) - describes GDP/SFA in general; only says SFA can also be used in encoding, not that the encoder stacks GDP and SFA; ad16fe5389 (6) - contribution list, not encoder-specific; d9db643d73 (19) - abstract, not encoder-specific; dbfb8986c6 (27) - results discussion; fc1d99ad29 (59) - ablation variants; 00ac5367ca (90) - decoder; d85f3b7e45 (102) - ablation results; bca8c0c39b (519) - figure caption title; d815d9d246 (620) - table
- **E2E-M208/G2**: 1a85883fa0 (334) - reference entry; 036947c69c (566) - table
- **E2E-M209/G1**: caa680ff9c (104) - loss terms presuppose the primary/secondary/detailed outputs; no statement of the hierarchy
- **E2E-M210/G1**: f22212bbb2 (27) - ablation table caption; 364d44253b (31) - implementation k values; 6fe4097a42 (83) - geometry-aware block, not the feature extractor; 4b3f9cb849 (213) - ablation; 54c8a35f55 (524) - table caption; 5d56a937e8 (541) - table
- **E2E-M211/G1**: f73b8a402e (76) - about pair-wise models generally, not transformer query-key interactions; 6180653896 (131) - about graph convolution; 167abee5a2 (151) - ablation caption
- **E2E-M211/G2**: 364d44253b (59) - only a kNN k value for the block; e0d6499e8d (91) - figure-label residue ('kNN query')
- **E2E-M212/G1**: 5cc65376aa (90) - caption labels P1 as 512 points in a visualisation; does not state the merged seed count N0
- **E2E-M213/G1**: a9540366b0 (10) - robustness test ratios 25/50/75%, not the 25% train/test setting; c2421a30e4 (15) - table 5 ratios; 57c6ed3a67 (77) - 6.25% of points in sampling; dd4406ad38 (187) - figure caption
- **E2E-M216/G1**: 6e2eed40f1 (2) - names the auxiliary denoising task as an ablation step; no content; c3a8526ac3 (3) - says a denoising task stabilises optimisation; no content; fc633bc453 (6) - 'we devise an advanced denoising task'; no content; 9040cf6a4f (17) - abstract names the task; no content; c61caf9b28 (21) - contribution list names the task; no content; f540bf7f4f (188) - ablation table text; 4c8d0cfdcf (463) - loss of the denoising task; does not state that noised queries are fed with the adaptive queries to the decoder
- **E2E-M216/G2**: 6ca19dd4e3 (14) - about Context Encoder; 380981ce88 (509) - equation fragment (loss weights)
- **E2E-M217/G2**: a07b8d91e8 (47) - ablation removes the partial matching loss; purpose (preserve input structure) not stated
- **E2E-M218/G2**: 15e6be0f4f (18) - about other methods using colour images; 5f3ef3df15 (34) - about cross-modal methods; f89f24fcd9 (128) - about MVCNN
- **E2E-M219/G1**: d3562bb320 (44) - introduces CD and EMD as candidate losses but does not state which term uses which; b9d6b552ed (96) - figure labels; 1b002ed183 (381) - evaluation metrics; d30a14313a (401) - figure caption; e8f44819c9 (429) - table; d141b4f19f (438) - results discussion; 029c7f27ed (440) - architecture appendix; f3d8e28b90 (470) - robustness; 24c5083353 (491) - figure caption; a222890b63 (501) - table; bf9b57941c (600) - figure caption; bd25e5e95f (660) - appendix ablation; e77a3072b6 (695) - appendix ablation
- **E2E-M220/G2**: 9fbd2b391b (729) - equation-only chunk (LaTeX)
- **E2E-M221/G1**: f30b476856 (79) - discriminator similar to CMLP; 4d75a1295c (308) - ablation; 0c5ab9a047 (362) - table caption; b1952719de (375) - figure caption; 9d56dbeac3 (421) - classification analysis; ecb4b66873 (733) - table
- **E2E-M227/G1**: 151cd89243 (7) - results vs PoinTr; e09ffd196d (14) - table caption; 93eec1b507 (16) - figure caption; d08869815a (83) - related work, not the dataset credit; 16fe514bfd (119) - complexity; 7b558a6444 (128) - reference; eb9c7d0fff (164) - table; 79c138650b (183) - intro critique; 4e91641898 (196) - table; 204e2092a2 (226) - table; 0bea2bb382 (238) - results; 841a8fafe3 (320) - method; 259e4496ef (336) - table; 530c4046bb (339) - table; 34b7596941 (361) - caption; 0ca3662dff (417) - table
- **E2E-M228/G1**: 37b32e8271 (35) - residue; 08493794f2 (68) - caption; 4e4a5041bb (75) - ablation; 787440567d (94) - caption; 1159cb9403 (241) - reference; 77d7b92af0 (335) - labels; 6b80ace508 (434) - table; 772f98cc04 (523) - table; a7861178b4 (536) - table; 900470d16f (537) - table; 6d57f549df (586) - table
- **E2E-M229/G1**: 6b870c9b42 (42) - reference; 314587afe9 (145) - caption; 9a0ca9258f (161) - intro mentions PCN but not 'first learning-based architecture focusing on shape completion'; 36271aa4d4 (180) - baseline list; d0eddeab8a (276) - table
- **E2E-M231/G1**: c318f95253 (64) - reference
- **E2E-M232/G1**: c7abb05768 (16) - ablation; a693072368 (178) - table; 3c5a054edb (192) - reference; 772f98cc04 (481) - table
- **E2E-M233/G1**: d08869815a (3) - set-to-set statement, not the upsample-transformer claim; 16fe514bfd (16) - complexity; 151cd89243 (35) - results; 0ca3662dff (46) - table; eb9c7d0fff (72) - table; 4e91641898 (88) - table; 204e2092a2 (107) - table; 259e4496ef (110) - table; 97ee36b954 (114) - reference; 530c4046bb (179) - table
- **E2E-M234/G1**: c3a8526ac3 (49) - figure-label residue
- **E2E-M236/G1**: 9040cf6a4f (18) - PoinTr point proxies generally; no concatenation statement; 1ad78d02a0 (34) - ablation; 8e1a78d29f (60) - PoinTr set-to-set formulation; 8dfbde8a90 (61) - appendix caption; 5abde1ff8b (86) - PoinTr pipeline caption; 20e16b95a1 (87) - intro + residue; ace2e86104 (121) - implementation numbers; 99adcec413 (135) - appendix; 16b778a66e (143) - appendix; 5a6fdbaf20 (168) - scene completion; 05af306b36 (174) - dynamic queries; f95bdcbf68 (179) - denoising queries; b36ec6e486 (198) - point proxy definition; 910c2ef1d6 (324) - decoder; 67722fbb35 (349) - caption; f5f8d87482 (366) - appendix ablation
- **E2E-M236/G2**: d15eb20be5 (16) - ablation says 256 seeds are suitable; does not state the generator produces 256
- **E2E-M241/G1**: eb70d41307 (22) - generic related-work definition plus figure-label residue; 5636f753af (26) - supplementary visualisation; 167abee5a2 (67) - ablation caption; c2fa8d2fc8 (80) - generic HGNN background; e1d8845dbc (84) - feature embedding; 6180653896 (123) - ablation discussion; 7aaf2f9d41 (132) - references; 7e99c933c4 (187) - references; 59e8906f94 (192) - references; 6c147b9cb2 (193) - references; 8e2e453c4d (205) - title fragment; faa5b96abc (206) - title fragment; e8da518585 (210) - references; ffcc6a08c5 (239) - related work on other tasks; 7d0b8752ac (240) - visualisation; 572f3d272c (296) - restatement tail; 993f0ed448 (325) - ablation; 4d4de9815a (395) - equation fragment; 6d19303faa (447) - equation fragment
- **E2E-M241/G2**: a629e4fd0a (6) - caption says cross-modal input only; no multi-view depth images; 4e4a5041bb (9) - ablation; 06de2c87f3 (34) - figure-label residue; f36efa93a9 (52) - ablation; de36c034b8 (226) - table caption
- **E2E-M242/G2**: 4fdbd54ee6 (3) - results caption
- **E2E-M243/G1**: 6ca19dd4e3 (81) - about Context Encoder; 380981ce88 (800) - equation fragment
- **E2E-M244/G1**: 7533746d18 (57) - figure-label residue

