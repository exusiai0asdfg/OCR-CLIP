# Experimental Design Notes / 实验设计记录

This document records the choices made during the RHOS Stage 0 assessment. See `report.md` for detailed methods and results.

本文档记录了在完成考核做出了的一些选择,理由以及一些没有实践的方案,本来是自用草稿,后来经过整理后成型

## 1. Dataset Organization / 数据集组织

Use an object-level Dataset rather than an image-level Dataset.

- Each image may contain multiple objects with independent category, attribute and affordance annotations.
- Fixed-shape samples can be directly collated by PyTorch DataLoader.
- Use audited crop coordinates and multi-hot labels for attribute and affordance prediction.

采用以物体实例为单位的 Dataset,理由如上,
因此，batch_size = 4 表示每批处理四个物体实例。

## 2. CLIP Backbone / 模型选择

**Current baseline:** OpenAI CLIP ViT-B/32.

- Relatively low computational cost.
- Suitable for local GPU inference.
- Official pretrained model is readily available.
- Sufficient for establishing an initial zero-shot baseline.

选择 ViT-B/32 主要考虑计算成本、实现便利性及建立基础实验结果，目前尚未进行不同 backbone 之间的性能比较。


## 3. Prompt Design / 提示词设计

| Task | Baseline template |
|---|---|
| Category | `a photo of a {category}` |
| Attribute | `a photo of an object that is {attribute}` |
| Affordance | `a photo of an object that can be used to {affordance['word'][0]}` |

Category prediction uses the original Top 10 categories. Attribute and affordance prompts follow the original vocabulary order. The first synonym of each affordance entry is used in the baseline.

三项任务均采用统一的自然语言模板：Category描述物体是什么，Attribute描述物体具有什么特征，Affordance 描述物体可以用于什么行为。初始方案只使用每个 affordance 的第一个同义词，便于建立 baseline。

### Known Problem: Prompt Collision / 提示词冲突

During the initial inference test, I found that different affordance labels may share the same first synonym, for example:

- `bend`: form a curve
- `bend`: change direction

Using `word[0]` results in only 147 unique prompts for 170 affordance labels, with 44 labels involved in collisions. Identical prompts produce identical text features and thus cannot distinguish the corresponding labels by similarity scores. I kept this implementation as V1 for later comparisons.

初次推理时发现，同一个动词可能对应不同词义(如 bend 的“形成曲线”和“改变方向”)。只取 `word[0]` 会使 44 个标签涉及提示词冲突，模型无法根据相似度区分它们。因此保留原方案作为 V1，并尝试引入词义定义解决这一问题。

由于当前方案只使用 `word[0]`，部分具有不同含义的 Affordance 被映射为完全相同的 Prompt 和文本特征，导致语义信息丢失。


## 4. Text Feature Precomputation / 文本特征预计算

All candidate prompts are tokenized and encoded before batch inference.

由于所有物体共享相同的候选标签，因此只需要计算一次文本特征，避免在每次图像推理时重复运行 Text Encoder。

## 5. Similarity Calculation / 相似度计算

Both image and text embeddings are converted to FP32 and L2-normalized before matrix multiplication. Cosine similarity is used directly as the prediction score, without softmax, since all evaluation metrics are ranking-based.

对图像和文本特征进行 L2 归一化，通过矩阵乘法得到余弦相似度，直接用于排序，不额外进行 Softmax。类别是单标签分类，属性和 Affordance 是多标签排序任务。

## 6. Evaluation Protocol / 评估方案

- Category: Top-1, Top-3 and Top-5 accuracy over all valid instances.
- Attribute and affordance: per-label AP over the entire subset, followed by macro averaging over labels with positives (93/114 attributes and 161/170 affordances).
- Use positive-label prevalence as a constant-score baseline to account for differences in label density.

Ground-truth boxes are used for cropping, but ground-truth category labels are not used to assist attribute or affordance prediction. All experiments use the same 11,231 valid instances.

类别采用 Top-k Accuracy，属性和 Affordance 采用 mAP(仅统计有正样本的标签)。额外计算正样本比例基准，避免仅凭 mAP 的绝对大小判断模型能力。所有实验使用同一有效实例子集，且不向属性或功能预测提供真实类别。

## 7. Affordance Prompt Ablation / 提示词对照实验

To investigate prompt collisions, I compared three variants using the same CLIP ViT-B/32 backbone, cached image embeddings and evaluation subset. Only affordance prompts were changed.

All variants share the prefix `a photo of an object that can be used to`.

| Variant | Prompt suffix |
|---|---|
| V1: Word | `word[0]` |
| V2: Definition | `define.strip()` |
| V3: Word + Definition | `word[0] (define.strip())` |

假设引入词义定义可以消除同词异义造成的冲突，同时补充单个动词缺少的语义信息。保持模型、图像特征、样本和评估方式不变，对比仅使用单词、仅使用定义及两者结合三种方案。

| Metric | V1 | V2 | V3 |
|---|---:|---:|---:|
| Unique prompts / token sequences | 147 | 165 | 170 |
| Overall mAP | 31.43% | 31.97% | 32.57% |
| Original collision mAP | 37.14% | 38.64% | 38.03% |
| Original non-collision mAP | 29.28% | 29.47% | 30.52% |

V3 eliminates all exact prompt and token-sequence collisions and achieves the highest overall mAP, while V2 performs best on the original collision labels. Improvements on non-collision labels suggest that definitions may also provide useful semantic context. The experiment does not isolate the effects of semantics from other changes in prompt wording.

结果：V3 完全消除了提示词及 token 序列的重复，整体 mAP 提升 1.14 个百分点；但冲突组表现最好的是 V2。V3 在非冲突组也有改善，说明引入定义的作用可能不局限于消歧，也可能提供了额外语义信息，具体机制尚未验证。

## 8. Pairwise Semantic Disambiguation / 同词异义区分实验

AP alone cannot directly measure whether two ambiguous labels are distinguishable.  Introducing a supplementary pairwise balanced accuracy metric:

- For each original collision pair, evaluate only objects positive for exactly one of the two labels.
- Count the prediction as correct if the true label receives the higher score. Differences within `1e-6` count as ties and receive 0.5 credit.
- Weight the two exclusive label groups equally, then average across valid pairs. A pair requires samples from both groups.

为直接检验模型能否区分同词异义的标签，设计了 Pairwise 指标：只保留两个标签中恰好具有一个标签的物体，判断真实标签的相似度是否更高，平分计 0.5；两侧样本组等权，再对所有有效标签对取平均，避免样本数量不平衡影响结果。

Of the 25 original collision pairs, 24 were valid.

| Method | Pairwise balanced accuracy |
|---|---:|
| V1 | 50.00% |
| V2 | 53.83% |
| V3 | 54.65% |

结果：V1 无法区分重复提示词，得分为 50%；V3 提升至 54.65%。即使完全消除文本冲突，实际语义区分能力的提升仍然有限。

This is a custom diagnostic, not an official OCL metric. Both supplementary experiments are exploratory, conducted on the same test subset without independent validation or statistical significance testing.

两个补充实验均为同一测试子集上的探索性实验，尚未验证泛化能力；Pairwise 也并非公认指标。

## 9. Engineering Decisions / 工程设计

Keep notebooks for exploration and provide `main.py` for reproducible CLI experiments. Bind the shared image-feature cache to verified data, ordered vocabularies, model weights, encoding configuration and software versions; prompt mode is excluded from the image cache key. Save experiment configurations and metrics as JSON.

代码审查后加强数据与缓存校验，统一 Notebook 和 CLI 的实验逻辑。清空缓存重新完成 2,808 个 GPU 推理批次后，图像特征、标签及三组实验指标与历史结果一致。工程修复未改变实验设计，验证仍限于本地 Windows CUDA 环境。













