# OCL & CLIP 实验报告

本项目使用预训练 OpenAI CLIP，在 OCL 官方测试集的 Top 10 类别子集上进行零样本(zero-shot)物体类别、属性和可供性(affordance)预测，并针对可供性提示词进行补充实验。没有进行模型训练或微调。英文版见下。

## 1. 具体实现

### 1.1 数据处理

读取官方 `OCL_annot_test.pkl` 及类别、属性、可供性词表，按测试集中的物体实例数量选出前十个类别：car、broccoli、boat、horse、dog、banana、truck、motorcycle、zebra、sheep，共 11,388 个实例，涉及 4,151 张图片。

使用官方资源和自定义脚本获取 ImageNet、COCO 和 Web 图片，检查图像能否正常解码、文件哈希、尺寸及标注框，修正 ImageNet 坐标并排除缺失图片和异常标注框，最终生成 `valid_instances.csv`。

| 来源 | 所需图片 | 已获取图片 | 有效图片 | 有效实例 / 原始实例 |
|---|---:|---:|---:|---:|
| ImageNet | 309 | 309 | 309 | 361 / 361 |
| COCO | 3,359 | 3,359 | 3,359 | 10,108 / 10,108 |
| Web | 483 | 401 | 397 | 762 / 919 |
| 合计 | 4,151 | 4,069 | 4,065 | 11,231 / 11,388 |

基于 PyTorch 的 `Dataset` 实现 `OCLDataset`，以物体实例为单位读取图片、转换为 RGB、按真实标注框裁剪，并使用 CLIP 自带的预处理得到 `(3, 224, 224)` 图像张量。类别标签为整数索引，属性和可供性标签分别为 114 维和 170 维 multi-hot 向量。使用原生 `DataLoader`，设置 `batch_size=4`、`shuffle=False`、`drop_last=False`，保留最后不足四个实例的批次。

### 1.2 模型调用与推理

通过官方 `clip.load()` 加载 ViT-B/32 及预处理函数，将模型置于 GPU，调用 `eval()`，冻结全部参数，并在 `torch.inference_mode()` 下推理。

首先为候选标签构造文本提示词：

| 任务 | Baseline Prompt |
|---|---|
| 类别 | `a photo of a {category}` |
| 属性 | `a photo of an object that is {attribute}` |
| 可供性 | `a photo of an object that can be used to {word[0]}` |

使用 `clip.tokenize()` 和 `encode_text()` 一次性计算所有文本特征，再逐批调用 `encode_image()` 提取图像特征。两类特征均转换为 FP32 并进行 L2 归一化，通过矩阵乘法 `image_features @ text_features.T` 计算余弦相似度。直接使用相似度进行排序。属性和可供性预测未使用真实类别作为额外输入。

将图像特征与对应标签缓存到 CPU，便于重复实验。通过 `main.py` 提供独立的命令行入口，支持三种可供性提示词方案、缓存复用及 JSON 结果保存。

### 1.3 评估指标

- **Top-k Accuracy：** 若真实类别位于相似度最高的 k 个候选类别中，则该实例计为正确；对全部有效实例取平均，报告 Top-1、Top-3 和 Top-5。
- **AP 与 mAP：** 对每个属性或可供性标签，使用全部实例在该标签上的相似度和真实二值标签计算 AP(`sklearn.metrics.average_precision_score`，即按召回率增量加权求和的 precision)。随后对有正样本的标签取宏平均，得到 mAP。有效属性为 93/114，有效可供性为 161/170。不对每批的 AP 分别取平均。
- **Pairwise Disambiguation(补充指标)：** 对原始提示词冲突的每一对标签，仅选择恰好具有其中一个标签的实例。真实标签得分较高计 1，较低计 0，分差绝对值不超过 `1e-6` 计 0.5。两组独占正例分别取平均后等权合并，再对所有有效标签对取平均。25 对中有 24 对满足两组均非空的条件。尝试使用该指标用于衡量同词异义标签的区分能力。

## 2. 实验结果与分析

### 2.1 Zero-shot Baseline

| 指标 | 结果 |
|---|---:|
| Top-1 Accuracy | 88.42% |
| Top-3 Accuracy | 97.66% |
| Top-5 Accuracy | 99.22% |
| Attribute mAP | 13.87% |
| Affordance mAP | 31.43% |

在十个候选类别及真实物体裁剪框的条件下，CLIP 取得了较好的类别识别结果。为进一步理解多标签预测性能，计算了恒定分数预测器的 mAP，即有效标签正样本比例的平均值：属性为 8.75%，可供性为 27.89%。CLIP 分别高出 5.12 和 3.54 个百分点。

部分属性(如 `striped`、`quadruped`、`ripe`)的 AP 较高，分别为 87.25%、87.01% 和 75.80%。这些预测可能受物体类别先验影响；极低 AP 的标签大多缺少正样本。

### 2.2 可供性提示词对照实验

初始方案发现不同可供性标签可能对应相同单词(见第 3 节)。因此使用官方词义定义，比较 Word(V1)、Definition(V2)、Word + Definition(V3)三种提示词，保持模型、图像特征、标签及评估子集不变。

| 指标 | V1 | V2 | V3 |
|---|---:|---:|---:|
| 唯一 Prompt / token 序列数 | 147 | 165 | 170 |
| Affordance mAP | 31.43% | 31.97% | 32.57% |
| 原冲突标签 mAP | 37.14% | 38.64% | 38.03% |
| 原非冲突标签 mAP | 29.28% | 29.47% | 30.52% |
| Pairwise Disambiguation | 50.00% | 53.83% | 54.65% |

V3 消除了全部文本及 token 序列的重复，整体 mAP 提高 1.14 个百分点，Pairwise 提高 4.65 个百分点。V2 在原冲突标签上的 mAP 最好，V3 在非冲突标签上也取得改善，说明定义可能同时具有消歧和补充语义信息的作用。不过消除提示词冲突后的区分能力仍然有限。

### 2.3 结果复核

工程修复后，从空缓存重新执行了全部 2,808 个 GPU 推理批次。新生成的图像特征及三组标签与历史缓存逐元素一致，三个 Prompt mode 的 CLI 指标与原始实验一致。Notebook 与 CLI 共用缓存，类别及属性指标在三组实验中保持不变。

## 3. 遇到的问题、解决方法及尚未解决的问题

### 3.1 主要困难与解决方法

| 问题 | 解决方法与结果 |
|---|---|
| 官方数据准备流程涉及多种来源，完整下载 ImageNet 和 COCO 成本较高 | 从官方测试标注提取 Top 10 所需图片路径，编写针对子集的下载脚本。COCO 按文件名和原始来源下载，最终获得全部 3,359 张。 |
| ImageNet 难以直接按文件名获取指定图片 | 使用 Hugging Face ImageNet 数据集，解决访问授权后，利用 Parquet 元数据与 HTTP Range 请求获取指定 JPEG 字节，没有通过下载完整数据集，最终获得全部 309 张。后续增加固定 revision 和哈希校验。 |
| Web 图片大量失效、重定向或无法访问 | 结合原始 URL(237 张)、官方 `from_web.tgz`(123 张)、Wayback Machine(10 张)及人工恢复(31 张),也尝试了 Common Crawl。最后部分图片通过浏览器手动下载或通过手动查找历史域名成功恢复。最终获得 401/483 张。 |
| ImageNet 标注框与原图坐标不一致 | 与 ImageNet 定位标注对比，发现较强证据支持 256×256 参考坐标系，分别按原图宽、高进行缩放。72 个参考框的中位 IoU 由 0.243 提高至 0.931。 |
| 部分边界框异常或与实际图片不匹配 | 在推理前统一验证坐标：COCO 和 Web 采用原图像素 XYXY；轻微越界框裁剪至图像边界，无法合理修复的框排除。最终排除 8 个异常 Web 实例。 |
| 不同物体具有不同数量的标签，批量推理及标签对齐不便 | 将数据组织为固定形状的物体级样本，使用 multi-hot 标签，并保持标签索引与官方词表顺序严格一致。AP 按标签列汇总全部实例，排除无正例标签。 |
| 不同 affordance 标签共享同一个 Prompt | 170 个标签中有 44 个涉及冲突，例如 `bend` 可表示 `form a curve` 或 `change direction`。保留原方案，引入词义定义进行对照实验，并设计 Pairwise 指标直接检验区分能力。 |
| 重复推理耗时，旧缓存可能在数据或词表变化时被错误复用 | 缓存图像特征，并将经过验证的数据快照、有序词表、模型权重、编码配置及软件版本绑定到缓存键。校验实际文件哈希、特征及标签；损坏或不匹配的缓存重新计算，统一 Notebook 与 CLI 逻辑，并采用原子写入。 |

数据下载、恢复、审计脚本及最终工程审查与修复过程中使用了 Codex 协助。

### 3.2 尚未解决的问题与局限

- 仍有 82 张 Web 图片(涉及 149 个物体实例)无法获取，另有 8 个异常实例被排除。最终有效实例覆盖率为 98.62%(11,231/11,388)。缺失样本可能造成评估偏差。
- ImageNet 的 256×256 坐标约定具有较强实验依据，但无法确认是否合理。
- 单张静态图像对部分物理功能的信息表达有限，类别先验可能影响属性及可供性预测。目前没有进一步分离这些因素。Prompt 实验基于同一测试子集，没能进行独立验证或统计显著性检验。
- 已在本地 Windows CUDA 环境完成重新推理及故障测试，但尚未验证全新环境安装、跨平台端到端运行或重新在线获取完整数据快照。

更多实验设计依据见 `design.md`，数据准备及运行方法见 `README.md`，历史进程见 `timeline.md`。

---

# Experiment Report

This project evaluates pretrained OpenAI CLIP on object category, attribute and affordance prediction in the Top 10 category subset of the official OCL test set. A supplementary affordance prompt experiment is included. No training or fine-tuning was performed.

## 1. Implementation

### 1.1 Data Processing

I loaded the official `OCL_annot_test.pkl` and label vocabularies, and selected the ten most frequent object categories in the test set: car, broccoli, boat, horse, dog, banana, truck, motorcycle, zebra and sheep. They contain 11,388 target instances across 4,151 images.

Using official resources and custom scripts, I obtained images from ImageNet, COCO and the Web. I checked image decoding, hashes, dimensions and bounding boxes, corrected ImageNet coordinates, and excluded missing images and invalid boxes. The resulting instance manifest is stored in `valid_instances.csv`.

| Source | Required images | Acquired images | Valid images | Valid / original instances |
|---|---:|---:|---:|---:|
| ImageNet | 309 | 309 | 309 | 361 / 361 |
| COCO | 3,359 | 3,359 | 3,359 | 10,108 / 10,108 |
| Web | 483 | 401 | 397 | 762 / 919 |
| Total | 4,151 | 4,069 | 4,065 | 11,231 / 11,388 |

`OCLDataset` inherits from PyTorch's `Dataset` and treats each object as one sample. Images are converted to RGB, cropped using ground-truth boxes, and transformed into `(3, 224, 224)` tensors by CLIP's preprocessing function. Category labels are integer indices, while attribute and affordance labels are 114- and 170-dimensional multi-hot vectors. The native `DataLoader` uses `batch_size=4`, `shuffle=False` and `drop_last=False`.

### 1.2 Model and Inference

I loaded pretrained CLIP ViT-B/32 through the official `clip.load()` API, moved it to GPU, called `eval()`, froze all parameters and performed inference under `torch.inference_mode()`.

| Task | Baseline prompt |
|---|---|
| Category | `a photo of a {category}` |
| Attribute | `a photo of an object that is {attribute}` |
| Affordance | `a photo of an object that can be used to {word[0]}` |

All candidate texts are tokenized and encoded once using `clip.tokenize()` and `encode_text()`. Images are encoded batch by batch using `encode_image()`. Both embeddings are converted to FP32 and L2-normalized. The matrix product `image_features @ text_features.T` gives cosine similarity scores, which are used directly for ranking without softmax. Ground-truth categories are not supplied to attribute or affordance prediction.

Image embeddings and aligned labels are cached on CPU for repeated experiments. `main.py` provides a command-line interface for all three prompt variants, cache reuse and JSON result saving.

### 1.3 Evaluation Metrics

- **Top-k accuracy:** An instance is correct if its true category appears among the k highest-scoring candidates. Accuracy is averaged over all valid instances for k = 1, 3 and 5.
- **AP and mAP:** For each attribute or affordance label, AP is computed over all instances using `sklearn.metrics.average_precision_score`, which weights precision by increments in recall. Macro mAP averages AP over labels with at least one positive instance (93/114 attributes and 161/170 affordances). AP is not averaged across batches.
- **Pairwise Disambiguation (supplementary):** For each original prompt-collision pair, evaluate instances positive for exactly one label. A higher score for the true label receives one point; a lower score receives zero; ties within `1e-6` receive half a point. The two exclusive-positive groups are weighted equally, followed by a macro average over valid pairs. Of 25 original pairs, 24 have both groups nonempty. This is a custom diagnostic rather than an official OCL metric.

## 2. Results and Analysis

### 2.1 Zero-shot Baseline

| Metric | Result |
|---|---:|
| Top-1 Accuracy | 88.42% |
| Top-3 Accuracy | 97.66% |
| Top-5 Accuracy | 99.22% |
| Attribute mAP | 13.87% |
| Affordance mAP | 31.43% |

CLIP performs well at category recognition in this ten-class, ground-truth-crop setting. To interpret multi-label performance, I calculated a constant-score mAP baseline by averaging positive-label prevalence over valid labels. The baselines are 8.75% for attributes and 27.89% for affordances. CLIP improves on them by 5.12 and 3.54 percentage points, respectively. Thus, higher raw affordance mAP does not necessarily indicate stronger affordance understanding.

Attributes such as `striped`, `quadruped` and `ripe` achieve APs of 87.25%, 87.01% and 75.80%. These results may partly reflect category priors. Very low-AP labels generally have few positive instances, so their results should also be interpreted cautiously.

### 2.2 Affordance Prompt Comparison

After observing that distinct affordance labels can share identical prompts (Section 3), I compared three variants using the official semantic definitions: Word (V1), Definition (V2), and Word + Definition (V3). The backbone, image features, labels and evaluation subset remained fixed.

| Metric | V1 | V2 | V3 |
|---|---:|---:|---:|
| Unique prompts / token sequences | 147 | 165 | 170 |
| Affordance mAP | 31.43% | 31.97% | 32.57% |
| Original collision-label mAP | 37.14% | 38.64% | 38.03% |
| Original non-collision-label mAP | 29.28% | 29.47% | 30.52% |
| Pairwise Disambiguation | 50.00% | 53.83% | 54.65% |

V3 eliminates all exact prompt and token-sequence collisions, improving overall mAP by 1.14 percentage points and pairwise accuracy by 4.65 percentage points. V2 performs best on the original collision labels, while V3 also improves non-collision labels. This suggests that definitions may provide semantic context in addition to disambiguation. However, actual semantic discrimination remains limited, and the mechanism and generalizability of the improvement have not been verified.

### 2.3 Verification

After the engineering fixes, I reran all 2,808 GPU inference batches from an empty cache. The resulting image embeddings and three label tensors matched the historical cache element by element. All three command-line experiments reproduced the original metrics, and the notebooks and CLI shared the new cache. Category and attribute metrics remained unchanged across prompt variants.

## 3. Problems, Solutions and Remaining Limitations

### 3.1 Main Difficulties and Solutions

| Problem | Solution and outcome |
|---|---|
| Large datasets and multiple data sources | Extracted required image paths for the Top 10 subset from official annotations and wrote targeted download scripts. All 3,359 required COCO images were obtained. The complete official `prepare_data.sh` was not executed. |
| Difficulty retrieving specific ImageNet images | Obtained access to the Hugging Face ImageNet dataset and used Parquet metadata and HTTP Range requests to retrieve the original JPEG bytes of all 309 target images without downloading the full dataset. Fixed-revision selection and hash verification were added later. |
| Unavailable or redirected Web images | Combined original URLs (237 images), the official `from_web.tgz` archive (123), Wayback Machine (10) and manual recovery (31). Common Crawl was also investigated. Some images were downloaded manually and some old news images were searched through historical domains manually. In total, 401/483 images were acquired. |
| ImageNet bounding boxes did not match original image coordinates | Compared against ImageNet localization annotations and found strong evidence for a 256 × 256 reference coordinate system. Rescaled each axis independently, improving median IoU from 0.243 to 0.931 on 72 reference boxes. |
| Invalid or mismatched bounding boxes | Audited boxes before inference. Used original-pixel XYXY for COCO and Web, clipped minor boundary violations and excluded boxes that could not be reasonably corrected. Eight abnormal Web instances were excluded. |
| Variable numbers of objects and labels, and label alignment | Used fixed-shape object-level samples, multi-hot labels and the original vocabulary order. Calculated AP across instances for each label and excluded labels without positives from mAP. |
| Affordance prompt collisions | Found 44 affected labels, such as `bend` meaning either `form a curve` or `change direction`. Preserved the original baseline, compared definition-based prompts and introduced a pairwise diagnostic to assess discrimination directly. |
| Repeated inference and unsafe cache reuse | Bound the image-feature cache to the verified data snapshot, ordered vocabularies, model weights, encoding settings and software versions. Added file-hash and tensor validation, invalid-cache recomputation, atomic writes and shared notebook/CLI logic. |

Codex assisted with data acquisition, recovery and audit scripts, as well as the final engineering review and fixes.

### 3.2 Remaining Limitations

- 82 Web images covering 149 instances remain unavailable; another eight invalid instances were excluded. The final valid-instance coverage is 98.62% (11,231/11,388). Missing data may introduce bias, and historical image sources may become unavailable again.
- The inferred ImageNet 256 × 256 coordinate convention has strong empirical support but has no external confirmation.
- Some physical affordances are difficult to infer from static images, and category priors may influence predictions. These factors have not been isolated. The prompt experiments use the same test subset without independent validation or significance testing.
- Fresh inference and failure-case tests were completed in the local Windows CUDA environment. Fresh-environment installation, cross-platform end-to-end execution and online reconstruction of the exact data snapshot have not been verified.

See `design.md` for design decisions, `README.md` for data preparation and execution instructions, and `timeline.md` for the development history.
