# OCL & CLIP Experiment Report

For clarity, I divide the experiment into several parts, each covering one or more steps in the original assignment.

为了更清晰地记录实验过程，我将本次测试拆分为几个部分，每个部分对应原测试要求中的一个或多个步骤。

1. Data Preparation / 数据准备（Part of Step 1）
2. Dataset and DataLoader（Step 1）
3. Model Preparation（Step 2）
4. Inference and Evaluation（Step 3）

## 1. Data Preparation / 数据准备

### 1.1 Dataset Exploration / 数据集探索

First, I read the OCL paper and explored its official repository, focusing on the dataset structure, task definition and baseline experiments.

I loaded the test annotations from `OCL_annot_test.pkl` and inspected the category, attribute, affordance and bounding box annotations.

首先阅读 OCL 论文及其官方仓库，重点关注数据集结构、任务定义和 baseline 实验。通过读取 pickle 和 JSON 文件，了解了 OCL 的图片、物体实例及三种标签的组织方式。

The original test set contains 25,617 object instances from 221 categories. Following the assignment requirements, I selected the 10 categories with the largest number of instances.

原始测试集共有 25,617 个物体实例，涉及 221 个实际出现的类别。根据测试要求，按照物体实例数量筛选出 Top 10 类别。

| Category | Instances |
|---|---:|
| car | 1599 |
| broccoli | 1282 |
| boat | 1271 |
| horse | 1229 |
| dog | 1056 |
| banana | 1050 |
| truck | 1042 |
| motorcycle | 1011 |
| zebra | 928 |
| sheep | 920 |
| Total | 11388 |

After filtering, 4,151 images containing 11,388 target object instances were required.

筛选后共需要 4,151 张图片，涉及 11,388 个目标物体实例。生成 `image_manifest.json`，记录所有需要获取的图片路径。

### 1.2 Image Acquisition / 图片获取

The required images come from three different sources: COCO, ImageNet and the Web. Different acquisition methods were used for each source.

所需图片来自三个不同的数据源，因此分别采用不同的获取方案。

#### COCO

Images were downloaded directly from the official COCO image server according to their filenames. Only the required images were downloaded instead of the entire dataset.

All 3,359 required images were successfully downloaded.

根据 OCL 标注中提供的文件名构造下载 URL，直接从 COCO 官方服务器获取所需图片。

#### ImageNet

Downloading the entire ImageNet dataset would be inefficient, since only 309 images were required.

With the assistance of Codex, I used the Hugging Face ImageNet dataset, Parquet metadata and HTTP Range requests to locate and retrieve the required images without downloading the entire dataset.

All 309 images were successfully retrieved, with the source JPEG bytes preserved.


由于只需要 309 张 ImageNet 图片，完整下载 ImageNet 的成本过高。在 Codex 的协助下，通过 Hugging Face 镜像、Parquet 文件索引及 HTTP Range 请求实现了目标图片的定向获取。

#### Web Images

Web images were the most difficult part of data acquisition. Many original URLs were no longer accessible because of website migration, resource deletion or access restrictions.

Several methods were used:

- Downloading from the original URLs.
- Extracting images from the official `from_web.tgz` archive.
- Recovering historical images through the Wayback Machine.
- Searching Common Crawl archives.
- Manually checking failed URLs and recovering images through browsers.

82 images remain unavailable after the recovery attempts.

Web 图片是数据获取中最困难的部分。由于网站迁移、资源删除、访问限制等问题，大量原始 URL 无法直接使用。

尝试了原始 URL、官方存档、互联网历史档案以及人工浏览器恢复等多种方法，最终获取 401 张图片。但仍有82张图片未能获取(主要源于某几个网站的关停或者重建)。

| Source | Recovered images |
|---|---:|
| Original URLs | 237 |
| Official archive | 123 |
| Wayback Machine | 10 |
| Manual recovery | 31 |
| Total | 401 |

### 1.3 Bounding Box Validation / 标注框验证

During data preparation, I found that the OCL bounding boxes for ImageNet did not match the original image coordinates.

The investigation showed strong empirical evidence that these annotations use a 256 × 256 coordinate space.


The coordinates are transformed as follows:

- x' = x × W / 256
- y' = y × H / 256

On 72 validation reference boxes, the median IoU increased from 0.243 to 0.931 after coordinate transformation.

Although this coordinate convention was not explicitly documented in the official repository, the experimental evidence supports its use for the current ImageNet subset.

For COCO and Web images, the original pixel coordinates in XYXY format were used. Slightly out-of-bound boxes were clipped, while eight abnormal Web instances were excluded.

在数据准备过程中发现 ImageNet 图片的原始标注框与实际图像坐标存在偏差。通过对比 ImageNet 定位标注，发现将 OCL 标注视为基于 256 × 256 画布的坐标可以显著改善框的位置。

### 1.4 Final Dataset Statistics / 最终数据统计

| Source | Expected images | Acquired images | Valid images | Expected instances | Valid instances |
|---|---:|---:|---:|---:|---:|
| ImageNet | 309 | 309 | 309 | 361 | 361 |
| COCO | 3359 | 3359 | 3359 | 10108 | 10108 |
| Web | 483 | 401 | 397 | 919 | 762 |
| Total | 4151 | 4069 | 4065 | 11388 | 11231 |

- Image acquisition rate: 98.02%
- Valid image coverage: 97.93%
- Valid instance coverage: 98.62%

The original Top 10 categories and annotations were preserved. All missing images and invalid instances were recorded in the final audit files.

原始 Top 10 类别及标注保持不变，所有缺失图片及异常实例均已记录。后续实验将在 11,231 个有效实例上进行。

### 1.5 Limitations / 局限性

- 82 Web images involving 149 object instances remain unavailable.
- Eight additional Web instances were excluded due to invalid or mismatched bounding boxes.
- The ImageNet coordinate transformation is supported by empirical evidence but lacks direct confirmation from the dataset authors.
- Missing instances are not uniformly distributed across categories, which may introduce bias into the evaluation results.

Data acquisition and validation scripts were developed with the assistance of Codex. The original annotations, images, processing records and experimental evidence were preserved for reproducibility.

- 仍有部分图片获取失败
- 部分边框无效/不匹配
- ImageNet的坐标转换为自行处理,没有任何项目文件可以支持这一操作的合法性
- 缺失实例不均匀,可能影响评估

所有的操作脚本(包括手写和codeex辅助)均已经保留, 见/scripts和/tools