包: numpy pillow scikit-learn tqdm matplotlib requests torch torchvision

clip :
> python -m pip install git+https://github.com/openai/CLIP.git

## OCL Stage 0 数据准备

在 Python 3.11 的 `ocl` 环境中运行数据脚本。官方 OCL 仓库默认位于相邻的 `../ObjectConceptLearning`，校验脚本从其 `data/resources/` 读取测试集标注、属性词表和可供性词表。路径不同时可以传入 `--resources`。

`notebooks/top-10.ipynb` 记录原始 Top 10 筛选。校验脚本从官方测试集复算每个类别的实例数，并核对 `data/image_manifest.json` 的路径集合。图片按该清单的原始相对路径放在 `data/ImageNet/`、`data/COCO/`、`data/from_web/`。

- **ImageNet** 已收齐目标子集。`scripts/download_imagenet.py` 和 `scripts/imagenet_parquet.py` 保留原始 JPEG，并使用 Hugging Face 本机登录与分片范围读取。历史证据位于 `data/imagenet_download_report.json` 和 `data/imagenet_bbox_audit/`。
- **COCO** 已收齐原始清单的 3,359 张目标 JPEG，其中 train2014 有 1,308 张，val2014 有 2,051 张。运行 `python scripts/download_coco.py` 可从官方 S3 源和原图片站补齐本地缺失文件。默认使用 8 个并发，保留 split 路径与原始字节，并跳过已经完整解码的文件。续跑使用同一命令，进度、SHA-256 和失败原因记录在 `data/coco_download_report.json`。
- **Web** 人工恢复已收尾。`scripts/download_web.py`、`scripts/recover_web.py`、`tools/recover_web.html` 和 `data/web_recovery/` 保留自动与人工恢复依据。`data/web_download_report.json` 记录每张图的来源和 SHA-256，`data/web_missing.csv` 记录最终 Web 缺失项。历史坐标抽查证据位于 `data/web_bbox_audit/`。

### 离线收录和校验

本次浏览器记录已保存到 `data/final_audit/manual_browser_progress.json`。它包含一个 `statuses` 对象，键为 manifest 路径，值为 `success`、`invalid` 或 `skipped`。收录脚本用人工成功标记、官方 URL 和 Windows 下载文件的来源信息核对对应关系。默认输出预检结果，传入 `--apply` 后才复制原始字节并更新来源报告，可用 `--downloads` 指定下载目录。

```powershell
python scripts/import_manual_web.py
python scripts/import_manual_web.py --apply
python scripts/audit_dataset.py
```

`data/final_audit/` 存放当前结果。

| 文件 | 用途 |
| --- | --- |
| `summary.json` | 三个来源和 Top 10 的覆盖率、坐标规则及证据、重复检查 |
| `valid_images.json` | 与原始 manifest 同结构，每张图至少有一个有效目标实例 |
| `valid_instances.csv` | 有效实例、原始对象索引、标签 ID 和可直接使用的裁剪框 |
| `images.csv` | 各目标路径的获取、解码和有效状态，以及尺寸、格式和哈希 |
| `missing_images.csv` / `anomalies.json` | 缺失项与受影响类别、解码或坐标异常及格式警告 |
| `manual_web_import.json` | 人工下载文件与官方 URL 的匹配证据、逐框校验以及待核对文件 |

### 后续 Dataset 取样口径

按 `valid_instances.csv` 的 `path` 和从 0 开始的 `object_index` 取样。同一张图可能同时含有效框与异常框。按图像文件实际格式解码，保持原始像素方向。

- ImageNet 用 `x × W / 256` 和 `y × H / 256` 回到原图坐标。此前对 72 个验证集定位框的中位 IoU 从 0.243 提高到 0.931。
- Web 用原图像素 XYXY。此前抽查已确认偏移的两个版本及本次发现的严重越界、空裁剪均从有效实例中排除。
- COCO 在 OCL 中使用原图像素 XYXY。审计了全部 10,108 个目标框，按 XYXY 解释均在图像范围内。按原生 COCO API 的 XYWH 解释时有 8,253 个框会越界。统计见 `data/final_audit/summary.json` 的 `bbox_evidence.COCO`。[COCO 官方 API](https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/coco.py#L267-L269)

轻微越界时，先裁到图像范围，再对左上角向下取整、右下角向上取整。标签使用官方词表里的零基索引。可直接读取报告中的 `crop_xyxy`，无需在 Dataset 中再次映射。
