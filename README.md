# OCL & CLIP — RHOS Stage 0 Assessment

Zero-shot object category, attribute and affordance prediction on an audited subset of the OCL test set using pretrained OpenAI CLIP.

- OCL: [Beyond Object Recognition: A New Benchmark towards Object Concept Learning (ICCV 2023)](https://openaccess.thecvf.com/content/ICCV2023/html/Li_Beyond_Object_Recognition_A_New_Benchmark_towards_Object_Concept_Learning_ICCV_2023_paper.html)
- [Official OCL repository](https://github.com/silicx/ObjectConceptLearning)
- [OpenAI CLIP](https://github.com/openai/CLIP)

## 1. Results 

CLIP ViT-B/32, 10 candidate object categories, 11,231 valid object instances, no training or fine-tuning.

| Metric | Baseline (word) |
|---|---:|
| Top-1 Accuracy | 88.42% |
| Top-3 Accuracy | 97.66% |
| Top-5 Accuracy | 99.22% |
| Attribute mAP | 13.87% |
| Affordance mAP | 31.43% |

| Affordance prompt | Unique prompts | mAP | Pairwise disambiguation accuracy |
|---|---:|---:|---:|
| Word only | 147 | 31.43% | 50.00% |
| Definition only | 165 | 31.97% | 53.83% |
| Word + definition | 170 | 32.57% | 54.65% |

The evaluation covers 98.62% of the original Top 10 object instances. These are zero-shot results on the selected audited subset, not directly comparable with full-benchmark OCL results. Prompt comparisons are exploratory.

## 2. Environment 

The experiments were developed in a Conda environment with Python 3.11 and tested locally using CUDA.

```bash
conda create -n ocl python=3.11 -y
conda activate ocl
python -m pip install -r requirements.txt
```

For GPU inference, install a compatible CUDA-enabled PyTorch and torchvision build using the instructions at [pytorch.org](https://pytorch.org/get-started/locally/). OpenAI CLIP should be installed from its official repository, rather than an unrelated package named `clip` on PyPI:

```bash
python -m pip install git+https://github.com/openai/CLIP.git
```

If the installed requirements already include the Git dependency, the last command is unnecessary. Git is required to install CLIP from GitHub.

## 3. Data preparation 

Clone the official OCL repository separately, or obtain its original resource files. The data preparation scripts were developed with the official repository at `../ObjectConceptLearning`. For evaluation, the default resource directory is `data/resources/` in this project; `--resource-root` can point to the official resource directory instead.

`notebooks/top-10.ipynb` records the original Top 10 selection. The original annotation and `data/image_manifest.json` define the target subset. Keep image paths relative to `data/`, including the original source directories.

```text
data/
├── ImageNet/
├── COCO/
├── from_web/
├── resources/
│   ├── OCL_class_attribute.json
│   └── OCL_class_affordance.json
├── image_manifest.json
├── final_audit/
│   └── valid_instances.csv
└── cache/                         # generated during evaluation
```

The final evaluation additionally depends on the audited image files referenced in `valid_instances.csv`. The original `OCL_annot_test.pkl` and other official resource files are required for data preparation, although inference reads the audited CSV and the two label vocabularies directly.

### Image acquisition

| Source | Expected images | Acquired | Valid images |
|---|---:|---:|---:|
| ImageNet | 309 | 309 | 309 |
| COCO | 3,359 | 3,359 | 3,359 |
| Web | 483 | 401 | 397 |
| Total | 4,151 | 4,069 | 4,065 |

- **ImageNet:** `scripts/download_imagenet.py` and `scripts/imagenet_parquet.py` retrieve the required original JPEG bytes using Hugging Face authentication, Parquet metadata and HTTP Range requests. See `data/imagenet_download_report.json` and `data/imagenet_bbox_audit/` for previous results and coordinate evidence. Access to the gated dataset may require authorization.
- **COCO:** `python scripts/download_coco.py` downloads the 3,359 target images from the original sources, preserves the split paths and skips already decodable files. Results and SHA-256 hashes are recorded in `data/coco_download_report.json`.
- **Web:** `scripts/download_web.py`, `scripts/recover_web.py`, `tools/recover_web.html` and `data/web_recovery/` document automatic and manual recovery. `data/web_download_report.json` records sources and hashes; `data/web_missing.csv` records remaining missing images. The original `from_web.tgz`, original URLs, Wayback Machine and manual browser recovery were used.

The exact recovered Web subset cannot be guaranteed by downloading again: URLs and archive availability may change, and 31 images were recovered manually. Obtain matching image bytes where possible and verify against the recorded hashes. Follow the original datasets' access and redistribution terms.

### Manual import and final audit 

The browser progress record in `data/final_audit/manual_browser_progress.json` contains a `statuses` object with manifest paths and `success`, `invalid` or `skipped` statuses. The import script checks the recorded URLs and downloaded files; `--apply` performs the actual import. `--downloads` specifies an alternative download directory. The audit script accepts `--resources` when the official annotation path differs from the default.

```bash
python scripts/import_manual_web.py
python scripts/import_manual_web.py --apply
python scripts/audit_dataset.py
```

These are the original workflow commands; check the corresponding scripts' options and prerequisites before rerunning. Manual import requires the matching previously downloaded files or a new recovery attempt.

| Audit artifact | Purpose |
|---|---|
| `summary.json` | Coverage, coordinate rules, evidence and duplicate checks |
| `valid_images.json` | Images containing at least one valid target object |
| `valid_instances.csv` | Valid object instances, label IDs and final crop boxes |
| `images.csv` | Image status, dimensions, format and hashes |
| `missing_images.csv`, `anomalies.json` | Missing images and invalid instances |
| `manual_web_import.json` | Manual import and matching evidence |

### Bounding boxes 

ImageNet boxes were empirically mapped from a 256 × 256 coordinate space using `x * W / 256` and `y * H / 256`. On 72 validation reference boxes, median IoU improved from 0.243 to 0.931. This convention is inferred from evidence, not confirmed by the original authors. COCO and Web boxes use original pixel XYXY coordinates. All 10,108 COCO target boxes were in bounds under XYXY; interpreting them as native COCO XYWH would make 8,253 boxes out of bounds.

Slightly out-of-bounds boxes are clipped and rounded in the audit; invalid boxes are excluded. Decode actual image formats and retain original pixel orientation. Inference uses `crop_xyxy` directly, without a second coordinate transformation.

## 4. Run evaluation 

Run from the repository root after preparing the audited data and label vocabularies:

```bash
python main.py --help
python main.py --model ViT-B/32 --batch-size 4 --device cuda --prompt-mode word --use-cache
python main.py --model ViT-B/32 --batch-size 4 --device cuda --prompt-mode definition --use-cache
python main.py --model ViT-B/32 --batch-size 4 --device cuda --prompt-mode combined --use-cache
```

| Option | Purpose |
|---|---|
| `--model` | CLIP backbone; default `ViT-B/32` |
| `--batch-size` | Number of object instances per batch; default 4 |
| `--device` | `cuda` or `cpu` |
| `--prompt-mode` | `word`, `definition` or `combined`; default `word` |
| `--data-root` | Alternative data directory |
| `--resource-root` | Alternative vocabulary directory |
| `--output-dir` | Alternative result directory |
| `--use-cache` | Load or save cached image features |

Example using vocabularies from the official repository:

```bash
python main.py --resource-root ../ObjectConceptLearning/data/resources --use-cache
```

With caching enabled, the first run generates image embeddings and the following runs reuse them if the cache key matches. Prompt changes do not require re-encoding images. Omit `--use-cache` to perform fresh image inference. Results and configuration are saved as timestamped JSON files under `outputs/` by default.

The cache key incorporates the model name, audited CSV, vocabularies and relevant source code. If actual image bytes, model weights or preprocessing dependencies change, regenerate the cache or rerun without caching; the current key does not independently track all of these changes.

## 5. Project organization 

| Path | Purpose |
|---|---|
| `main.py` | Command-line evaluation entry point |
| `src/dataset.py` | Object-level Dataset and image preprocessing |
| `src/model.py` | CLIP loading and normalized feature extraction |
| `src/prompts.py` | Prompt construction |
| `src/metrics.py` | Top-k accuracy, AP and mAP |
| `src/evaluate.py` | Batch inference and feature collection |
| `src/cache.py` | Image feature cache |
| `notebooks/` | Data exploration, implementation tests and supplementary experiments |
| `scripts/`, `tools/` | Data preparation and recovery tools |
| `outputs/` | Evaluation results |
| [report.md](report.md) | Methods, results and limitations |
| [design.md](design.md) | Design decisions and exploratory experiments |
| [timeline.md](timeline.md) | Development timeline |

## 6. Evaluation notes

The Dataset returns one cropped object per sample. Attribute and affordance AP are calculated across all valid object instances for each label, then averaged over labels with at least one positive instance. The pairwise disambiguation metric is a supplementary diagnostic, not an official OCL metric; its implementation and experiments are recorded in the prompt analysis notebook and report.

Local command-line reproduction of the three prompt experiments has been tested. Exact independent reproduction of the recovered Web subset and execution in a fresh environment have not yet been verified.
