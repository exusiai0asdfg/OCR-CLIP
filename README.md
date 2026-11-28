# OCL & CLIP — RHOS Stage 0 Assessment

Zero-shot object category, attribute and affordance prediction on an audited subset of the OCL test set using pretrained OpenAI CLIP.

- [OCL: Beyond Object Recognition: A New Benchmark towards Object Concept Learning (ICCV 2023)](https://openaccess.thecvf.com/content/ICCV2023/html/Li_Beyond_Object_Recognition_A_New_Benchmark_towards_Object_Concept_Learning_ICCV_2023_paper.html)
- [Official OCL repository](https://github.com/silicx/ObjectConceptLearning)
- [OpenAI CLIP](https://github.com/openai/CLIP)

## 1. Results

CLIP ViT-B/32, 10 candidate object categories, 11,231 valid object instances, ground-truth bounding boxes, no training or fine-tuning.

| Metric | Baseline (word) |
|---|---:|
| Top-1 Accuracy | 88.42% |
| Top-3 Accuracy | 97.66% |
| Top-5 Accuracy | 99.22% |
| Attribute mAP (93/114 valid labels) | 13.87% |
| Affordance mAP (161/170 valid labels) | 31.43% |

| Affordance prompt | Unique prompts | mAP | Pairwise disambiguation accuracy |
|---|---:|---:|---:|
| Word only | 147 | 31.43% | 50.00% |
| Definition only | 165 | 31.97% | 53.83% |
| Word + definition | 170 | 32.57% | 54.65% |

The evaluation covers 98.62% of the original Top 10 object instances. These are zero-shot results on the selected audited subset, not directly comparable with full-benchmark OCL results. Prompt comparisons are exploratory. See [report.md](report.md) for analysis and limitations.

## 2. Environment

The tested environment was Windows, Python 3.11.18, NVIDIA GeForce RTX 5060 Laptop GPU, PyTorch 2.15.0+cu132 and torchvision 0.29.2+cu132 (CUDA 13.2 build).

```bash
conda create -n ocl python=3.11 -y
conda activate ocl
```

Install a compatible CUDA-enabled PyTorch and torchvision build **before** installing the remaining requirements. Use the installation instructions and package index at [pytorch.org](https://pytorch.org/get-started/locally/), selecting the tested versions and CUDA build where available. Other supported builds may work but have not been tested here. For CPU inference, use a compatible CPU build instead.

```bash
# Core inference (includes the fixed OpenAI CLIP Git commit)
python -m pip install -r requirements.txt

# Optional: data preparation and recovery
python -m pip install -r requirements-data.txt

# Optional: run the research notebooks (also needed to generate the manifest)
python -m pip install -r requirements-notebook.txt
```

Git is required to install CLIP from its official repository. Do not replace the pinned Git dependency with the unrelated PyPI package named `clip`.

Verify the active interpreter and CUDA availability:

```bash
python -c "import sys, torch; print(sys.version); print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

A fresh-environment installation has not yet been independently verified.

## 3. Data preparation

The original OCL resources and images are not distributed in this repository. Data preparation uses official annotations and archives together with custom scripts that retrieve only the required Top 10 subset; the complete official `prepare_data.sh` was not executed.

### 3.1. Resources and manifest

Clone the official OCL repository separately, or obtain its original resource files following the [official data preparation instructions](https://github.com/silicx/ObjectConceptLearning/tree/main/data). Place its `data/resources/` contents in this project's `data/resources/`. All current entry points use this location by default; explicit path overrides are available.

Open `notebooks/01-top-10.ipynb` in the `ocl` Python environment and run the cells through manifest generation. It selects the ten most frequent object categories from the original test annotations and generates `data/image_manifest.json`, or verifies an existing manifest. Keep the original relative image paths.

The expected layout for evaluation is:

```text
data/
├── ImageNet/
├── COCO/
├── from_web/
├── resources/
│   ├── OCL_annot_test.pkl
│   ├── OCL_class_object.json
│   ├── OCL_class_attribute.json
│   ├── OCL_class_affordance.json
│   └── url_to_filename.json       # required for Web acquisition
├── image_manifest.json
├── final_audit/
│   ├── snapshot.json
│   └── valid_instances.csv
└── cache/                         # generated during evaluation
```

The CLI requires the four official annotation/vocabulary files, the manifest, the local audit snapshot and instance CSV, and all images referenced by the CSV. Even though inference reads labels from the audited CSV, the official resources are still required for hash verification. The URL mapping is needed for data preparation, not inference.

### 3.2. Image acquisition

| Source | Expected images | Acquired | Valid images | Valid / original instances |
|---|---:|---:|---:|---:|
| ImageNet | 309 | 309 | 309 | 361 / 361 |
| COCO | 3,359 | 3,359 | 3,359 | 10,108 / 10,108 |
| Web | 483 | 401 | 397 | 762 / 919 |
| Total | 4,151 | 4,069 | 4,065 | 11,231 / 11,388 |

From the repository root, after generating the manifest:

```bash
python scripts/download_coco.py
python scripts/download_imagenet.py --revision 49e2ee26f3810fb5a7536bbf732a7b07389a47b5
```

- **COCO:** `download_coco.py` retrieves the 3,359 target images from the original sources, preserves split paths and skips already decodable files. Results and hashes are recorded in `data/coco_download_report.json`.
- **ImageNet:** `download_imagenet.py` and `imagenet_parquet.py` retrieve the required original JPEG bytes from the gated Hugging Face `ILSVRC/imagenet-1k` dataset using Parquet metadata and HTTP Range requests. Obtain dataset access and authenticate locally (e.g. `hf auth login`) first. The revision above is the one recorded for this experiment. The download report is `data/imagenet_download_report.json`. Use `--check-only` to check local files without downloading; see `--help` for other options, including the transfer budget.

For Web images, obtain the official `from_web.tgz` archive following the official OCL instructions, place it in `data/`, and run:

```bash
python scripts/download_web.py --wayback
```

If additional recovery is necessary, inspect the options for the extended recovery tool:

```bash
python scripts/recover_web.py --help
```

The original experiment used the official archive (123 images), original URLs (237), Wayback Machine (10) and manual browser recovery (31). Common Crawl was also investigated. `scripts/download_web.py`, `scripts/recover_web.py` and `tools/recover_web.html` retain the recovery workflow; local reports are stored under `data/` (including `web_download_report.json`, `web_missing.csv` and `web_recovery/`).

### 3.3. Manual recovery and final audit

The browser progress record, `data/final_audit/manual_browser_progress.json`, contains a `statuses` object with manifest paths and `success`, `invalid` or `skipped` statuses. The original Windows import workflow checks download metadata and recorded URLs; `--apply` performs the actual import and `--downloads` specifies an alternative download directory. This step requires the corresponding local progress record and downloaded files.

```bash
# Optional: import confirmed manual downloads on Windows
python scripts/import_manual_web.py
python scripts/import_manual_web.py --apply
```

For verified manual recoveries, the script also supports exporting and checking a portable source/hash record. These commands require the corresponding local recovery records or an existing source list:

```bash
python scripts/import_manual_web.py --export-sources data/final_audit/manual_web_sources.json
python scripts/import_manual_web.py --verify-sources data/final_audit/manual_web_sources.json
```

Once the available images are in their manifest paths, run the final audit:

```bash
python scripts/audit_dataset.py
```

The script defaults to `data/resources/` and writes to `data/final_audit/`. Use `--resources`, `--data-root` or `--output-dir` for alternative paths.

| Audit artifact | Purpose |
|---|---|
| `summary.json` | Coverage, coordinate rules and audit evidence |
| `snapshot.json` | Resource, manifest and instance CSV fingerprints; ordered categories |
| `valid_images.json` | Images containing at least one valid target object |
| `valid_instances.csv` | Valid instances, label IDs, image hashes and final crop boxes |
| `images.csv` | Image status, dimensions, format and hashes |
| `missing_images.csv`, `anomalies.json` | Missing images and invalid instances |
| `manual_web_import.json` | Manual import and matching evidence, if generated |

The committed [reproducibility/snapshot.json](reproducibility/snapshot.json) identifies the original evaluation snapshot. Compare the locally generated audit snapshot against it, especially the resource, manifest and instance CSV SHA-256 values. The committed fingerprint alone is not a replacement for the audited CSV or recovered image files.

### 3.4. Bounding boxes

ImageNet boxes were empirically mapped from a 256 × 256 reference coordinate space using `x * W / 256` and `y * H / 256`. On 72 validation reference boxes, median IoU improved from 0.243 to 0.931. This convention is inferred from evidence, not confirmed by the original authors.

COCO and Web boxes use original pixel XYXY coordinates. All 10,108 COCO target boxes were in bounds under XYXY; interpreting them as native COCO XYWH would make 8,253 boxes out of bounds. Slightly out-of-bounds boxes are clipped and rounded during the audit; invalid boxes are excluded. Image formats are determined by their contents, and original pixel orientation is retained. Inference uses `crop_xyxy` directly, without a second coordinate transformation, followed by RGB conversion and CLIP's standard image preprocessing (including center cropping).

The supporting audit code is in `scripts/audit_imagenet_boxes.py` and `scripts/audit_web_boxes.py`.

### 3.5. Data reproducibility limitations

Historical URLs and archive availability may change. In particular, the exact 31 manually recovered Web images may not be obtainable on another machine. The original images, raw annotations and local recovery records are intentionally excluded from Git. Follow the original datasets' access and redistribution terms; do not substitute visually similar images and assume they belong to the original snapshot.

The workflow can be rerun, but exact reconstruction of the original data snapshot is not guaranteed. If the newly audited snapshot differs from the committed fingerprint, report results as a different evaluation subset. The original snapshot may be needed to reproduce the exact published numbers.

## 4. Run evaluation

From the repository root, after preparing the audited data and official resources:

```bash
python main.py --help
python main.py --model ViT-B/32 --batch-size 4 --device cuda --prompt-mode word --use-cache
python main.py --model ViT-B/32 --batch-size 4 --device cuda --prompt-mode definition --use-cache
python main.py --model ViT-B/32 --batch-size 4 --device cuda --prompt-mode combined --use-cache
```

| Option | Purpose |
|---|---|
| `--model` | OpenAI CLIP model name or local checkpoint; default `ViT-B/32` |
| `--batch-size` | Object instances per batch; default 4 |
| `--device` | `cuda` or `cpu`; automatically selected if omitted |
| `--prompt-mode` | `word`, `definition` or `combined`; default `word` |
| `--data-root` | Alternative data directory |
| `--resource-root` | Alternative official annotation and vocabulary directory |
| `--output-dir` | Alternative JSON result directory |
| `--cache-dir` | Alternative feature-cache directory (default: `data/cache/` under the selected data root) |
| `--weights-dir` | Directory for OpenAI CLIP checkpoint downloads |
| `--use-cache` | Reuse a verified image-feature cache or save a new one |

For example, to use official resources from the adjacent clone:

```bash
python main.py --resource-root ../ObjectConceptLearning/data/resources --use-cache
```

Named CLIP checkpoints are downloaded on first use if not already cached. A local checkpoint can also be supplied through `--model`.

### Cache and output

With caching enabled, the first run generates image embeddings; later runs reuse them when the cache identity matches. The identity binds the verified data snapshot, ordered vocabularies, model weights, preprocessing, encoding settings, runtime environment and relevant source code. Image bytes and resource hashes are checked before cache reuse, and cached tensor shapes and label alignment are validated. Invalid caches are recomputed, and new caches are saved using atomic replacement. Changing only the text prompts does not invalidate the image-feature cache.

Omit `--use-cache` to perform fresh image inference. Timestamped JSON files containing configurations, snapshot and weight fingerprints, cache information and metrics are saved under `outputs/` by default. Original and verification results are included in [outputs/](outputs/).

### Supplementary experiments

The CLI reports Top-k Accuracy and both mAP metrics for each prompt mode. Run [notebooks/04-prompt-ablation.ipynb](notebooks/04-prompt-ablation.ipynb) to reproduce the unique-prompt counts, collision/non-collision mAP and pairwise disambiguation results. It uses the same image-feature cache and prompt builders as the CLI.

Open notebooks from within the project workspace using the `ocl` kernel. For alternative locations, the notebooks support `OCL_PROJECT_ROOT`, `OCL_DATA_ROOT` and `OCL_RESOURCE_ROOT`; the prompt ablation notebook also supports `OCL_CACHE_DIR`.

## 5. Project organization

| Path | Purpose |
|---|---|
| `main.py` | Command-line evaluation entry point |
| `src/dataset.py` | Object-level Dataset, input validation and image preprocessing |
| `src/model.py` | CLIP loading and normalized feature extraction |
| `src/prompts.py` | Prompt construction |
| `src/metrics.py` | Top-k accuracy, AP, mAP and pairwise disambiguation |
| `src/evaluate.py` | Batch inference and feature collection |
| `src/cache.py` | Verified image-feature cache |
| `notebooks/01-top-10.ipynb` | Annotation exploration and manifest generation |
| `notebooks/02-dataset_loader.ipynb` | Dataset and DataLoader tests |
| `notebooks/03-clip.ipynb` | Baseline inference and metric analysis |
| `notebooks/04-prompt-ablation.ipynb` | Prompt comparison and disambiguation analysis |
| `scripts/`, `tools/` | Data acquisition, recovery and audit tools |
| `reproducibility/snapshot.json` | Original evaluation snapshot fingerprint |
| `outputs/` | Original and verification result JSON files |
| [report.md](report.md) | Methods, results and limitations |
| [design.md](design.md) | Design decisions and exploratory experiments |
| [timeline.md](timeline.md) | Development timeline |

## 6. Evaluation and verification notes

The Dataset returns one ground-truth-cropped object per sample. Top-k Accuracy is averaged over all valid instances. Attribute and affordance AP are calculated across the entire subset for each label, then averaged over labels with at least one positive instance. Pairwise disambiguation is a supplementary diagnostic, not an official OCL metric. Ground-truth category labels are not supplied to attribute or affordance prediction.

Following the engineering review, all 2,808 GPU batches were rerun from an empty cache on the original Windows CUDA environment. The resulting image features and three label tensors matched the historical cache element by element, and all three CLI experiments reproduced the original metrics. The notebooks and CLI now share the verified cache mechanism. Local fault-case tests were also performed.

Fresh-environment installation, cross-platform end-to-end execution and independent online reconstruction of the exact data snapshot have not been verified. Possible overlap between CLIP pretraining data and the test images is unknown. Codex assisted with data acquisition, recovery, auditing and final engineering review and fixes.
