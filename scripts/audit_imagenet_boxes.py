"""Measure OCL bounding-box coordinate hypotheses without modifying inputs."""

import argparse
from collections import Counter, defaultdict
import csv
from hashlib import sha256
import json
from pathlib import Path
import pickle
import random
import re
import tarfile

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SEED = 20261125


class AnnotationReader(pickle.Unpickler):
    def find_class(self, module, name):
        raise ValueError(f"Unexpected pickle class {module}.{name}")


def json_read(path):
    return json.loads(path.read_bytes())


def json_write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def envelope(boxes):
    return (min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes))


def iou(a, b):
    area = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    a_area = (a[2] - a[0]) * (a[3] - a[1])
    b_area = (b[2] - b[0]) * (b[3] - b[1])
    union = a_area + b_area - area
    return area / union if union > 0 else 0.0


def map_box(box, scales):
    return [x * scale for x, scale in zip(box, scales * 2)]


def from_square(row, canvas):
    w, h = row["image_size"]
    return map_box(row["ocl_xyxy"], (w / canvas, h / canvas))


def clip(box, size):
    return [max(0, min(pos, edge)) for pos, edge in zip(box, size * 2)]


def excess(box, size):
    x0, y0, x1, y1 = box
    w, h = size
    return {"left": max(0, -x0), "top": max(0, -y0), "right": max(0, x1 - w), "bottom": max(0, y1 - h)}


def score(box, row):
    refs = row["reference_candidates"]
    if not refs:
        return None
    return max(iou(box, ref) for ref in refs)


def load_rows(data_root, reference, annotation):
    manifest = json_read(data_root / "image_manifest.json")["ImageNet"]
    records = json_read(data_root / "imagenet_download_report.json")["files"]
    external = json_read(reference)
    with annotation.open("rb") as file:
        annotation = {x["name"]: x for x in AnnotationReader(file).load() if x.get("source") == "ImageNet"}
    images, rows = {}, []
    for relative in manifest:
        path = Path(relative)
        split = path.parts[1]
        data = (data_root / path).read_bytes()
        if sha256(data).hexdigest() != records[relative]["sha256"]:
            raise ValueError(f"Download hash mismatch: {relative}")
        with Image.open(data_root / path) as image:
            size = list(image.size)
            orientation = image.getexif().get(274, 1)
        ref = external[split]["matches"].get(path.name)
        ref_objects = ref["objects"] if ref else []
        ref_size = [ref["width"], ref["height"]] if ref else None
        if ref is None:
            status = "unavailable"
        elif ref_size != size:
            status = "dimension_mismatch"
        else:
            status = "matched"
        synset = re.search(r"(n\d{8})\.JPEG$", records[relative]["source_image_path"]).group(1)
        boxes = [obj["xyxy"] for obj in ref_objects if obj["synset"] == synset]
        boxes = [b for b in boxes if b[2] > b[0] and b[3] > b[1]]
        if not boxes and status == "matched":
            status = "missing_class_box"
        image_row = {
            "split": split, "size": size, "orientation": orientation, "synset": synset,
            "reference_status": status, "reference_size": ref_size,
            "reference_boxes": boxes, "reference_objects": ref_objects,
        }
        images[relative] = image_row
        for index, obj in enumerate(annotation[relative]["objects"]):
            if "box" not in obj:
                continue
            bbox = obj["box"]
            if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                raise ValueError(f"Degenerate source OCL box: {relative}, {index}")
            canvas_excess = excess(bbox, [256, 256])
            candidates = boxes + ([envelope(boxes)] if len(boxes) > 1 else [])
            rows.append({
                "path": relative, "split": split, "object_index": index, "object": obj["obj"],
                "image_size": size, "ocl_xyxy": bbox, "outside_256": any(canvas_excess.values()),
                "canvas_excess": canvas_excess, "raw_excess": excess(bbox, size),
                "reference_status": status, "reference_box_count": len(boxes),
                "reference_candidates": candidates if status == "matched" else [],
            })
    return images, rows, external


def fit_training_square(rows):
    rows = [r for r in rows if r["split"] == "train" and r["reference_candidates"]]
    scores = {canvas: float(np.mean([score(from_square(r, canvas), r) for r in rows])) for canvas in range(224, 289)}
    return max(scores, key=scores.get), scores


def measure(rows, canvas_fit):
    for row in rows:
        w, h = row["image_size"]
        raw = row["ocl_xyxy"]
        scaled = from_square(row, 256)
        mapped = {
            "raw": raw,
            "square_256": scaled,
            "square_256_clipped": clip(scaled, [w, h]),
            "square_train_fit": from_square(row, canvas_fit),
            "square_262": from_square(row, 262),
            "short_edge_256": map_box(raw, (min(w,h)/256,) * 2),
            "long_edge_256": map_box(raw, (max(w,h)/256,) * 2),
        }
        row["transformed_xyxy"] = mapped
        row["iou"] = {name: score(box, row) for name, box in mapped.items()}
        row["mapped_256_excess_px"] = excess(scaled, [w, h])
        if row["outside_256"] and row["reference_candidates"]:
            reference = max(row["reference_candidates"], key=lambda box: iou(mapped["square_256_clipped"], box))
            gaps = {"left": reference[0], "top": reference[1], "right": w-reference[2], "bottom": h-reference[3]}
            limits = dict(left=w, right=w, top=h, bottom=h)
            row["reference_boundary_gap_px"] = {edge: gaps[edge] for edge, value in row["canvas_excess"].items() if value > 0}
            row["reference_same_edge_within_3pct"] = all(
                gaps[edge] <= max(3, 0.03 * limits[edge]) for edge in row["reference_boundary_gap_px"]
            )


def metric_summary(rows, mode):
    values = np.array([row["iou"][mode] for row in rows if row["iou"][mode] is not None])
    if not len(values):
        return {"n": 0}
    low, q1, median, q3, high = np.percentile(values, [10, 25, 50, 75, 90]).round(5).tolist()
    return {
        "n": len(values), "mean": round(float(values.mean()), 5), "min": round(float(values.min()), 5),
        "p10": low, "p25": q1, "median": median, "p75": q3, "p90": high,
        "at_0_5": int((values >= 0.5).sum()), "at_0_75": int((values >= 0.75).sum()),
        "at_0_9": int((values >= 0.9).sum()),
    }


def save_overlay(images, rows, chosen, filename, title, data_root, output_dir):
    by_image = defaultdict(list)
    for row in rows:
        by_image[row["path"]].append(row)
    height = 8.1 if len(chosen) > 4 else 4.6
    fig, axes = plt.subplots((len(chosen) + 3)//4, 4, figsize=(15.8, height), layout="constrained")
    for ax, relative in zip(axes.flat, chosen):
        case = images[relative]
        with Image.open(data_root / relative) as image:
            ax.imshow(np.asarray(image.convert("RGB")))
        w, h = case["size"]
        scores = []
        for row in by_image[relative]:
            raw, scaled = (row["transformed_xyxy"][mode] for mode in ("raw","square_256"))
            for (a, b, c, d), (color, style) in [(raw, ("#ff4160", "--")), (scaled, ("#c9ff18", "-"))]:
                ax.add_patch(Rectangle((a,b), c-a, d-b, fill=False, edgecolor=color, linewidth=1.6, linestyle=style))
            if row["iou"]["raw"] is not None:
                scores.append(f'{row["iou"]["raw"]:.2f} → {row["iou"]["square_256"]:.2f}')
        for a,b,c,d in case["reference_boxes"]:
            ax.add_patch(Rectangle((a,b), c-a, d-b, fill=False, edgecolor="#5ee7ff", linewidth=1.4, linestyle=":"))
        detail = " / ".join(scores) if scores else "No usable LOC reference"
        excess_value = max(max(r["canvas_excess"].values()) for r in by_image[relative])
        if excess_value:
            detail += f"  |  +{excess_value}px on 256 canvas"
        ax.set_title(f'{Path(relative).name}\n{case["split"]}  ·  {w} × {h}\n{detail}', fontsize=9)
        right = max(r["mapped_256_excess_px"]["right"] for r in by_image[relative])
        bottom = max(r["mapped_256_excess_px"]["bottom"] for r in by_image[relative])
        ax.set_xlim(-1, w + right + 1)
        ax.set_ylim(h + bottom + 1, -1)
        ax.axis("off")
    handles = [
        Line2D([0],[0], color="#ff4160", linestyle="--", label="OCL raw"),
        Line2D([0],[0], color="#84aa00", label="OCL mapped from 256 × 256"),
        Line2D([0],[0], color="#0089a7", linestyle=":", label="ILSVRC localization"),
    ]
    fig.suptitle(title, fontsize=15, fontweight="bold", y=1.045)
    fig.legend(handles=handles, loc="outside lower center", ncol=3, frameon=False)
    fig.savefig(output_dir / filename, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_distributions(subsets, train_canvas, output_dir):
    fig, axes = plt.subplots(1, 2, sharey=True, figsize=(10.6, 4.5), layout="constrained")
    colors = {"raw": "#d0485c", "square_256": "#0666ae", "square_256_clipped": "#1f8f67",
              "square_262": "#b87b21", "short_edge_256": "#8456a6", "long_edge_256": "#626d7b"}
    labels = {"raw": "Raw", "square_256": "256 × 256", "square_256_clipped": "256 × 256 + clip",
              "square_262": "262 × 262", "short_edge_256": "Short edge 256", "long_edge_256": "Long edge 256"}
    styles = {"square_256_clipped": "dashed", "square_262": "dashdot",
              "short_edge_256": "dotted", "long_edge_256": "dashed"}
    if train_canvas != 256:
        colors["square_train_fit"] = "#8a5c4e"
        labels["square_train_fit"] = f"{train_canvas} × {train_canvas} (train fit)"
        styles["square_train_fit"] = "dotted"
    for split, ax in zip(("train","val"), axes):
        for name, color in colors.items():
            values = sorted(r["iou"][name] for r in subsets[split] if r["iou"][name] is not None)
            ax.step([0] + values + [1], [0] + [(i+1)/len(values) for i in range(len(values))] + [1], where="post",
                    color=color, label=labels[name], linestyle=styles.get(name,"solid"))
        ax.set(title=f"{split} · {len(values)} matched OCL boxes", xlabel="Best IoU", xlim=(0,1), ylim=(0,1.01))
        ax.grid(alpha=.23)
    axes[0].set_ylabel("Cumulative fraction of boxes")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False)
    fig.savefig(output_dir / "iou_distribution.png", dpi=155, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def prepare_visuals(images, rows, data_root, output_dir):
    rng = random.Random(SEED)
    anomaly = {r["path"] for r in rows if r["outside_256"]}
    ordinary = [
        chosen for split in ("train","val")
        for chosen in rng.sample(sorted(p for p, img in images.items() if img["split"] == split and p not in anomaly), 4)
    ]
    ranked = sorted(
        (r for r in rows if r["outside_256"]),
        key=lambda r: (-max(r["canvas_excess"].values()), r["path"]),
    )
    selected_anomalies = list(dict.fromkeys(r["path"] for r in ranked[:2]))
    for split in ("val", "train"):
        candidates = [r["path"] for r in ranked if r["split"] == split and r["path"] not in selected_anomalies]
        selected_anomalies.extend(rng.sample(sorted(set(candidates)), min(3, len(set(candidates)))))
    if len(selected_anomalies) < 8:
        selected_anomalies.extend(rng.sample(sorted(anomaly - set(selected_anomalies)), 8-len(selected_anomalies)))
    worst = sorted((r for r in rows if r["iou"]["square_256"] is not None),
                   key=lambda r: (r["iou"]["square_256"] - r["iou"]["raw"], r["path"]))
    unusual = list(dict.fromkeys(r["path"] for r in worst))[:2]
    unusual += sorted(p for p,img in images.items() if max(img["size"]) > 1800)
    save_overlay(images, rows, ordinary, "random_train_val.png", "Random train and validation images", data_root, output_dir)
    save_overlay(images, rows, selected_anomalies, "out_of_canvas_examples.png", "OCL boxes outside the candidate 256 × 256 canvas", data_root, output_dir)
    save_overlay(images, rows, unusual, "other_exceptions.png", "Reference mismatches and images over 1800 px", data_root, output_dir)
    return {"seed":SEED,"random":ordinary,"anomalies":selected_anomalies,"other_exceptions":unusual}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--resources", type=Path)
    parser.add_argument("--reference", type=Path, help="ILSVRC localization reference JSON")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    data_root = args.data_root
    audit_dir = data_root / "imagenet_bbox_audit"
    output_dir = args.output_dir or audit_dir
    annotation = (args.resources or data_root / "resources") / "OCL_annot_test.pkl"
    images, rows, external = load_rows(data_root, args.reference or audit_dir / "localization_reference.json", annotation)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_canvas, training_grid = fit_training_square(rows)
    measure(rows, train_canvas)
    anomaly = [r for r in rows if r["outside_256"]]
    comparable = [r for r in rows if r["reference_candidates"]]
    subsets = {
        "all": rows, "train": [r for r in rows if r["split"] == "train"], "val": [r for r in rows if r["split"] == "val"],
        "outside_256": anomaly, "inside_256": [r for r in rows if not r["outside_256"]],
        "single_reference": [r for r in rows if r["reference_box_count"] == 1],
    }
    metrics = {group: {mode: metric_summary(subset, mode) for mode in rows[0]["iou"]} for group, subset in subsets.items()}
    annotation_sha = sha256(annotation.read_bytes()).hexdigest()
    archive = annotation.with_name("OCL_annot.tgz")
    with tarfile.open(archive, "r:gz") as tar:
        original_sha = sha256(tar.extractfile("OCL_annot_test.pkl").read()).hexdigest()
    if annotation_sha != original_sha:
        raise ValueError("Local OCL test annotations differ from the release archive.")
    clipped_change = [r for r in comparable if r["iou"]["square_256_clipped"] != r["iou"]["square_256"]]
    stats = {
        "images": len(images), "boxes":len(rows), "source_annotation_sha256":annotation_sha,
        "archive_member_matches_local":annotation_sha == original_sha,
        "method": {
            "reference": "ILSVRC localization CSV copy with dimensions. Compare only matching filenames, dimensions, and ILSVRC synsets.",
            "metric": "Maximum IoU against a same-synset localization box or their enclosing box. A separate single-reference subset is reported.",
            "calibration": "Fit an integer square canvas from 224 to 288 by mean training IoU, then evaluate on held-out ImageNet validation.",
            "behavior": "Independent annotations can choose a different object in the same image. The coordinate test cannot establish how ImageNet-150K files were encoded or packaged.",
        },
        "reference_images_by_split": {split: dict(Counter(i["reference_status"] for i in images.values() if i["split"] == split))
                                      for split in ("train","val")},
        "reference_download_bytes":{split: detail["network_body_bytes"] for split, detail in external.items()},
        "non_normal_exif_orientation":{p: i["orientation"] for p,i in images.items() if i["orientation"] != 1},
        "images_triggering_official_loader_half_size": [p for p,i in images.items() if max(i["size"]) > 1800],
        "width_height_ranges": [[min(i["size"][j] for i in images.values()),max(i["size"][j] for i in images.values())] for j in (0,1)],
        "dimension_mismatches":{p:i for p,i in images.items() if i["reference_status"] == "dimension_mismatch"},
        "outside_canvas_count": len(anomaly), "outside_canvas_by_split": dict(Counter(r["split"] for r in anomaly)),
        "outside_canvas_sides": dict(Counter(side for r in anomaly for side,v in r["canvas_excess"].items() if v)),
        "overflow_canvas_range":[min(max(r["canvas_excess"].values()) for r in anomaly),max(max(r["canvas_excess"].values()) for r in anomaly)],
        "outside_canvas_reference_near_same_edge":sum(r.get("reference_same_edge_within_3pct",False) for r in anomaly),
        "outside_canvas_with_reference":len([r for r in anomaly if r["reference_candidates"]]),
        "outside_original_image_raw_count":sum(any(r["raw_excess"].values()) for r in rows),
        "max_mapped_overflow_px":max(max(r["mapped_256_excess_px"].values()) for r in anomaly),
        "scaled_better_than_raw":sum(r["iou"]["square_256"]>r["iou"]["raw"] for r in comparable),
        "scaled_worse_than_raw":[r for r in comparable if r["iou"]["square_256"]<r["iou"]["raw"]],
        "clipping_changes_on_reference":len(clipped_change),
        "clipping_improves_on_reference":sum(r["iou"]["square_256_clipped"]>r["iou"]["square_256"] for r in clipped_change),
        "training_fit_square":train_canvas,"training_fit_grid":training_grid,
        "ious":metrics,"visualized":prepare_visuals(images, rows, data_root, output_dir),
    }
    save_distributions(subsets, train_canvas, output_dir)
    json_write(output_dir/"summary.json", stats)
    json_write(output_dir/"anomalies.json", anomaly)
    json_write(output_dir/"image_reference_checks.json", images)
    with (output_dir/"per_box.csv").open("w",encoding="utf-8-sig", newline="") as csv_file:
        columns = ["path","split","object_index","object","image_size","ocl_xyxy","outside_256","reference_status","reference_box_count"]
        modes = list(rows[0]["iou"])
        writer = csv.DictWriter(csv_file, fieldnames=columns + [f"iou_{name}" for name in modes])
        writer.writeheader()
        for row in rows:
            output = {key:row[key] for key in columns}
            output.update({f"iou_{mode}":round(value,6) if value is not None else None for mode,value in row["iou"].items()})
            writer.writerow(output)
    for name in ("reference_images_by_split","outside_canvas_count","outside_canvas_by_split",
                 "outside_canvas_reference_near_same_edge","outside_canvas_with_reference","scaled_better_than_raw",
                 "clipping_changes_on_reference","clipping_improves_on_reference","training_fit_square"):
        print(name,stats[name])
    for group in ("all","train","val","outside_256","single_reference"):
        print(group,{name: (s["n"],s.get("median")) for name,s in metrics[group].items()})
    print("Representative outliers:")
    for row in anomaly:
        if row["reference_candidates"]:
            print(row["path"],row["ocl_xyxy"],row["image_size"],row["reference_boundary_gap_px"],
                  round(row["iou"]["raw"],3),round(row["iou"]["square_256"],3),round(row["iou"]["square_256_clipped"],3))


if __name__ == "__main__":
    main()
