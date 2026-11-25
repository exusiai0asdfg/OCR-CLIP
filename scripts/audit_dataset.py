"""Audit the OCL Top 10 manifest and export usable image and instance lists."""

import argparse
from collections import Counter, defaultdict
import csv
from hashlib import sha256
import io
import json
import math
from pathlib import Path, PurePosixPath
import pickle
import re

from PIL import Image

from download_web import IMAGE_ERRORS, RESOURCE, ROOT, image_info, read_json, write_json
from download_imagenet import atomic_write


SOURCES = ("ImageNet", "COCO", "from_web")
PATH_PATTERNS = {
    "ImageNet": re.compile(r"ImageNet/(?:train/(n\d{8})/\1_\d+|val/ILSVRC2012_val_\d{8})\.JPEG"),
    "COCO": re.compile(r"COCO/((?:train|val)2014)/COCO_\1_\d{12}\.jpg"),
    "from_web": re.compile(r"from_web/test_\d+\.(?:jpg|jpeg|png)"),
}
FORMATS = {".jpeg": "JPEG", ".jpg": "JPEG", ".png": "PNG"}
# The previous overlay review found displaced boxes in these exact renditions.
DISPLACED_WEB = {
    "from_web/test_0374.jpg": "b9d87e1e4639afac461f9df786a9adc07f84fecbb8ded6d7bf75f0b906a10372",
    "from_web/test_0411.jpg": "0bef29aa99ec3f674982c60043f9af149fb582c638cdef469472320aaa8bddc1",
}
BOX_RULES = {
    "ImageNet": "XYXY on 256×256, independently scale x by W/256 and y by H/256.",
    "COCO": "OCL XYXY in original pixels. The native COCO API uses XYWH.",
    "from_web": "OCL XYXY in original pixels. Exclude known displaced renditions.",
    "clipping": "ImageNet up to 6 px on the 256 canvas; COCO/Web up to max(3 px, 2% of the corresponding axis). Floor the clipped start, ceil the clipped end.",
}
COCO_API = "https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/coco.py#L267-L269"


class AnnotationReader(pickle.Unpickler):
    def find_class(self, module, name):
        raise ValueError(f"Unexpected OCL pickle object {module}.{name}")


def label_issues(obj, sizes):
    return [
        f"invalid_{key}_indices"
        for key, size in sizes.items()
        if not isinstance(obj.get(key), (list, tuple))
        or any(type(i) is not int or not 0 <= i < size for i in obj[key])
    ]


def load_selection(data, resources):
    manifest_bytes = (data / "image_manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes.decode("utf-8-sig"))
    annotation_bytes = (resources / "OCL_annot_test.pkl").read_bytes()
    rows = AnnotationReader(io.BytesIO(annotation_bytes)).load()
    categories = Counter(obj["obj"] for row in rows for obj in row["objects"])
    top = dict(categories.most_common(10))
    selected = {source: {} for source in SOURCES}
    sizes = {
        key: len(read_json(resources / f"OCL_class_{name}.json"))
        for key, name in (("attr", "attribute"), ("aff", "affordance"))
    }
    release_labels = Counter()
    release_label_errors = []
    for row in rows:
        path, source = row["name"], row["source"]
        objects = []
        for index, obj in enumerate(row["objects"]):
            errors = label_issues(obj, sizes)
            release_labels.update(errors)
            if errors:
                release_label_errors.append({"path": path, "object_index": index, "issues": errors})
            if obj["obj"] in top:
                objects.append((index, obj))
        if not objects:
            continue
        if source not in selected or not PATH_PATTERNS[source].fullmatch(path):
            raise ValueError(f"Unexpected OCL source/path: {source}: {path}")
        if path in selected[source]:
            raise ValueError(f"Duplicate OCL annotation path: {path}")
        selected[source][path] = objects
    if set(manifest) != set(SOURCES):
        raise ValueError(f"Unexpected manifest sources: {list(manifest)}")
    for source in SOURCES:
        paths = manifest[source]
        if len(set(paths)) != len(paths) or set(paths) != set(selected[source]):
            delta = sorted(set(paths) ^ set(selected[source]))
            raise ValueError(f"{source} manifest differs from the OCL Top 10 paths: {delta[:12]}")
    evidence = {
        "manifest_sha256": sha256(manifest_bytes).hexdigest(),
        "annotation_sha256": sha256(annotation_bytes).hexdigest(),
        "vocabulary_sizes": sizes,
        "release_instances_checked_for_labels": sum(categories.values()),
        "release_label_issue_counts": dict(release_labels),
        "release_label_issues": release_label_errors,
        "manifest_matches_ocl_top10": True,
    }
    return manifest, selected, top, evidence


def raw_box(box):
    if (
        not isinstance(box, (list, tuple))
        or len(box) != 4
        or any(type(x) not in (int, float) or not math.isfinite(x) for x in box)
        or box[0] >= box[2]
        or box[1] >= box[3]
    ):
        return None
    return list(box)


def check_box(box, source, size):
    xyxy = raw_box(box)
    if xyxy is None:
        return {"issue": "invalid_raw_box"}
    if size is None:
        return {}
    w, h = size
    axes = (w, h, w, h)
    mapped = [x * (axis / 256) for x, axis in zip(xyxy, axes)] if source == "ImageNet" else xyxy
    x0, y0, x1, y1 = mapped
    overflow = [max(0, -x0), max(0, -y0), max(0, x1 - w), max(0, y1 - h)]
    limit = [axis * 6 / 256 if source == "ImageNet" else max(3, axis * .02) for axis in axes]
    result = {"mapped_xyxy": mapped, "overflow_px": overflow}
    clipped = [max(0, x0), max(0, y0), min(w, x1), min(h, y1)]
    if clipped[0] >= clipped[2] or clipped[1] >= clipped[3]:
        result["issue"] = "empty_crop"
    elif any(over > allowance + 1e-5 for over, allowance in zip(overflow, limit)):
        result["issue"] = "severe_overflow"
    else:
        result["crop_xyxy"] = [math.floor(x) for x in clipped[:2]] + [math.ceil(x) for x in clipped[2:]]
        result["clipped"] = any(x > 1e-5 for x in overflow)
    return result


def read_image(path):
    if not path.exists():
        return {}, "missing_image"
    try:
        body = path.read_bytes()
        details = image_info(body)
        with Image.open(io.BytesIO(body)) as image:
            pixels = sha256(f"{image.width}x{image.height}:RGBA:".encode("ascii"))
            for frame in range(details["frames"]):
                image.seek(frame)
                pixels.update(image.convert("RGBA").tobytes())
            details["pixel_sha256"] = pixels.hexdigest()
        return details, None
    except IMAGE_ERRORS as error:
        return {"decode_error": f"{type(error).__name__}: {error}"}, "corrupt_or_unreadable_image"


def csv_output(path, rows, columns):
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({
            field: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value
            for field, value in row.items()
        })
    atomic_write(path, output.getvalue().encode("utf-8-sig"))


def image_warnings(path, source, meta, provenance):
    warnings = []
    if meta["format"] != FORMATS.get(PurePosixPath(path).suffix.lower()):
        warnings.append("extension_format_mismatch")
    if meta["exif_orientation"] != 1:
        warnings.append("nonstandard_exif_orientation")
    if meta["frames"] > 1:
        warnings.append("animated_image")
    if source in ("ImageNet", "from_web") and not provenance.get("sha256"):
        warnings.append("unrecorded_provenance")
    elif provenance.get("sha256") and provenance["sha256"] != meta["sha256"]:
        warnings.append("provenance_sha256_mismatch")
    if DISPLACED_WEB.get(path) == meta["sha256"]:
        warnings.append("known_spatial_mismatch")
    return warnings


def audit(data, resources, output):
    manifest, selected, top, evidence = load_selection(data, resources)
    provenance = {}
    for name in ("imagenet_download_report.json", "web_download_report.json", "coco_download_report.json"):
        if (data / name).is_file():
            provenance.update(read_json(data / name)["files"])

    stats = {source: Counter() for source in SOURCES}
    format_counts = {source: Counter() for source in SOURCES}
    issues = {source: Counter() for source in SOURCES}
    valid_images = {source: [] for source in SOURCES}
    counts = Counter()
    assets, missing, anomalies, valid_instances = [], [], [], []
    digests = defaultdict(list)
    pixel_digests = defaultdict(list)
    coco_comparison = []
    coco_local_counts = Counter()
    coco_fit_combinations = Counter()
    coco_endpoint_shapes = Counter()
    for source in SOURCES:
        tally = stats[source]
        for path in manifest[source]:
            objects = selected[source][path]
            tally.update(expected_images=1, expected_instances=len(objects))
            meta, problem = read_image(data / path)
            if problem != "missing_image":
                tally["acquired_images"] += 1
            upstream = provenance.get(path, {})
            warnings = []
            if problem is None:
                tally.update(decoded_images=1, instances_on_decoded_images=len(objects))
                format_counts[source][meta["format"]] += 1
                digests[meta["sha256"]].append(path)
                pixel_digests[meta["pixel_sha256"]].append(path)
                warnings = image_warnings(path, source, meta, upstream)
            issues[source].update(warnings)
            if problem:
                issues[source][problem] += 1
            if problem == "missing_image":
                missing.append({
                    "source": source, "path": path, "instance_count": len(objects),
                    "categories": dict(Counter(obj["obj"] for _, obj in objects)),
                    "reason": upstream.get("closure_reason", upstream.get("reason", "absent_at_manifest_path")),
                })
                tally.update(missing_images=1, missing_instances=len(objects))
            elif problem:
                tally.update(corrupt_images=1, instances_on_corrupt_images=len(objects))

            valid_count = 0
            coordinate_issues_on_image = False
            size = (meta["width"], meta["height"]) if problem is None else None
            for index, obj in objects:
                labels = label_issues(obj, evidence["vocabulary_sizes"])
                checked = check_box(obj.get("box"), source, size)
                box_issue = checked.get("issue")
                blockers = [x for x in warnings if x in {
                    "unrecorded_provenance", "provenance_sha256_mismatch", "known_spatial_mismatch",
                }]
                if source == "COCO" and (box := raw_box(obj.get("box"))) is not None:
                    coco_endpoint_shapes["ordered_ocl_xyxy"] += 1
                    if max(box[2:]) > 256:
                        coco_endpoint_shapes["endpoint_above_256"] += 1
                    if size is not None:
                        x, y, third, fourth = box
                        xyxy_fits = 0 <= x < third <= size[0] and 0 <= y < fourth <= size[1]
                        xywh_fits = x >= 0 and y >= 0 and x + third <= size[0] and y + fourth <= size[1]
                        coco_local_counts.update(compared=1, xyxy_fits=xyxy_fits, xywh_fits=xywh_fits)
                        coco_fit_combinations[xyxy_fits, xywh_fits] += 1
                        if len(coco_comparison) < 5 and xyxy_fits != xywh_fits:
                            coco_comparison.append({
                                "path": path, "object_index": index, "size": size, "ocl_box": box,
                                "xyxy_fits": xyxy_fits, "xywh_fits": xywh_fits,
                            })
                if box_issue:
                    blockers.append(box_issue)
                    tally["coordinate_abnormal_instances"] += 1
                elif "known_spatial_mismatch" in blockers:
                    tally["coordinate_abnormal_instances"] += 1
                if box_issue or "known_spatial_mismatch" in blockers:
                    coordinate_issues_on_image = True
                blockers.extend(labels)
                issues[source].update(labels + ([box_issue] if box_issue else []))
                if checked.get("clipped"):
                    tally["mild_clipping_instances"] += 1
                detail = {
                    "path": path, "source": source, "object_index": index, "category": obj["obj"],
                    "raw_xyxy": obj.get("box"), **checked,
                }
                if blockers:
                    anomalies.append({**detail, "issues": blockers})
                if problem or blockers:
                    continue
                valid_count += 1
                counts[obj["obj"]] += 1
                valid_instances.append({
                    **detail, "width": size[0], "height": size[1], "sha256": meta["sha256"],
                    "attribute_ids": obj["attr"], "affordance_ids": obj["aff"],
                })
            tally["valid_instances"] += valid_count
            if valid_count:
                valid_images[source].append(path)
                tally["valid_images"] += 1
            elif problem is None:
                tally["decoded_without_valid_instance"] += 1
            if coordinate_issues_on_image:
                tally["images_with_coordinate_issue"] += 1
            status = "valid" if valid_count else "coordinate_abnormal" if coordinate_issues_on_image else "excluded"
            if warnings or (problem and problem != "missing_image"):
                anomalies.append({"path": path, "source": source, "issues": warnings or [problem], **meta})
            assets.append({
                "source": source, "path": path, "expected_instances": len(objects),
                "valid_instances": valid_count,
                "status": problem or status,
                "provenance": upstream.get("source") or ("ILSVRC/imagenet-1k" if upstream and source == "ImageNet" else ""),
                "warnings": warnings, **meta,
            })

    duplicates = [{"sha256": digest, "paths": paths} for digest, paths in digests.items() if len(paths) > 1]
    duplicate_pixels = [
        {"pixel_sha256": digest, "paths": paths} for digest, paths in pixel_digests.items() if len(paths) > 1
    ]
    extras = {
        source: sorted(
            p.relative_to(data).as_posix()
            for p in (data / source).rglob("*")
            if p.is_file() and p.relative_to(data).as_posix() not in selected[source]
        )
        for source in SOURCES
    }
    required_keys = (
        "expected_images", "acquired_images", "decoded_images", "valid_images", "missing_images",
        "corrupt_images", "decoded_without_valid_instance", "expected_instances", "instances_on_decoded_images",
        "instances_on_corrupt_images", "valid_instances", "missing_instances", "coordinate_abnormal_instances",
        "images_with_coordinate_issue", "mild_clipping_instances",
    )
    overall = {key: sum(stats[source][key] for source in SOURCES) for key in required_keys}
    summary = {
        "inputs": evidence,
        "by_source": {
            source: {**{key: stats[source][key] for key in required_keys},
                     "formats": dict(format_counts[source]), "issue_counts": dict(issues[source])}
            for source in SOURCES
        },
        "overall": {
            **overall,
            "acquired_image_coverage": overall["acquired_images"] / overall["expected_images"],
            "valid_image_coverage": overall["valid_images"] / overall["expected_images"],
            "valid_instance_coverage": overall["valid_instances"] / overall["expected_instances"],
        },
        "top10": {
            cat: {"expected_instances": expected, "valid_instances": counts[cat], "coverage": counts[cat] / expected}
            for cat, expected in top.items()
        },
        "bbox_rules": BOX_RULES,
        "bbox_evidence": {
            "ImageNet": "Existing imagenet_bbox_audit/summary.json. ILSVRC localization compared at original and scaled coordinates.",
            "from_web": "Existing web_bbox_audit and overlay review. Two Dreamstime renditions show displacement in bounds.",
            "COCO": {
                "native_api": COCO_API,
                "ocl_consumer": "ObjectConceptLearning/models/base_models.py:193-205 passes OCL boxes to MultiScaleRoIAlign as XYXY.",
                "target_geometry": dict(coco_endpoint_shapes),
                "local_fit_counts": dict(coco_local_counts),
                "format_comparison": {
                    "both_fit": coco_fit_combinations[True, True],
                    "xyxy_only": coco_fit_combinations[True, False],
                    "xywh_only": coco_fit_combinations[False, True],
                    "neither_fits": coco_fit_combinations[False, False],
                },
                "local_file_examples": coco_comparison,
            },
        },
        "same_sha256_groups": duplicates,
        "same_decoded_pixel_groups": duplicate_pixels,
        "extra_files_under_source_roots": extras,
    }
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "summary.json", summary)
    write_json(output / "valid_images.json", valid_images)
    write_json(output / "anomalies.json", anomalies)
    csv_output(output / "missing_images.csv", missing, ("source", "path", "instance_count", "categories", "reason"))
    csv_output(output / "images.csv", assets, (
        "source", "path", "status", "expected_instances", "valid_instances", "width", "height",
        "format", "frames", "exif_orientation", "bytes", "sha256", "pixel_sha256", "provenance", "warnings",
        "decode_error",
    ))
    csv_output(output / "valid_instances.csv", valid_instances, (
        "source", "path", "object_index", "category", "width", "height", "sha256", "raw_xyxy",
        "mapped_xyxy", "crop_xyxy", "clipped", "attribute_ids", "affordance_ids",
    ))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    parser.add_argument("--resources", type=Path, default=RESOURCE)
    args = parser.parse_args()
    report = audit(args.data, args.resources, args.data / "final_audit")
    print(json.dumps(report["overall"], indent=2))


if __name__ == "__main__":
    main()
