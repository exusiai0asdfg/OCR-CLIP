"""Check unmodified OCL Web images against the released boxes."""

import argparse
from collections import Counter, defaultdict
import csv
from hashlib import sha256
import json
from pathlib import Path
import pickle
import random
import tarfile

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image


if __package__:
    from . import download_web as web
else:
    import download_web as web

SEED = 20261125


class AnnotationReader(pickle.Unpickler):
    def find_class(self, module, name):
        raise ValueError(f"Unexpected pickle type {module}.{name}")


def quantiles(values):
    if not len(values):
        return None
    return dict(zip(("min","p10","median","p90","max"), np.percentile(values,[0,10,50,90,100]).round(3).tolist()))


def get_annotations(targets, resources):
    annotation_file = resources / "OCL_annot_test.pkl"
    digest = sha256(annotation_file.read_bytes()).hexdigest()
    with tarfile.open(resources/"OCL_annot.tgz","r:gz") as source:
        if sha256(source.extractfile("OCL_annot_test.pkl").read()).hexdigest() != digest:
            raise ValueError("Local OCL test annotations differ from the release archive.")
    with annotation_file.open("rb") as file:
        rows = {r["name"]:r["objects"] for r in AnnotationReader(file).load() if r.get("source") == "from_web"}
    return {name:rows[name] for name in targets}, digest


def overflow(box, size):
    x0,y0,x1,y1 = box
    w,h = size
    return [max(0,-x0),max(0,-y0),max(0,x1-w),max(0,y1-h)]


def collect(annotations, report, data, use_verified_metadata=False):
    rows, images = [], {}
    for name, objects in annotations.items():
        entry = report["files"][name]
        available = entry["status"] == "available"
        size = None
        if available:
            if use_verified_metadata:
                if (data/name).stat().st_size != entry["bytes"]:
                    raise ValueError(f"Image size differs from download report: {name}")
                details = entry
            else:
                details = web.image_info((data/name).read_bytes())
                if details["sha256"] != entry["sha256"]:
                    raise ValueError(f"Image differs from download report: {name}")
            size = (details["width"], details["height"])
            images[name] = {**details,"source":entry["source"],"box_count":len(objects)}
        for index,obj in enumerate(objects):
            box = obj.get("box")
            over = overflow(box,size) if box and size else None
            outside = max(over) if over else None
            big = bool(over and any(v>max(3,length*.02) for v,length in zip(over, size*2)))
            rows.append({
                "path":name,"index":index,"category":obj["obj"],"available":available,
                "width":size[0] if size else None, "height":size[1] if size else None,
                "xyxy":box, "within_bounds":outside<=1e-5 if outside is not None else None,
                "overflow_ltrb":over,"significant_overflow":big,
                "invalid_area": bool(box and (box[2]<=box[0] or box[3]<=box[1])),
            })
    return rows, images


def diagnostic(rows, img, selected, filename, title, data, output, *, columns=4):
    plots = (len(selected)+columns-1)//columns
    fig, axes = plt.subplots(plots,columns,figsize=(4.05*columns,4*plots),squeeze=False,layout="constrained")
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["path"]].append(row)
    for ax in axes.flat:
        ax.axis("off")
    for ax,name in zip(axes.flat,selected):
        with Image.open(data/name) as source:
            ax.imshow(np.asarray(source.convert("RGB")))
        width,height = img[name]["width"],img[name]["height"]
        cases = grouped[name]
        outside = max((max(r["overflow_ltrb"] or [0]) for r in cases),default=0)
        for row in cases:
            if row["xyxy"] is None:
                continue
            a,b,c,d = row["xyxy"]
            color = "#caff25" if row["within_bounds"] else "#ff5171"
            ax.add_patch(Rectangle((a,b),c-a,d-b,fill=False,color=color,linewidth=1.55))
        right = max((r["overflow_ltrb"][2] if r["overflow_ltrb"] else 0 for r in cases),default=0)
        bottom = max((r["overflow_ltrb"][3] if r["overflow_ltrb"] else 0 for r in cases),default=0)
        ax.set_xlim(-0.5,width+min(right,width*.10))
        ax.set_ylim(height+min(bottom,height*.10),-0.5)
        extra = f"  ·  overflow {outside:.1f} px" if outside > 1e-5 else ""
        categories = ", ".join(sorted({r["category"] for r in cases}))
        ax.set_title(f"{Path(name).name}  ·  {width} × {height}\n{categories}{extra}",fontsize=9)
    fig.suptitle(title, fontsize=15, fontweight="bold")
    fig.legend(handles=[Line2D([0],[0],color="#78a100",label="OCL raw box within image"),
                        Line2D([0],[0],color="#ff5171",label="OCL raw box crossing image edge")],
               loc="outside lower center",ncol=2,frameon=False)
    fig.savefig(output/filename,dpi=145,bbox_inches="tight",facecolor="white")
    plt.close(fig)


def select_visuals(rows, images, data, output):
    rng = random.Random(SEED)
    random_set = rng.sample(sorted(images),min(12,len(images)))
    outside = [r for r in rows if r["available"] and r["within_bounds"] is False]
    extreme = sorted(outside,key=lambda r:(-max(r["overflow_ltrb"]), r["path"]))
    severe_names = list(dict.fromkeys(r["path"] for r in extreme if r["significant_overflow"]))
    selected = severe_names[:8]
    low = min(images,key=lambda p:(min(images[p]["width"],images[p]["height"]),p))
    high = min(images,key=lambda p:(-max(images[p]["width"],images[p]["height"]),p))
    selected = list(dict.fromkeys(selected+[low,high]+[r["path"] for r in extreme]))[:12]
    diagnostic(rows,images,random_set,"random_boxes.png","OCL Web images · random sample with raw boxes",data,output)
    diagnostic(rows,images,selected,"outlier_boxes.png","OCL Web images · edge crossings and resolution extremes",data,output)
    return {"seed":SEED,"random":random_set,"outliers":selected,"smallest_short_edge_path":low,"largest_long_edge_path":high,
            "severe_paths":severe_names}


def write_missing(report, annotations, mapping, data):
    missing_names = report["missing"]
    assert set(missing_names)=={n for n,r in report["files"].items() if r["status"]=="missing"}
    categories = Counter()
    unlisted_categories = Counter()
    reasons = Counter()
    web.atomic_write(data/"web_missing.txt",("\n".join(missing_names)+"\n" if missing_names else "").encode())
    with (data/"web_missing.csv").open("w",newline="",encoding="utf-8-sig") as f:
        fieldnames = ["path","mapped_url","reason","object_count","categories","evidence"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for name in missing_names:
            objects = Counter(o["obj"] for o in annotations[name])
            attempts = report["files"][name]["attempts"]
            category = "no_official_url" if name not in mapping else "mapped_source_and_recovery_unsuccessful"
            if name not in mapping:
                unlisted_categories.update(objects)
            categories.update(objects)
            reasons[category]+=1
            writer.writerow({"path":name,"mapped_url":" | ".join(mapping.get(name,[])),"reason":category,
                             "object_count":sum(objects.values()),"categories":json.dumps(dict(objects),ensure_ascii=False),
                             "evidence":json.dumps(attempts,ensure_ascii=False)})
    return {"counts_by_reason":dict(reasons),"images":len(missing_names),"objects":sum(categories.values()),
            "categories":dict(categories.most_common()),"categories_no_official_url":dict(unlisted_categories.most_common())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--use-verified-metadata", action="store_true", help="Use image metadata from a just-completed verified download.")
    parser.add_argument("--data-root", type=Path, default=web.ROOT / "data")
    parser.add_argument("--resources", type=Path)
    args = parser.parse_args()
    data = args.data_root
    resources = args.resources or data / "resources"
    output = data / "web_bbox_audit"
    output.mkdir(parents=True, exist_ok=True)
    targets,mapping = web.load_inputs(data/"image_manifest.json",resources/"url_to_filename.json")
    report = web.read_json(data/"web_download_report.json")
    annotations,digest = get_annotations(targets, resources)
    rows,images = collect(annotations,report,data,args.use_verified_metadata)
    assert set(images)=={p.relative_to(data).as_posix() for p in (data/"from_web").iterdir() if p.is_file()}
    missing = write_missing(report,annotations,mapping,data)
    available = [r for r in rows if r["available"]]
    boxes = [r for r in available if r["xyxy"] is not None]
    objects = [r for r in rows if r["xyxy"] is not None]
    outliers = [r for r in boxes if r["within_bounds"] is False]
    severe = [r for r in outliers if r["significant_overflow"]]
    visualized = None if args.no_plots else select_visuals(boxes,images,data,output)
    changed = [r for r in boxes if report["files"][r["path"]].get("archive_action") in ("added","replaced")]
    changed_outside = [r for r in changed if r["within_bounds"] is False]
    same_digest = defaultdict(list)
    for p, image in images.items():
        same_digest[image["sha256"]].append(p)
    summary = {
        "annotations_sha256":digest,"manifest_images":len(targets),"manifest_objects":len(rows),
        "available_images":len(images),"available_objects":len(available),"available_boxes":len(boxes),
        "images_by_source":dict(Counter(im["source"] for im in images.values())),
        "format_counts":dict(Counter(im["format"] for im in images.values())),
        "orientation_counts":dict(Counter(im["exif_orientation"] for im in images.values())),
        "dimensions":{k:quantiles([im[k] for im in images.values()]) for k in ("width","height")},
        "short_edge_below_160":[p for p,im in images.items() if min(im["width"],im["height"])<160],
        "long_edge_above_1800":[p for p,im in images.items() if max(im["width"],im["height"])>1800],
        "aspect_ratio_above_2_5":[p for p,im in images.items() if max(im["width"],im["height"])/min(im["width"],im["height"])>2.5],
        "annotation_ranges_all":{k:quantiles([r["xyxy"][i] for r in objects]) for i,k in enumerate(("x0","y0","x1","y1"))},
        "annotation_ranges_available":{k:quantiles([r["xyxy"][i] for r in boxes]) for i,k in enumerate(("x0","y0","x1","y1"))},
        "coordinate_above_256_all":sum(max(r["xyxy"])>256 for r in objects),
        "coordinate_above_256_available":sum(max(r["xyxy"])>256 for r in boxes),
        "within_raw_image_bounds":sum(r["within_bounds"] for r in boxes),
        "invalid_box_area_all":sum(r["invalid_area"] for r in rows),
        "out_of_bounds_boxes":len(outliers),"out_of_bounds_images":len({r["path"] for r in outliers}),
        "significant_overflow_boxes":len(severe),"significant_overflow_images":len({r["path"] for r in severe}),
        "significant_rule":"An edge exceeds both 3 pixels and 2% of the corresponding image dimension.",
        "new_or_replaced_archive_images":len({r["path"] for r in changed}),
        "new_or_replaced_archive_boxes":len(changed),
        "new_or_replaced_archive_out_of_bounds_boxes":len(changed_outside),
        "new_or_replaced_archive_out_of_bounds_images":len({r["path"] for r in changed_outside}),
        "new_or_replaced_archive_dimensions":{k:quantiles([images[p][k] for p in {r["path"] for r in changed}])
                                               for k in ("width","height")},
        "visualization_status":"Previous PNGs cover the URL-era subset." if args.no_plots else "Rendered for current inventory.",
        "visualized":visualized,"missing":missing,
        "identical_byte_groups":[p for p in same_digest.values() if len(p)>1],
    }
    web.write_json(output/"summary.json",summary)
    web.write_json(output/"out_of_bounds.json",outliers)
    with (output/"boxes.csv").open("w",newline="",encoding="utf-8-sig") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({k:summary[k] for k in ("available_images","available_boxes","dimensions",
                      "within_raw_image_bounds","out_of_bounds_boxes","significant_overflow_boxes","missing",
                      "identical_byte_groups")},ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
