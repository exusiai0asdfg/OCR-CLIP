"""Download the original COCO 2014 JPEGs selected by the OCL image manifest."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack
from pathlib import Path, PurePosixPath
from queue import SimpleQueue
import re
from urllib.parse import urlsplit

import requests
from tqdm import tqdm

from download_web import (
    FetchError, IMAGE_ERRORS, MAX_IMAGE_BYTES, ROOT, atomic_write,
    fetch, image_info, read_json, same_url, write_json,
)


COCO_PATH = re.compile(r"COCO/((?:train|val)2014)/COCO_\1_\d{12}\.jpg")


def get_coco_url(name):
    path = PurePosixPath(name)
    return f"http://images.cocodataset.org/{path.parent.name}/{path.name}"


def source_urls(name):
    canonical = get_coco_url(name)
    # The official image host is a DNS alias for this public S3 bucket.
    resource = urlsplit(canonical).path
    return (f"https://s3.amazonaws.com/images.cocodataset.org{resource}", canonical)


def manifest_targets(manifest, root):
    targets = read_json(manifest)["COCO"]
    if not isinstance(targets, list) or not targets:
        raise ValueError("Expected a nonempty COCO list in the manifest.")
    if any(not isinstance(name, str) or not COCO_PATH.fullmatch(name) for name in targets):
        raise ValueError("COCO paths must use the matched 2014 split and original JPEG filename.")
    if len(set(targets)) != len(targets):
        raise ValueError("Duplicate COCO paths in the manifest.")
    for name in targets:
        if not (root / name).resolve().is_relative_to(root):
            raise ValueError(f"COCO path escapes the data root: {name}")
    return targets


def jpeg_info(body):
    info = image_info(body)
    if info["format"] != "JPEG" or info["frames"] != 1:
        raise ValueError("COCO image payload is not a single JPEG.")
    return info


def existing_image(target, previous):
    if not target.is_file():
        return None, "missing"
    if previous.get("status") != "available":
        previous = {}
    try:
        if target.stat().st_size > MAX_IMAGE_BYTES:
            return None, "oversized_local_file"
        info = jpeg_info(target.read_bytes())
    except IMAGE_ERRORS as error:
        return None, f"invalid_local_image:{type(error).__name__}"
    if previous.get("sha256") and previous["sha256"] != info["sha256"]:
        return None, "local_sha256_mismatch"
    return {"source": "existing_local", **previous, **info, "status": "available"}, None


def download_image(name, output, sessions, local_state):
    attempts = []
    session = sessions.get()
    try:
        for url in source_urls(name):
            try:
                body, resolved = fetch(session, url, limit=MAX_IMAGE_BYTES, retries=2, image=True)
                if not same_url(url, resolved):
                    raise FetchError("redirect_changed_resource")
                try:
                    info = jpeg_info(body)
                except IMAGE_ERRORS as error:
                    raise FetchError(f"invalid_jpeg:{type(error).__name__}") from error
            except FetchError as error:
                attempts.append({"url": url, "reason": error.reason})
                continue
            try:
                atomic_write(output, body)
            except OSError as error:
                attempts.append({"url": url, "reason": f"cannot_save:{type(error).__name__}"})
                break
            return {
                **info, "status": "available", "source": "coco_official",
                "canonical_url": get_coco_url(name), "requested_url": url, "resolved_url": resolved,
                "attempts": attempts,
                "previous_local_state": local_state,
            }
    finally:
        sessions.put(session)
    return {"status": "missing", "reason": attempts[0]["reason"], "attempts": attempts,
            "previous_local_state": local_state}


def save_report(path, targets, names, files, downloaded, skipped):
    available = [row for row in files.values() if row["status"] == "available"]
    missing = [name for name in names if files.get(name, {}).get("status") == "missing"]
    report = {
        "source": "images.cocodataset.org",
        "manifest_coco_targets": len(targets),
        "selected": len(names),
        "processed": len(files),
        "available": len(available),
        "downloaded_this_run": downloaded,
        "skipped_existing": skipped,
        "missing_count": len(missing),
        "unprocessed_count": len(names) - len(files),
        "missing": missing,
        "failures_by_reason": dict(Counter(files[name]["reason"] for name in missing)),
        "available_image_bytes": sum(row["bytes"] for row in available),
        "downloaded_image_bytes": sum(row["bytes"] for row in available if row["run_action"] == "downloaded"),
        "files": files,
    }
    write_json(path, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/image_manifest.json")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--workers", type=int, choices=range(1, 17), default=8, metavar="1..16")
    parser.add_argument("--limit", type=int, help="Process the first N targets from each split as a smoke test.")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive.")
    root = (args.data_root or args.manifest.parent).resolve()
    targets = manifest_targets(args.manifest, root)
    names = targets
    if args.limit is not None:
        names = [
            name for split in ("train2014", "val2014")
            for name in [path for path in targets if PurePosixPath(path).parent.name == split][:args.limit]
        ]
    report_path = root / ("coco_smoke_report.json" if args.limit else "coco_download_report.json")
    prior = {}
    for previous_path in (root / "coco_smoke_report.json", root / "coco_download_report.json"):
        if previous_path.is_file():
            prior.update(read_json(previous_path)["files"])

    files, pending = {}, {}
    for name in names:
        stored, local_state = existing_image(root / name, prior.get(name, {}))
        if stored:
            files[name] = {**stored, "run_action": "skipped"}
        else:
            pending[name] = local_state
    skipped = len(files)
    downloaded = 0
    with ExitStack() as stack, tqdm(total=len(names), initial=skipped, unit="image", desc="COCO", mininterval=1) as bar:
        if pending:
            sessions = SimpleQueue()
            for _ in range(min(args.workers, len(pending))):
                sessions.put(stack.enter_context(requests.Session()))
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures = {
                    pool.submit(download_image, name, root / name, sessions, local_state): name
                    for name, local_state in pending.items()
                }
                try:
                    for index, future in enumerate(as_completed(futures), 1):
                        name = futures[future]
                        row = future.result()
                        row["run_action"] = "downloaded" if row["status"] == "available" else "failed"
                        files[name] = row
                        downloaded += row["status"] == "available"
                        bar.update()
                        if index % 100 == 0:
                            bar.set_postfix(downloaded=downloaded, failed=index - downloaded)
                            save_report(report_path, targets, names, files, downloaded, skipped)
                except KeyboardInterrupt:
                    for future in futures:
                        future.cancel()
                    save_report(report_path, targets, names, files, downloaded, skipped)
                    raise
    report = save_report(report_path, targets, names, files, downloaded, skipped)
    print(f"COCO: {report['available']}/{len(names)} available, {downloaded} downloaded, {skipped} skipped, "
          f"{report['missing_count']} failed. Report: {report_path}", flush=True)
    return int(bool(report["missing_count"]))


if __name__ == "__main__":
    raise SystemExit(main())
