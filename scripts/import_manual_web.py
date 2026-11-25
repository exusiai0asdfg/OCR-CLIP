"""Import verified Windows browser downloads into the OCL Web manifest offline."""

import argparse
from collections import Counter, defaultdict
from datetime import datetime
from hashlib import sha256
import json
import os
from pathlib import Path, PurePosixPath
import re
import tempfile
from urllib.parse import unquote, urlsplit, urlunsplit

from audit_dataset import check_box, csv_output, load_selection
from download_imagenet import atomic_write
from download_web import (
    IMAGE_ERRORS, RESOURCE, ROOT, image_info, load_inputs, read_json, same_url, url_ok, write_json,
)


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".avif", ".bmp", ".tif", ".tiff", ".jfif"}
# Observed in the Downloads origin metadata after the human marked this item successful.
OBSERVED_MIGRATION = {"from_web/test_0766.jpg": ("www.ybxww.com", "past.ybxww.com")}


def download_urls(path):
    try:
        text = Path(f"{path}:Zone.Identifier").read_text(encoding="utf-8-sig")
    except (OSError, UnicodeError):
        return {}
    fields = {}
    for line in text.splitlines():
        key, _, raw = line.partition("=")
        url = raw.strip("\x00 \t")
        if key in {"HostUrl", "ReferrerUrl"} and url_ok(url):
            fields[key] = url
    return fields


def match_origin(path, observed, mapped):
    if not observed:
        return None
    if same_url(observed, mapped):
        return "browser_origin_matches_official_url"
    hosts = OBSERVED_MIGRATION.get(path)
    parsed = urlsplit(mapped)
    if hosts and parsed.hostname == hosts[0]:
        migrated = urlunsplit(parsed._replace(netloc=hosts[1]))
        if same_url(observed, migrated):
            return "human_confirmed_same_site_migration_exact_resource_path"
    return None


def copy_new(target, body):
    # Publish a complete file without replacing an existing destination.
    with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as tmp:
        tmp.write(body)
    temporary = Path(tmp.name)
    try:
        if os.name == "nt":
            temporary.rename(target)
        else:
            os.link(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def scan(downloads, since, paths, urls, progress, baseline):
    names = defaultdict(set)
    candidates = defaultdict(list)
    unmatched = []
    pending = {name for name in paths if baseline["files"][name]["status"] != "available"}
    approved = {name for name, status in progress.items() if status == "success"}
    for name in pending | approved:
        names[PurePosixPath(name).name.casefold()].add(name)
        for url in urls[name]:
            names[PurePosixPath(unquote(urlsplit(url).path)).name.casefold()].add(name)
    for file in sorted(downloads.iterdir()):
        if not file.is_file():
            continue
        normalized = re.sub(r" \(\d+\)(?=\.[^.]+$)", "", file.name.casefold())
        by_name = names.get(normalized, set())
        recent = since is not None and file.suffix.lower() in IMAGE_SUFFIXES and file.stat().st_mtime >= since
        if not by_name and not recent:
            continue
        origin = download_urls(file)
        possible = by_name or approved
        by_origin = [
            (name, url, evidence)
            for name in sorted(possible)
            for url in urls[name]
            if (evidence := match_origin(name, origin.get("HostUrl"), url))
        ]
        matches = {name for name, _, _ in by_origin}
        if len(matches) == 1 and matches <= approved:
            name, url, evidence = by_origin[0]
            candidates[name].append((file, url, origin["HostUrl"], evidence))
        else:
            unmatched.append({
                "filename": file.name, "possible_targets": sorted(matches or by_name),
                "reason": "ambiguous_origin" if len(matches) > 1 else
                "no_matching_success_mark" if matches else "no_matching_browser_origin",
            })
    return approved, candidates, unmatched


def update_web_report(data, targets, urls, objects, progress, report, result):
    missing = [name for name in targets if report["files"][name]["status"] != "available"]
    missing_paths = set(missing)
    available = [report["files"][name] for name in targets if name not in missing_paths]
    sources = Counter(entry.get("source") for entry in available)
    categories = Counter()
    reasons = Counter()
    rows = []
    for name in missing:
        entry = report["files"][name]
        boxes = Counter(obj["obj"] for _, obj in objects[name])
        categories.update(boxes)
        state = progress.get(name, "unmarked")
        reason = {
            "invalid": "manually_marked_invalid",
            "skipped": "manually_skipped",
            "success": "marked_success_but_download_unverified",
        }.get(state, "not_recovered_at_close")
        reasons[reason] += 1
        entry.update(manual_status=state, closure_reason=reason)
        rows.append({
            "path": name, "mapped_url": " | ".join(urls[name]), "reason": reason,
            "object_count": sum(boxes.values()), "categories": dict(boxes),
            "evidence": entry.get("attempts", []), "manual_status": state,
            "previous_reason": entry.get("reason", ""),
        })
    report.update(
        status="manual_recovery_closed", available=len(available),
        coverage=len(available) / len(targets), missing_count=len(missing), missing=missing,
        available_image_bytes=sum(entry["bytes"] for entry in available),
        counts_by_source=dict(sources),
        missing_instances=sum(categories.values()), missing_instances_by_category=dict(categories),
        missing_images_by_reason=dict(reasons), additional_network_recovery_attempted_this_run=False,
        historical_bbox_report=report.get("historical_bbox_report", report.get("bbox_report")),
        bbox_report="final_audit/summary.json",
        manual_recovery={
            "report": "final_audit/manual_web_import.json",
            "confirmed": sources["manual_browser"],
            "unmatched_downloads": len(result["unmatched_downloads"]),
            "manual_status_counts": dict(Counter(progress.values())),
            "internet_recovery_closed": True,
        },
    )
    write_json(data / "web_download_report.json", report)
    atomic_write(data / "web_missing.txt", "".join(f"{name}\n" for name in missing).encode("utf-8"))
    csv_output(data / "web_missing.csv", rows, (
        "path", "mapped_url", "reason", "object_count", "categories", "evidence", "manual_status", "previous_reason",
    ))


def run(args):
    data = args.data
    output = data / "final_audit"
    snapshot = read_json(args.progress or output / "manual_browser_progress.json")
    progress = snapshot["statuses"]
    targets, urls = load_inputs(data / "image_manifest.json", args.resources / "url_to_filename.json")
    if not set(progress).issubset(targets) or not set(progress.values()).issubset({"success", "invalid", "skipped"}):
        raise ValueError("Unexpected path or status in the offline recovery snapshot.")
    _, objects, _, _ = load_selection(data, args.resources)
    report = read_json(data / "web_download_report.json")
    since = args.since.timestamp() if args.since else None
    approved, candidates, unmatched = scan(args.downloads, since, targets, urls, progress, report)
    result = {
        "network_requests": 0, "apply": args.apply, "manual_mark_counts": dict(Counter(progress.values())),
        "confirmed": [], "unmatched_downloads": unmatched, "success_marks_without_verified_file": [],
    }
    for name in sorted(approved):
        matched = candidates[name]
        if len(matched) != 1:
            result["success_marks_without_verified_file"].append({
                "path": name, "reason": "multiple_matching_downloads" if matched else "no_matching_download",
                "candidate_filenames": [row[0].name for row in matched],
            })
            continue
        file, url, origin, method = matched[0]
        try:
            body = file.read_bytes()
            meta = image_info(body)
        except IMAGE_ERRORS as exc:
            unmatched.append({"path": name, "filename": file.name, "reason": "invalid_image", "error": str(exc)})
            result["success_marks_without_verified_file"].append({"path": name, "reason": "invalid_image"})
            continue
        target = data / name
        if target.exists():
            try:
                existing_digest = sha256(target.read_bytes()).hexdigest()
            except OSError:
                existing_digest = None
            if existing_digest != meta["sha256"]:
                unmatched.append({"path": name, "filename": file.name, "reason": "different_or_unreadable_existing_file"})
                result["success_marks_without_verified_file"].append({
                    "path": name, "reason": "different_or_unreadable_existing_file",
                })
                continue
        boxes = [
            {"object_index": index, "category": obj["obj"], "xyxy": obj.get("box"),
             **check_box(obj.get("box"), "from_web", (meta["width"], meta["height"]))}
            for index, obj in objects["from_web"][name]
        ]
        details = {"path": name, "download_filename": file.name, "mapped_url": url, "browser_host_url": origin,
                   "match_evidence": method, **meta}
        result["confirmed"].append({**details, "boxes": boxes})
        if args.apply:
            if not target.exists():
                copy_new(target, body)
            entry = report["files"][name]
            if entry.get("status") != "available":
                entry["previous_reason"] = entry.pop("reason", None)
                entry.pop("closure_reason", None)
                entry.update(status="available", source="manual_browser", manual_status="success", **details)
    if args.apply:
        output.mkdir(parents=True, exist_ok=True)
        write_json(output / "manual_web_import.json", result)
        update_web_report(data, targets, urls, objects["from_web"], progress, report, result)
    print(json.dumps({
        "confirmed": len(result["confirmed"]),
        "unmatched_downloads": len(unmatched),
        "success_marks_without_verified_file": len(result["success_marks_without_verified_file"]),
        "boxes_with_clipping": sum(b.get("clipped", False) for r in result["confirmed"] for b in r["boxes"]),
        "boxes_with_severe_issue": sum("issue" in b for r in result["confirmed"] for b in r["boxes"]),
    }, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    parser.add_argument("--resources", type=Path, default=RESOURCE)
    parser.add_argument("--downloads", type=Path, default=Path.home() / "Downloads")
    parser.add_argument("--progress", type=Path, help="Snapshot of offline recovery progress, with a statuses object.")
    parser.add_argument("--since", type=datetime.fromisoformat, help="List unmatched image downloads since a local ISO timestamp.")
    parser.add_argument("--apply", action="store_true", help="Archive the confirmed bytes and refresh Web provenance and missing lists.")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
