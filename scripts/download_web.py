"""Get OCL Web images from the release archive, original URLs or exact captures."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import gzip
from hashlib import sha256
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import sys
import tarfile
import tempfile
from threading import BoundedSemaphore
import time
from urllib.parse import unquote, urlsplit, urlunsplit

from PIL import Image
import requests

if __package__:
    from .download_imagenet import atomic_write, write_json
else:
    from download_imagenet import atomic_write, write_json


ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_FOLDER = "https://drive.google.com/drive/folders/1MDBiw45vN4qvjOA8ZuGmFhjMVoSBnF"
MAX_IMAGE_BYTES = 32 * 1024 * 1024
WAYBACK_API = "https://archive.org/wayback/available"
WAYBACK_CDX = "https://web.archive.org/cdx/search/cdx"
CAPTURE = re.compile(r"https?://web\.archive\.org/web/([0-9]{14})(?:[a-z]+_)?/(https?://.+)")
TRANSIENT_STATUS = {408, 429, 500, 502, 503, 504}
IMAGE_ERRORS = (OSError, ValueError, SyntaxError, EOFError, RuntimeError, Image.DecompressionBombError)


class FetchError(Exception):
    def __init__(self, reason, *, status=None, retryable=False):
        super().__init__(reason)
        self.reason = reason
        self.status = status
        self.retryable = retryable


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def load_inputs(manifest, mapping):
    targets = read_json(manifest)["from_web"]
    if not isinstance(targets, list) or not targets:
        raise ValueError("Expected a nonempty from_web list in the manifest.")
    for path in targets:
        if not isinstance(path, str) or not re.fullmatch(r"from_web/test_[0-9]+\.(?:jpg|jpeg|png)", path):
            raise ValueError(f"Invalid Web test manifest path: {path!r}")
    if len(set(targets)) != len(targets):
        raise ValueError("Duplicate Web image paths in the manifest.")
    selected = set(targets)
    index = defaultdict(list)
    for url, path in read_json(mapping):
        if path in selected and url not in index[path]:
            index[path].append(url)
    return targets, index


def image_info(data):
    if len(data) > MAX_IMAGE_BYTES:
        raise ValueError("Image exceeds 32 MiB.")
    with Image.open(io.BytesIO(data)) as im:
        width, height = im.size
        fmt = im.format
        if min(width,height) <= 2:
            raise ValueError(f"Placeholder-sized image: {width}×{height}.")
        im.verify()
    with Image.open(io.BytesIO(data)) as im:
        frames = getattr(im, "n_frames", 1)
        if frames > 128:
            raise ValueError("Image exceeds 128 animation frames.")
        for i in range(frames):
            im.seek(i)
            im.load()
        im.seek(0)
        orientation = im.getexif().get(274, 1)
    return {"width":width,"height":height,"format":fmt,"exif_orientation":orientation,
            "frames":frames,"bytes":len(data),"sha256":sha256(data).hexdigest()}


def url_ok(url):
    parsed = urlsplit(url)
    return parsed.scheme in {"http","https"} and parsed.hostname and not parsed.username and not parsed.password


def read_response(response, limit):
    size = response.headers.get("Content-Length", "")
    if size.isdecimal() and int(size) > limit:
        raise FetchError("body_too_large")
    data = bytearray()
    deadline = time.monotonic() + 30
    for chunk in response.iter_content(64 * 1024):
        data.extend(chunk)
        if len(data) > limit:
            raise FetchError("body_too_large")
        if time.monotonic() > deadline:
            raise FetchError("body_deadline", retryable=True)
    return bytes(data)


def fetch(session, url, *, limit, retries=0, image=False, params=None):
    if not url_ok(url):
        raise FetchError("unsupported_url")
    for attempt in range(retries + 1):
        try:
            with session.get(url, params=params, timeout=(5, 9), stream=True) as response:
                code = response.status_code
                if code != 200:
                    raise FetchError(f"http_{code}", status=code, retryable=code in TRANSIENT_STATUS)
                if image:
                    content_type = response.headers.get("Content-Type", "").lower()
                    if content_type.startswith(("text/html", "application/xhtml+xml", "application/json")):
                        raise FetchError("non_image_response")
                return read_response(response, limit), response.url
        except (requests.Timeout, requests.ConnectionError, requests.exceptions.ChunkedEncodingError) as error:
            reason = "timeout" if isinstance(error, requests.Timeout) else type(error).__name__
            failed = FetchError(reason, retryable=not isinstance(error, requests.exceptions.SSLError))
        except requests.RequestException as error:
            failed = FetchError(type(error).__name__)
        except FetchError as error:
            failed = error
        if not failed.retryable or attempt == retries:
            raise failed from None
        time.sleep(0.5)


def candidates(url):
    parsed = urlsplit(url)
    if parsed.scheme == "http":
        yield urlunsplit(parsed._replace(scheme="https"))
    yield url


def store_url(session, url, output, source, retries):
    body, final_url = fetch(session, url, limit=MAX_IMAGE_BYTES, retries=retries, image=True)
    if source == "wayback":
        expected, actual = CAPTURE.fullmatch(url), CAPTURE.fullmatch(final_url)
        if not expected or not actual or not same_url(expected[2], actual[2]):
            raise FetchError("archive_replay_changed_image_url")
    try:
        info = image_info(body)
    except IMAGE_ERRORS as error:
        raise FetchError(f"invalid_image:{type(error).__name__}") from None
    atomic_write(output, body)
    return {"status":"available","source":source,"requested_url":url,"resolved_url":final_url,**info}


def download_original(urls, output):
    errors = []
    with requests.Session() as session:
        for original in urls:
            for url in candidates(original):
                try:
                    info = store_url(session, url, output, "original_url", 1)
                    info["mapped_url"] = original
                    return info, errors
                except FetchError as error:
                    errors.append({"source":"original_url","url":url,"reason":error.reason})
    return None, errors


def same_url(left, right):
    def normalized(url):
        parsed = urlsplit(url)
        default = 443 if parsed.scheme == "https" else 80
        port = parsed.port if parsed.port not in (None, default) else None
        return parsed.hostname, port, unquote(parsed.path or "/"), parsed.query
    return normalized(left) == normalized(right)


def find_wayback(session, original):
    body, _ = fetch(session, WAYBACK_API, params={"url":original,"timestamp":"20230820"}, limit=256*1024)
    try:
        snapshot = json.loads(body)["archived_snapshots"].get("closest", {})
    except (ValueError, KeyError, TypeError):
        raise FetchError("invalid_archive_index") from None
    if not snapshot or not snapshot.get("available") or str(snapshot.get("status")) != "200":
        raise FetchError("no_exact_capture")
    match = CAPTURE.fullmatch(snapshot.get("url", ""))
    if not match or not same_url(original, match[2]):
        raise FetchError("capture_url_differs")
    return f"https://web.archive.org/web/{match[1]}id_/{match[2]}"


def image_captures(session, original):
    params = {
        "url":original, "matchType":"exact", "output":"json", "fl":"timestamp,original",
        "from":"2014", "to":"2024", "filter":["statuscode:200","mimetype:image/.*"],
        "collapse":"digest", "limit":"3",
    }
    body, _ = fetch(session, WAYBACK_CDX, params=params, limit=256*1024)
    try:
        rows = json.loads(body)
        return [f"https://web.archive.org/web/{timestamp}id_/{url}" for timestamp,url in rows[1:]
                if re.fullmatch(r"[0-9]{14}",timestamp) and url_ok(url) and same_url(original,url)]
    except (ValueError, TypeError):
        raise FetchError("invalid_archive_cdx_index") from None


def download_wayback(urls, output):
    errors = []
    with requests.Session() as session:
        for original in urls:
            tried = set()
            try:
                url = find_wayback(session, original)
                tried.add(url)
                return {**store_url(session,url,output,"wayback",0),"mapped_url":original}, errors
            except FetchError as error:
                errors.append({"source":"wayback_availability","url":next(iter(tried),original),"reason":error.reason})
            try:
                captures = image_captures(session, original)
            except FetchError as error:
                errors.append({"source":"wayback_cdx","url":original,"reason":error.reason})
                continue
            for url in captures:
                if url in tried:
                    continue
                try:
                    return {**store_url(session,url,output,"wayback",0),"mapped_url":original}, errors
                except FetchError as error:
                    errors.append({"source":"wayback_replay","url":url,"reason":error.reason})
            if not captures:
                errors.append({"source":"wayback_cdx","url":original,"reason":"no_exact_image_capture"})
    return None, errors


class ArchiveReader:
    def __init__(self, stream, digest=None):
        self.stream = stream
        self.digest = digest
        self.last = b""

    def read(self, size=-1):
        self.last = self.stream.read(size)
        if self.digest is not None:
            self.digest.update(self.last)
        return self.last


def read_archive(path, wanted, files, data_root):
    detail = {"official_folder":OFFICIAL_FOLDER, "path":str(path), "targets_found":0, "targets_valid":0, "errors":[]}
    if not path.is_file():
        return {**detail,"status":"archive_unavailable","targets_found":None,"targets_valid":None}
    seen, valid, duplicates = set(), {}, set()
    roots, directories, actions = Counter(), Counter(), Counter()
    archive_digest = sha256()
    data_root.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.TemporaryDirectory(dir=data_root, prefix=".from-web-stage-") as folder:
            stage = Path(folder)
            with path.open("rb") as source, gzip.GzipFile(fileobj=ArchiveReader(source, archive_digest)) as gzip_stream:
                tar_stream = ArchiveReader(gzip_stream)
                with tarfile.open(fileobj=tar_stream, mode="r|", bufsize=512) as archive:
                    for member in archive:
                        rel = PurePosixPath(member.name)
                        if rel.parts:
                            roots[rel.parts[0]] += 1
                        if member.isdir():
                            directories[rel.as_posix()] += 1
                        if not member.isfile() or rel.is_absolute() or ".." in rel.parts:
                            continue
                        detail["regular_file_count"] = detail.get("regular_file_count", 0) + 1
                        if len(rel.parts) > 1 and rel.parts[-2] != "from_web":
                            continue
                        name = f"from_web/{rel.name}"
                        if name not in wanted:
                            continue
                        seen.add(name)
                        try:
                            if member.size > MAX_IMAGE_BYTES:
                                raise ValueError("Archive member exceeds 32 MiB.")
                            with archive.extractfile(member) as file:
                                raw = file.read()
                            digest = sha256(raw).hexdigest()
                            if name in valid:
                                if valid[name][0]["sha256"] != digest:
                                    duplicates.add(name)
                                continue
                            old = files[name]
                            action = "added"
                            if old["status"] == "available":
                                action = "identical" if old["sha256"] == digest else "replaced"
                            if action == "identical":
                                info = {key: old[key] for key in ("width","height","format","exif_orientation","frames","bytes","sha256")}
                            else:
                                info = image_info(raw)
                                (stage / rel.name).write_bytes(raw)
                            valid[name] = (info, member.name, action)
                        except IMAGE_ERRORS as error:
                            detail["errors"].append({"path":name,"reason":type(error).__name__})
                # tarfile stops at the first zero block. Drain through the gzip
                # trailer before publishing files, so CRC/truncation is checked.
                if tar_stream.last != bytes(512) or gzip_stream.read(512) != bytes(512):
                    raise tarfile.ReadError("Missing TAR end marker.")
                while block := gzip_stream.read(1024 * 1024):
                    if block.strip(b"\0"):
                        raise tarfile.ReadError("Unexpected data after the TAR end marker.")
            for name in duplicates:
                del valid[name]
                detail["errors"].append({"path":name,"reason":"conflicting_archive_members"})
            for name, (info, member, action) in valid.items():
                old = files[name]
                entry = {"status":"available","source":"official_archive","archive_member":member,"archive_action":action,**info}
                if old["status"] == "available" and old["source"] != "official_archive":
                    entry["prior_version"] = {k:old[k] for k in ("source","sha256","width","height") if k in old}
                if action != "identical":
                    target = data_root / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    os.replace(stage / PurePosixPath(name).name,target)
                files[name] = entry
                actions[action] += 1
        detail.update(status="scanned", integrity="gzip CRC and TAR end markers verified",
                      archive_sha256=archive_digest.hexdigest(), archive_bytes=path.stat().st_size)
    except (tarfile.TarError, OSError, EOFError) as error:
        detail["status"] = "archive_read_error"
        detail["errors"].append({"reason":type(error).__name__})
    detail.update(targets_found=len(seen), targets_valid=len(valid), actions=dict(actions),
                  top_level_entries=dict(roots), directories=dict(directories))
    return detail


def write_report(path, archive, files, mapping, selection):
    sources = Counter(v["source"] for v in files.values() if v["status"] == "available")
    missing = [p for p,v in files.items() if v["status"] == "missing"]
    write_json(path,{
        "selected":len(files),"manifest_web_targets":selection,
        "with_mapped_url":sum(p in mapping for p in files),"available":sum(sources.values()),
        "counts_by_source":dict(sources),"missing_count":len(missing),"missing":missing,
        "available_image_bytes":sum(v["bytes"] for v in files.values() if v["status"] == "available"),
        "official_archive":archive,"files":files,
    })
    return sources, missing


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--mapping", type=Path, help="Override the official URL mapping")
    parser.add_argument("--resources", type=Path, help="Defaults to resources under the data root")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--archive", type=Path, help="Official from_web.tgz file (default: data-root/from_web.tgz).")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--workers", type=int, default=12, choices=range(1,25), metavar="1..24")
    parser.add_argument("--limit", type=int, help="Only process the first N manifest paths for a smoke test.")
    parser.add_argument("--no-direct", action="store_true", help="Skip live URL downloads.")
    parser.add_argument("--wayback", action="store_true", help="Try exact historical image URLs after live failures.")
    parser.add_argument("--wayback-limit", type=int, default=360, help="Upper limit on missing paths queried for captures.")
    args = parser.parse_args()
    if (args.limit is not None and args.limit <= 0) or args.wayback_limit < 0:
        parser.error("The sample limit must be positive and the archive lookup limit nonnegative.")
    root = (args.data_root or (args.manifest.parent if args.manifest else ROOT / "data")).resolve()
    report = args.report or root / ("web_sample_report.json" if args.limit else "web_download_report.json")
    resources = args.resources or root / "resources"
    targets, mapping = load_inputs(args.manifest or root / "image_manifest.json", args.mapping or resources / "url_to_filename.json")
    names = targets[:args.limit]
    prior = read_json(report).get("files", {}) if report.is_file() else {}
    files = {}
    for name in names:
        output = root / name
        if not output.resolve().is_relative_to(root):
            parser.error(f"Output escapes data root: {name}")
        previous = prior.get(name, {})
        try:
            if output.stat().st_size > MAX_IMAGE_BYTES:
                raise ValueError("Image exceeds 32 MiB.")
            raw = output.read_bytes()
            known = previous if previous.get("status") == "available" and sha256(raw).hexdigest() == previous.get("sha256") else {}
            info = {} if known else image_info(raw)
            files[name] = {"status":"available","source":"existing",**known,**info}
        except IMAGE_ERRORS:
            reason = "source_unavailable" if name in mapping else "no_mapped_url"
            files[name] = {"status":"missing","reason":reason,"attempts":previous.get("attempts", [])}
    archive = read_archive(args.archive or root/"from_web.tgz", set(names), files, root)

    def run_downloads(method, selected, label):
        host_slots = {
            urlsplit(mapping[p][0]).hostname: BoundedSemaphore(3)
            for p in selected
        }

        def download(path):
            if label == "original":
                with host_slots[urlsplit(mapping[path][0]).hostname]:
                    return method(mapping[path], root/path)
            return method(mapping[path], root/path)

        with ThreadPoolExecutor(max_workers=args.workers if label == "original" else min(args.workers,6)) as pool:
            pending = {pool.submit(download, p):p for p in selected}
            for i, future in enumerate(as_completed(pending), 1):
                name = pending[future]
                try:
                    info, errors = future.result()
                except OSError as error:
                    info, errors = None, [{"source":label,"reason":f"cannot_save:{type(error).__name__}"}]
                if info:
                    files[name] = {**info,"attempts":files[name].get("attempts", []) + errors}
                else:
                    prior_attempts = files[name]["attempts"]
                    prior_attempts.extend(error for error in errors if error not in prior_attempts)
                if i % 16 == 0 or i == len(pending):
                    write_report(report, archive, files, mapping, len(targets))
                    print(f"{label}: {i}/{len(pending)} paths tried; {sum(x['status']=='available' for x in files.values())} available", flush=True)

    def unresolved():
        return [p for p in names if files[p]["status"] == "missing" and p in mapping]

    # Round-robin across hosts so one slow CDN does not occupy all workers.
    per_host = defaultdict(list)
    for name in unresolved():
        per_host[urlsplit(mapping[name][0]).hostname].append(name)
    grouped = list(per_host.values())
    order = [group[i] for i in range(max(map(len,grouped),default=0)) for group in grouped if i<len(group)]
    if not args.no_direct:
        run_downloads(download_original, order, "original")
    if args.wayback:
        run_downloads(download_wayback, unresolved()[:args.wayback_limit], "wayback")
    sources, missing = write_report(report, archive, files, mapping, len(targets))
    print(f"Available: {dict(sources)}. Missing: {len(missing)}. Report: {report.resolve()}")
    return int(bool(missing))


if __name__ == "__main__":
    raise SystemExit(main())
