"""Extract the OCL manifest subset from ILSVRC/imagenet-1k or a local original tree.

For Hugging Face, install pyarrow and thriftpy2, accept access to
ILSVRC/imagenet-1k, and use HF_TOKEN or a local login. Start with --sample-per-split 2.
"""

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import io
from itertools import islice
import json
import os
from pathlib import Path, PurePosixPath
import re
import sys
import tempfile
from threading import Lock
import time

from PIL import Image
import requests
from urllib3.exceptions import HTTPError


PROJECT_ROOT = Path(__file__).resolve().parents[1]
HUB = "https://huggingface.co"
REPOSITORY = "ILSVRC/imagenet-1k"
MIB = 1024 * 1024
SPLITS = {"train": "train", "val": "validation"}
TRAIN_PATH = re.compile(r"ImageNet/train/(n[0-9]{8})/\1_[0-9]+\.JPEG")
VAL_PATH = re.compile(r"ImageNet/val/ILSVRC2012_val_[0-9]{8}\.JPEG")
# This HF Parquet release appends the class synset to the stored image.path.
SOURCE_TRAIN_NAME = re.compile(r"(n[0-9]{8})_([0-9]+)_\1\.JPEG")
SOURCE_VAL_NAME = re.compile(r"(ILSVRC2012_val_[0-9]{8})_n[0-9]{8}\.JPEG")
CONTENT_RANGE = re.compile(r"bytes ([0-9]+)-([0-9]+)/([0-9]+)")
RETRYABLE_HTTP = {408, 429, 500, 502, 503, 504}


class DownloadError(Exception):
    pass


class BudgetExceeded(DownloadError):
    pass


class Transfer:
    """Count response body bytes read and share a hard limit between workers."""

    def __init__(self, limit):
        self.limit = limit
        self.received = 0
        self.range_received = 0
        self.range_responses = 0
        self.largest_range = 0
        self.reserved = 0
        self.lock = Lock()

    def read(self, response, limit, *, is_range=False):
        with self.lock:
            if self.received + self.reserved + limit > self.limit:
                raise BudgetExceeded("Download budget reached; rerun to resume.")
            self.reserved += limit
        data = bytearray()
        try:
            while len(data) < limit:
                block = response.raw.read(min(64 * 1024, limit - len(data)))
                if not block:
                    break
                data.extend(block)
            return bytes(data)
        finally:
            with self.lock:
                self.received += len(data)
                if is_range:
                    self.range_received += len(data)
                    self.range_responses += 1
                    self.largest_range = max(self.largest_range, len(data))
                self.reserved -= limit

    def get(self, session, url, label, interval=None):
        headers = {"Accept-Encoding": "identity"}
        if interval is not None:
            start, length = interval
            if not 0 < length <= 64 * MIB:
                raise DownloadError(f"{label}: requested range exceeds 64 MiB.")
            headers["Range"] = f"bytes={start}-{start + length - 1}"

        for attempt in range(3):
            try:
                with session.get(url, headers=headers, stream=True, timeout=(10, 30)) as response:
                    status = response.status_code
                    if status in RETRYABLE_HTTP and attempt < 2:
                        retry = response.headers.get("Retry-After", "")
                        delay = min(int(retry), 30) if retry.isdecimal() else 2 ** attempt
                    elif status in (401, 403):
                        if response.headers.get("X-Error-Code") == "GatedRepo" or status == 401:
                            raise DownloadError(
                                f"{label}: HTTP {status}. Accept the access terms for {REPOSITORY} "
                                "and use an authorized Hugging Face token."
                            )
                        raise DownloadError(f"{label}: HTTP 403 from the source or storage endpoint.")
                    elif interval is not None:
                        match = CONTENT_RANGE.fullmatch(response.headers.get("Content-Range", ""))
                        if status != 206 or not match:
                            raise DownloadError(f"{label}: expected HTTP 206 with Content-Range, got {status}.")
                        first, last, size = map(int, match.groups())
                        if (first, last) != (start, start + length - 1) or size <= last:
                            raise DownloadError(f"{label}: incorrect Content-Range.")
                        if response.headers.get("Content-Encoding", "identity").lower() != "identity":
                            raise DownloadError(f"{label}: encoded Range responses are unsupported.")
                        body = self.read(response, length, is_range=True)
                        if len(body) != length:
                            raise OSError("truncated range")
                        return body, size
                    elif status == 200:
                        limit = MIB
                        body = self.read(response, limit + 1)
                        if len(body) > limit:
                            raise DownloadError(f"{label}: metadata exceeds 1 MiB.")
                        return body
                    else:
                        raise DownloadError(f"{label}: unexpected HTTP {status}.")
            except (requests.RequestException, HTTPError, OSError) as error:
                if attempt == 2:
                    # The original HTTP exception can contain a signed storage URL.
                    raise DownloadError(f"{label}: transport error ({type(error).__name__}).") from None
                delay = 2 ** attempt
            time.sleep(delay)


class RangeFile(io.RawIOBase):
    """Pass exact seek/read operations from PyArrow to HTTP Range requests."""

    def __init__(self, transfer, session, url, label, size):
        super().__init__()
        if not isinstance(size, int) or size <= 0:
            _, size = transfer.get(session, url, label, (0, 1))
        self.transfer = transfer
        self.session = session
        self.url = url
        self.label = label
        self.size = size
        self.offset = 0
        self.tail = b""

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.offset

    def seek(self, offset, whence=io.SEEK_SET):
        self._checkClosed()
        if whence == io.SEEK_CUR:
            offset += self.offset
        elif whence == io.SEEK_END:
            offset += self.size
        elif whence != io.SEEK_SET:
            raise ValueError(f"Unsupported seek mode {whence}")
        if offset < 0:
            raise ValueError("Negative seek offset")
        self.offset = offset
        return self.offset

    def read(self, size=-1):
        self._checkClosed()
        remaining = self.size - self.offset
        size = remaining if size < 0 else min(size, remaining)
        if size <= 0:
            return b""
        if self.tail and self.offset >= self.size - len(self.tail):
            index = self.offset - (self.size - len(self.tail))
            self.offset += size
            return self.tail[index:index + size]
        body, total = self.transfer.get(self.session, self.url, self.label, (self.offset, size))
        if total != self.size:
            raise DownloadError(f"{self.label}: remote file size changed.")
        self.offset += len(body)
        if self.offset == self.size and len(body) <= 2 * MIB:
            self.tail = body
        return body

    def readinto(self, buffer):
        data = self.read(len(buffer))
        buffer[:len(data)] = data
        return len(data)


def hub_token():
    for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        if token := os.environ.get(name, "").strip():
            return token
    cache = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
    home = Path(os.environ.get("HF_HOME") or cache / "huggingface")
    path = Path(os.environ.get("HF_TOKEN_PATH") or home / "token").expanduser()
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def hub_session(token):
    session = requests.Session()
    if token:
        session.headers["Authorization"] = f"Bearer {token}"
    return session


def atomic_write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".imagenet-", delete=False) as file:
        temporary = Path(file.name)
        try:
            file.write(data)
            file.close()
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def write_json(path, value):
    atomic_write(path, (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode("utf-8"))


def manifest_paths(path):
    entries = json.loads(path.read_text(encoding="utf-8-sig"))["ImageNet"]
    if not isinstance(entries, list) or not entries:
        raise ValueError("Manifest ImageNet must be a nonempty list.")
    split_paths = {split: {} for split in SPLITS}
    for entry in entries:
        if not isinstance(entry, str) or not (TRAIN_PATH.fullmatch(entry) or VAL_PATH.fullmatch(entry)):
            raise ValueError(f"Invalid ImageNet manifest path: {entry!r}")
        path = PurePosixPath(entry)
        split, name = path.parts[1], path.name
        if name in split_paths[split]:
            raise ValueError(f"Duplicate ImageNet filename: {entry}")
        split_paths[split][name] = entry
    return split_paths


def image_info(data):
    with Image.open(io.BytesIO(data)) as image:
        width, height = image.size
        image_format = image.format
        if image_format not in {"JPEG", "PNG"} or min(width, height) < 1:
            raise ValueError(f"Unexpected image format or dimensions: {image_format} {image.size}")
        image.verify()
    with Image.open(io.BytesIO(data)) as image:
        image.load()
    return {
        "width": width, "height": height, "format": image_format,
        "bytes": len(data), "sha256": sha256(data).hexdigest(),
    }


def source_filename(path):
    if not isinstance(path, str):
        return None
    name = path.replace("\\", "/").rsplit("/", 1)[-1]
    if match := SOURCE_TRAIN_NAME.fullmatch(name):
        return f"{match[1]}_{match[2]}.JPEG"
    if match := SOURCE_VAL_NAME.fullmatch(name):
        return f"{match[1]}.JPEG"
    return name


def shard_locations(parquet, cache_path, names):
    try:
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if isinstance(cached, dict):
            return cached
    except (OSError, ValueError):
        pass
    locations = {}
    for group in range(parquet.num_row_groups):
        rows = parquet.read_row_group(group, columns=["image.path"], use_threads=False)
        for index, path in enumerate(rows["image"].combine_chunks().field("path").to_pylist()):
            name = source_filename(path)
            if name in names:
                if name in locations:
                    raise DownloadError(f"Duplicate source filename in shard: {name}")
                locations[name] = [group, index]
    write_json(cache_path, locations)
    return locations


def scan_shard(pq, transfer, token, revision, shard, names, pending, cache_dir):
    path = shard["rfilename"]
    cache_path = cache_dir / (PurePosixPath(path).name + ".json")
    if cache_path.is_file():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if isinstance(cached, dict) and pending.isdisjoint(cached):
                return {}, {}, None
        except (OSError, ValueError):
            pass
    url = f"{HUB}/datasets/{REPOSITORY}/resolve/{revision}/{path}"
    found, failures = {}, {}
    with hub_session(token) as session:
        try:
            with RangeFile(transfer, session, url, path, shard.get("size")) as file:
                with pq.ParquetFile(file, buffer_size=0, pre_buffer=False) as parquet:
                    columns = {parquet.schema.column(i).path for i in range(len(parquet.schema))}
                    if not {"image.path", "image.bytes"} <= columns:
                        raise DownloadError(f"{path}: image.path or image.bytes is missing.")
                    locations = shard_locations(parquet, cache_path, names)
                    matched = defaultdict(list)
                    for name in pending.intersection(locations):
                        group, index = locations[name]
                        matched[group].append((name, index))
                    if not matched:
                        return found, failures, None
                    try:
                        if __package__:
                            from . import imagenet_parquet
                        else:
                            import imagenet_parquet
                    except ImportError:
                        raise DownloadError("Install the Parquet index reader dependency: python -m pip install thriftpy2") from None
                    footer = imagenet_parquet.read_footer(file)
                    for group, matches in sorted(matched.items()):
                        try:
                            rows = parquet.read_row_group(group, columns=["image.path"], use_threads=False)
                            paths = rows["image"].combine_chunks().field("path")
                            bodies = imagenet_parquet.read_image_bytes(
                                pq, file, parquet, footer, group, [index for _, index in matches],
                            )
                            for name, index in matches:
                                body, source_path = bodies[index], paths[index].as_py()
                                if source_filename(source_path) == name and body:
                                    found[name] = (body, {
                                        "source_file": path, "source_image_path": source_path,
                                        "source_row_group": group, "source_row": index,
                                    })
                                else:
                                    failures[name] = "Source row has a mismatched filename or empty bytes."
                        except (DownloadError, OSError, ValueError) as error:
                            failures.update({name: str(error) for name, _ in matches})
                            if isinstance(error, BudgetExceeded):
                                return found, failures, error
        except (DownloadError, OSError, ValueError) as error:
            return found, failures, error
    return found, failures, None


def local_images(root, split, pending):
    # Validation images may be flat (official TAR layout) or grouped by WNID.
    val_dirs = list((root / "val").glob("n[0-9]" + "[0-9]" * 7)) if split == "val" else []
    for name, relative in pending.items():
        source = root.joinpath(*PurePosixPath(relative).parts[1:])
        if not source.is_file() and val_dirs:
            matches = [folder / name for folder in val_dirs if (folder / name).is_file()]
            if len(matches) > 1:
                yield name, None, "Ambiguous source filename in local validation tree."
                continue
            if matches:
                source = matches[0]
        try:
            yield name, source.read_bytes(), None
        except OSError as error:
            yield name, None, f"Cannot read local {split} file: {error.strerror}"


def hub_shards(transfer, token):
    with hub_session(token) as session:
        payload = transfer.get(
            session, f"{HUB}/api/datasets/{REPOSITORY}/revision/main?blobs=true", REPOSITORY,
        )
    info = json.loads(payload)
    revision = info.get("sha", "")
    if not re.fullmatch("[0-9a-f]{40}", revision):
        raise DownloadError("Missing or invalid Hugging Face dataset revision.")
    files = info["siblings"]
    shards = {}
    for split, prefix in SPLITS.items():
        pattern = re.compile(rf"data/{prefix}-([0-9]+)-of-([0-9]+)\.parquet")
        selected = []
        for file in files:
            if match := pattern.fullmatch(file["rfilename"]):
                selected.append((int(match[1]), int(match[2]), file))
        selected.sort(key=lambda row: row[0])
        if not selected or any((index, total) != (i, len(selected)) for i, (index, total, _) in enumerate(selected)):
            raise DownloadError(f"Incomplete shard listing for split {split}.")
        shards[split] = [file for _, _, file in selected]
    return revision, shards


def save_image(root, relative, body):
    info = image_info(body)
    atomic_write(root / relative, body)
    return {"status": "downloaded", **info}


def run_hub(root, split_paths, pending, transfer, workers, update, record):
    token = hub_token()
    revision, shards = hub_shards(transfer, token)
    update(revision=revision)
    # Check data access before starting a scan of hundreds of shards.
    first_split = next(split for split in pending if pending[split])
    path = shards[first_split][0]["rfilename"]
    with hub_session(token) as session:
        transfer.get(session, f"{HUB}/datasets/{REPOSITORY}/resolve/{revision}/{path}", path, (0, 1))

    try:
        import pyarrow.parquet as pq
    except ImportError:
        raise DownloadError("Install PyArrow in the active environment: python -m pip install pyarrow") from None

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for split, remaining in pending.items():
            if not remaining:
                continue
            names = split_paths[split]
            manifest_id = sha256("\n".join(sorted(names)).encode()).hexdigest()
            cache_dir = root / ".imagenet-index" / "v2" / revision / manifest_id
            split_shards = iter(shards[split])
            checked = 0
            incomplete = False
            while remaining and (batch := list(islice(split_shards, workers))):
                wanted = frozenset(remaining)
                jobs = [
                    pool.submit(scan_shard, pq, transfer, token, revision, shard, names, wanted, cache_dir)
                    for shard in batch
                ]
                budget_reached = False
                for shard, job in zip(batch, jobs):
                    found, failures, error = job.result()
                    for name, (body, provenance) in found.items():
                        if name in remaining and record(remaining[name], body, provenance):
                            remaining.pop(name)
                    for name, reason in failures.items():
                        if name in remaining:
                            record(remaining[name], None, reason)
                    if error:
                        incomplete = True
                        budget_reached |= isinstance(error, BudgetExceeded)
                        update(source_error={"shard": shard["rfilename"], "reason": str(error)})
                checked += len(batch)
                update()
                if checked == len(batch) or checked % 24 == 0 or not remaining:
                    print(f"{split}: {checked}/{len(shards[split])} shards checked, {len(remaining)} still needed", flush=True)
                if budget_reached:
                    raise BudgetExceeded("Download budget reached; rerun to resume.")
            for relative in remaining.values():
                record(relative, None, ("Source scan incomplete; see source_errors." if incomplete
                                        else "Exact filename not found in this split."), replace=False)


class Report:
    def __init__(self, path, source, files, transfer):
        self.path = path
        self.transfer = transfer
        self.files = files
        self.details = {"source": source, "revision": None, "source_errors": []}

    def __call__(self, *, revision=None, source_error=None):
        if revision is not None:
            self.details["revision"] = revision
        if source_error is not None:
            self.details["source_errors"].append(source_error)
        downloaded = sum(item["status"] == "downloaded" for item in self.files.values())
        existing = sum(item["status"] == "existing" for item in self.files.values())
        missing = [path for path, item in self.files.items() if item["status"] == "missing"]
        write_json(self.path, {
            **self.details, "selected": len(self.files), "downloaded": downloaded, "existing": existing,
            "missing_count": len(missing), "network_body_bytes": self.transfer.received,
            "range_body_bytes": self.transfer.range_received, "range_responses": self.transfer.range_responses,
            "largest_range_body_bytes": self.transfer.largest_range,
            "written_image_bytes": sum(item["bytes"] for item in self.files.values() if item["status"] == "downloaded"),
            "missing": missing, "files": self.files,
        })
        return downloaded, existing, len(missing)


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("Expected a positive integer.")
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "data" / "image_manifest.json")
    parser.add_argument("--data-root", type=Path, help="Output directory (default: manifest directory).")
    parser.add_argument("--source-root", type=Path, help="Local original source directory containing train/ and val/.")
    parser.add_argument("--report", type=Path, help="Override the JSON report path.")
    parser.add_argument("--sample-per-split", type=positive_int, metavar="N", help="Select the first N names per split.")
    parser.add_argument("--check-only", action="store_true", help="Validate local files without network access.")
    parser.add_argument("--max-download-mib", type=positive_int, default=512, help="Response body budget (default: 512 MiB).")
    parser.add_argument("--workers", type=positive_int, default=6, help="Parallel shard readers (default: 6).")
    args = parser.parse_args()
    root = (args.data_root or args.manifest.parent).resolve()
    report_path = args.report or root / ("imagenet_sample_report.json" if args.sample_per_split else "imagenet_download_report.json")
    if args.source_root is not None and not args.source_root.is_dir():
        parser.error("--source-root must be an existing ImageNet directory.")
    try:
        split_paths = manifest_paths(args.manifest)
    except (OSError, KeyError, ValueError) as error:
        parser.error(str(error))

    results = {}
    pending = {split: {} for split in SPLITS}
    for split, names in split_paths.items():
        for name, relative in islice(names.items(), args.sample_per_split):
            target = root / relative
            if not target.resolve().is_relative_to(root):
                parser.error(f"Destination escapes data root: {relative}")
            try:
                results[relative] = {"status": "existing", **image_info(target.read_bytes())}
            except FileNotFoundError:
                results[relative] = {"status": "missing", "reason": "File is absent."}
            except (OSError, ValueError, SyntaxError, Image.DecompressionBombError) as error:
                results[relative] = {"status": "missing", "reason": f"Existing file is invalid: {error}"}
            if results[relative]["status"] == "missing":
                pending[split][name] = relative

    transfer = Transfer(args.max_download_mib * MIB)
    source = REPOSITORY
    if args.check_only:
        source = "local verification"
    elif args.source_root:
        source = str(args.source_root.resolve())
    report = Report(report_path, source, results, transfer)

    def record(relative, body, detail, *, replace=True):
        if body is None:
            if replace or results[relative].get("reason") == "File is absent.":
                results[relative] = {"status": "missing", "reason": detail}
            return False
        try:
            results[relative] = {**save_image(root, relative, body), **detail}
            return True
        except (OSError, ValueError, SyntaxError, Image.DecompressionBombError) as error:
            results[relative] = {"status": "missing", "reason": f"Cannot validate or save source image: {error}"}
            return False

    interrupted = False
    try:
        if not args.check_only and any(pending.values()):
            if args.source_root is None:
                run_hub(root, split_paths, pending, transfer, args.workers, report, record)
            else:
                for split, remaining in pending.items():
                    for name, body, reason in local_images(args.source_root, split, remaining):
                        detail = reason if body is None else {"source_file": "local original tree"}
                        record(remaining[name], body, detail)
    except (DownloadError, OSError, ValueError, KeyError) as error:
        detail = str(error)
        report(source_error={"reason": detail})
        for remaining in pending.values():
            for relative in remaining.values():
                record(relative, None, detail, replace=False)
        print(detail, file=sys.stderr)
    except KeyboardInterrupt:
        interrupted = True
        report(source_error={"reason": "Interrupted; rerun to resume."})
    downloaded, existing, missing = report()
    print(f"Saved {downloaded}, verified existing {existing}, missing or invalid {missing}. "
          f"Read {transfer.received:,} HTTP body bytes. Report: {report_path.resolve()}")
    return 130 if interrupted else int(bool(missing))


if __name__ == "__main__":
    raise SystemExit(main())
