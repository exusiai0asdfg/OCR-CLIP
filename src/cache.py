import hashlib
import json
import os
import pickle
import platform
import tempfile
from importlib.metadata import distribution, version
from inspect import isfunction
from pathlib import Path

import torch
from torch.utils.data import SequentialSampler
from torchvision.transforms import Compose

from .dataset import file_sha256
from .evaluate import collect_features


def cache_identity(dataset, model, loader, weights_sha256):
    device = next(model.parameters()).device
    clip_dist = distribution("clip")
    clip_origin = json.loads(clip_dist.read_text("direct_url.json") or "{}")
    source_root = Path(__file__).parent
    preprocess = dataset.preprocess
    transforms = preprocess.transforms if isinstance(preprocess, Compose) else [preprocess]
    # CLIP Compose contains a Python function whose repr includes a process-local address.
    transform_spec = [
        f"{step.__module__}.{step.__qualname__}" if isfunction(step) else repr(step)
        for step in transforms
    ]
    return {
        "schema": 1,
        "dataset": dataset.snapshot_id,
        "weights_sha256": weights_sha256,
        "preprocess": transform_spec,
        "embedding_size": model.visual.output_dim,
        "batch_size": loader.batch_size,
        "default_dtype": str(torch.get_default_dtype()),
        "device": str(device),
        "hardware": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
        "cuda": torch.version.cuda if device.type == "cuda" else None,
        "cudnn": torch.backends.cudnn.version() if device.type == "cuda" else None,
        "cudnn_tf32": torch.backends.cudnn.allow_tf32,
        "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
        "matmul_precision": torch.get_float32_matmul_precision(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "platform": [platform.system(), platform.machine()],
        "python": platform.python_version(),
        "packages": {
            name: version(name) for name in ("torch", "torchvision", "Pillow", "numpy", "clip")
        },
        "clip_commit": clip_origin.get("vcs_info", {}).get("commit_id"),
        "clip_sources": {
            name: file_sha256(clip_dist.locate_file(f"clip/{name}")) for name in ("clip.py", "model.py")
        },
        "encoder_sources": {
            name: file_sha256(source_root / name) for name in ("dataset.py", "model.py", "evaluate.py")
        },
    }


def build_cache_key(identity):
    payload = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def validate_results(results, targets, embedding_size):
    if not isinstance(results, dict) or results.keys() != {"features", *targets}:
        raise ValueError("unexpected cached tensor names")
    features = results["features"]
    if (
        not isinstance(features, torch.Tensor)
        or features.shape != (len(targets["category"]), embedding_size)
        or features.dtype != torch.float32
        or not torch.isfinite(features).all()
    ):
        raise ValueError("invalid image feature shape, dtype or values")
    for label, expected in targets.items():
        value = results[label]
        if not isinstance(value, torch.Tensor) or value.dtype != expected.dtype or not torch.equal(value, expected):
            raise ValueError(f"{label} labels differ from the ordered audit rows")


def load_cache(path, identity, targets):
    if not path.is_file():
        return None
    try:
        artifact = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(artifact, dict) or artifact.get("identity") != identity:
            raise ValueError("cache identity mismatch")
        results = artifact.get("tensors")
        validate_results(results, targets, identity["embedding_size"])
    except (OSError, EOFError, RuntimeError, ValueError, TypeError, IndexError, pickle.UnpicklingError) as exc:
        print(f"Invalid cache at {path}: {exc}. Recomputing image features.")
        return None
    print(f"Cache hit: {path}")
    return results


def save_cache(path, identity, results):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".features-", suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        torch.save({"identity": identity, "tensors": results}, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Cache saved: {path}")


def get_image_features(model, loader, device, weights_sha256, *, use_cache, cache_dir):
    if type(loader.sampler) is not SequentialSampler or loader.drop_last:
        raise ValueError("Image feature collection requires the complete dataset in audit order")
    dataset = loader.dataset
    identity = cache_identity(dataset, model, loader, weights_sha256)
    key = build_cache_key(identity)
    path = Path(cache_dir) / f"{key}.pt"
    results = load_cache(path, identity, dataset.targets) if use_cache else None
    hit = results is not None
    if not hit:
        results = collect_features(model, loader, device)
        validate_results(results, dataset.targets, identity["embedding_size"])
        if use_cache:
            save_cache(path, identity, results)
    return results, key, hit
