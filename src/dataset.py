import csv
import hashlib
import io
import json
import re
from pathlib import Path, PurePosixPath

import torch
from PIL import Image
from torch.utils.data import Dataset


def file_sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def checked_bytes(path, digest, description):
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != digest:
        raise ValueError(f"{description} SHA-256 mismatch: {path}")
    return data


class OCLDataset(Dataset):
    def __init__(self, data_root, resource_root, preprocess):
        self.data_root = Path(data_root).resolve()
        self.preprocess = preprocess
        audit_root = self.data_root / "final_audit"
        snapshot_path = audit_root / "snapshot.json"
        if not snapshot_path.is_file():
            raise ValueError(f"Missing {snapshot_path}. Restore or generate the audited snapshot first.")
        snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
        if snapshot.get("schema") != 1:
            raise ValueError(f"Unsupported audit snapshot schema: {snapshot_path}")

        inputs = snapshot["inputs"]
        resource_hashes = inputs["resource_sha256"]
        resource_root = Path(resource_root)
        for name in ("OCL_annot_test.pkl", "OCL_class_object.json"):
            if file_sha256(resource_root / name) != resource_hashes[name]:
                raise ValueError(f"Official resource SHA-256 mismatch: {resource_root / name}")
        self.attributes = json.loads(checked_bytes(
            resource_root / "OCL_class_attribute.json",
            resource_hashes["OCL_class_attribute.json"], "Attribute vocabulary",
        ))
        self.affordances = json.loads(checked_bytes(
            resource_root / "OCL_class_affordance.json",
            resource_hashes["OCL_class_affordance.json"], "Affordance vocabulary",
        ))
        self.categories = snapshot["categories"]
        if len(self.categories) != 10 or len(set(self.categories)) != 10:
            raise ValueError(f"Expected ten unique categories in {snapshot_path}")
        self.category_to_id = {name: i for i, name in enumerate(self.categories)}
        if file_sha256(self.data_root / "image_manifest.json") != inputs["manifest_sha256"]:
            raise ValueError("Image manifest SHA-256 mismatch against the audit snapshot")

        csv_bytes = checked_bytes(
            audit_root / "valid_instances.csv", snapshot["instances_sha256"], "Instance CSV",
        )
        rows = list(csv.DictReader(io.StringIO(csv_bytes.decode("utf-8-sig"), newline="")))
        if not rows or len(rows) != snapshot["instances"]:
            raise ValueError("Instance CSV is empty or its row count differs from the audit snapshot")
        self.samples = []
        self.targets = {
            "category": torch.empty(len(rows), dtype=torch.long),
            "attribute": torch.zeros(len(rows), len(self.attributes), dtype=torch.float32),
            "affordance": torch.zeros(len(rows), len(self.affordances), dtype=torch.float32),
        }
        verified_images = {}
        instance_ids = set()
        for i, row in enumerate(rows):
            location = f"{row.get('path')} object {row.get('object_index')}"
            try:
                rel = PurePosixPath(row["path"])
                image_path = self.data_root.joinpath(*rel.parts)
                if rel.is_absolute() or ".." in rel.parts or "\\" in row["path"]:
                    raise ValueError("expected a relative POSIX image path")
                index = int(row["object_index"])
                if index < 0 or (rel, index) in instance_ids:
                    raise ValueError("negative or duplicate object index")
                instance_ids.add((rel, index))
                category = self.category_to_id[row["category"]]
                width, height = int(row["width"]), int(row["height"])
                box = json.loads(row["crop_xyxy"])
                if (
                    not isinstance(box, list) or len(box) != 4
                    or any(type(x) is not int for x in box)
                    or not (0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height)
                ):
                    raise ValueError("invalid XYXY crop")
                digest = row["sha256"]
                if not re.fullmatch(r"[0-9a-f]{64}", digest):
                    raise ValueError("invalid image SHA-256")
                metadata = (digest, width, height)
                if image_path not in verified_images:
                    if not image_path.resolve().is_relative_to(self.data_root):
                        raise ValueError("image resolves outside the data root")
                    body = checked_bytes(image_path, digest, "Image")
                    with Image.open(io.BytesIO(body)) as image:
                        if image.size != (width, height):
                            raise ValueError("image dimensions differ from audit")
                    verified_images[image_path] = metadata
                elif verified_images[image_path] != metadata:
                    raise ValueError("conflicting metadata for the same image")
                self.targets["category"][i] = category
                for label in ("attribute", "affordance"):
                    ids = json.loads(row[f"{label}_ids"])
                    target = self.targets[label]
                    if not isinstance(ids, list) or any(
                        type(j) is not int or not 0 <= j < target.shape[1] for j in ids
                    ):
                        raise ValueError(f"invalid {label} indices")
                    target[i, ids] = 1
                self.samples.append((image_path, digest, box))
            except (ValueError, KeyError, TypeError, OSError) as exc:
                raise ValueError(f"Invalid audited instance {location}: {exc}") from exc

        if len(verified_images) != snapshot["images"]:
            raise ValueError("Unique image count differs from the audit snapshot")
        # The CSV digest also identifies the ordered rows and the verified image hashes.
        identity = {"schema": snapshot["schema"], "categories": self.categories,
                    "instances_sha256": snapshot["instances_sha256"], "inputs": inputs}
        self.snapshot_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        self.image_count = len(verified_images)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, digest, box = self.samples[idx]
        body = checked_bytes(path, digest, "Image")
        with Image.open(io.BytesIO(body)) as img:
            image = img.convert("RGB").crop(box)
        return {
            "image": self.preprocess(image),
            **{label: target[idx] for label, target in self.targets.items()},
        }
