"""Show example labels from the official OCL resources."""

import argparse
import json
from pathlib import Path

from audit_dataset import AnnotationReader


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path(__file__).resolve().parents[1] / "data")
    parser.add_argument("--resources", type=Path)
    args = parser.parse_args()
    resources = args.resources or args.data_root / "resources"
    with (resources / "OCL_annot_test.pkl").open("rb") as stream:
        annotations = AnnotationReader(stream).load()
    print(f"Test images: {len(annotations)}")
    for name in ("object", "attribute", "affordance"):
        labels = json.loads((resources / f"OCL_class_{name}.json").read_text(encoding="utf-8-sig"))
        print(f"{name} ({len(labels)}): {labels[0]}")


if __name__ == "__main__":
    main()
