import argparse
import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.cache import get_image_features
from src.dataset import OCLDataset
from src.metrics import mean_average_precision, topk_accuracy
from src.model import encode_text, load_model
from src.prompts import build_affordance_prompts, build_attribute_prompts, build_category_prompts


def parse_args():
    parser = argparse.ArgumentParser(description="OCL zero-shot evaluation with CLIP")
    parser.add_argument("--model", default="ViT-B/32", help="OpenAI CLIP model name or local checkpoint")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--prompt-mode", choices=["word", "definition", "combined"], default="word")
    parser.add_argument("--data-root", type=Path, help="Defaults to the project's data directory")
    parser.add_argument("--resource-root", type=Path, help="Defaults to resources under the data root")
    parser.add_argument("--output-dir", type=Path, help="Defaults to the project's outputs directory")
    parser.add_argument("--cache-dir", type=Path, help="Defaults to cache under the data root")
    parser.add_argument("--weights-dir", type=Path, help="Directory for OpenAI CLIP checkpoints")
    parser.add_argument("--use-cache", action="store_true", help="Reuse a verified cache or save a new one")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable in this Python environment")
    return args


def main():
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    data_root = args.data_root or project_root / "data"
    resource_root = args.resource_root or data_root / "resources"
    output_dir = args.output_dir or project_root / "outputs"
    cache_dir = args.cache_dir or data_root / "cache"
    device = torch.device(args.device)
    print(f"Model: {args.model} | Device: {device} | Batch size: {args.batch_size} | Prompt: {args.prompt_mode}")

    # Verify the audited inputs before loading the model or trying a cache.
    dataset = OCLDataset(data_root, resource_root, preprocess=None)
    model, dataset.preprocess, weights_sha256 = load_model(args.model, device, args.weights_dir)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False)
    print(f"Verified {len(dataset)} instances and {dataset.image_count} images")
    results, cache_key, cache_hit = get_image_features(
        model, loader, device, weights_sha256, use_cache=args.use_cache, cache_dir=cache_dir,
    )

    category_features = encode_text(model, build_category_prompts(dataset.categories), device).cpu()
    category_scores = results["features"] @ category_features.T
    metrics = {}
    for k in (1, 3, 5):
        accuracy = topk_accuracy(category_scores, results["category"], k=k)
        metrics[f"top{k}_accuracy"] = accuracy
        print(f"Top-{k} Accuracy: {accuracy:.2%}")

    prompts = {
        "attribute": build_attribute_prompts(dataset.attributes),
        "affordance": build_affordance_prompts(dataset.affordances, mode=args.prompt_mode),
    }
    for label, texts in prompts.items():
        text_features = encode_text(model, texts, device).cpu()
        scores = results["features"] @ text_features.T
        score, ap = mean_average_precision(scores, results[label])
        valid = int(np.isfinite(ap).sum())
        metrics[f"{label}_map"] = score
        metrics[f"valid_{label}_classes"] = valid
        formatted_score = "undefined" if score is None else f"{score:.2%}"
        print(f"{label.capitalize()} mAP: {formatted_score} ({valid}/{len(ap)} classes)")

    experiment = {
        "config": {
            "model": args.model,
            "batch_size": args.batch_size,
            "device": args.device,
            "prompt_mode": args.prompt_mode,
            "num_instances": len(dataset),
            "num_images": dataset.image_count,
            "dataset_snapshot": dataset.snapshot_id,
            "weights_sha256": weights_sha256,
            "cache_key": cache_key,
            "cache_hit": cache_hit,
        },
        "metrics": metrics,
    }
    model_label = Path(args.model).name if Path(args.model).is_file() else args.model
    model_name = re.sub(r"[^A-Za-z0-9._-]", "_", model_label)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{timestamp}_{model_name}_{args.prompt_mode}.json"
    with output_path.open("w", encoding="utf-8") as stream:
        json.dump(experiment, stream, indent=4, allow_nan=False)
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
