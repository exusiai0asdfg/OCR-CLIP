from pathlib import Path

import torch
import torch.nn.functional as F
import clip
from clip.clip import _MODELS

from .dataset import file_sha256


def load_model(model_name, device, weights_dir=None):
    local_path = None if model_name in _MODELS else Path(model_name)
    # CLIP verifies named checkpoints against the SHA-256 embedded in the URL.
    weights_sha256 = (
        file_sha256(local_path) if local_path is not None else _MODELS[model_name].split("/")[-2]
    )
    model, preprocess = clip.load(
        model_name,
        device=device,
        download_root=str(weights_dir) if weights_dir is not None else None,
    )
    if local_path is not None and file_sha256(local_path) != weights_sha256:
        raise ValueError(f"Checkpoint changed while loading: {local_path}")

    model.eval()

    for param in model.parameters():
        param.requires_grad_(False)

    return model, preprocess, weights_sha256

@torch.inference_mode()
def encode_text(model, prompts, device):

    tokens = clip.tokenize(prompts).to(device)

    features = model.encode_text(tokens)

    features = F.normalize(
        features.float(),
        p=2,
        dim=-1
    )

    return features

@torch.inference_mode()
def encode_image(model, images, device):

    images = images.to(device)

    features = model.encode_image(images)
    
    features = F.normalize(
        features.float(),
        p=2,
        dim = -1
    )
    return features
