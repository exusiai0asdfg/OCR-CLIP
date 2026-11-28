import torch
from tqdm.auto import tqdm

from .model import encode_image

@torch.inference_mode()
def collect_features(model, dataloader, device):

    all_features = []
    all_categories = []
    all_attributes = []
    all_affordances = []

    for batch in tqdm(dataloader, desc="Encoding images"):

        features = encode_image(
            model,
            batch["image"],
            device
        )

        all_features.append(features.cpu())

        all_categories.append(batch["category"])
        all_attributes.append(batch["attribute"])
        all_affordances.append(batch["affordance"])

    return {
        "features": torch.cat(all_features, dim=0),
        "category": torch.cat(all_categories, dim=0),
        "attribute": torch.cat(all_attributes, dim=0),
        "affordance": torch.cat(all_affordances, dim=0)
    }