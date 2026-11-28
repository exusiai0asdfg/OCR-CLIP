import numpy as np
from sklearn.metrics import average_precision_score


def topk_accuracy(scores, targets, k=1):
    if scores.ndim != 2 or targets.shape != (scores.shape[0],) or not scores.shape[0]:
        raise ValueError("Top-k requires nonempty scores [instances, classes] and targets [instances]")
    if type(k) is not int or not 1 <= k <= scores.shape[1]:
        raise ValueError(f"k must be between 1 and {scores.shape[1]}, got {k}")
    _, indices = scores.topk(k, dim=1)

    correct = indices == targets.unsqueeze(1)

    correct = correct.any(dim=1)

    accuracy = correct.float().mean()

    return accuracy.item()

def average_precision(scores, targets):

    if targets.sum() == 0:
        return np.nan

    return average_precision_score(
        targets,
        scores
    )


def mean_average_precision(scores, targets):

    scores = scores.detach().cpu().numpy()
    targets = targets.detach().cpu().numpy()

    if scores.ndim != 2 or scores.shape != targets.shape:
        raise ValueError("mAP requires matching scores and targets [instances, classes]")

    num_classes = scores.shape[1]

    ap_list = []

    for i in range(num_classes):
        i_score = scores[:,i]
        i_label = targets[:,i]
        ap_list.append(average_precision(i_score,i_label))

    ap_list = np.array(ap_list)

    valid = np.isfinite(ap_list)
    return (float(np.nanmean(ap_list)) if valid.any() else None), ap_list


def pairwise_disambiguation(scores, targets, pairs):
    if scores.ndim != 2 or scores.shape != targets.shape:
        raise ValueError("Pairwise disambiguation requires matching scores and targets")
    accuracies = []
    for a, b in pairs:
        if a == b or not (0 <= a < scores.shape[1] and 0 <= b < scores.shape[1]):
            raise ValueError(f"Expected distinct valid class indices, got {(a, b)}")
        label_a, label_b = targets[:, a], targets[:, b]
        mask_a = (label_a == 1) & (label_b == 0)
        mask_b = (label_a == 0) & (label_b == 1)
        if not mask_a.any() or not mask_b.any():
            continue

        diff = scores[:, a] - scores[:, b]
        correct_a = ((diff[mask_a] > 1e-6).float() + 0.5 * (diff[mask_a].abs() <= 1e-6).float()).mean()
        correct_b = ((diff[mask_b] < -1e-6).float() + 0.5 * (diff[mask_b].abs() <= 1e-6).float()).mean()
        accuracies.append(((correct_a + correct_b) / 2).item())
    return (float(np.mean(accuracies)) if accuracies else None), len(accuracies)
