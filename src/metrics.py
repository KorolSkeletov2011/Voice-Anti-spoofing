from __future__ import annotations

import numpy as np
import torch


def bonafide_scores(logits: torch.Tensor) -> torch.Tensor:
    return logits[:, 1] - logits[:, 0]


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    return float((logits.argmax(dim=1) == labels).float().mean().item())


def eer(scores: np.ndarray, labels: np.ndarray, positive_label: int = 1) -> float:
    """Compute EER for scores where larger means more likely bonafide."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels)
    positives = labels == positive_label
    negatives = ~positives
    n_pos = int(positives.sum())
    n_neg = int(negatives.sum())
    if n_pos == 0 or n_neg == 0:
        raise ValueError("EER requires both bonafide and spoof samples")

    order = np.argsort(scores)
    sorted_labels = positives[order].astype(np.int64)
    cum_pos = np.concatenate(([0], np.cumsum(sorted_labels)))
    cum_neg = np.concatenate(([0], np.cumsum(1 - sorted_labels)))

    # Threshold is placed between sorted samples. Samples below it are rejected.
    fnr = cum_pos / n_pos
    fpr = (n_neg - cum_neg) / n_neg
    idx = int(np.argmin(np.abs(fnr - fpr)))
    return float((fnr[idx] + fpr[idx]) / 2.0)
