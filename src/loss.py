import torch
import torch.nn.functional as F
from torch import nn


class ASoftmaxLoss(nn.Module):
    def __init__(self, lambda_max: float = 1000.0, lambda_min: float = 5.0, lambda_decay: float = 0.1):
        super().__init__()
        self.lambda_max = float(lambda_max)
        self.lambda_min = float(lambda_min)
        self.lambda_decay = float(lambda_decay)
        self.register_buffer("iteration", torch.zeros((), dtype=torch.long))

    def forward(self, logits: torch.Tensor, margin_logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        target_mask = F.one_hot(labels.long(), num_classes=logits.shape[1]).bool()
        if self.training:
            self.iteration.add_(1)
        angular_lambda = max(
            self.lambda_min,
            self.lambda_max / (1.0 + self.lambda_decay * float(self.iteration.item())),
        )
        annealed = (angular_lambda * logits + margin_logits) / (1.0 + angular_lambda)
        training_logits = torch.where(target_mask, annealed, logits)
        return F.cross_entropy(training_logits, labels)
