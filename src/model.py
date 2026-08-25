import torch
import torch.nn.functional as F
from torch import nn


class MFM(nn.Module):
    """Max-Feature-Map activation used in Light CNN."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        first, second = torch.chunk(x, chunks=2, dim=1)
        return torch.maximum(first, second)


class AngularLinear(nn.Module):
    def __init__(self, in_features: int = 80, out_features: int = 2, m: int = 4):
        super().__init__()
        self.m = int(m)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_normal_(self.weight)

    def forward(self, embeddings: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        norm = torch.linalg.vector_norm(embeddings, ord=2, dim=1, keepdim=True).clamp_min(1e-7)
        embeddings = embeddings / norm
        weight = F.normalize(self.weight, p=2, dim=1)
        cos_theta = F.linear(embeddings, weight).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        theta = torch.acos(cos_theta)
        k = torch.floor(self.m * theta / torch.pi).detach()
        sign = torch.where(k.to(torch.int64) % 2 == 0, torch.ones_like(k), -torch.ones_like(k))
        phi_theta = sign * torch.cos(self.m * theta) - 2.0 * k
        return norm * cos_theta, norm * phi_theta


class LCNN(nn.Module):
    """Light CNN classifier for fixed-size ASVspoof log-power spectrograms."""

    def __init__(self, angular_margin: int = 4, dropout: float = 0.75):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, 5, padding=2), MFM(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 1), MFM(), nn.BatchNorm2d(32),
            nn.Conv2d(32, 96, 3, padding=1), MFM(), nn.MaxPool2d(2, 2), nn.BatchNorm2d(48),
            nn.Conv2d(48, 96, 1), MFM(), nn.BatchNorm2d(48),
            nn.Conv2d(48, 128, 3, padding=1), MFM(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 1), MFM(), nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, padding=1), MFM(), nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, 1), MFM(), nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, 3, padding=1), MFM(), nn.MaxPool2d(2, 2),
        )
        self.embedding = nn.Sequential(
            nn.Linear(32 * 53 * 37, 160),
            MFM(),
            nn.BatchNorm1d(80),
            nn.Dropout(dropout),
        )
        self.angular_head = AngularLinear(80, 2, angular_margin)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, spectrogram: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.features(spectrogram)
        x = torch.flatten(x, start_dim=1)
        embeddings = self.embedding(x)
        logits, margin_logits = self.angular_head(embeddings)
        return {"embeddings": embeddings, "logits": logits, "margin_logits": margin_logits}
