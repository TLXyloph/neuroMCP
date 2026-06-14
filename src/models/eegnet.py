import torch
import torch.nn as nn


class EEGNet(nn.Module):
    """EEGNet for motor imagery classification (Lawhern et al. 2018).

    Input: (batch, n_channels, n_times)
    Output: (batch, n_classes) — raw logits
    """

    def __init__(
        self,
        n_channels: int = 64,
        n_times: int = 160,
        n_classes: int = 3,
        F1: int = 8,
        D: int = 2,
        dropout_rate: float = 0.5,
    ) -> None:
        super().__init__()
        F2 = F1 * D

        # Block 1: temporal conv + depthwise spatial conv
        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, 64), padding="same", bias=False),
            nn.BatchNorm2d(F1),
            nn.Conv2d(F1, F2, (n_channels, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout_rate),
        )

        # Block 2: separable conv (depthwise + pointwise)
        self.block2 = nn.Sequential(
            nn.Conv2d(F2, F2, (1, 16), padding="same", groups=F2, bias=False),
            nn.Conv2d(F2, F2, (1, 1), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout_rate),
        )

        # Dynamically compute flattened size
        with torch.no_grad():
            probe = torch.zeros(1, 1, n_channels, n_times)
            probe = self.block1(probe)
            probe = self.block2(probe)
            flat_size = probe.flatten(1).shape[1]

        self.classifier = nn.Linear(flat_size, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, n_channels, n_times)
        x = x.unsqueeze(1)        # (batch, 1, n_channels, n_times)
        x = self.block1(x)
        x = self.block2(x)
        return self.classifier(x.flatten(1))
