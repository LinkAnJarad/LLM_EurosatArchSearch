import torch
from torch import nn

NUM_CLASSES = 10  # EuroSAT land-cover classes


class DummyModel(nn.Module):
    """Small 3-block CNN over 64x64 EuroSAT tiles: 64 -> 32 -> 16 -> 8, then a linear head."""

    def __init__(self, hidden_dim: int = 32, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(3, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(hidden_dim * 2, hidden_dim * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim * 4),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(hidden_dim * 4, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.cnn(x))
