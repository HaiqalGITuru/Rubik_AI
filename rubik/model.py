"""
Jaringan saraf untuk memprediksi gerakan berikutnya.

Input : state kubus, 54 angka warna (0..5) -> diubah ke one-hot (324)
Output: 12 logit, satu untuk setiap gerakan seperempat (U, U', R, R', ...)

Arsitektur: MLP dengan residual block (mirip EfficientCube / DeepCubeA,
tapi diperkecil supaya bisa dilatih di CPU laptop).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.bn1 = nn.BatchNorm1d(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.bn2 = nn.BatchNorm1d(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.bn1(self.fc1(x)))
        h = self.bn2(self.fc2(h))
        return F.relu(x + h)


class CubeNet(nn.Module):
    def __init__(self, hidden1: int = 1024, hidden2: int = 512, num_blocks: int = 4, num_moves: int = 12,
                 num_stickers: int = 54):
        super().__init__()
        self.config = dict(hidden1=hidden1, hidden2=hidden2, num_blocks=num_blocks, num_moves=num_moves,
                           num_stickers=num_stickers)
        self.input = nn.Sequential(
            nn.Linear(num_stickers * 6, hidden1), nn.BatchNorm1d(hidden1), nn.ReLU(),
            nn.Linear(hidden1, hidden2), nn.BatchNorm1d(hidden2), nn.ReLU(),
        )
        self.blocks = nn.Sequential(*[ResidualBlock(hidden2) for _ in range(num_blocks)])
        self.output = nn.Linear(hidden2, num_moves)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """states: LongTensor (B, stiker) berisi warna 0..5 -> logits (B, jumlah gerakan)"""
        x = F.one_hot(states.long(), num_classes=6).float().flatten(1)
        x = self.input(x)
        x = self.blocks(x)
        return self.output(x)


def save_model(model: CubeNet, path: str, **extra) -> None:
    torch.save({"config": model.config, "state_dict": model.state_dict(), **extra}, path)


def pick_device(name: str = "auto") -> torch.device:
    """'auto' -> GPU (cuda) kalau tersedia, selain itu CPU."""
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


def load_model(path: str, device: str = "cpu") -> CubeNet:
    ckpt = torch.load(path, map_location="cpu")
    model = CubeNet(**ckpt["config"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model.to(pick_device(device))