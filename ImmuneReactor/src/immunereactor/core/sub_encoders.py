"""
core/sub_encoders.py
Wyspecjalizowane sub-encodery dla różnych domen sensorycznych.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class TerrainSubEncoder(nn.Module):
    IN_DIM = 4; OUT_DIM = 4
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(self.IN_DIM, self.OUT_DIM)
        self.norm = nn.LayerNorm(self.OUT_DIM)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.tanh(x / torch.tensor([1013.0, 500.0, 5.0, 50.0], device=x.device))
        return self.norm(torch.tanh(self.fc(x)))

class MotionSubEncoder(nn.Module):
    IN_DIM = 6; OUT_DIM = 4
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(self.IN_DIM + 1, 16), nn.ReLU(), nn.Linear(16, self.OUT_DIM))
        self.norm = nn.LayerNorm(self.OUT_DIM)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        acc_norm = x[..., :3].norm(dim=-1, keepdim=True)
        x_aug = torch.cat([x, acc_norm], dim=-1)
        return self.norm(torch.tanh(self.fc(x_aug)))

class AirSignatureEncoder(nn.Module):
    IN_DIM = 5; OUT_DIM = 5
    _RANGES = torch.tensor([[-40.0, 60.0], [0.0, 100.0], [300.0, 2000.0], [870.0, 1084.0], [0.0, 50.0]])
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(self.IN_DIM, 16), nn.ReLU(), nn.Linear(16, self.OUT_DIM))
        self.norm = nn.LayerNorm(self.OUT_DIM)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ranges = self._RANGES.to(x.device)
        x_norm = (x - ranges[:, 0]) / (ranges[:, 1] - ranges[:, 0] + 1e-8)
        return self.norm(torch.tanh(self.fc(x_norm.clamp(0.0, 1.0))))

class MagRFEncoder(nn.Module):
    IN_DIM = 4; OUT_DIM = 3
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(self.IN_DIM + 2, 16), nn.ReLU(), nn.Linear(16, self.OUT_DIM))
        self.norm = nn.LayerNorm(self.OUT_DIM)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B_norm = x[..., :3].norm(dim=-1, keepdim=True)
        azimuth = torch.atan2(x[..., 1:2], x[..., 0:1])
        x_aug = torch.cat([x, B_norm, azimuth], dim=-1)
        return self.norm(torch.tanh(self.fc(x_aug)))

# Stała używana przez główny enkoder
TOTAL_CONCAT_DIM = (
    TerrainSubEncoder.OUT_DIM + 
    MotionSubEncoder.OUT_DIM + 
    AirSignatureEncoder.OUT_DIM + 
    MagRFEncoder.OUT_DIM
)