"""
core/terrain_air_encoder.py  — FIXED
Zmiany:
  - Importy zmienione z relative (.sub_encoders) na absolute (core.sub_encoders)
    aby działały zarówno jako pakiet jak i bezpośredni import
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from core.sub_encoders import (
    TerrainSubEncoder,
    MotionSubEncoder,
    AirSignatureEncoder,
    MagRFEncoder,
    TOTAL_CONCAT_DIM,
)


@dataclass
class RegionFingerprint:
    embedding:  torch.Tensor
    drift:      float
    ema:        torch.Tensor
    variance:   float
    timestamp:  int
    host_id:    int

    @property
    def is_anomalous(self) -> bool:
        return self.drift > 2.0

    def to_trust_input(self) -> torch.Tensor:
        drift_t = torch.tensor([self.drift], device=self.embedding.device)
        return torch.cat([self.embedding, drift_t])


class FingerprintMemory:
    def __init__(self, embed_dim: int, alpha: float = 0.1, device: str = "cpu"):
        self.alpha     = alpha
        self.device    = device
        self.embed_dim = embed_dim
        self.ema: Optional[torch.Tensor] = None
        self._drift_history: list[float] = []
        self._window = 50

    def update(self, embedding: torch.Tensor) -> tuple[float, float]:
        embedding = embedding.detach().to(self.device)
        if self.ema is None:
            self.ema = embedding.clone()
            return 0.0, 0.0
        sigma_ema = self.ema.std().item() + 1e-8
        drift = (embedding - self.ema).norm().item() / sigma_ema
        self.ema = self.alpha * embedding + (1.0 - self.alpha) * self.ema
        self._drift_history.append(drift)
        if len(self._drift_history) > self._window:
            self._drift_history.pop(0)
        variance = float(torch.tensor(self._drift_history).var().item()) \
                   if len(self._drift_history) > 1 else 0.0
        return drift, variance

    def reset(self) -> None:
        self.ema = None
        self._drift_history.clear()


class TerrainAirEncoder(nn.Module):
    def __init__(self, embed_dim: int = 64, ema_alpha: float = 0.1, device: str = "cpu"):
        super().__init__()
        self.embed_dim = embed_dim
        self.device    = device

        self.terrain_enc = TerrainSubEncoder()
        self.motion_enc  = MotionSubEncoder()
        self.air_enc     = AirSignatureEncoder()
        self.mag_rf_enc  = MagRFEncoder()

        self.fusion = nn.Sequential(
            nn.Linear(TOTAL_CONCAT_DIM, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.Tanh(),
        )
        self._memories: dict[int, FingerprintMemory] = {}
        self._ema_alpha = ema_alpha
        self.to(device)
        self.eval()

    def _get_memory(self, host_id: int) -> FingerprintMemory:
        if host_id not in self._memories:
            self._memories[host_id] = FingerprintMemory(self.embed_dim, self._ema_alpha, self.device)
        return self._memories[host_id]

    @staticmethod
    def _to_tensor(data, device: str) -> torch.Tensor:
        if isinstance(data, torch.Tensor):
            return data.float().to(device)
        return torch.tensor(data, dtype=torch.float32, device=device)

    @torch.no_grad()
    def encode(self, host_id: int, sensors: dict[str, torch.Tensor]) -> RegionFingerprint:
        t = self._to_tensor(sensors["terrain"], self.device)
        m = self._to_tensor(sensors["motion"],  self.device)
        a = self._to_tensor(sensors["air"],     self.device)
        r = self._to_tensor(sensors["mag_rf"],  self.device)

        concat    = torch.cat([self.terrain_enc(t), self.motion_enc(m),
                               self.air_enc(a),     self.mag_rf_enc(r)])
        embedding = self.fusion(concat)

        memory = self._get_memory(host_id)
        drift, variance = memory.update(embedding)

        return RegionFingerprint(
            embedding = embedding.cpu(),
            drift     = drift,
            ema       = memory.ema.cpu().clone(),
            variance  = variance,
            timestamp = int(time.time() * 1000),
            host_id   = host_id,
        )

    def reset_host(self, host_id: int) -> None:
        if host_id in self._memories:
            self._memories[host_id].reset()

    def reset_all(self) -> None:
        for mem in self._memories.values():
            mem.reset()

    @property
    def num_hosts_tracked(self) -> int:
        return len(self._memories)
