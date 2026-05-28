import torch
import torch.nn as nn

class StreamMemory(nn.Module):
    def __init__(self, num_hosts: int, feature_dim: int, device: str = "cpu"):
        super().__init__()
        self.device = device
        self.feature_dim = feature_dim
        
        # Buffer pamięci: [num_hosts, feature_dim]
        self.register_buffer("ema", torch.zeros((num_hosts, feature_dim)))
        
        # Mapowanie identyfikatorów
        self.hid_to_idx = {}
        self.next_idx = 0
        self.num_hosts = num_hosts

    def get_idx(self, hid: str) -> int:
        if hid not in self.hid_to_idx:
            self.hid_to_idx[hid] = self.next_idx
            self.next_idx = (self.next_idx + 1) % self.num_hosts
        return self.hid_to_idx[hid]

    def update(self, hid: str, x: torch.Tensor) -> torch.Tensor:
        """Aktualizuje EMA i zwraca błąd (energię) stanu."""
        idx = self.get_idx(hid)
        
        prev = self.ema[idx].clone()
        # Exponential Moving Average update
        self.ema[idx] = 0.9 * prev + 0.1 * x.detach()
        
        # Obliczenie energii (norma błędu)
        return torch.norm(x - prev)