import torch
import torch.nn as nn
import torch.nn.functional as F

class TrustEngine(nn.Module):
    def __init__(self, num_hosts: int, embed_dim: int, device: str = "cpu"):
        super().__init__()
        self.device = device
        # Identity: wektor tożsamości każdego hosta
        self.identity = nn.Parameter(torch.randn(num_hosts, embed_dim) * 0.05)
        
        self.hid_to_idx = {}
        self.next_idx = 0
        self.num_hosts = num_hosts

    def get_idx(self, hid: str) -> int:
        if hid not in self.hid_to_idx:
            self.hid_to_idx[hid] = self.next_idx
            self.next_idx = (self.next_idx + 1) % self.num_hosts
        return self.hid_to_idx[hid]

    def forward(self, hid: str, emb: torch.Tensor, mem_energy: torch.Tensor, 
                graph_risk: torch.Tensor, drift: torch.Tensor) -> torch.Tensor:
        """
        Główna logika zaufania.
        """
        idx = self.get_idx(hid)
        
        # 1. Similarity (kosinusowe dopasowanie)
        sim = (F.cosine_similarity(emb, self.identity[idx].unsqueeze(0), dim=-1) + 1) / 2
        
        # 2. Energy (Anomaly Scoring)
        energy = (
            0.3 * (1.0 - sim) + 
            0.3 * mem_energy +    
            0.2 * graph_risk +    
            0.2 * torch.tanh(drift) 
        )
        
        # 3. Trust (Boltzmann distribution)
        return torch.exp(-energy)