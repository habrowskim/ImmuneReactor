"""
core/encoder.py
Główny enkoder systemowy odpowiedzialny za rzutowanie surowych cech 
na przestrzeń ukrytą (embedding space).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    """
    Enkoder typu MLP (Multi-Layer Perceptron) z aktywacją GELU.
    Zwraca znormalizowane wektory na sferze jednostkowej, 
    co jest kluczowe dla obliczania odległości (driftu) w systemie.
    """
    def __init__(self, feature_dim: int = 16, embed_dim: int = 64):
        super().__init__()
        self.feature_dim = feature_dim
        self.embed_dim = embed_dim
        
        self.net = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.GELU(),
            nn.Linear(128, embed_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Przetwarza wejście i normalizuje wyjście.
        
        Args:
            x: Tensor o kształcie [..., feature_dim]
            
        Returns:
            Tensor o kształcie [..., embed_dim] (znormalizowany L2)
        """
        # Obliczenie sieci
        features = self.net(x)
        
        # Normalizacja do sfery jednostkowej (p=2 oznacza normę L2)
        # Dzięki temu drift między dronami jest mierzalny w sposób stabilny
        return F.normalize(features, p=2, dim=-1)

    def get_config(self) -> dict:
        """Zwraca konfigurację enkodera dla potrzeb logowania."""
        return {
            "feature_dim": self.feature_dim,
            "embed_dim": self.embed_dim
        }