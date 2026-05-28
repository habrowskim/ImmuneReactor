"""
data/generator.py
Symuluje strumień danych z sensorów dronów.
"""
import torch
import random
from dataclasses import dataclass

@dataclass
class DroneEvent:
    host_id: int
    features: torch.Tensor

def generate_stream(num_hosts: int, steps: int):
    """Generuje losowe zdarzenia w roju."""
    for _ in range(steps):
        hid = random.randint(0, num_hosts - 1)
        # Symulacja danych sensorowych (16 cech)
        features = torch.randn(16) 
        yield DroneEvent(host_id=hid, features=features)