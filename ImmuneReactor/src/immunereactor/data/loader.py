"""
data/loader.py
Narzędzie do wczytywania historycznych logów sensorów.
"""
import json
import torch
from typing import Dict, List

def load_sensor_logs(filepath: str) -> List[Dict[str, torch.Tensor]]:
    """
    Wczytuje logi z pliku JSON i konwertuje wartości na tensory PyTorch.
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    processed_logs = []
    for entry in data:
        # Konwersja list z JSONa na tensory
        sensors = {k: torch.tensor(v, dtype=torch.float32) for k, v in entry["sensors"].items()}
        processed_logs.append({
            "host_id": entry["host_id"],
            "sensors": sensors
        })
    return processed_logs