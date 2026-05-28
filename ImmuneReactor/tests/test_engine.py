import pytest
import time
import base64
import torch
from immunereactor.runtime.engine import ImmuneEngine

def test_engine_initialization():
    # Zwróć uwagę: num_hosts musi być zgodne z konstruktorem w engine.py
    engine = ImmuneEngine(num_hosts=5)
    assert engine.device == "cpu"

def test_gatekeeper_logic():
    engine = ImmuneEngine(num_hosts=5, trusted_root_pub=b"fake_pub_key")
    hid = "TEST-DRONE"
    x = torch.randn(16)
    
    current_time = int(time.time())
    # Generujemy dummy signature w formacie Base64
    dummy_sig = base64.b64encode(b"not_a_real_signature").decode('utf-8')
    
    trust = engine.step(
        hid, 
        x, 
        cert_data={
            "payload": {
                "subject": "WRONG",
                "nonce": "123",
                "timestamp": current_time
            },
            "signature": dummy_sig # Dodajemy brakujący klucz
        }, 
        expected_nonce="123"
    )
    # Testujemy, czy system odrzucił (zwrócił 0.0)
    assert trust == 0.0

def test_anomaly_detection():
    engine = ImmuneEngine(num_hosts=5)
    hid = "DRONE-01"
    
    # Warmup - zwiększamy do 60, aby enkoder załapał "normalność"
    for _ in range(60):
        engine.step(hid, torch.randn(16))
        
    # Anomalia - wejście o dużej wariancji
    # Zgodnie z Twoim engine.py, trust < 0.5 jest uznawany za anomalię
    trust_val = engine.step(hid, torch.randn(16) * 50) 
    assert trust_val < 0.5