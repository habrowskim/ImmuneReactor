"""
main.py
Główny punkt startowy aplikacji.
"""
import torch
from immunereactor.runtime.engine import ImmuneEngine
from immunereactor.data.generator import generate_stream

def main():
    num_hosts = 20
    engine = ImmuneEngine(num_hosts=num_hosts, device="cpu")
    
    print("[*] Starting Immune Reactor simulation...")
    
    for i, event in enumerate(generate_stream(num_hosts, steps=100)):
        trust = engine.step(str(event.host_id), event.features)
        if i % 10 == 0:
            print(f"Step {i} | Host {event.host_id} | Trust: {trust:.4f}")

if __name__ == "__main__":
    main()