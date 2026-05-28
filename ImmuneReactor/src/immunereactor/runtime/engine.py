"""
runtime/engine.py
Silnik główny systemu ImmuneReactor.
Odpowiada za cykl życia atestacji, detekcję anomalii i zarządzanie zaufaniem.
"""
from __future__ import annotations

import collections
import logging
from typing import Optional, Dict

import torch

from immunereactor.core.DroneIdentity import Drone
from immunereactor.core.memory import StreamMemory
from immunereactor.core.graph import SparseGraph
from immunereactor.core.trust import TrustEngine
from immunereactor.core.encoder import Encoder

# Konfiguracja logowania
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ImmuneEngine")

class ImmuneEngine:
    def __init__(
        self, 
        num_hosts: int, 
        feature_dim: int = 16, 
        embed_dim: int = 64, 
        device: str = "cpu", 
        trusted_root_pub: Optional[bytes] = None
    ):
        self.device = device
        self.trusted_root_pub = trusted_root_pub

        # Inicjalizacja komponentów (moduły core)
        self.memory = StreamMemory(num_hosts, feature_dim, device)
        self.encoder = Encoder().to(device)
        self.trust_engine = TrustEngine(num_hosts, embed_dim, device)
        self.graph = SparseGraph(num_hosts, k=5, device=device)

        # Optymalizator
        self.optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.trust_engine.parameters()), 
            lr=1e-3
        )
        
        self.last_hid: Optional[str] = None
        self.step_count = 0
        self.trust_history = collections.defaultdict(lambda: collections.deque(maxlen=20))

    def step(
        self, 
        hid: str, 
        x: torch.Tensor, 
        cert_data: Optional[Dict] = None, 
        expected_nonce: Optional[str] = None
    ) -> float:
        """
        Główna pętla przetwarzania zdarzeń.
        """
        if hid is None:
            return 0.0
        
        # Wymuszenie urządzenia
        x = x.to(self.device)

        # 1. Gatekeeper: Weryfikacja atestacji
        if cert_data is not None and self.trusted_root_pub is not None:
            nonce = expected_nonce or cert_data.get("nonce", "")
            valid = Drone.verify_attestation(
                cert_data, self.trusted_root_pub, nonce, max_age=60
            )
            if not valid or cert_data.get("payload", {}).get("subject") != hid:
                logger.warning(f"[SECURITY] Invalid cert or subject mismatch: {hid}")
                return 0.0

        # 2. Graph Quarantine Check
        if hasattr(self.graph, "is_quarantined") and self.graph.is_quarantined(hid):
            logger.info(f"[QUARANTINE] Host {hid} is isolated.")
            return 0.0

        # 3. Graph Interaction Update
        if self.last_hid is not None:
            self.graph.update_interaction(self.last_hid, hid)

        # 4. Memory/Drift Calculation
        mem_energy = self.memory.update(hid, x)
        drift_val = torch.clamp(mem_energy.detach(), 0.0, 10.0)

        # 5. Encoding
        emb = self.encoder(x.unsqueeze(0))
        graph_risk = self.graph.get_risk(hid)

        # 6. Trust Computation
        trust = self.trust_engine(hid, emb, mem_energy, graph_risk, drift_val)
        trust_val = float(trust.item())

        # 7. Temporal Smoothing
        hist = self.trust_history[hid]
        hist.append(trust_val)
        smoothed_trust = sum(hist) / len(hist)
        trend = (hist[-1] - hist[-3]) if len(hist) > 3 else 0.0

        # 8. Adaptive Training Gate
        # Uczymy się tylko wtedy, gdy stan jest stabilny (nie anomaly)
        is_stable = (smoothed_trust > 0.45 and drift_val.item() < 5.0 and abs(trend) < 0.2)
        if self.step_count < 500 or is_stable:
            loss = -torch.log(trust + 1e-6)
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.encoder.parameters()) + list(self.trust_engine.parameters()), 1.0
            )
            self.optimizer.step()
        else:
            logger.info(f"[IMMUNE] Anomaly threshold triggered for {hid} (Training skipped)")

        # 9. Graph Risk Update
        idx = self.graph._get_idx(hid)
        self.graph.risk[idx] = (1.0 - trust.detach()).squeeze().to(self.graph.device)
        self.graph.update()

        # 10. Update State
        self.last_hid = hid
        self.step_count += 1
        
        return trust_val