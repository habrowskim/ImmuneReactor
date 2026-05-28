"""
core/swarm_orchestrator.py
Orkiestrator roju: zarządza sensorami, enkoderami i konsensusem.
"""
from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass
from typing import Optional, Protocol, Dict, List

import torch

from core.terrain_air_encoder import TerrainAirEncoder, RegionFingerprint
from core.swarm_consensus import (
    SwarmConsensus, FingerprintMessage, JammingAlert,
    ConsensusResult, SwarmDecision,
)

logger = logging.getLogger("SwarmOrchestrator")

class ImmuneEngineProtocol(Protocol):
    def step(self, host_id: int, features: torch.Tensor) -> float: ...

@dataclass
class OrchestratorStep:
    host_id: int
    fingerprint: RegionFingerprint
    local_trust: float
    consensus: ConsensusResult
    alert_sent: Optional[JammingAlert]
    step: int

    @property
    def decision(self) -> SwarmDecision:
        return self.consensus.decision

class SwarmOrchestrator:
    def __init__(
        self,
        num_hosts: int,
        embed_dim: int = 64,
        min_quorum: int = 3,
        neighbor_radius: int = 2,
        immune_engine: Optional[ImmuneEngineProtocol] = None,
        device: str = "cpu",
        ema_alpha: float = 0.1,
    ):
        self.num_hosts = num_hosts
        self.device = device

        self.encoder = TerrainAirEncoder(embed_dim=embed_dim, ema_alpha=ema_alpha, device=device)
        self.trust_engine = immune_engine
        
        self.consensus_nodes: Dict[int, SwarmConsensus] = {
            hid: SwarmConsensus(host_id=hid, embed_dim=embed_dim, min_quorum=min_quorum)
            for hid in range(num_hosts)
        }
        
        assert len(self.consensus_nodes) == num_hosts, "Consensus nodes initialization mismatch!"

        self._neighbors: Dict[int, List[int]] = {
            hid: [(hid + offset) % num_hosts for offset in range(-neighbor_radius, neighbor_radius + 1) if offset != 0]
            for hid in range(num_hosts)
        }
        self._last_fp_msgs: Dict[int, Optional[FingerprintMessage]] = {hid: None for hid in range(num_hosts)}
        self._global_step = itertools.count(1)

    def step(self, host_id: int, sensors: Dict[str, torch.Tensor]) -> OrchestratorStep:
        current_step = next(self._global_step)
        
        # 1. Kodowanie cech
        fp = self.encoder.encode(host_id, sensors)

        # 2. Obliczenie zaufania lokalnego
        trust_input = fp.to_trust_input().to(self.device)
        local_trust = float(max(0.0, min(1.0, self.trust_engine.step(host_id, trust_input))))

        # 3. Konsensus (Pobieranie wiadomości od sąsiadów)
        my_consensus = self.consensus_nodes[host_id]
        for neighbor_id in self._neighbors[host_id]:
            msg = self._last_fp_msgs.get(neighbor_id)
            if msg:
                my_consensus.receive_fingerprint(msg)

        # 4. Decyzja konsensusu (Zapewnienie zgodności urządzeń tensorów)
        result = my_consensus.step(
            my_embedding=fp.embedding.to(self.device),
            my_ema=fp.ema.to(self.device),
            local_trust=local_trust,
            current_step=current_step
        )

        # 5. Obsługa alertów
        if result.decision == SwarmDecision.ISOLATE:
            alert = JammingAlert(source_id=host_id, step=current_step, 
                                 regional_trust=result.regional_trust, local_drift=fp.drift)
            for neighbor_id in self._neighbors[host_id]:
                self.consensus_nodes[neighbor_id].receive_alert(alert, current_step)
        else:
            alert = None

        # 6. Aktualizacja stanu
        self._last_fp_msgs[host_id] = FingerprintMessage(
            host_id=host_id,
            embedding=fp.embedding.detach().clone(),
            drift=fp.drift,
            region_ema=fp.ema.detach().clone(),
            timestamp=fp.timestamp,
            step=current_step,
        )

        return OrchestratorStep(host_id, fp, local_trust, result, alert, current_step)

    def reset(self) -> None:
        self.encoder.reset_all()
        for node in self.consensus_nodes.values():
            node.reset()
        self._last_fp_msgs = {hid: None for hid in range(self.num_hosts)}
        self._global_step = itertools.count(1)