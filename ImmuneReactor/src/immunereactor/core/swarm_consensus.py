"""
core/swarm_consensus.py
Logika podejmowania decyzji o zaufaniu wewnątrz roju (Quorum/Consensus).
"""
from __future__ import annotations

import time
import logging
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Tuple

import torch
import torch.nn.functional as F

# Konfiguracja logowania
logger = logging.getLogger("SwarmConsensus")

class SwarmDecision(str, Enum):
    ALLOW      = "ALLOW"
    QUARANTINE = "QUARANTINE"
    ISOLATE    = "ISOLATE"

@dataclass
class FingerprintMessage:
    host_id:    int
    embedding:  torch.Tensor
    drift:      float
    region_ema: torch.Tensor
    timestamp:  int
    step:       int

@dataclass
class ConsensusResult:
    host_id:        int
    decision:       SwarmDecision
    regional_trust: float
    local_trust:    float
    final_trust:    float
    quorum_size:    int
    quorum_ok:      bool
    step:           int
    alert_active:   bool = False

class MessageBuffer:
    def __init__(self, max_age_steps: int = 5, buffer_per_host: int = 3):
        self.max_age  = max_age_steps
        self.buf_size = buffer_per_host
        self._store: dict[int, deque[FingerprintMessage]] = defaultdict(
            lambda: deque(maxlen=self.buf_size)
        )

    def put(self, msg: FingerprintMessage) -> None:
        self._store[msg.host_id].append(msg)

    def get_fresh(self, current_step: int) -> List[FingerprintMessage]:
        result = []
        for msgs in self._store.values():
            if not msgs: continue
            latest = msgs[-1]
            if (current_step - latest.step) <= self.max_age:
                result.append(latest)
        return result

    def clear(self) -> None:
        self._store.clear()

class SwarmConsensus:
    def __init__(
        self,
        host_id: int,
        threshold_allow: float = 0.70,
        threshold_quarantine: float = 0.50,
        threshold_isolate: float = 0.30,
        ema_weight_vote: float = 0.3,
    ):
        self.host_id = host_id
        self._th_allow = threshold_allow
        self._th_quarantine = threshold_quarantine
        self._th_isolate = threshold_isolate
        self._ema_w = ema_weight_vote
        self._buffer = MessageBuffer()
        self._history: List[ConsensusResult] = []

    def _compute_regional_trust(
        self, my_embedding: torch.Tensor, my_ema: torch.Tensor, current_step: int
    ) -> Tuple[float, int]:
        fresh_msgs = self._buffer.get_fresh(current_step)
        if not fresh_msgs:
            return 1.0, 0

        # Normalizacja i spójność urządzeń
        device = my_embedding.device
        my_vec = (1 - self._ema_w) * F.normalize(my_embedding, dim=0) + \
                 self._ema_w * F.normalize(my_ema.to(device), dim=0)
        
        similarities = []
        for msg in fresh_msgs:
            neighbor_vec = (1 - self._ema_w) * F.normalize(msg.embedding.to(device), dim=0) + \
                           self._ema_w * F.normalize(msg.region_ema.to(device), dim=0)
            sim = float(F.cosine_similarity(my_vec.unsqueeze(0), neighbor_vec.unsqueeze(0)).item())
            similarities.append((sim + 1.0) / 2.0)

        return float(sum(similarities) / len(similarities)), len(similarities)

    def _make_decision(
        self, regional_trust: float, local_trust: float, quorum_ok: bool, alert_delta: float
    ) -> Tuple[SwarmDecision, float]:
        
        final_trust = local_trust if not quorum_ok else min(regional_trust, local_trust)
        
        # Korekta progów o alerty
        th_q = self._th_quarantine + alert_delta
        th_i = self._th_isolate + alert_delta

        if final_trust >= self._th_allow:
            decision = SwarmDecision.ALLOW
        elif final_trust >= th_q:
            decision = SwarmDecision.QUARANTINE
        elif final_trust >= th_i:
            decision = SwarmDecision.QUARANTINE # Zgodnie z założeniem o strefie pośredniej
        else:
            decision = SwarmDecision.ISOLATE
            
        return decision, final_trust

    def step(self, my_embedding: torch.Tensor, my_ema: torch.Tensor, 
             local_trust: float, current_step: int, alert_delta: float = 0.0) -> ConsensusResult:
        
        regional_trust, quorum_size = self._compute_regional_trust(my_embedding, my_ema, current_step)
        decision, final_trust = self._make_decision(regional_trust, local_trust, quorum_size >= 3, alert_delta)
        
        result = ConsensusResult(
            host_id=self.host_id, decision=decision,
            regional_trust=regional_trust, local_trust=local_trust,
            final_trust=final_trust, quorum_size=quorum_size,
            quorum_ok=quorum_size >= 3, step=current_step
        )
        self._history.append(result)
        logger.debug(f"Host {self.host_id} consensus: {decision}")
        return result