"""
core/drone_dna.py
Plik napisany od nowa w tej sesji — dodany do struktury projektu.
Podpisuje RegionFingerprint zamiast czystego nonce.
Wymaga: pip install liboqs-python
"""
from __future__ import annotations

import hashlib
import struct
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.terrain_air_encoder import RegionFingerprint

try:
    from oqs import Signature as OQSSignature
    _OQS_AVAILABLE = True
except ImportError:
    _OQS_AVAILABLE = False

PQC_ALGORITHM       = "Dilithium2"
TIMESTAMP_WINDOW_MS = 10_000


def _serialize_fingerprint(drone_id: str, fp: "RegionFingerprint") -> bytes:
    id_bytes  = drone_id.encode("utf-8")
    id_len    = struct.pack(">H", len(id_bytes))
    host_part = struct.pack(">I", fp.host_id)
    ts_part   = struct.pack(">Q", fp.timestamp)
    embed_hash= hashlib.sha256(fp.embedding.numpy().astype("float32").tobytes()).digest()
    drift_part= struct.pack(">f", fp.drift)
    return id_len + id_bytes + host_part + ts_part + embed_hash + drift_part


@dataclass
class SignedFingerprintMessage:
    fingerprint: "RegionFingerprint"
    signature:   bytes
    public_key:  bytes
    drone_id:    str
    algorithm:   str = PQC_ALGORITHM

    def get_payload(self) -> bytes:
        return _serialize_fingerprint(self.drone_id, self.fingerprint)

    def is_fresh(self, window_ms: int = TIMESTAMP_WINDOW_MS) -> bool:
        return abs(int(time.time() * 1000) - self.fingerprint.timestamp) <= window_ms

    def __repr__(self) -> str:
        return (f"SignedFingerprintMessage(drone_id={self.drone_id!r}, "
                f"host_id={self.fingerprint.host_id}, drift={self.fingerprint.drift:.4f}, "
                f"fresh={self.is_fresh()})")


class DroneDNA:
    def __init__(self, drone_id: str, algorithm: str = PQC_ALGORITHM):
        if not _OQS_AVAILABLE:
            raise ImportError("Zainstaluj: pip install liboqs-python")
        self.drone_id  = drone_id
        self.algorithm = algorithm
        self._signer   = OQSSignature(algorithm)
        self.public_key: bytes = self._signer.generate_keypair()

    def sign_fingerprint(self, fp: "RegionFingerprint") -> SignedFingerprintMessage:
        payload   = _serialize_fingerprint(self.drone_id, fp)
        signature = self._signer.sign(payload)
        return SignedFingerprintMessage(
            fingerprint=fp, signature=signature,
            public_key=self.public_key, drone_id=self.drone_id,
            algorithm=self.algorithm,
        )

    def __del__(self) -> None:
        if hasattr(self, "_signer") and self._signer is not None:
            try:
                self._signer.free()
            except Exception:
                pass

    def __repr__(self) -> str:
        return f"DroneDNA(id={self.drone_id!r}, pk={self.public_key.hex()[:16]}...)"


class FingerprintVerifier:
    def __init__(self, window_ms: int = TIMESTAMP_WINDOW_MS):
        if not _OQS_AVAILABLE:
            raise ImportError("Zainstaluj: pip install liboqs-python")
        self.window_ms = window_ms

    def verify(self, msg: SignedFingerprintMessage, verbose: bool = False) -> bool:
        if not msg.is_fresh(self.window_ms):
            if verbose:
                print(f"[VERIFY FAIL] {msg.drone_id}: stara wiadomość")
            return False

        payload     = msg.get_payload()
        id_len      = struct.unpack(">H", payload[:2])[0]
        hash_offset = 2 + id_len + 4 + 8
        payload_hash= payload[hash_offset: hash_offset + 32]
        computed    = hashlib.sha256(msg.fingerprint.embedding.numpy().astype("float32").tobytes()).digest()
        if payload_hash != computed:
            if verbose:
                print(f"[VERIFY FAIL] {msg.drone_id}: hash embeddingu nie zgadza się")
            return False

        try:
            with OQSSignature(msg.algorithm) as v:
                is_valid = v.verify(payload, msg.signature, msg.public_key)
        except Exception as e:
            if verbose:
                print(f"[VERIFY FAIL] {msg.drone_id}: {e}")
            return False

        if not is_valid and verbose:
            print(f"[VERIFY FAIL] {msg.drone_id}: nieprawidłowy podpis Dilithium2")
        return is_valid

    def verify_batch(self, messages: list[SignedFingerprintMessage],
                     verbose: bool = False) -> list[SignedFingerprintMessage]:
        return [m for m in messages if self.verify(m, verbose=verbose)]
