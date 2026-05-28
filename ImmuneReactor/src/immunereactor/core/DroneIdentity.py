from __future__ import annotations
import hashlib
import time
import json
import base64
from typing import Dict, Any, Optional
from cryptography.hazmat.primitives.asymmetric import ed25519

# --- Funkcje pomocnicze ---
def b64(data: bytes) -> str: return base64.b64encode(data).decode()
def b64d(data: str) -> bytes: return base64.b64decode(data.encode())
def canonical_json(data: Dict[str, Any]) -> bytes: 
    return json.dumps(data, sort_keys=True, separators=(",", ":")).encode()

class Signature:
    def __init__(self, private_key: Optional[ed25519.Ed25519PrivateKey] = None):
        self.private_key = private_key or ed25519.Ed25519PrivateKey.generate()
        self.public_key = self.private_key.public_key()
    
    def sign(self, data: bytes) -> bytes: 
        return self.private_key.sign(data)
    
    @staticmethod
    def verify(data: bytes, sig: bytes, pub_bytes: bytes) -> bool:
        try:
            pub = ed25519.Ed25519PublicKey.from_public_bytes(pub_bytes)
            pub.verify(sig, data)
            return True
        except (ValueError, TypeError, KeyError):
            return False

class Drone:
    def __init__(self, serial: str, hw_uuid: str, firmware: str):
        self.serial = serial
        self.signer = Signature()
        self.public_key = self.signer.public_key.public_bytes_raw()
        self._firmware_hash = hashlib.sha256(firmware.encode()).hexdigest()
        self._hw_uuid = hw_uuid

    def generate_attestation(self, nonce: str) -> Dict[str, Any]:
        payload = {
            "nonce": nonce,
            "subject": self.serial,
            "hw_uuid": self._hw_uuid,
            "firmware_hash": self._firmware_hash,
            "timestamp": int(time.time())
        }
        signature = self.signer.sign(canonical_json(payload))
        return {"payload": payload, "signature": b64(signature)}

    @staticmethod
    def verify_attestation(attestation: Dict, public_key: bytes, expected_nonce: str, max_age: int = 60) -> bool:
        payload = attestation["payload"]
        if payload["nonce"] != expected_nonce:
            return False
        if (int(time.time()) - payload["timestamp"]) > max_age:
            return False
        return Signature.verify(
            canonical_json(payload), 
            b64d(attestation["signature"]), 
            public_key
        )