from __future__ import annotations

import hashlib


class FakeBiometricProtector:
    def __init__(self, failing_purpose: str | None = None) -> None:
        self.failing_purpose = failing_purpose

    @staticmethod
    def _transform(value: bytes, purpose: str) -> bytes:
        key = hashlib.sha256(purpose.encode("utf-8")).digest()
        return bytes(byte ^ key[index % len(key)] for index, byte in enumerate(value))

    def protect(self, value: bytes, purpose: str) -> bytes:
        if purpose == self.failing_purpose:
            raise OSError("test encryption failure")
        return self._transform(value, purpose)

    def unprotect(self, value: bytes, purpose: str) -> bytes:
        return self._transform(value, purpose)
