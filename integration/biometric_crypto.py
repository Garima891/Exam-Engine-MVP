from __future__ import annotations

import ctypes
import os
from ctypes import wintypes
from typing import Protocol


ENCRYPTED_BIOMETRIC_BLOB_PREFIX = b"EEBI\x01"


class BiometricProtector(Protocol):
    def protect(self, value: bytes, purpose: str) -> bytes: ...

    def unprotect(self, value: bytes, purpose: str) -> bytes: ...


def seal_biometric_payload(
    value: bytes,
    purpose: str,
    protector: BiometricProtector,
) -> bytes:
    if not value:
        raise ValueError("Biometric payload must not be empty.")
    return ENCRYPTED_BIOMETRIC_BLOB_PREFIX + protector.protect(value, purpose)


def open_biometric_payload(
    value: bytes,
    purpose: str,
    protector: BiometricProtector | None = None,
    *,
    allow_legacy_plaintext: bool = False,
) -> bytes:
    if value.startswith(ENCRYPTED_BIOMETRIC_BLOB_PREFIX):
        if protector is None:
            raise RuntimeError("A biometric protector is required to decrypt data.")
        return protector.unprotect(
            value[len(ENCRYPTED_BIOMETRIC_BLOB_PREFIX):],
            purpose,
        )
    if allow_legacy_plaintext:
        return value
    raise ValueError("Biometric payload is not DPAPI-protected.")


class WindowsDpapiProtector:
    """Protect biometric bytes using the current Windows user's DPAPI key."""

    _CRYPTPROTECT_UI_FORBIDDEN = 0x1

    class _DataBlob(ctypes.Structure):
        _fields_ = [
            ("cbData", wintypes.DWORD),
            ("pbData", ctypes.POINTER(ctypes.c_ubyte)),
        ]

    def __init__(self) -> None:
        if os.name != "nt":
            raise RuntimeError(
                "Biometric storage requires Windows DPAPI; no plaintext "
                "fallback is available."
            )

    @classmethod
    def _blob(cls, value: bytes) -> tuple[ctypes.Array, _DataBlob]:
        buffer = ctypes.create_string_buffer(value, len(value))
        blob = cls._DataBlob(
            len(value),
            ctypes.cast(buffer, ctypes.POINTER(ctypes.c_ubyte)),
        )
        return buffer, blob

    @staticmethod
    def _entropy_blob(purpose: str) -> tuple[ctypes.Array, _DataBlob]:
        entropy = ("ExamEngineBiometric:v1:" + purpose).encode("utf-8")
        buffer = ctypes.create_string_buffer(entropy, len(entropy))
        blob = WindowsDpapiProtector._DataBlob(
            len(entropy),
            ctypes.cast(buffer, ctypes.POINTER(ctypes.c_ubyte)),
        )
        return buffer, blob

    @classmethod
    def _crypt(cls, value: bytes, purpose: str, *, protect: bool) -> bytes:
        if not value:
            raise ValueError("Biometric payload must not be empty.")

        input_buffer, input_blob = cls._blob(value)
        entropy_buffer, entropy_blob = cls._entropy_blob(purpose)
        output_blob = cls._DataBlob()
        crypt32 = ctypes.WinDLL("crypt32", use_last_error=True)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

        if protect:
            operation = crypt32.CryptProtectData
            operation.argtypes = [
                ctypes.POINTER(cls._DataBlob),
                wintypes.LPCWSTR,
                ctypes.POINTER(cls._DataBlob),
                ctypes.c_void_p,
                ctypes.c_void_p,
                wintypes.DWORD,
                ctypes.POINTER(cls._DataBlob),
            ]
            operation.restype = wintypes.BOOL
            succeeded = operation(
                ctypes.byref(input_blob),
                "Exam Engine biometric",
                ctypes.byref(entropy_blob),
                None,
                None,
                cls._CRYPTPROTECT_UI_FORBIDDEN,
                ctypes.byref(output_blob),
            )
        else:
            operation = crypt32.CryptUnprotectData
            operation.argtypes = [
                ctypes.POINTER(cls._DataBlob),
                ctypes.c_void_p,
                ctypes.POINTER(cls._DataBlob),
                ctypes.c_void_p,
                ctypes.c_void_p,
                wintypes.DWORD,
                ctypes.POINTER(cls._DataBlob),
            ]
            operation.restype = wintypes.BOOL
            succeeded = operation(
                ctypes.byref(input_blob),
                None,
                ctypes.byref(entropy_blob),
                None,
                None,
                cls._CRYPTPROTECT_UI_FORBIDDEN,
                ctypes.byref(output_blob),
            )

        if not succeeded:
            raise ctypes.WinError(ctypes.get_last_error())

        try:
            return ctypes.string_at(output_blob.pbData, output_blob.cbData)
        finally:
            kernel32.LocalFree.argtypes = [ctypes.c_void_p]
            kernel32.LocalFree.restype = ctypes.c_void_p
            kernel32.LocalFree(ctypes.cast(output_blob.pbData, ctypes.c_void_p))
            del input_buffer, entropy_buffer

    def protect(self, value: bytes, purpose: str) -> bytes:
        return self._crypt(value, purpose, protect=True)

    def unprotect(self, value: bytes, purpose: str) -> bytes:
        return self._crypt(value, purpose, protect=False)
