import hashlib


def sha256_hash(value: str, salt: str = "") -> str:
    """SHA-256 ile hash'le. Salt ön eke eklenir (rainbow table koruması)."""
    salted = salt + value
    return hashlib.sha256(salted.encode("utf-8")).hexdigest()
