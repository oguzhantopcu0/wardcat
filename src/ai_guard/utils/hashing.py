import hashlib


def sha256_hash(value: str, salt: str = "") -> str:
    """Hash with SHA-256. Salt is prepended (rainbow table protection)."""
    salted = salt + value
    return hashlib.sha256(salted.encode("utf-8")).hexdigest()
