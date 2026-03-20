from ai_guard.utils.hashing import sha256_hash


def test_deterministic():
    assert sha256_hash("test") == sha256_hash("test")


def test_salt_changes_output():
    assert sha256_hash("test", salt="tuz1") != sha256_hash("test", salt="tuz2")


def test_empty_salt_still_hashes():
    result = sha256_hash("hello")
    assert len(result) == 64   # SHA-256 hex digest length


def test_rainbow_table_protection():
    # Same value, different salt → different hash
    without_salt = sha256_hash("password123")
    with_salt    = sha256_hash("password123", salt="gizli")
    assert without_salt != with_salt
