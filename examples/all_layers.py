"""
All three detection layers on one guard: regex + SpaCy NER + on-prem LLM.

- regex — deterministic patterns with checksum validation (IBAN, credit card,
  TC ID, email, phone, IP, ...). Highest confidence, always wins overlaps.
- ner   — SpaCy models for contextual entities (PERSON, ORG, ADDRESS). One
  detector per language; match the model to the text's language.
- llm   — an on-prem LLM (Ollama here). With adjudicate=True it verifies the
  regex/NER candidates (dropping false positives, relabeling mistakes) and
  adds contextual PII the other layers miss — all in a single call.

Requires:
    pip install "wardcat[ner]"
    python -m spacy download en_core_web_sm   # tr_core_news_md auto-downloads
    ollama pull gemma3:12b                    # or any chat model; set below

If Ollama is not running, the scan still completes with the regex + NER
layers; the LLM layer's absence is reported on ``result.scan_error``.
"""

import os

from wardcat import Action, Backend, Entity, Wardcat


def resolve_salt() -> str:
    """Get the hashing salt from the environment.

    wardcat never reads env vars itself — that boundary is deliberate: the
    library takes the salt as a plain argument, and *your* application supplies
    it (from an env var, a secrets manager, a vault, …). This example plays the
    role of that app. In production, fail if it is unset rather than fall back.
    """
    salt = os.environ.get("WARDCAT_SALT")
    if not salt:
        print("⚠  WARDCAT_SALT not set — using a throwaway demo salt (dev only).")
        salt = "demo-only-salt"
    return salt


# Bilingual sample: Turkish + English, several PII types in each.
SAMPLE = """\
Merhaba, ben Ayşe Kaya. TC kimliğim 10987654321, telefonum 0555 000 00 00.
Ödemeyi TR330006100519786457841326 IBAN'ına yapabilirsiniz.

Hi, this is John Carter from the billing team. Reach me at john.carter@acme.com.
The card on file is 4111 1111 1111 1111 and the gateway IP is 192.168.1.10.
"""


def build_guard() -> Wardcat:
    """One Wardcat with all three layers active, built fluently."""
    return (
        Wardcat(salt=resolve_salt())
        # NER: one SpaCy model per language of the text. Running only the
        # English model over Turkish (or vice versa) produces false positives.
        .with_ner(spacy_model=["tr_core_news_md", "en_core_web_sm"])
        # LLM: local Ollama. Loopback HTTP is allowed with no allow_http needed.
        # adjudicate=True → the LLM double-checks the regex/NER candidates too.
        .with_llm(
            backend=Backend.OLLAMA,
            model="gemma3:12b",
            adjudicate=True,
            timeout=120,
        )
        # Checksum-validated identifiers → irreversible salted hash.
        .add_entities(
            [Entity.CREDIT_CARD, Entity.IBAN, Entity.TC_ID],
            action=Action.HASH,
        )
        # Contact details → entity-aware partial mask (u***@acme.com, ...4567).
        .add_entities([Entity.EMAIL, Entity.PHONE], action=Action.MASK)
        # Names come from the model layers (NER + LLM) → plain [PERSON] label.
        .add_entity(Entity.PERSON, Action.REDACT)
        # Internal IPs: report only, leave the text untouched.
        .add_entity(Entity.IP_ADDRESS, Action.WARN)
    )


def main() -> None:
    guard = build_guard()
    result = guard.scan(SAMPLE)

    print("Sanitized text:")
    print(result.sanitized_text)

    if result.scan_error:
        print(f"(degraded scan — a layer was unavailable: {result.scan_error})")

    print(f"\nViolations ({len(result.violations)}):")
    print(f"{'ENTITY':<13} {'ORIGINAL':<30} {'ACTION':<8} CONFIDENCE")
    for v in sorted(result.violations, key=lambda v: v.start):
        print(f"{v.entity_type:<13} {v.original!r:<30} {v.action:<8} {v.confidence:.2f}")


if __name__ == "__main__":
    main()
