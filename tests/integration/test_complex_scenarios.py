"""
Complex scenario tests.

Scope:
  S01 — Realistic customer support prompt (all entity types)
  S02 — Different card formats (spaced, dashed, contiguous)
  S03 — Multiple entities of the same type
  S04 — PII embedded in JSON / code blocks
  S05 — Multi-line conversation (LLM chat format)
  S06 — Mixed language (TR + EN)
  S07 — False positive check (clean texts)
  S08 — Warn-only mode — text should not change
  S09 — Hash consistency and salt effect
  S10 — Multiple salt changes (idempotency)
  S11 — Correct resolution of overlapping entities
  S12 — TC identity boundary values
  S13 — No false positives between card numbers
  S14 — Batch scan mixed scenario
  S15 — Large/dense text performance
  S16 — IBAN case insensitivity
  S17 — International phone formats
  S18 — Address with Turkish special characters
  S19 — Postal code boundary values
  S20 — CLI JSON output schema validation
"""

from __future__ import annotations

import json
import time

import pytest

from ai_guard import AIGuard
from ai_guard.__main__ import _build_parser, cmd_scan
from ai_guard.core.models import Action

# ── helper ───────────────────────────────────────────────────────────────────


def _types(result) -> set[str]:
    return {v.entity_type for v in result.violations}


def _originals(result) -> list[str]:
    return [v.original for v in result.violations]


def _guard(**kw) -> AIGuard:
    return AIGuard(use_ner=False, **kw)


# ── S01: Realistic customer support prompt ───────────────────────────────────

CUSTOMER_SUPPORT_PROMPT = """
[SYSTEM] Aşağıdaki müşteri bilgileri ile işlem yap.

Müşteri Adı  : Ali Veli
TC Kimlik    : 12345678950
Telefon      : 0532 999 00 11
E-posta      : ali.veli@musteri.com
IBAN         : TR330006100519786457841326
Kredi Kartı  : 4532 0151 1283 0366
IP Adresi    : 192.168.10.55
Adres        : Atatürk Caddesi No:15 Kat:3, 34100 İstanbul
""".strip()


class TestS01CustomerSupportPrompt:
    def test_all_entity_types_detected(self):
        result = _guard().scan(CUSTOMER_SUPPORT_PROMPT)
        found = _types(result)
        assert "TC_ID" in found
        assert "PHONE" in found
        assert "EMAIL" in found
        assert "IBAN" in found
        assert "CREDIT_CARD" in found
        assert "IP_ADDRESS" in found

    def test_hashed_entities_removed_from_text(self):
        result = _guard().scan(CUSTOMER_SUPPORT_PROMPT)
        # Default: CC, IBAN, TC_ID → hash
        assert "4532 0151 1283 0366" not in result.sanitized_text
        assert "TR330006100519786457841326" not in result.sanitized_text
        assert "12345678950" not in result.sanitized_text

    def test_placeholders_in_sanitized_text(self):
        result = _guard().scan(CUSTOMER_SUPPORT_PROMPT)
        assert "[CREDIT_CARD:" in result.sanitized_text
        assert "[IBAN:" in result.sanitized_text
        assert "[TC_ID:" in result.sanitized_text

    def test_violation_positions_are_valid(self):
        result = _guard().scan(CUSTOMER_SUPPORT_PROMPT)
        for v in result.violations:
            assert v.start >= 0
            assert v.end > v.start
            assert CUSTOMER_SUPPORT_PROMPT[v.start : v.end] == v.original


# ── S02: Different card formats ──────────────────────────────────────────────


class TestS02CardFormats:
    @pytest.mark.parametrize(
        "card",
        [
            "4111111111111111",  # Visa contiguous
            "4111 1111 1111 1111",  # Visa spaced
            "4111-1111-1111-1111",  # Visa dashed
            "5500 0000 0000 0004",  # MasterCard spaced
            "3782 822463 10005",  # Amex (4-6-5 group)
            "6011 1111 1111 1117",  # Discover spaced
        ],
    )
    def test_card_detected(self, card):
        result = _guard().scan(f"kartım: {card} ödeme")
        assert "CREDIT_CARD" in _types(result), f"Not detected: {card}"

    @pytest.mark.parametrize(
        "not_card",
        [
            "1234 5678",  # 8 digits — not a card
            "0000 0000 0000 0000",  # zero-starting — not Visa
            "sipariş: 2024031801234",  # order number
        ],
    )
    def test_non_card_not_detected(self, not_card):
        result = _guard().scan(not_card)
        assert "CREDIT_CARD" not in _types(result), f"False positive: {not_card}"


# ── S03: Multiple entities of the same type ───────────────────────────────────


class TestS03MultipleEntities:
    def test_three_emails(self):
        text = "a@foo.com ile b@bar.com ve c@baz.org"
        result = _guard().scan(text)
        emails = [v for v in result.violations if v.entity_type == "EMAIL"]
        assert len(emails) == 3

    def test_two_credit_cards(self):
        text = "birinci: 4111111111111111 ikinci: 5500000000000004"
        result = _guard().scan(text)
        cards = [v for v in result.violations if v.entity_type == "CREDIT_CARD"]
        assert len(cards) == 2

    def test_mixed_phones(self):
        text = "0532 111 22 33 veya +90 533 444 55 66 ara"
        result = _guard().scan(text)
        phones = [v for v in result.violations if v.entity_type == "PHONE"]
        assert len(phones) == 2


# ── S04: PII in JSON / code blocks ───────────────────────────────────────────


class TestS04EmbeddedInCode:
    def test_json_payload(self):
        payload = '{"user": "ali@test.com", "card": "4111111111111111", "tc": "12345678950"}'
        result = _guard().scan(payload)
        found = _types(result)
        assert "EMAIL" in found
        assert "CREDIT_CARD" in found
        assert "TC_ID" in found

    def test_python_dict(self):
        code = """
config = {
    "db_password": "gizli123",
    "admin_email": "admin@firma.com",
    "server_ip": "10.20.30.40",
}
"""
        result = _guard().scan(code)
        found = _types(result)
        assert "EMAIL" in found
        assert "IP_ADDRESS" in found

    def test_curl_command_with_iban(self):
        cmd = 'curl -d \'{"iban":"TR330006100519786457841326"}\' https://api.example.com'
        result = _guard().scan(cmd)
        assert "IBAN" in _types(result)


# ── S05: Multi-line LLM chat format ─────────────────────────────────────────

CHAT_TEXT = """\
[SYSTEM] Sen bir banka asistanısın.
[USER] Merhaba, hesabıma 0532 888 77 66 numarasından ulaştım.
       IBAN numaram TR330006100519786457841326.
[ASSISTANT] Anlıyorum, hesabınızı sorguluyorum.
[USER] Ayrıca kartım 4111 1111 1111 1111 bloke oldu, TC: 12345678950
"""


class TestS05MultilineChat:
    def test_detects_across_lines(self):
        result = _guard().scan(CHAT_TEXT)
        found = _types(result)
        assert "PHONE" in found
        assert "IBAN" in found
        assert "CREDIT_CARD" in found
        assert "TC_ID" in found

    def test_line_positions_are_correct(self):
        result = _guard().scan(CHAT_TEXT)
        for v in result.violations:
            extracted = CHAT_TEXT[v.start : v.end]
            assert extracted == v.original, (
                f"{v.entity_type}: expected '{v.original}', position {v.start}:{v.end} → '{extracted}'"
            )


# ── S06: Mixed language (TR + EN) ────────────────────────────────────────────

MIXED_LANG = (
    "Dear support team, my Turkish ID is 12345678950 "
    "and my e-mail is user@example.com. "
    "Ayrıca kartım: 5500 0000 0000 0004. "
    "My IBAN: TR330006100519786457841326."
)


class TestS06MixedLanguage:
    def test_all_detected_in_mixed_text(self):
        result = _guard().scan(MIXED_LANG)
        found = _types(result)
        assert "TC_ID" in found
        assert "EMAIL" in found
        assert "CREDIT_CARD" in found
        assert "IBAN" in found


# ── S07: False positive check ────────────────────────────────────────────────


class TestS07FalsePositives:
    @pytest.mark.parametrize(
        "clean_text",
        [
            "Bugün hava çok güzel, pikniğe gidelim.",
            "Python 3.13 sürümü yayınlandı.",
            "Toplantı saat 14:30'da başlayacak.",
            "2024 yılında 1500 satış hedefliyoruz.",
            "Sipariş no: 2024031800001 takip edilebilir.",
            "Ürün kodu: ABC-12345 stokta mevcut.",
            "Versiyon: 10.2.3 güncelleme mevcut.",
            "Koordinat: 41.0082, 28.9784 (İstanbul)",
        ],
    )
    def test_clean_text_has_no_violations(self, clean_text):
        result = _guard().scan(clean_text)
        assert result.is_clean, (
            f"False positive! '{clean_text}' → {[(v.entity_type, v.original) for v in result.violations]}"
        )


# ── S08: Warn-only mode ──────────────────────────────────────────────────────


class TestS08WarnOnlyMode:
    def test_sanitized_text_unchanged_when_all_warn(self):
        guard = (
            _guard()
            .add_entity("CREDIT_CARD", action="warn")
            .add_entity("EMAIL", action="warn")
            .add_entity("TC_ID", action="warn")
            .add_entity("IBAN", action="warn")
        )
        text = "kart: 4111111111111111 mail: a@b.com TC: 12345678950"
        result = guard.scan(text)
        assert result.sanitized_text == text
        assert len(result.violations) > 0
        assert all(v.action == Action.WARN for v in result.violations)
        assert all(v.replacement is None for v in result.violations)


# ── S09: Hash consistency and salt effect ────────────────────────────────────


class TestS09HashConsistency:
    def test_same_input_same_hash(self):
        guard = _guard(salt="sabit-tuz")
        text = "kart: 4111111111111111"
        r1 = guard.scan(text)
        r2 = guard.scan(text)
        assert r1.sanitized_text == r2.sanitized_text

    def test_different_salt_different_hash(self):
        text = "TC: 12345678950"
        g1 = _guard(salt="tuz-a")
        g2 = _guard(salt="tuz-b")
        assert g1.scan(text).sanitized_text != g2.scan(text).sanitized_text

    def test_empty_salt_still_deterministic(self):
        guard = _guard(salt="")
        text = "mail: x@y.com"
        assert guard.scan(text).sanitized_text == guard.scan(text).sanitized_text

    def test_hash_placeholder_length_stable(self):
        # Placeholder format: [TYPE:8hexchars]
        guard = _guard(salt="tuz")
        guard.add_entity("EMAIL", action="hash")
        result = guard.scan("a@b.com")
        v = next(v for v in result.violations if v.entity_type == "EMAIL")
        # [EMAIL:xxxxxxxx] → 8 hex characters
        assert v.replacement is not None
        hex_part = v.replacement.split(":")[1].rstrip("]")
        assert len(hex_part) == 16


# ── S10: Salt change idempotency ─────────────────────────────────────────────


class TestS10SaltIdempotency:
    def test_set_salt_twice_uses_last(self):
        guard = _guard(salt="ilk")
        guard.set_salt("son")
        r1 = guard.scan("TC: 12345678950")
        guard.set_salt("son")
        r2 = guard.scan("TC: 12345678950")
        assert r1.sanitized_text == r2.sanitized_text


# ── S11: Overlapping entity resolution ───────────────────────────────────────


class TestS11OverlapResolution:
    def test_longer_span_wins(self):
        # TC_ID (11 digits) and PHONE should not overlap
        # 12345678950 → TC_ID (11 digits), PHONE not possible because no 0 prefix
        result = _guard().scan("numara: 12345678950 sonu")
        types = _types(result)
        assert "TC_ID" in types
        assert "PHONE" not in types

    def test_no_duplicate_violations_for_same_span(self):
        result = _guard().scan("4111111111111111")
        cc_violations = [v for v in result.violations if v.entity_type == "CREDIT_CARD"]
        assert len(cc_violations) == 1


# ── S12: TC identity boundary values ─────────────────────────────────────────


class TestS12TCIDBoundary:
    @pytest.mark.parametrize(
        "valid",
        [
            "12345678950",  # standard
            "10000000078",  # minimum (starts with 1)
            "99999999990",  # maximum
        ],
    )
    def test_valid_tc_detected(self, valid):
        result = _guard().scan(f"TC: {valid}")
        assert "TC_ID" in _types(result)

    @pytest.mark.parametrize(
        "invalid",
        [
            "1234567890",  # 10 digits — too short
            "012345678950",  # 12 digits — too long
            "01234567890",  # starts with 0 — invalid
        ],
    )
    def test_invalid_tc_not_detected(self, invalid):
        result = _guard().scan(f"TC: {invalid}")
        assert "TC_ID" not in _types(result)


# ── S13: False positives between card numbers ─────────────────────────────────


class TestS13CardFalsePositives:
    def test_order_number_not_card(self):
        result = _guard().scan("Sipariş: ORD-20240318-00001")
        assert "CREDIT_CARD" not in _types(result)

    def test_partial_card_not_matched(self):
        result = _guard().scan("son 4 hane: 1234")
        assert "CREDIT_CARD" not in _types(result)

    def test_price_not_card(self):
        result = _guard().scan("fiyat: 4.500,00 TL")
        assert "CREDIT_CARD" not in _types(result)


# ── S14: Batch scan mixed scenario ───────────────────────────────────────────


class TestS14BatchMixedScenario:
    LINES = [
        "Merhaba, nasılsın?",  # clean
        "kartım: 4111111111111111",  # CC
        "ali@test.com",  # email
        "TC kimliğim: 12345678950",  # TC_ID
        "IBAN: TR330006100519786457841326",  # IBAN
        "Tel: 0532 123 45 67",  # phone
        "Sunucu: 10.0.0.1",  # IP
        "Başka temiz metin.",  # clean
    ]

    def test_correct_result_count(self):
        results = _guard().scan_batch(self.LINES)
        assert len(results) == len(self.LINES)

    def test_clean_lines_identified(self):
        results = _guard().scan_batch(self.LINES)
        assert results[0].is_clean
        assert results[7].is_clean

    def test_each_entity_detected_in_correct_line(self):
        results = _guard().scan_batch(self.LINES)
        assert "CREDIT_CARD" in _types(results[1])
        assert "EMAIL" in _types(results[2])
        assert "TC_ID" in _types(results[3])
        assert "IBAN" in _types(results[4])
        assert "PHONE" in _types(results[5])
        assert "IP_ADDRESS" in _types(results[6])

    def test_batch_isolation(self):
        """Detection in one line should not affect other lines."""
        results = _guard().scan_batch(self.LINES)
        for i, result in enumerate(results):
            if i in (0, 7):
                continue
            # Each line's original_text should equal its LINES entry
            assert result.original_text == self.LINES[i]


# ── S15: Large / dense text performance ──────────────────────────────────────


class TestS15LargeText:
    def test_dense_pii_text_scanned_under_2s(self):
        pii_block = "TC: 12345678950 kart: 4111111111111111 mail: a@b.com tel: 0532 123 45 67\n"
        big_text = pii_block * 200  # 200 repetitions

        guard = _guard(salt="perf-test")
        start = time.perf_counter()
        result = guard.scan(big_text)
        elapsed = time.perf_counter() - start

        assert len(result.violations) > 0
        assert elapsed < 2.0, f"Scan too slow: {elapsed:.2f}s"


# ── S16: IBAN case insensitivity ─────────────────────────────────────────────


class TestS16IBANCaseInsensitive:
    @pytest.mark.parametrize(
        "iban",
        [
            "TR330006100519786457841326",  # uppercase (standard)
            "tr330006100519786457841326",  # lowercase
            "Tr330006100519786457841326",  # mixed
        ],
    )
    def test_iban_detected_regardless_of_case(self, iban):
        result = _guard().scan(f"IBAN: {iban}")
        assert "IBAN" in _types(result), f"Not detected: {iban}"


# ── S17: International phone formats ─────────────────────────────────────────


class TestS17PhoneFormats:
    @pytest.mark.parametrize(
        "phone",
        [
            "0532 123 45 67",  # national spaced
            "05321234567",  # national contiguous
            "+90 532 123 45 67",  # international spaced
            "+905321234567",  # international contiguous
            "90 532 123 45 67",  # starting with 90
            "0(532)1234567",  # with parentheses
        ],
    )
    def test_phone_detected(self, phone):
        result = _guard().scan(f"tel: {phone} ara")
        assert "PHONE" in _types(result), f"Not detected: {phone}"

    @pytest.mark.parametrize(
        "non_phone",
        [
            "1234567890",  # no prefix → should not be false positive
            "12 34 56",  # too short
            "2024-03-18",  # date
        ],
    )
    def test_non_phone_not_detected(self, non_phone):
        result = _guard().scan(non_phone)
        assert "PHONE" not in _types(result), f"False positive: {non_phone}"


# ── S18: Address with Turkish special characters ──────────────────────────────


class TestS18TurkishAddressChars:
    @pytest.mark.parametrize(
        "address",
        [
            "Çiğdem Sokak No:3",
            "Güneş Mahallesi",
            "İstiklal Caddesi No:45",
            "Şehit Ömer Bulvarı",
        ],
    )
    def test_turkish_address_detected(self, address):
        result = _guard().scan(address)
        assert "ADDRESS" in _types(result), f"Not detected: {address}"


# ── S19: Postal code boundary values ─────────────────────────────────────────


class TestS19PostalCodeBoundary:
    @pytest.mark.parametrize(
        "code",
        [
            "01000",  # minimum (Adana)
            "81999",  # maximum (Duzce)
            "34000",  # Istanbul
            "06100",  # Ankara
        ],
    )
    def test_valid_postal_code(self, code):
        result = _guard().scan(f"posta kodu: {code}")
        assert "POSTAL_CODE" in _types(result), f"Not detected: {code}"

    @pytest.mark.parametrize(
        "invalid",
        [
            "00000",  # 00xxx invalid
            "82000",  # 82+ invalid
            "99999",  # invalid
        ],
    )
    def test_invalid_postal_code_not_detected(self, invalid):
        result = _guard().scan(f"posta: {invalid}")
        assert "POSTAL_CODE" not in _types(result), f"False positive: {invalid}"


# ── S20: CLI JSON output schema validation ────────────────────────────────────


class TestS20CLIJsonSchema:
    def test_scan_json_schema(self, capsys):
        args = _build_parser().parse_args(
            ["scan", "--text", CUSTOMER_SUPPORT_PROMPT, "--no-ner", "--format", "json"]
        )
        cmd_scan(args)
        data = json.loads(capsys.readouterr().out)

        assert isinstance(data["is_clean"], bool)
        assert isinstance(data["sanitized_text"], str)
        assert isinstance(data["violations"], list)

        for v in data["violations"]:
            assert "entity_type" in v
            assert "original" in v
            assert "start" in v
            assert "end" in v
            assert "action" in v
            assert "replacement" in v
            assert v["action"] in ("warn", "hash")
            assert v["end"] > v["start"]


# ── S21: New global entity types ─────────────────────────────────────────────


class TestS21GlobalEntityTypes:
    """UUID, SSN, MAC_ADDRESS, JWT, IPv6, NIN detection tests (regex-based)."""

    def test_uuid_detected(self):
        result = _guard().scan("User UUID: 550e8400-e29b-41d4-a716-446655440000")
        assert "UUID" in _types(result)

    def test_ssn_detected(self):
        result = _guard().scan("SSN: 123-45-6789")
        assert "SSN" in _types(result)

    def test_mac_address_detected(self):
        result = _guard().scan("Device MAC: 00:1A:2B:3C:4D:5E")
        assert "MAC_ADDRESS" in _types(result)

    def test_mac_address_dash_separator_detected(self):
        result = _guard().scan("MAC: 00-1A-2B-3C-4D-5E")
        assert "MAC_ADDRESS" in _types(result)

    def test_jwt_detected(self):
        result = _guard().scan("session: eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ1c2VyIn0.abc123")
        assert "JWT" in _types(result)

    def test_ipv6_detected(self):
        result = _guard().scan("IPv6: 2001:db8::8a2e:0370:7334")
        assert "IPv6" in _types(result)

    def test_nin_detected(self):
        result = _guard().scan("NIN: AB123456C")
        assert "NIN" in _types(result)

    def test_mixed_new_entities_in_one_text(self):
        text = (
            "Device MAC: 00:1A:2B:3C:4D:5E, "
            "IPv6: 2001:db8::1, "
            "session: eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ1c2VyIn0.abc123"
        )
        result = _guard().scan(text)
        found = _types(result)
        assert "MAC_ADDRESS" in found
        assert "IPv6" in found
        assert "JWT" in found

    def test_uuid_ssn_nin_in_one_text(self):
        text = "User UUID: 550e8400-e29b-41d4-a716-446655440000, SSN: 123-45-6789, NIN: AB123456C"
        result = _guard().scan(text)
        found = _types(result)
        assert "UUID" in found
        assert "SSN" in found
        assert "NIN" in found

    def test_violation_positions_valid_for_new_entities(self):
        text = "UUID: 550e8400-e29b-41d4-a716-446655440000 SSN: 123-45-6789"
        result = _guard().scan(text)
        for v in result.violations:
            assert v.start >= 0
            assert v.end > v.start
            assert text[v.start : v.end] == v.original
