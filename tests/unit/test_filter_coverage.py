"""Coverage added to the regex layer: wallets, IMEI, routing, NHS, EU IDs, secrets.

Every value used here is either a published test vector (the Bitcoin genesis
address, the BIP-173/350 examples, the NHS and IMEI test numbers, JPMorgan's
routing number) or is derived from the scheme's own check rule, so a failure
means the implementation is wrong rather than the fixture.
"""

from __future__ import annotations

import logging

import pytest

from wardcat import Action, Entity, Wardcat
from wardcat.detectors.regex_detector import (
    CONF_CHECKSUM,
    CONF_UNCUED,
    RegexDetector,
    _validate_aba_routing,
    _validate_crypto_wallet,
    _validate_eu_national_id,
    _validate_imei,
    _validate_nhs_number,
)


@pytest.fixture
def detector():
    """A regex detector with every regex-layer entity switched on."""
    from wardcat.core.registry import REGEX_ENTITIES

    return RegexDetector(set(REGEX_ENTITIES))


def types_in(detector, text: str) -> set[str]:
    return {span.entity_type for span in detector.detect(text)}


# ─────────────────────────────────────────────────────────────────────────────
# CRYPTO_WALLET
# ─────────────────────────────────────────────────────────────────────────────


class TestCryptoWallet:
    @pytest.mark.parametrize(
        "address",
        [
            "1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa",  # P2PKH — the genesis address
            "3J98t1WpEZ73CNmQviecrnyiWrnqRhWNLy",  # P2SH
            "bc1qar0srrr7xfkvy5l643lydnw9re59gtzzwf5mdq",  # bech32, BIP-173
            "bc1p5d7rjq7g6rdk2yhzks9smlaqtedr4dekq08ge8ztwac72sfr9rusxg3297",  # bech32m
            "tb1qw508d6qejxtdg4y5r3zarvary0c5xw7kxpjzsx",  # testnet bech32
            "0x52908400098527886E0F7030069857D2E4169EE7",  # Ethereum account
        ],
    )
    def test_valid_addresses_detected(self, detector, address):
        assert "CRYPTO_WALLET" in types_in(detector, f"send funds to {address} now")

    def test_base58_checksum_rejects_a_single_changed_character(self, detector):
        """The last four bytes are a double-SHA-256 of the payload, so one edit fails."""
        assert not _validate_crypto_wallet("1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNb")
        assert "CRYPTO_WALLET" not in types_in(detector, "pay 1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNb")

    def test_bech32_checksum_rejects_a_single_changed_character(self, detector):
        assert not _validate_crypto_wallet("bc1qar0srrr7xfkvy5l643lydnw9re59gtzzwf5mdr")

    def test_mixed_case_bech32_rejected(self):
        """BIP-173 forbids mixing cases — the checksum is defined over one of them."""
        assert not _validate_crypto_wallet("bc1QAr0srrr7xfkvy5l643lydnw9re59gtzzwf5mdq")

    def test_ethereum_wrong_length_rejected(self, detector):
        assert not _validate_crypto_wallet("0x529084000985278")
        assert "CRYPTO_WALLET" not in types_in(detector, "ref 0x529084000985278")

    def test_ordinary_prose_is_not_a_wallet(self, detector):
        assert "CRYPTO_WALLET" not in types_in(
            detector, "The invoice reference is 3000000000000000000000000000 exactly."
        )


# ─────────────────────────────────────────────────────────────────────────────
# IMEI
# ─────────────────────────────────────────────────────────────────────────────


class TestIMEI:
    @pytest.mark.parametrize(
        "text",
        [
            "IMEI: 490154203237518",
            "imei 490154203237518",
            "IMEI No. 490154203237518",
            "IMEI numarası: 490154203237518",
        ],
    )
    def test_keyword_forms_detected(self, detector, text):
        assert "IMEI" in types_in(detector, text)

    def test_luhn_failure_rejected(self, detector):
        assert not _validate_imei("490154203237519")
        assert "IMEI" not in types_in(detector, "IMEI: 490154203237519")

    def test_bare_run_is_a_candidate_not_a_finding(self, detector):
        """One bare 15-digit run in ten passes Luhn, so it is scored, not gated."""
        spans = [
            s
            for s in detector.detect("reference 490154203237518 logged")
            if s.entity_type == "IMEI"
        ]
        assert spans, "the bare form must still be seen"
        assert all(s.confidence == CONF_UNCUED for s in spans)

    def test_bare_run_is_not_acted_on_at_the_default_floor(self):
        guard = Wardcat(salt="s").add_entity(Entity.IMEI, Action.HASH)
        assert guard.scan("reference 490154203237518 logged").is_clean

    def test_bare_run_is_acted_on_when_the_floor_is_lowered(self):
        guard = Wardcat(salt="s").add_entity(Entity.IMEI, Action.HASH).with_min_confidence(0.6)
        result = guard.scan("reference 490154203237518 logged")
        assert "490154203237518" not in result.sanitized_text

    def test_wrong_length_rejected(self):
        assert not _validate_imei("49015420323751")


# ─────────────────────────────────────────────────────────────────────────────
# BANK_ROUTING
# ─────────────────────────────────────────────────────────────────────────────


class TestBankRouting:
    @pytest.mark.parametrize(
        "text",
        [
            "routing number 021000021",
            "ABA: 021000021",
            "RTN 021000021",
            "routing transit no. 021000021",
        ],
    )
    def test_keyword_forms_detected(self, detector, text):
        assert "BANK_ROUTING" in types_in(detector, text)

    def test_bad_check_digit_rejected(self, detector):
        assert not _validate_aba_routing("021000022")
        assert "BANK_ROUTING" not in types_in(detector, "routing number 021000022")

    def test_prefix_outside_the_federal_reserve_ranges_rejected(self):
        """A checksum-valid run whose leading pair is not a routing symbol in issue."""
        assert not _validate_aba_routing("991000029")

    def test_bare_run_is_a_candidate_not_a_finding(self, detector):
        spans = [
            s
            for s in detector.detect("invoice 021000021 settled")
            if s.entity_type == "BANK_ROUTING"
        ]
        assert spans
        assert all(s.confidence == CONF_UNCUED for s in spans)

    def test_bare_run_is_not_acted_on_at_the_default_floor(self):
        guard = Wardcat(salt="s").add_entity(Entity.BANK_ROUTING, Action.HASH)
        assert guard.scan("invoice 021000021 settled").is_clean


# ─────────────────────────────────────────────────────────────────────────────
# NHS_NUMBER
# ─────────────────────────────────────────────────────────────────────────────


class TestNHSNumber:
    @pytest.mark.parametrize(
        "text",
        [
            "NHS 943 476 5919",
            "NHS No: 9434765919",
            "nhs number 943-476-5919",
        ],
    )
    def test_keyword_forms_detected(self, detector, text):
        assert "NHS_NUMBER" in types_in(detector, text)

    def test_bad_check_digit_rejected(self, detector):
        assert not _validate_nhs_number("9434765910")
        assert "NHS_NUMBER" not in types_in(detector, "NHS 943 476 5910")

    def test_uncued_form_scores_below_the_default_floor(self, detector):
        """The NHS 3-3-4 grouping is also the US phone grouping.

        Mod-11 lets roughly one US-format number in eleven through, so the bare
        form cannot be treated as proven. It is still matched — the confidence
        tier, not a keyword gate, is what keeps it out of the output.
        """
        spans = [
            s for s in detector.detect("record 943 476 5919 filed") if s.entity_type == "NHS_NUMBER"
        ]
        assert spans
        assert all(s.confidence == CONF_UNCUED for s in spans)

    def test_a_phone_number_is_still_reported_as_a_phone_number(self):
        """Overlap resolution settles the collision: PHONE outranks an uncued NHS."""
        guard = (
            Wardcat(salt="s")
            .add_entities([Entity.PHONE, Entity.NHS_NUMBER], action=Action.WARN)
            .with_min_confidence(0.6)
        )
        found = {v.entity_type for v in guard.scan("call 0532 123 45 67 now").violations}
        assert found == {"PHONE"}

    def test_uncued_form_is_not_acted_on_at_the_default_floor(self):
        guard = Wardcat(salt="s").add_entity(Entity.NHS_NUMBER, Action.HASH)
        assert guard.scan("record 943 476 5919 filed").is_clean


# ─────────────────────────────────────────────────────────────────────────────
# EU_NATIONAL_ID — every scheme now carries its own check
# ─────────────────────────────────────────────────────────────────────────────


class TestEUNationalIDChecksums:
    @pytest.mark.parametrize(
        ("valid", "invalid"),
        [
            ("12345678Z", "12345678S"),  # Spanish DNI — check letter
            ("X1234567L", "X1234567M"),  # Spanish NIE
            ("180027512345676", "180027512345677"),  # French INSEE — 2-digit key
        ],
    )
    def test_check_character_decides(self, detector, valid, invalid):
        assert "EU_NATIONAL_ID" in types_in(detector, f"ID {valid} on file")
        assert "EU_NATIONAL_ID" not in types_in(detector, f"ID {invalid} on file")

    def test_dutch_bsn_elfproef(self, detector):
        assert "EU_NATIONAL_ID" in types_in(detector, "BSN 111222333 registered")
        assert "EU_NATIONAL_ID" not in types_in(detector, "BSN 111222334 registered")

    def test_polish_pesel_check_digit(self, detector):
        assert "EU_NATIONAL_ID" in types_in(detector, "PESEL 44051401359")
        assert "EU_NATIONAL_ID" not in types_in(detector, "PESEL 44051401358")

    @pytest.mark.parametrize("bare", ["111222333", "44051401359"])
    def test_bare_digit_runs_need_the_keyword(self, detector, bare):
        """Nine- and eleven-digit runs are ordinary in business text."""
        assert "EU_NATIONAL_ID" not in types_in(detector, f"order {bare} shipped")

    def test_validator_rejects_an_unrecognised_shape(self):
        assert not _validate_eu_national_id("not-an-id")


# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM_SECRET — provider tokens added to the existing set
# ─────────────────────────────────────────────────────────────────────────────


# Every value below is fabricated, but a good fixture has to carry the real
# shape — which is also what a credential scanner keys on. Assembling each one
# from a prefix and a body keeps the literal out of the file, so scanning this
# repository does not report a finding for its own test data.
def _token(prefix: str, body: str) -> str:
    return f"{prefix}{body}"


class TestSecretCoverage:
    @pytest.mark.parametrize(
        "secret",
        [
            _token("github_" + "pat_", "11ABCDEFG0abcdefghijklm_" + "A" * 59),
            _token("hf_", "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefg"),
            _token("shp" + "at_", "0123456789abcdef" * 2),
            _token("dop_" + "v1_", "a" * 64),
            _token("xa" + "pp-", "1-A01BCDEFGHI-1234567890123-abcdef"),
        ],
    )
    def test_provider_tokens_detected(self, detector, secret):
        assert "CUSTOM_SECRET" in types_in(detector, f"the token is {secret} ok")

    def test_azure_storage_account_key(self, detector):
        key = "AccountKey=" + "A" * 86 + "=="
        assert "CUSTOM_SECRET" in types_in(
            detector, f"DefaultEndpointsProtocol=https;AccountName=acme;{key};"
        )

    def test_aws_secret_key_matched_under_its_name(self, detector):
        """The key has no prefix, only a 40-character shape, so the name gates it."""
        key = _token("wJalrXUtnFEMI/K7MDENG", "bPxRfiCYEXAMPLEKEY12")
        text = f'aws_secret_access_key = "{key}"'
        assert "CUSTOM_SECRET" in types_in(detector, text)

    def test_google_service_account_key_id(self, detector):
        text = '"private_key_id": "a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0",'
        assert "CUSTOM_SECRET" in types_in(detector, text)

    def test_sentry_dsn_is_a_secret_not_an_email(self, detector):
        """Regression: the DSN's public key sits where an email's local part would.

        Before the DSN branch existed the EMAIL pattern claimed it, so a secret
        was reported under EMAIL's `warn` action and left in the text.
        """
        dsn = "https://abc123def456789012345678901234ab@o12345.ingest.sentry.io/678901"
        guard = Wardcat(salt="s").add_entities(
            [Entity.CUSTOM_SECRET, Entity.EMAIL], action=Action.HASH
        )
        found = {v.entity_type for v in guard.scan(f"SENTRY_DSN={dsn}").violations}
        assert found == {"CUSTOM_SECRET"}

    def test_an_ordinary_email_is_still_an_email(self, detector):
        assert "EMAIL" in types_in(detector, "write to ali.veli@example.com please")


# ─────────────────────────────────────────────────────────────────────────────
# Confidence tier — a checksum-gated match must not be overridable
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("text", "entity"),
    [
        ("wallet 1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa", "CRYPTO_WALLET"),
        ("IMEI: 490154203237518", "IMEI"),
        ("routing number 021000021", "BANK_ROUTING"),
        ("NHS 943 476 5919", "NHS_NUMBER"),
        ("DNI 12345678Z", "EU_NATIONAL_ID"),
    ],
)
def test_checksum_entities_report_full_confidence(detector, text, entity):
    spans = [s for s in detector.detect(text) if s.entity_type == entity]
    assert spans, f"{entity} not detected in {text!r}"
    assert all(s.confidence == CONF_CHECKSUM for s in spans)


# ─────────────────────────────────────────────────────────────────────────────
# End-to-end through the guard
# ─────────────────────────────────────────────────────────────────────────────


class TestThroughTheGuard:
    def test_new_entities_are_configurable_by_constant(self):
        guard = (
            Wardcat(salt="s")
            .add_entity(Entity.CRYPTO_WALLET, Action.HASH)
            .add_entity(Entity.IMEI, Action.REDACT)
        )
        result = guard.scan("wallet 1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa, IMEI: 490154203237518")
        assert "1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa" not in result.sanitized_text
        assert "490154203237518" not in result.sanitized_text
        assert "[IMEI]" in result.sanitized_text

    def test_entity_all_includes_the_new_types(self):
        guard = Wardcat(salt="s").add_entity(Entity.ALL, Action.WARN)
        enabled = guard.enabled_entities()
        for name in ("CRYPTO_WALLET", "IMEI", "BANK_ROUTING", "NHS_NUMBER"):
            assert name in enabled

    def test_group_helpers_carry_the_new_types(self):
        from wardcat import financial_entities, network_entities, uk_entities

        assert {"BANK_ROUTING", "CRYPTO_WALLET"} <= financial_entities()
        assert "IMEI" in network_entities()
        assert "NHS_NUMBER" in uk_entities()


# ─────────────────────────────────────────────────────────────────────────────
# Config loader — phone_regions was a valid default but an "unknown" YAML key
# ─────────────────────────────────────────────────────────────────────────────


def test_phone_regions_is_a_recognised_config_key(tmp_path, caplog):
    """Regression: setting it in YAML logged a typo warning for a supported key."""
    cfg = tmp_path / "policy.yaml"
    cfg.write_text('phone_regions: ["GB"]\n', encoding="utf-8")
    with caplog.at_level(logging.WARNING, logger="wardcat.config.loader"):
        from wardcat.config.loader import load_config

        loaded = load_config(cfg)
    assert loaded["phone_regions"] == ["GB"]
    assert "Unknown configuration key" not in caplog.text


# ─────────────────────────────────────────────────────────────────────────────
# The confidence floor itself
# ─────────────────────────────────────────────────────────────────────────────


class TestConfidenceFloor:
    def test_default_sits_between_the_uncued_tier_and_everything_else(self):
        from wardcat.config.loader import DEFAULT_CONFIG
        from wardcat.detectors.regex_detector import CONF_FUZZY

        floor = DEFAULT_CONFIG["min_confidence"]
        assert CONF_UNCUED < floor <= CONF_FUZZY

    def test_a_zero_floor_acts_on_everything_reported(self):
        guard = Wardcat(salt="s").add_entity(Entity.NHS_NUMBER, Action.HASH).with_min_confidence(0)
        assert not guard.scan("record 943 476 5919 filed").is_clean

    def test_raising_the_floor_drops_the_weaker_tiers(self):
        """A fuzzy address (0.90) goes when the floor is put above it."""
        text = "adres: Moda Caddesi No:42"
        kept = Wardcat(salt="s").add_entity(Entity.ADDRESS, Action.REDACT)
        dropped = (
            Wardcat(salt="s").add_entity(Entity.ADDRESS, Action.REDACT).with_min_confidence(0.95)
        )
        assert not kept.scan(text).is_clean
        assert dropped.scan(text).is_clean

    def test_a_checksummed_match_survives_any_floor(self):
        guard = (
            Wardcat(salt="s").add_entity(Entity.CREDIT_CARD, Action.HASH).with_min_confidence(1.0)
        )
        assert not guard.scan("card 4111 1111 1111 1111").is_clean

    @pytest.mark.parametrize("bad", [-0.1, 1.5, "high", True, None])
    def test_invalid_floor_rejected(self, bad):
        from wardcat.exceptions import ConfigError

        with pytest.raises(ConfigError, match="min_confidence"):
            Wardcat(salt="s").with_min_confidence(bad)


# ─────────────────────────────────────────────────────────────────────────────
# A credential introduced by the word for it, rather than by its own prefix
# ─────────────────────────────────────────────────────────────────────────────


class TestKeywordCuedCredentials:
    @pytest.mark.parametrize(
        ("text", "secret"),
        [
            ("örnek erişim parolası ise TestPass!2026 olarak", "TestPass!2026"),
            ("kullanıcı şifresi: Gizli!42x", "Gizli!42x"),
            ("şifreniz = Bahar2026!", "Bahar2026!"),
            ("parolam Yaz-2026x", "Yaz-2026x"),
            ("password is hunter2X", "hunter2X"),
            ("PASSWORD: Sup3rSecret", "Sup3rSecret"),
            ("api_key = abc123XYZdef", "abc123XYZdef"),
            ("access token: eyJhbGciOiJIUzI1", "eyJhbGciOiJIUzI1"),
            ("erişim kodu ALPHA-BRAVO-42", "ALPHA-BRAVO-42"),
            ('passphrase "correct-Horse-9"', "correct-Horse-9"),
        ],
    )
    def test_the_value_is_found(self, detector, text, secret):
        spans = [s for s in detector.detect(text) if s.entity_type == "CUSTOM_SECRET"]
        assert [s.text for s in spans] == [secret]

    def test_only_the_value_is_taken_not_the_keyword(self):
        """The word introducing the secret stays, so the redacted line still reads."""
        guard = Wardcat(salt="s").add_entity(Entity.CUSTOM_SECRET, Action.REDACT)
        out = guard.scan("kullanıcı şifresi: Gizli!42x").sanitized_text
        assert out == "kullanıcı şifresi: [CUSTOM_SECRET]"

    @pytest.mark.parametrize(
        "text",
        [
            "şifre yanlış girildi",
            "parola değiştirildi",
            "password is unknown",
            "şifresi unutulmuş",
            "parolanız sıfırlandı",
            "the password is not set",
            "api key rotation policy",
            "access code generation failed",
            "password reset requested",
            "şifre politikası güncellendi",
            "kullanıcı parolasını hatırlamıyor",
        ],
    )
    def test_an_ordinary_word_after_the_keyword_is_not_a_secret(self, detector, text):
        """These sentences all put a plain word where a credential would go.

        A credential mixes character classes; a lower-case word does not, and
        that is the whole gate — there is no shape to key on otherwise.
        """
        assert "CUSTOM_SECRET" not in types_in(detector, text)

    def test_scored_as_the_heuristic_it_is(self, detector):
        """The keyword is the only evidence, so it must not claim a prefix match's tier."""
        from wardcat.detectors.regex_detector import CONF_FUZZY

        spans = [
            s for s in detector.detect("şifresi: Gizli!42x") if s.entity_type == "CUSTOM_SECRET"
        ]
        assert spans and all(s.confidence == CONF_FUZZY for s in spans)

    def test_a_prefixed_token_still_reports_at_its_own_tier(self, detector):
        """A recognisable token is proof in itself, cue or no cue."""
        from wardcat.detectors.regex_detector import CONF_STRUCTURAL

        spans = [
            s
            for s in detector.detect("token ghp_ABCDEFGHIJKLMNOPQRSTUV0123456789")
            if s.entity_type == "CUSTOM_SECRET"
        ]
        assert spans and all(s.confidence == CONF_STRUCTURAL for s in spans)

    def test_dropped_when_the_floor_is_raised_above_the_heuristic_tier(self):
        guard = (
            Wardcat(salt="s")
            .add_entity(Entity.CUSTOM_SECRET, Action.REDACT)
            .with_min_confidence(0.95)
        )
        assert guard.scan("şifresi: Gizli!42x").is_clean


# ─────────────────────────────────────────────────────────────────────────────
# USERNAME — the other identifier with no shape of its own
# ─────────────────────────────────────────────────────────────────────────────


class TestUsername:
    @pytest.mark.parametrize(
        ("text", "handle"),
        [
            ("Şirket sistemindeki kullanıcı adı ahmet.yilmaz, örnek", "ahmet.yilmaz"),
            ("kullanıcı adınız: ahmet_y42", "ahmet_y42"),
            ("kullanıcı kodu AY-1042", "AY-1042"),
            ("hesap adı jsmith", "jsmith"),
            ("username: jsmith42", "jsmith42"),
            ("user name = ahmet.yilmaz", "ahmet.yilmaz"),
            ("login jdoe", "jdoe"),
            ("nickname: kedi_2026", "kedi_2026"),
            ("user id: 4471xyz", "4471xyz"),
        ],
    )
    def test_the_handle_is_found(self, detector, text, handle):
        spans = [s for s in detector.detect(text) if s.entity_type == "USERNAME"]
        assert [s.text for s in spans] == [handle]

    @pytest.mark.parametrize(
        "text",
        [
            "kullanıcı adı boş bırakılamaz",
            "kullanıcı adı zorunlu alandır",
            "kullanıcı adı bulunamadı",
            "kullanıcı adı yanlış girildi",
            "kullanıcı adı geçersiz",
            "kullanıcı adı güncellendi",
            "kullanıcı adı silindi",
            "kullanıcı adı hatalı",
            "kullanıcı adı tanımsız",
            "kullanıcı adı ile şifre eşleşmiyor",
            "username is required",
            "username not found",
            "username invalid",
            "login failed",
        ],
    )
    def test_an_outcome_word_is_not_a_handle(self, detector, text):
        """Nothing but the stoplist separates 'ahmetyilmaz' from 'bulunamadı'.

        Both are lower-case letter runs after the same keyword, so no shape rule
        can tell them apart — which is why the list exists and is tested.
        """
        assert "USERNAME" not in types_in(detector, text)

    def test_a_turkish_word_is_not_cut_into_a_handle(self, detector):
        """Handles are ASCII, so 'yanlış' must not arrive as 'yanlı'."""
        spans = [s for s in detector.detect("kullanıcı adı yanlış") if s.entity_type == "USERNAME"]
        assert spans == []

    def test_an_email_local_part_is_left_to_the_email_filter(self):
        guard = Wardcat(salt="s").add_entities(
            [Entity.USERNAME, Entity.EMAIL], action=Action.REDACT
        )
        result = guard.scan("mail ahmet.yilmaz@example.com")
        assert result.sanitized_text == "mail [EMAIL]"
        assert {v.entity_type for v in result.violations} == {"EMAIL"}

    def test_only_the_handle_is_taken(self):
        guard = Wardcat(salt="s").add_entity(Entity.USERNAME, Action.REDACT)
        out = guard.scan("Şirket sistemindeki kullanıcı adı ahmet.yilmaz kayıtlıdır").sanitized_text
        assert out == "Şirket sistemindeki kullanıcı adı [USERNAME] kayıtlıdır"

    def test_scored_as_the_heuristic_it_is(self, detector):
        from wardcat.detectors.regex_detector import CONF_FUZZY

        spans = [s for s in detector.detect("username: jsmith42") if s.entity_type == "USERNAME"]
        assert spans and all(s.confidence == CONF_FUZZY for s in spans)
