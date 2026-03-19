"""
Adversarial / kaçınma girdi testleri.

Amaç: Kütüphanenin gerçek dünya kaçınma tekniklerine karşı davranışını
belgeler. Tespit EDİLEMEYEN durumlar `xfail` ile işaretlenip neden
yakalanmadığı açıklanmıştır — bunlar bilinen sınırlar, bug değil.
"""
from __future__ import annotations

import pytest

from ai_guard import LLMGuard


@pytest.fixture
def g():
    return LLMGuard(use_ner=False)


# ══════════════════════════════════════════════════════════════════════════
# A01: Separator varyasyonları — kart numarası
# ══════════════════════════════════════════════════════════════════════════

class TestCardSeparatorVariants:
    def test_plain_digits(self, g):
        assert "CREDIT_CARD" in {v.entity_type for v in g.scan("4111111111111111").violations}

    def test_single_space_separator(self, g):
        assert "CREDIT_CARD" in {v.entity_type for v in g.scan("4111 1111 1111 1111").violations}

    def test_dash_separator(self, g):
        assert "CREDIT_CARD" in {v.entity_type for v in g.scan("4111-1111-1111-1111").violations}

    def test_mixed_separator(self, g):
        assert "CREDIT_CARD" in {v.entity_type for v in g.scan("4111 1111-1111 1111").violations}

    @pytest.mark.xfail(reason="Çift boşluk kaçınması: regex yalnızca tek separator destekler")
    def test_double_space_separator_evades(self, g):
        result = g.scan("4111  1111  1111  1111")
        assert "CREDIT_CARD" in {v.entity_type for v in result.violations}

    @pytest.mark.xfail(reason="Nokta separator: regex [\\s\\-]? ile nokta eşleşmez")
    def test_dot_separator_evades(self, g):
        result = g.scan("4111.1111.1111.1111")
        assert "CREDIT_CARD" in {v.entity_type for v in result.violations}


# ══════════════════════════════════════════════════════════════════════════
# A02: Unicode / görsel yanıltma
# ══════════════════════════════════════════════════════════════════════════

class TestUnicodeAdversarial:
    def test_normal_email_detected(self, g):
        assert "EMAIL" in {v.entity_type for v in g.scan("ali@test.com").violations}

    @pytest.mark.xfail(reason="Kiril 'а' görsel olarak Latin 'a' ile aynı görünür; regex ASCII'ye özgü")
    def test_cyrillic_homoglyph_evades(self, g):
        """Kiril karakterli e-posta: аli@test.com ('а' = U+0430, Kiril)"""
        cyrillic_a = "\u0430"   # Kiril küçük а
        email = f"{cyrillic_a}li@test.com"
        result = g.scan(email)
        assert "EMAIL" in {v.entity_type for v in result.violations}

    def test_emoji_in_text_does_not_break_detection(self, g):
        """Emojiler arasındaki PII hâlâ tespit edilmeli."""
        result = g.scan("🔒 TC: 12345678950 🔒")
        assert "TC_ID" in {v.entity_type for v in result.violations}

    def test_arabic_text_around_pii(self, g):
        """Arapça metin PII tespitini bozmamalı."""
        result = g.scan("مرحبا 4111111111111111 شكرا")
        assert "CREDIT_CARD" in {v.entity_type for v in result.violations}

    def test_cjk_text_around_pii(self, g):
        """CJK karakterler arasındaki PII hâlâ yakalanmalı."""
        result = g.scan("你好 a@b.com 谢谢")
        assert "EMAIL" in {v.entity_type for v in result.violations}


# ══════════════════════════════════════════════════════════════════════════
# A03: Gürültü / bağlam içine gömme
# ══════════════════════════════════════════════════════════════════════════

class TestEmbeddedInNoise:
    def test_pii_in_long_paragraph(self, g):
        prefix = "Sevgili müşterimiz, " * 30
        suffix = " lütfen bilgilerinizi güncel tutun." * 20
        text   = prefix + "Kartınız: 4111111111111111" + suffix
        result = g.scan(text)
        assert "CREDIT_CARD" in {v.entity_type for v in result.violations}

    def test_pii_after_many_numbers(self, g):
        """Sahte sayıların arasındaki gerçek PII seçilmeli."""
        text = " ".join([f"ref-{i:04d}" for i in range(50)]) + " TC: 12345678950"
        result = g.scan(text)
        tc_violations = [v for v in result.violations if v.entity_type == "TC_ID"]
        assert len(tc_violations) == 1
        assert tc_violations[0].original == "12345678950"

    def test_pii_split_across_label_and_value(self, g):
        """Label–değer formatında PII tespit edilmeli."""
        for fmt in [
            "Email: ali@test.com",
            "E-posta : ali@test.com",
            "EMAIL=ali@test.com",
            '"email":"ali@test.com"',
        ]:
            result = g.scan(fmt)
            assert "EMAIL" in {v.entity_type for v in result.violations}, \
                f"Format yakalanmadı: {fmt!r}"


# ══════════════════════════════════════════════════════════════════════════
# A04: Arka arkaya / bitişik entity'ler
# ══════════════════════════════════════════════════════════════════════════

class TestConsecutiveEntities:
    def test_two_emails_comma_separated(self, g):
        result = g.scan("a@b.com,c@d.com")
        emails = [v for v in result.violations if v.entity_type == "EMAIL"]
        assert len(emails) == 2

    def test_email_immediately_after_phone(self, g):
        """Aralarında separator olan durumda her ikisi de tespit edilmeli."""
        result = g.scan("0532 111 22 33 a@b.com")   # boşluk ile ayrılmış
        types = {v.entity_type for v in result.violations}
        assert "PHONE" in types
        assert "EMAIL" in types

    @pytest.mark.xfail(
        reason="Bitişik phone+email: email regex '33a@b.com' yakalıyor, "
               "phone ile çakışıyor, overlap çözümü telefonu koruyor (daha uzun span)"
    )
    def test_email_glued_to_phone_evades(self, g):
        """Separator olmadan bitişik phone+email: email kaçınma vektörü."""
        result = g.scan("0532 111 22 33a@b.com")
        types = {v.entity_type for v in result.violations}
        assert "PHONE" in types
        assert "EMAIL" in types

    @pytest.mark.xfail(reason="İki IBAN arasında separator yoksa kelime sınırı bozulur")
    def test_two_ibans_no_separator(self, g):
        ibans = "TR330006100519786457841326TR330006100519786457841327"
        result = g.scan(ibans)
        iban_violations = [v for v in result.violations if v.entity_type == "IBAN"]
        assert len(iban_violations) == 2


# ══════════════════════════════════════════════════════════════════════════
# A05: Kısmi / kesilmiş veriler
# ══════════════════════════════════════════════════════════════════════════

class TestPartialData:
    def test_partial_card_not_detected(self, g):
        result = g.scan("son 4 hane: **1234")
        assert "CREDIT_CARD" not in {v.entity_type for v in result.violations}

    def test_masked_card_not_detected(self, g):
        result = g.scan("kart: 4111 **** **** 1111")
        assert "CREDIT_CARD" not in {v.entity_type for v in result.violations}

    def test_partial_tc_10_digits_not_detected(self, g):
        result = g.scan("numara: 1234567890")   # 10 hane — TC değil
        assert "TC_ID" not in {v.entity_type for v in result.violations}

    def test_partial_iban_not_detected(self, g):
        # IBAN minimum 15 karakter — bunun altı eşleşmemeli
        result = g.scan("TR330006100")   # 11 karakter — regex minimumunun altı
        assert "IBAN" not in {v.entity_type for v in result.violations}

    @pytest.mark.xfail(
        reason="15 karakterlik dizi IBAN regex minimumunu (2+2+4+7) karşılar; "
               "uzunluk doğrulaması regex kapsamı dışında"
    )
    def test_15_char_iban_lookalike_evades(self, g):
        """15 karakter minimum IBAN örüntüsünü karşılar ama gerçek IBAN değildir."""
        result = g.scan("TR3300061005197")
        assert "IBAN" not in {v.entity_type for v in result.violations}


# ══════════════════════════════════════════════════════════════════════════
# A06: Büyük/küçük harf varyasyonları
# ══════════════════════════════════════════════════════════════════════════

class TestCaseVariants:
    def test_email_mixed_case(self, g):
        result = g.scan("ALI@TEST.COM")
        assert "EMAIL" in {v.entity_type for v in result.violations}

    def test_iban_lowercase(self, g):
        result = g.scan("tr330006100519786457841326")
        assert "IBAN" in {v.entity_type for v in result.violations}

    def test_iban_mixed_case(self, g):
        result = g.scan("Tr330006100519786457841326")
        assert "IBAN" in {v.entity_type for v in result.violations}


# ══════════════════════════════════════════════════════════════════════════
# A07: Sanitized metin bütünlüğü — replacement sonrası konum kayması
# ══════════════════════════════════════════════════════════════════════════

class TestSanitizedIntegrity:
    def test_all_original_originals_extractable_from_original_text(self, g):
        """Her violation'ın original alanı, original_text'ten çıkarılabilmeli."""
        texts = [
            "TC: 12345678950 kart: 4111111111111111 mail: x@y.com",
            "IBAN: TR330006100519786457841326 tel: 0532 111 22 33",
            "ip: 10.0.0.1 posta: 34000 adres: Atatürk Caddesi No:5",
        ]
        for text in texts:
            result = g.scan(text)
            for v in result.violations:
                extracted = text[v.start:v.end]
                assert extracted == v.original, (
                    f"[{v.entity_type}] beklenen={v.original!r}, "
                    f"konumdan çıkarılan={extracted!r} pos=[{v.start}:{v.end}]"
                )

    def test_sanitized_text_length_accounts_for_replacements(self, g):
        g2 = LLMGuard(use_ner=False)
        g2.configure_entity("EMAIL", enabled=True, action="hash")
        text   = "prefix a@b.com suffix"
        result = g2.scan(text)
        v = next(v for v in result.violations if v.entity_type == "EMAIL")
        # len(replacement) - len(original) = fark
        expected_delta = len(v.replacement) - len(v.original)
        actual_delta   = len(result.sanitized_text) - len(result.original_text)
        assert actual_delta == expected_delta

    def test_non_pii_parts_unchanged_in_sanitized(self, g):
        """Hassas olmayan metin bölümleri sanitized_text'te korunmalı."""
        prefix = "BAŞLANGIÇ "
        suffix = " SONUÇ"
        text   = prefix + "4111111111111111" + suffix
        result = g.scan(text)
        # CREDIT_CARD varsayılan → hash; prefix ve suffix değişmemeli
        assert result.sanitized_text.startswith(prefix)
        assert result.sanitized_text.endswith(suffix)
