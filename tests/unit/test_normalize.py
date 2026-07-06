"""Confusable-folding normalizer tests.

Covers the offset-preserving folder (wardcat.utils.normalize) and its wiring
into RegexDetector: homoglyph-obfuscated PII must be detected, spans must map
back to the *original* text, and all-ASCII input must be untouched.
"""

from __future__ import annotations

from wardcat.detectors.regex_detector import RegexDetector
from wardcat.utils.normalize import fold_confusables, has_confusables


class TestFoldConfusables:
    def test_length_is_preserved(self):
        # The offset invariant the whole design relies on.
        for s in ["ali@tеst.com", "４111", "٤٥٦", "ΑΒΕ mixed", "plain ascii"]:
            assert len(fold_confusables(s)) == len(s)

    def test_cyrillic_lookalikes_fold_to_latin(self):
        assert fold_confusables("аeо") == "aeo"  # Cyrillic а, Latin e, Cyrillic о
        assert fold_confusables("СОР") == "COP"  # all Cyrillic uppercase

    def test_greek_lookalikes_fold_to_latin(self):
        assert fold_confusables("ΑΒΕΗ") == "ABEH"  # Greek uppercase
        assert fold_confusables("ορι") == "opi"  # Greek lowercase

    def test_digit_lookalikes_fold_to_ascii(self):
        assert fold_confusables("٤١٠") == "410"  # Arabic-Indic
        assert fold_confusables("۴۱۰") == "410"  # Extended Arabic-Indic
        assert fold_confusables("４１０") == "410"  # Fullwidth

    def test_fullwidth_letters_fold_to_ascii(self):
        assert fold_confusables("ＡＢｃ") == "ABc"

    def test_plain_ascii_is_identity(self):
        s = "Regular text: ali@test.com 4111111111111111"
        assert fold_confusables(s) is s or fold_confusables(s) == s
        assert fold_confusables(s) == s

    def test_has_confusables(self):
        assert has_confusables("ali@tеst.com") is True  # Cyrillic е
        assert has_confusables("ali@test.com") is False


class TestRegexDetectorFolding:
    def test_cyrillic_domain_email_is_detected(self):
        d = RegexDetector({"EMAIL"})
        text = "reach me at ali@tеst.com please"  # Cyrillic е in domain
        spans = d.detect(text)
        assert len(spans) == 1
        assert spans[0].entity_type == "EMAIL"
        # The reported span slices back to the ORIGINAL (homoglyph) substring.
        s = spans[0]
        assert text[s.start : s.end] == s.text == "ali@tеst.com"

    def test_fullwidth_digit_card_is_detected_and_checksum_validated(self):
        d = RegexDetector({"CREDIT_CARD"})
        spans = d.detect("card ４111111111111111 end")  # fullwidth leading 4
        assert len(spans) == 1
        assert spans[0].entity_type == "CREDIT_CARD"

    def test_folding_can_be_disabled(self):
        d = RegexDetector({"EMAIL"}, fold_confusables_enabled=False)
        # With folding off, the Cyrillic domain bypasses the ASCII domain class.
        assert d.detect("ali@tеst.com") == []

    def test_all_ascii_behaviour_unchanged(self):
        on = RegexDetector({"EMAIL", "CREDIT_CARD"})
        off = RegexDetector({"EMAIL", "CREDIT_CARD"}, fold_confusables_enabled=False)
        text = "ali@test.com paid with 4111111111111111"
        got_on = [(s.entity_type, s.start, s.end) for s in on.detect(text)]
        got_off = [(s.entity_type, s.start, s.end) for s in off.detect(text)]
        assert got_on == got_off

    def test_offsets_survive_multi_span_mixed_input(self):
        d = RegexDetector({"EMAIL", "TC_ID"})
        text = "mail аli@example.com and tc 10000000146 done"  # Cyrillic а
        for s in d.detect(text):
            assert text[s.start : s.end] == s.text
