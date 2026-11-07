"""E-mail addresses written with a bracketed "at" to keep harvesters off them.

"name[at]example[dot]com" and "ali.veli(at)firma(nokta)com" are addresses all
the same; the plain pattern needs a literal "@". The bracket around the "at" is
what makes the form unambiguous, so only that form is matched: a spoken
"john dot doe at gmail dot com" reads like any sentence with "at" in it and is
left to the LLM layer.
"""

from __future__ import annotations

import pytest

from wardcat import Action, Entity, Wardcat
from wardcat.detectors.regex_detector import (
    _OBFUSCATED_EMAIL,
    CONF_STRUCTURAL,
    RegexDetector,
)
from wardcat.utils.regex_safety import is_catastrophic


def emails(text: str) -> list[str]:
    return [s.text for s in RegexDetector({"EMAIL"}).detect(text) if s.entity_type == "EMAIL"]


class TestBracketedAddresses:
    @pytest.mark.parametrize(
        ("text", "address"),
        [
            (
                "Contact: m.alvarez[at]protonmail[dot]com",
                "m.alvarez[at]protonmail[dot]com",
            ),
            (
                "Mailim ahmet.yilmaz(at)hotmail(nokta)com, oraya yazın.",
                "ahmet.yilmaz(at)hotmail(nokta)com",
            ),
            (
                "Write to info [at] example [dot] co [dot] uk today",
                "info [at] example [dot] co [dot] uk",
            ),
            ("jane(at)example.org", "jane(at)example.org"),
            ("support{at}mail.example.com.tr", "support{at}mail.example.com.tr"),
            ("kontakt[at]firma[punkt]de", "kontakt[at]firma[punkt]de"),
            ("J.DOE(AT)EXAMPLE(DOT)COM", "J.DOE(AT)EXAMPLE(DOT)COM"),
        ],
    )
    def test_the_address_is_found(self, text: str, address: str) -> None:
        assert emails(text) == [address]


class TestWhatIsNotAnAddress:
    @pytest.mark.parametrize(
        "text",
        [
            "meet at noon",
            "look (at) this",
            "price[at]5[dot]99",
            "see [at] page 3",
            "user at example dot com",
            "a[at]b",
            "@handle [dot] com",
        ],
    )
    def test_nothing_is_reported(self, text: str) -> None:
        assert emails(text) == []

    def test_a_plain_address_is_reported_once(self) -> None:
        assert emails("mail jane@example.org now") == ["jane@example.org"]


class TestConfidenceAndGuard:
    def test_the_form_is_structural(self) -> None:
        [span] = RegexDetector({"EMAIL"}).detect("jane(at)example.org")

        assert span.confidence == CONF_STRUCTURAL

    def test_off_when_email_is_not_enabled(self) -> None:
        assert RegexDetector({"PHONE"}).detect("jane(at)example.org") == []

    def test_redacted_through_the_guard(self) -> None:
        guard = Wardcat(salt="s").add_entity(Entity.EMAIL, Action.REDACT)

        result = guard.scan("Contact: m.alvarez[at]protonmail[dot]com, thanks")

        assert result.sanitized_text == "Contact: [EMAIL], thanks"


def test_the_pattern_passes_the_redos_screen() -> None:
    assert not is_catastrophic(_OBFUSCATED_EMAIL)
