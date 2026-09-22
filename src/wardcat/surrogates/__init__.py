"""Realistic stand-ins for the ``surrogate`` action.

A model reasons better over ``Mehmet Kaya`` than over ``[PERSON]``, and a
downstream system that expects an e-mail address in a field wants something
shaped like one. A surrogate keeps the shape of what it replaces — a name is a
name from the guard's locale, an e-mail address stays an address on a reserved
domain, a card number passes Luhn, an IBAN passes mod-97, a TC number passes
its check digits — while carrying none of the original's information.

Two rules make surrogates usable at all:

* **Deterministic per salt.** The same value under the same salt gets the same
  surrogate in every scan, so a person keeps one name across documents and
  a record can be linked on it. Different salts give unrelated surrogates.
* **Unique within a scan.** Two distinct values never share a surrogate in one
  result, so :meth:`~wardcat.ScanResult.restore` is never ambiguous.

A type with no generator falls back to ``tokenize``, and the violation records
that honestly. Surrogates are not placeholders: nothing marks them as
substitutes, which is the point and the risk — see the security guide.
"""

from __future__ import annotations

import hashlib
import random
from collections.abc import Callable

from wardcat.surrogates._pools import (
    EMAIL_DOMAINS,
    FIRST_NAMES,
    LAST_NAMES,
    ORG_WORDS,
    SUPPORTED_LOCALES,
)

__all__ = ["SUPPORTED_LOCALES", "SurrogateAllocator", "supports"]

# IBAN lengths per country, for a surrogate that passes the same checksum the
# regex layer applies. Mirrors the detector's registry.
from wardcat.detectors.regex_detector import _IBAN_LENGTHS  # noqa: E402

_MAX_ATTEMPTS = 16


def _luhn_check_digit(body: str) -> str:
    total = 0
    for i, ch in enumerate(reversed(body)):
        d = int(ch)
        if i % 2 == 0:  # the check digit itself would be position 0
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return str((10 - total % 10) % 10)


def _iban_check_digits(country: str, bban: str) -> str:
    rearranged = bban + country + "00"
    numeric = "".join(str(int(c, 36)) for c in rearranged)
    return f"{98 - int(numeric) % 97:02d}"


def _tc_from(rng: random.Random) -> str:
    d = [rng.randint(1, 9)] + [rng.randint(0, 9) for _ in range(8)]
    d10 = ((d[0] + d[2] + d[4] + d[6] + d[8]) * 7 - (d[1] + d[3] + d[5] + d[7])) % 10
    d11 = (sum(d) + d10) % 10
    return "".join(map(str, d + [d10, d11]))


def _fill_digits(template: str, digits: str) -> str:
    """Write *digits* into the digit positions of *template*, keeping its separators."""
    out = []
    it = iter(digits)
    for ch in template:
        out.append(next(it) if ch.isdigit() else ch)
    return "".join(out)


def _match_case(original: str, generated: str) -> str:
    letters = [c for c in original if c.isalpha()]
    if letters and all(c.isupper() for c in letters):
        return generated.upper()
    if letters and all(c.islower() for c in letters):
        return generated.lower()
    return generated


def _person(value: str, rng: random.Random, locale: str) -> str:
    first = rng.choice(FIRST_NAMES[locale])
    last = rng.choice(LAST_NAMES[locale])
    words = len(value.split())
    if words <= 1:
        name = first
    elif words == 2:
        name = f"{first} {last}"
    else:
        name = f"{first} {rng.choice(FIRST_NAMES[locale])} {last}"
    return _match_case(value, name)


def _org(value: str, rng: random.Random, locale: str) -> str:
    first, second, forms = ORG_WORDS[locale]
    name = f"{rng.choice(first)} {rng.choice(second)} {rng.choice(forms)}"
    return _match_case(value, name)


def _email(value: str, rng: random.Random, locale: str) -> str:
    first = rng.choice(FIRST_NAMES["en"]).lower()
    last = rng.choice(LAST_NAMES["en"]).lower()
    local = rng.choice(
        (f"{first}.{last}", f"{first}{last}", f"{first[0]}{last}{rng.randint(1, 99)}")
    )
    address = f"{local}@{rng.choice(EMAIL_DOMAINS)}"
    return address.upper() if value.isupper() else address


def _phone(value: str, rng: random.Random, locale: str) -> str:
    digits = [c for c in value if c.isdigit()]
    generated = [str(rng.randint(0, 9)) for _ in digits]
    # Keep what identifies the numbering plan rather than the subscriber: a
    # leading + country code, or the trunk 0 with its area or operator code.
    keep = 0
    stripped = value.lstrip("+ ")
    if value.startswith("+"):
        keep = 2 + (1 if len(digits) > 11 else 0)
    elif stripped.startswith("0"):
        keep = 4
    for i in range(min(keep, len(digits))):
        generated[i] = digits[i]
    # A ten-digit North American shape gets the 555-01xx fiction reserved for it.
    if len(digits) == 10 and keep == 0:
        generated[3:6] = list("555")
        generated[6:8] = list("01")
    return _fill_digits(value, "".join(generated))


def _card(value: str, rng: random.Random, locale: str) -> str:
    digits = [c for c in value if c.isdigit()]
    length = len(digits) if 12 <= len(digits) <= 19 else 16
    prefix = "37" if length == 15 else "4"
    body = prefix + "".join(str(rng.randint(0, 9)) for _ in range(length - len(prefix) - 1))
    return _fill_digits(value, body + _luhn_check_digit(body))


def _iban(value: str, rng: random.Random, locale: str) -> str:
    compact = value.replace(" ", "").upper()
    country = (
        compact[:2]
        if compact[:2] in _IBAN_LENGTHS
        else {"tr": "TR", "de": "DE", "fr": "FR"}.get(locale, "GB")
    )
    length = _IBAN_LENGTHS[country]
    bban_template = (
        compact[4:] if compact[:2] == country and len(compact) == length else "0" * (length - 4)
    )
    bban = "".join(
        str(rng.randint(0, 9)) if ch.isdigit() or ch == "0" else chr(rng.randint(65, 90))
        for ch in bban_template
    )
    generated = country + _iban_check_digits(country, bban) + bban
    if len(generated) == len(compact) and compact[:2] == country:
        # Keep the original's spacing and case.
        out, it = [], iter(generated)
        for ch in value:
            out.append(next(it) if ch != " " else " ")
        text = "".join(out)
        return text.lower() if value.islower() else text
    return generated


def _tc_id(value: str, rng: random.Random, locale: str) -> str:
    return _fill_digits(value, _tc_from(rng))


def _ssn(value: str, rng: random.Random, locale: str) -> str:
    # A shape the detector itself accepts (area 001–899 except 666), so a second
    # pass over the output still sees an SSN where one stood.
    area = rng.choice([a for a in range(1, 900) if a != 666])
    digits = f"{area:03d}{rng.randint(1, 99):02d}{rng.randint(1, 9999):04d}"
    return _fill_digits(value, digits)


def _ipv4(value: str, rng: random.Random, locale: str) -> str:
    block = rng.choice(("203.0.113", "198.51.100", "192.0.2"))  # TEST-NET ranges
    return f"{block}.{rng.randint(1, 254)}"


def _ipv6(value: str, rng: random.Random, locale: str) -> str:
    return f"2001:db8::{rng.randint(1, 0xFFFF):x}:{rng.randint(1, 0xFFFF):x}"


def _username(value: str, rng: random.Random, locale: str) -> str:
    first = rng.choice(FIRST_NAMES["en"]).lower()
    last = rng.choice(LAST_NAMES["en"]).lower()
    sep = "." if "." in value else "_" if "_" in value else ""
    return f"{first}{sep}{last}{rng.randint(1, 99) if any(c.isdigit() for c in value) else ''}"


Generator = Callable[[str, random.Random, str], str]

_GENERATORS: dict[str, Generator] = {
    "PERSON": _person,
    "ORG": _org,
    "EMAIL": _email,
    "PHONE": _phone,
    "CREDIT_CARD": _card,
    "IBAN": _iban,
    "TC_ID": _tc_id,
    "SSN": _ssn,
    "IP_ADDRESS": _ipv4,
    "IPv6": _ipv6,
    "USERNAME": _username,
}


def supports(entity_type: str) -> bool:
    """Whether the ``surrogate`` action has a generator for *entity_type*."""
    return entity_type in _GENERATORS


class SurrogateAllocator:
    """Hands out surrogates for one scan: deterministic per salt, unique per scan.

    .. warning::
        Holds the raw values it has seen for the lifetime of the instance.
    """

    def __init__(self, salt: str = "", locale: str = "en") -> None:
        if locale not in SUPPORTED_LOCALES:
            raise ValueError(f"unsupported surrogate locale {locale!r}; one of {SUPPORTED_LOCALES}")
        self.salt = salt
        self.locale = locale
        self._by_value: dict[tuple[str, str], str] = {}
        self._taken: dict[tuple[str, str], str] = {}

    def surrogate_for(self, entity_type: str, value: str) -> str | None:
        """The surrogate for *value*, or ``None`` when the type has no generator."""
        generator = _GENERATORS.get(entity_type)
        if generator is None:
            return None
        key = (entity_type, value)
        known = self._by_value.get(key)
        if known is not None:
            return known
        seed = hashlib.sha256(f"{self.salt}|{entity_type}|{value}".encode()).digest()
        for attempt in range(_MAX_ATTEMPTS):
            rng = random.Random(seed + bytes([attempt]))
            candidate = generator(value, rng, self.locale)
            owner = self._taken.get((entity_type, candidate))
            if owner is None or owner == value:
                break
        else:  # pragma: no cover - sixteen collisions in one scan
            candidate = f"{candidate}{attempt}"
        # A surrogate must differ from the value it stands for, or restore has
        # nothing to do and the value stays in the text.
        if candidate == value:
            candidate = generator(value, random.Random(seed + b"\xff"), self.locale)
        self._by_value[key] = candidate
        self._taken[(entity_type, candidate)] = value
        return candidate
