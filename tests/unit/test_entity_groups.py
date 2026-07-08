"""
Entity group helper tests.

Verifies that the entity group functions return correct, non-overlapping
frozensets and that all returned types are known entity types.
"""

from __future__ import annotations

from wardcat import (
    all_entities,
    core_entities,
    european_entities,
    financial_entities,
    identity_entities,
    network_entities,
    turkish_entities,
    uk_entities,
    us_entities,
)
from wardcat.core.models import KNOWN_ENTITY_TYPES


class TestEntityGroupBasics:
    def test_core_entities_non_empty(self):
        assert len(core_entities()) > 0

    def test_all_groups_return_frozensets(self):
        groups = [
            core_entities(),
            financial_entities(),
            turkish_entities(),
            european_entities(),
            uk_entities(),
            us_entities(),
            network_entities(),
            identity_entities(),
            all_entities(),
        ]
        for g in groups:
            assert isinstance(g, frozenset)

    def test_all_entities_contains_known_types(self):
        assert all_entities() == frozenset(KNOWN_ENTITY_TYPES)

    def test_groups_are_subsets_of_all(self):
        all_e = all_entities()
        for group_fn in [
            core_entities,
            financial_entities,
            turkish_entities,
            european_entities,
            uk_entities,
            us_entities,
            network_entities,
            identity_entities,
        ]:
            assert group_fn().issubset(all_e), f"{group_fn.__name__} has unknown entity"


class TestGroupContents:
    def test_core_contains_email_phone_cc_iban(self):
        core = core_entities()
        assert "EMAIL" in core
        assert "PHONE" in core
        assert "CREDIT_CARD" in core
        assert "IBAN" in core

    def test_turkish_contains_tc_id_and_postal(self):
        tr = turkish_entities()
        assert "TC_ID" in tr
        assert "POSTAL_CODE" in tr

    def test_turkish_includes_core(self):
        assert core_entities().issubset(turkish_entities())

    def test_uk_contains_nin_postal_passport(self):
        uk = uk_entities()
        assert "NIN" in uk
        assert "UK_POSTAL_CODE" in uk
        assert "PASSPORT" in uk

    def test_us_contains_ssn_zip_passport(self):
        us = us_entities()
        assert "SSN" in us
        assert "US_ZIP_CODE" in us
        assert "PASSPORT" in us

    def test_european_contains_eu_national_id_and_codice_fiscale(self):
        eu = european_entities()
        assert "EU_NATIONAL_ID" in eu
        assert "CODICE_FISCALE" in eu
        assert "PASSPORT" in eu

    def test_network_contains_ip_ipv6_mac_uuid_jwt(self):
        net = network_entities()
        assert "IP_ADDRESS" in net
        assert "IPv6" in net
        assert "MAC_ADDRESS" in net
        assert "UUID" in net
        assert "JWT" in net

    def test_financial_contains_cc_iban_ssn(self):
        fin = financial_entities()
        assert "CREDIT_CARD" in fin
        assert "IBAN" in fin
        assert "SSN" in fin

    def test_identity_contains_passport_nin_ssn_tc_id(self):
        idt = identity_entities()
        assert "PASSPORT" in idt
        assert "NIN" in idt
        assert "SSN" in idt
        assert "TC_ID" in idt
        assert "EU_NATIONAL_ID" in idt
        assert "CODICE_FISCALE" in idt


class TestEntityGroupsWithGuard:
    def test_configure_uk_group(self):
        from wardcat import Wardcat

        guard = Wardcat()
        for entity in uk_entities():
            guard.add_entity(entity, action="warn")
        result = guard.scan("Passport: AB1234567")
        assert not result.is_clean

    def test_configure_network_group(self):
        from wardcat import Wardcat, network_entities

        guard = Wardcat().add_entities(network_entities())
        result = guard.scan("Server IP: 192.168.1.100")
        assert any(v.entity_type == "IP_ADDRESS" for v in result.violations)
