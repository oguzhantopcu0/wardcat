"""
Kapsamlı filtre kapsama testi — Regex + NER (tr_core_news_lg) + LLM (gemma3:12b).

6 uzun, gerçekçi ve TAMAMEN KURGUSAL metin üzerinde tüm detektörleri çalıştırır.
Her metin için bir "ground truth" (beklenen PII) listesi tanımlıdır; script
tespit edilenleri bununla karşılaştırıp recall (yakalama oranı), kaçırılanlar,
yanlış etiketlenenler ve fazladan (beklenmeyen) tespitleri raporlar.

Tüm TC/IBAN değerleri checksum doğrulamasından geçen kurgusal değerlerdir;
kişiler, e-postalar ve sırlar uydurmadır — gerçek bir kişiye ait değildir.

Çalıştırma:
    uv run python filter_coverage_test.py 2>&1 | tee filter_coverage_results.txt
"""

from __future__ import annotations

import time

from wardcat import Wardcat

# ── Renk kodları ──────────────────────────────────────────────────────────────
GREEN, RED, YELLOW, CYAN, BOLD, DIM, RESET = (
    "\033[92m",
    "\033[91m",
    "\033[93m",
    "\033[96m",
    "\033[1m",
    "\033[2m",
    "\033[0m",
)

# ═══════════════════════════════════════════════════════════════════════════════
# TEST METİNLERİ + GROUND TRUTH
# Her giriş: (başlık, metin, [(entity_type, beklenen_değer), ...])
# ═══════════════════════════════════════════════════════════════════════════════

CASES: list[tuple[str, str, list[tuple[str, str]]]] = []

# ── 1. TR — Banka müşteri destek kaydı ─────────────────────────────────────────
CASES.append(
    (
        "TR · Banka Müşteri Destek Kaydı",
        """\
MÜŞTERİ HİZMETLERİ GÖRÜŞME KAYDI — Referans No: TLP-2024-08815

Görüşme Tarihi: 14 Mart 2024, Saat 11:42
Temsilci: Mehmet Kaya (Sicil: 4471)

Sayın müşterimiz Ayşe Yılmaz, kayıtlı cep telefonu 0532 123 4567 üzerinden
aradı. Kimlik doğrulaması için T.C. Kimlik numarası 62601815964 ve doğum
tarihi 15 Mart 1988 teyit edildi. Kayıtlı e-posta adresi
ayse.yilmaz@gmail.com olarak güncellendi.

Müşteri, IBAN TR33 0006 1005 1978 6457 8413 26 numaralı vadesiz hesabından
yapılan 45.000 TL tutarındaki havaleye itiraz etti. Ayrıca kredi kartı
4111 1111 1111 1111 ile yapılan bir işlemin kendisine ait olmadığını beyan
etti. Kart geçici olarak bloke edildi.

Müşterinin ikamet adresi: Atatürk Caddesi No 45 Daire 7, Çankaya 06680
Ankara. Servise bıraktığı aracın plakası 34 ABC 123 olarak kayıtlara geçti.

Aksiyon: İşlem araştırmaya alındı, müşteriye 3 iş günü içinde dönülecek.
""",
        [
            ("PERSON", "Mehmet Kaya"),
            ("PERSON", "Ayşe Yılmaz"),
            ("PHONE", "0532 123 4567"),
            ("TC_ID", "62601815964"),
            ("DATE_OF_BIRTH", "15 Mart 1988"),
            ("EMAIL", "ayse.yilmaz@gmail.com"),
            ("IBAN", "TR33 0006 1005 1978 6457 8413 26"),
            ("FINANCIAL_AMOUNT", "45.000 TL"),
            ("CREDIT_CARD", "4111 1111 1111 1111"),
            ("ADDRESS", "Atatürk Caddesi"),
            ("POSTAL_CODE", "06680"),
            ("VEHICLE_PLATE", "34 ABC 123"),
        ],
    )
)

# ── 2. EN — US İK / işe alım kaydı ─────────────────────────────────────────────
CASES.append(
    (
        "EN · US Employment / HR Record",
        """\
CONFIDENTIAL — EMPLOYEE ONBOARDING RECORD

New hire: John Anderson
Position: Senior Backend Engineer, Platform Team
Start date: April 1, 2024

The candidate John Anderson completed all pre-employment paperwork. His
Social Security Number 078-05-1120 was verified against the I-9 form.
Date of birth: 1985-07-22. Personal email john.anderson@example.com and
mobile phone +1 415 555 0132 were added to the HR system.

Home address on file: 123 Main Street, Springfield, IL 62704. The mailing
ZIP code 90210-1234 corresponds to the previous residence and should be
removed.

Compensation: the agreed annual base salary is $85,000 with a signing bonus
of $10,000. Direct deposit will be set up after the first pay cycle.
""",
        [
            ("PERSON", "John Anderson"),
            ("SSN", "078-05-1120"),
            ("DATE_OF_BIRTH", "1985-07-22"),
            ("EMAIL", "john.anderson@example.com"),
            ("PHONE", "+1 415 555 0132"),
            ("ADDRESS", "123 Main Street"),
            ("US_ZIP_CODE", "90210-1234"),
            ("FINANCIAL_AMOUNT", "$85,000"),
        ],
    )
)

# ── 3. Mixed — DevOps olay raporu (sırlar ağırlıklı) ───────────────────────────
CASES.append(
    (
        "Mixed · DevOps Incident Report (secrets)",
        """\
POST-MORTEM — Incident INC-2024-0312 (production outage)

Hazırlayan: Selin Aydın (selin.aydin@company.io)

Kök neden: yanlışlıkla commit edilen kimlik bilgileri. Sızan config:

DATABASE_URL=postgresql://app_admin:Pr0d_P4ssw0rd@db-master.internal:5432/core
AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE
OPENAI_API_KEY=sk-proj-9aZqLmNwX1234567890abcdef
GITHUB_TOKEN=ghp_AbCdEfGh1234567890IjKlMnOpQrStUv

Etkilenen sunucu IPv4 10.0.0.45, IPv6 adresi
2001:0db8:85a3:0000:0000:8a2e:0370:7334 ve ağ arayüzü MAC adresi
00:1A:2B:3C:4D:5E olarak tespit edildi. İlgili oturum tokeni
eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c
loglarda açık halde görüldü.

Olay kaydının dahili izleme kimliği (UUID):
550e8400-e29b-41d4-a716-446655440000. Tüm sızan anahtarlar rotate edildi.
""",
        [
            ("PERSON", "Selin Aydın"),
            ("EMAIL", "selin.aydin@company.io"),
            ("CUSTOM_SECRET", "Pr0d_P4ssw0rd"),
            ("CUSTOM_SECRET", "AKIAIOSFODNN7EXAMPLE"),
            ("CUSTOM_SECRET", "sk-proj-9aZqLmNwX1234567890abcdef"),
            ("CUSTOM_SECRET", "ghp_AbCdEfGh1234567890IjKlMnOpQrStUv"),
            ("IP_ADDRESS", "10.0.0.45"),
            ("IPv6", "2001:0db8:85a3:0000:0000:8a2e:0370:7334"),
            ("MAC_ADDRESS", "00:1A:2B:3C:4D:5E"),
            ("JWT", "eyJhbGciOiJIUzI1NiJ9"),
            ("UUID", "550e8400-e29b-41d4-a716-446655440000"),
        ],
    )
)

# ── 4. EN — UK / EU uyum belgesi ───────────────────────────────────────────────
CASES.append(
    (
        "EN · UK / EU Compliance Document",
        """\
DATA SUBJECT ACCESS REQUEST — Case GDPR-2024-117

Requestor: Oliver Bennett
Relationship: data subject (self)

We have located the following personal data for Oliver Bennett. His UK
National Insurance Number AB123456C is held in the payroll system. The
registered residential postcode is SW1A 1AA, London.

Identity was confirmed using passport no: U1234567 and a Spanish national
ID document 12345678Z provided during the 2019 onboarding. Banking details
on file include IBAN GB29 NWBK 6016 1331 9268 19.

Contact details: email oliver.bennett@protonmail.com and phone
+44 20 7946 0958. Please confirm within one calendar month per UK GDPR
Article 15.
""",
        [
            ("PERSON", "Oliver Bennett"),
            ("NIN", "AB123456C"),
            ("UK_POSTAL_CODE", "SW1A 1AA"),
            ("PASSPORT", "U1234567"),
            ("EU_NATIONAL_ID", "12345678Z"),
            ("IBAN", "GB29 NWBK 6016 1331 9268 19"),
            ("EMAIL", "oliver.bennett@protonmail.com"),
            ("PHONE", "+44 20 7946 0958"),
        ],
    )
)

# ── 5. TR — Tedarik sözleşmesi / e-ticaret ─────────────────────────────────────
CASES.append(
    (
        "TR · Tedarik Sözleşmesi",
        """\
TEDARİK VE HİZMET SÖZLEŞMESİ

İşbu sözleşme; satıcı sıfatıyla hareket eden tedarikçi ile alıcı arasında
imzalanmıştır.

ALICI YETKİLİSİ: Zeynep Demir
T.C. Kimlik No: 18301661332
İletişim: zeynep.demir@ornekfirma.com.tr — Tel: 0216 444 7788

Madde 4 — Ödeme
Toplam sözleşme bedeli 2.1 milyon TL olup, ödemeler tedarikçinin Almanya'daki
hesabına yapılacaktır: IBAN DE89 3704 0044 0532 0130 00. İlk taksit olarak
350.000 TL peşin ödenecektir.

Madde 9 — Teslimat
Mallar, alıcının 06 XY 4567 plakalı aracıyla teslim alınacaktır. Teslimat
adresi: Cumhuriyet Mahallesi Sanayi Sokak No 12, Kadıköy 34710 İstanbul.

Sözleşme 1 Şubat 2024 tarihinde yürürlüğe girmiştir.
""",
        [
            ("PERSON", "Zeynep Demir"),
            ("TC_ID", "18301661332"),
            ("EMAIL", "zeynep.demir@ornekfirma.com.tr"),
            ("PHONE", "0216 444 7788"),
            ("FINANCIAL_AMOUNT", "2.1 milyon TL"),
            ("IBAN", "DE89 3704 0044 0532 0130 00"),
            ("FINANCIAL_AMOUNT", "350.000 TL"),
            ("VEHICLE_PLATE", "06 XY 4567"),
            ("ADDRESS", "Cumhuriyet Mahallesi"),
            ("POSTAL_CODE", "34710"),
        ],
    )
)

# ── 6. Mixed — Sağlık / sigorta (İtalyan + Fransız kimlikleri) ─────────────────
CASES.append(
    (
        "Mixed · Sağlık / Sigorta Talebi",
        """\
ULUSLARARASI SAĞLIK SİGORTASI TAZMİNAT TALEBİ — Dosya: CLM-2024-5567

Sigortalı: Marco Rossi
Doğum tarihi: 10 Aralık 1985

Sigortalı Marco Rossi, İtalya vatandaşı olup vergi kimlik kodu (codice
fiscale) RSSMRA85T10A562S ile kayıtlıdır. Fransa'da ikamet ettiği dönemden
kalan sosyal güvenlik numarası 180057001012345 da dosyaya eklenmiştir.

Seyahat sağlık poliçesi kapsamında, pasaport numarası YA1122334 ile giriş
yaptığı ülkede acil tıbbi müdahale görmüştür. Toplam tedavi masrafı
€12.500 olarak faturalandırılmıştır.

İletişim: marco.rossi@example.it, telefon +39 06 6982 1234. Tazminat,
talep sahibinin beyan ettiği hesaba aktarılacaktır.
""",
        [
            ("PERSON", "Marco Rossi"),
            ("DATE_OF_BIRTH", "10 Aralık 1985"),
            ("CODICE_FISCALE", "RSSMRA85T10A562S"),
            ("EU_NATIONAL_ID", "180057001012345"),
            ("PASSPORT", "YA1122334"),
            ("FINANCIAL_AMOUNT", "€12.500"),
            ("EMAIL", "marco.rossi@example.it"),
            ("PHONE", "+39 06 6982 1234"),
        ],
    )
)


# ── 7. DE — Almanca müşteri / fatura belgesi ───────────────────────────────────
CASES.append(
    (
        "DE · Almanca Müşteri / Fatura",
        """\
RECHNUNG UND KUNDENDATEN — Vorgang Nr. RG-2024-4471

Sehr geehrter Herr Klaus Müller,

vielen Dank für Ihre Bestellung. Ihre Kundendaten wurden wie folgt
aktualisiert. Geburtsdatum: 15. März 1988. Ihre E-Mail-Adresse
klaus.mueller@beispiel.de und Ihre Mobilnummer 0151 23456789 sind nun
hinterlegt.

Die Zahlung erfolgt per Lastschrift von IBAN DE89 3704 0044 0532 0130 00.
Unsere Umsatzsteuer-Identifikationsnummer lautet USt-IdNr DE123456789.
Der Rechnungsbetrag beläuft sich auf €12.500.

Ihre Kreditkarte 4111 1111 1111 1111 wurde für zukünftige Zahlungen
vorgemerkt. Bei Rückfragen steht Ihnen unser Mitarbeiter zur Verfügung.
""",
        [
            ("PERSON", "Klaus Müller"),
            ("DATE_OF_BIRTH", "15. März 1988"),
            ("EMAIL", "klaus.mueller@beispiel.de"),
            ("PHONE", "0151 23456789"),
            ("IBAN", "DE89 3704 0044 0532 0130 00"),
            ("VAT_NUMBER", "DE123456789"),
            ("FINANCIAL_AMOUNT", "€12.500"),
            ("CREDIT_CARD", "4111 1111 1111 1111"),
        ],
    )
)

# ── 8. FR — Fransızca sözleşme / sağlık belgesi ───────────────────────────────
CASES.append(
    (
        "FR · Fransızca Sözleşme",
        """\
CONTRAT DE SERVICE ET DONNÉES PERSONNELLES — Dossier FR-2024-9981

Cliente : Madame Sophie Laurent
Née le 3 février 1990.

Nous confirmons les coordonnées de Madame Sophie Laurent. Son adresse
e-mail sophie.laurent@exemple.fr et son numéro de téléphone
01 23 45 67 89 ont été enregistrés. Son numéro de sécurité sociale
180057001012345 figure également au dossier.

Les paiements seront prélevés sur le compte IBAN
FR14 2004 1010 0505 0001 3M02 606. Le numéro de TVA de la société est
FRAB123456789. Le montant total du contrat s'élève à €45.000.

Merci de confirmer dans un délai d'un mois.
""",
        [
            ("PERSON", "Sophie Laurent"),
            ("DATE_OF_BIRTH", "3 février 1990"),
            ("EMAIL", "sophie.laurent@exemple.fr"),
            ("PHONE", "01 23 45 67 89"),
            ("EU_NATIONAL_ID", "180057001012345"),
            ("IBAN", "FR14 2004 1010 0505 0001 3M02 606"),
            ("VAT_NUMBER", "FRAB123456789"),
            ("FINANCIAL_AMOUNT", "€45.000"),
        ],
    )
)


# ═══════════════════════════════════════════════════════════════════════════════
# YARDIMCILAR
# ═══════════════════════════════════════════════════════════════════════════════


def _norm(s: str) -> str:
    """Boşluk/küçük-büyük harf farklarını yok sayan normalleştirme."""
    return "".join(s.split()).lower()


def _match(gt_val: str, found_val: str) -> bool:
    a, b = _norm(gt_val), _norm(found_val)
    return a in b or b in a


def detector_of(confidence: float) -> str:
    """confidence -> kaba detektör etiketi."""
    return "regex" if confidence >= 0.999 else "ner/llm"


# ═══════════════════════════════════════════════════════════════════════════════
# GUARD KURULUMU
# ═══════════════════════════════════════════════════════════════════════════════

print(f"{BOLD}{CYAN}Guard kuruluyor: Regex + NER(tr_core_news_lg) + LLM(gemma3:12b)…{RESET}")
guard = Wardcat(
    use_ner=True,
    spacy_model="tr_core_news_lg",
    salt="coverage-test-salt",
).with_llm(backend="ollama", model="gemma3:12b", timeout=120)
# Bağlamsal sırların LLM tarafından da yakalanması için CUSTOM_SECRET'i aç.
guard._config.setdefault("llm_detector", {}).setdefault("entities", {})["CUSTOM_SECRET"] = {
    "enabled": True,
    "action": "hash",
}
# FINANCIAL_AMOUNT varsayılan olarak kapalı (opt-in) — bu test için açıyoruz.
guard._config["entities"]["FINANCIAL_AMOUNT"] = {"enabled": True, "action": "redact"}
guard._rebuild()

# Hangi entity'ler gerçekten etkin? (rapora dahil)
active_regex = sorted(
    e
    for e in guard._config.get("entities", {})
    if guard._config["entities"][e].get("enabled", True)
)
print(f"{DIM}Aktif detektör sayısı: {len(guard._detectors)}{RESET}\n")


# ═══════════════════════════════════════════════════════════════════════════════
# ÇALIŞTIR + KARŞILAŞTIR
# ═══════════════════════════════════════════════════════════════════════════════

total_gt = 0
total_found = 0
per_type: dict[str, list[int]] = {}  # entity_type -> [found, total]
all_missed: list[tuple[str, str, str]] = []  # (case, etype, value)
all_extra: list[tuple[str, str, str]] = []  # (case, etype, value)

for title, text, ground_truth in CASES:
    print(f"{BOLD}{CYAN}{'═' * 70}{RESET}")
    print(f"{BOLD}{CYAN}  {title}{RESET}")
    print(f"{BOLD}{CYAN}{'═' * 70}{RESET}")

    t0 = time.time()
    result = guard.scan(text)
    elapsed = time.time() - t0

    violations = result.violations
    print(
        f"  {DIM}Metin uzunluğu: {len(text)} karakter — {len(violations)} tespit — ⏱ {elapsed:.1f}s{RESET}\n"
    )

    matched_violation_idx: set[int] = set()

    for etype, val in ground_truth:
        total_gt += 1
        per_type.setdefault(etype, [0, 0])[1] += 1

        # Doğru etiket + değer eşleşmesi
        hit_idx = None
        for i, v in enumerate(violations):
            if v.entity_type == etype and _match(val, v.original):
                hit_idx = i
                break

        if hit_idx is not None:
            v = violations[hit_idx]
            matched_violation_idx.add(hit_idx)
            total_found += 1
            per_type[etype][0] += 1
            det = detector_of(v.confidence)
            print(f"  {GREEN}✓{RESET} {etype:16} {DIM}{val[:38]:38}{RESET} [{det}]")
        else:
            # Değer başka etiketle bulundu mu? (yanlış etiketleme)
            mislabel = None
            for v in violations:
                if _match(val, v.original) and v.entity_type != etype:
                    mislabel = v.entity_type
                    break
            if mislabel:
                print(
                    f"  {YELLOW}~{RESET} {etype:16} {DIM}{val[:38]:38}{RESET} "
                    f"{YELLOW}→ '{mislabel}' olarak etiketlendi{RESET}"
                )
            else:
                print(f"  {RED}✗{RESET} {etype:16} {DIM}{val[:38]:38}{RESET} {RED}KAÇIRILDI{RESET}")
            all_missed.append((title, etype, val))

    # Fazladan (ground-truth'ta olmayan) tespitler — manuel inceleme için
    extras = [v for i, v in enumerate(violations) if i not in matched_violation_idx]
    if extras:
        print(f"\n  {DIM}Fazladan tespitler (ground-truth dışı):{RESET}")
        for v in extras:
            det = detector_of(v.confidence)
            print(
                f"    {YELLOW}+{RESET} {v.entity_type:16} {DIM}'{v.original[:40]}'{RESET} [{det}]"
            )
            all_extra.append((title, v.entity_type, v.original))
    print()


# ═══════════════════════════════════════════════════════════════════════════════
# RAPOR
# ═══════════════════════════════════════════════════════════════════════════════

print(f"{BOLD}{CYAN}{'═' * 70}{RESET}")
print(f"{BOLD}{CYAN}  RAPOR ÖZETİ{RESET}")
print(f"{BOLD}{CYAN}{'═' * 70}{RESET}\n")

recall = total_found / total_gt * 100 if total_gt else 0
print(f"  Toplam beklenen PII : {total_gt}")
print(f"  {GREEN}Yakalanan           : {total_found}{RESET}")
print(f"  {RED}Kaçırılan           : {total_gt - total_found}{RESET}")
print(f"  {BOLD}Recall (yakalama)   : {recall:.1f}%{RESET}\n")

print(f"  {BOLD}Entity tipine göre:{RESET}")
for etype in sorted(per_type):
    found, tot = per_type[etype]
    mark = GREEN if found == tot else (YELLOW if found else RED)
    print(f"    {mark}{etype:16} {found}/{tot}{RESET}")

if all_missed:
    print(f"\n  {BOLD}{RED}Kaçırılan / yanlış etiketlenenler:{RESET}")
    for case, etype, val in all_missed:
        print(f"    {RED}✗{RESET} [{etype}] '{val}'  {DIM}({case}){RESET}")

print(f"\n  {DIM}Fazladan tespit sayısı (manuel inceleme): {len(all_extra)}{RESET}")
print()
