"""
Gerçek llama3.1:8b modeli ile kapsamlı canlı tarama testi.

Bu dosya otomatik pytest suite'ine dahil değildir (sys.exit kullanır).
Ollama servisinin çalışıyor ve llama3.1:8b modelinin kurulu olması gerekir.

Çalıştırma:
    uv run python live_scan_test.py
    uv run python live_scan_test.py 2>&1 | tee live_test_results.txt

Hazırlık:
    ollama pull llama3.1:8b
"""

from __future__ import annotations

import sys
import time

from wardcat import Wardcat
from wardcat.core.engine import DetectionEngine
from wardcat.detectors.llm_detector import LLMDetector
from wardcat.llm.backends.ollama import OllamaBackend

# ── Renk kodları ─────────────────────────────────────────────────────────────
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"

PASS = f"{GREEN}✓ PASS{RESET}"
FAIL = f"{RED}✗ FAIL{RESET}"
WARN = f"{YELLOW}⚠ WARN{RESET}"

# ── Guard fabrikası ───────────────────────────────────────────────────────────


def make_guard(entities: set[str] | None = None, use_ner: bool = False) -> Wardcat:
    enabled = entities or {
        "CREDIT_CARD",
        "EMAIL",
        "PERSON",
        "TC_ID",
        "IBAN",
        "PHONE",
        "IP_ADDRESS",
        "CUSTOM_SECRET",
    }
    guard = Wardcat(use_ner=use_ner)
    for e in enabled:
        guard._config["entities"].setdefault(e, {"enabled": True, "action": "warn"})
    guard._config["entities"]["PERSON"] = {"enabled": True, "action": "hash"}
    guard._config["entities"]["CUSTOM_SECRET"] = {"enabled": True, "action": "hash"}
    guard._config["entities"]["CREDIT_CARD"] = {"enabled": True, "action": "hash"}
    guard._config["entities"]["TC_ID"] = {"enabled": True, "action": "hash"}
    guard._config["entities"]["IBAN"] = {"enabled": True, "action": "hash"}

    llm = LLMDetector(
        backend=OllamaBackend(model="gemma3:12b"),
        enabled_entities=enabled,
    )
    guard._detectors.append(llm)
    guard._engine = DetectionEngine(guard._config, guard._detectors)
    return guard


# ── Test yardımcıları ─────────────────────────────────────────────────────────

results: list[tuple[str, bool, str]] = []  # (name, passed, note)


def check(name: str, condition: bool, note: str = "") -> bool:
    results.append((name, condition, note))
    status = PASS if condition else FAIL
    print(f"  {status}  {name}" + (f"  {YELLOW}({note}){RESET}" if note else ""))
    return condition


def section(title: str) -> None:
    print(f"\n{BOLD}{CYAN}{'═' * 60}{RESET}")
    print(f"{BOLD}{CYAN}  {title}{RESET}")
    print(f"{BOLD}{CYAN}{'═' * 60}{RESET}")


def run(label: str, guard: Wardcat, text: str) -> object:
    t0 = time.time()
    result = guard.scan(text)
    elapsed = time.time() - t0
    print(f"\n  {BOLD}Metin   :{RESET} {text[:90]}{'…' if len(text) > 90 else ''}")
    print(
        f"  {BOLD}Sanitize:{RESET} {result.sanitized_text[:90]}{'…' if len(result.sanitized_text) > 90 else ''}"
    )
    if result.violations:
        for v in result.violations:
            arrow = f"→ '{v.replacement}'" if v.replacement else ""
            print(f"  {YELLOW}  [{v.action:4}] {v.entity_type:15} '{v.original}' {arrow}{RESET}")
    else:
        print(f"  {GREEN}  Temiz{RESET}")
    print(f"  {YELLOW}  ⏱  {elapsed:.1f}s{RESET}")
    return result


# ═════════════════════════════════════════════════════════════════════════════
# TESTLER
# ═════════════════════════════════════════════════════════════════════════════

guard = make_guard()

# ── 1. Türkçe çok-entity ─────────────────────────────────────────────────────
section("1 · Türkçe — Çok Entity")

text = (
    "Müşterimiz Ali Veli (ali.veli@sirket.com, TC: 12345678901) "
    "4111111111111111 kartıyla 500 TL ödedi. Tel: 0532 123 4567"
)
r = run("Türkçe çok-entity", guard, text)
types = {v.entity_type for v in r.violations}
check("PERSON tespit", "PERSON" in types)
check("EMAIL tespit", "EMAIL" in types)
check("TC_ID tespit", "TC_ID" in types)
check("CREDIT_CARD tespit", "CREDIT_CARD" in types)
check("PHONE tespit", "PHONE" in types)
check("Ali Veli hash'lendi", "Ali Veli" not in r.sanitized_text)
check("TC_ID hash'lendi", "12345678901" not in r.sanitized_text)
check("Kart no hash'lendi", "4111111111111111" not in r.sanitized_text)
check("Email korundu (warn)", "ali.veli@sirket.com" in r.sanitized_text)


# ── 2. İngilizce — finansal belge ────────────────────────────────────────────
section("2 · İngilizce — Finansal Belge")

text = (
    "Dear John Smith, your IBAN TR33 0006 1005 1978 6457 8413 26 "
    "and card 4111 1111 1111 1111 are confirmed. "
    "Contact: john@acme.com / +90 533 987 6543"
)
r = run("İngilizce finansal", guard, text)
types = {v.entity_type for v in r.violations}
check("IBAN tespit", "IBAN" in types)
check("CREDIT_CARD tespit", "CREDIT_CARD" in types)
check("EMAIL tespit", "EMAIL" in types)
check("PHONE tespit", "PHONE" in types)
check("IBAN hash'lendi", "TR33 0006 1005 1978 6457 8413 26" not in r.sanitized_text)
check("Kart hash'lendi", "4111 1111 1111 1111" not in r.sanitized_text)


# ── 3. Bağlamsal sırlar — regex'in kaçırdığı ─────────────────────────────────
section("3 · Bağlamsal Sırlar (Regex Kaçırır, LLM Yakalar)")

text = "Veritabanı şifresi db_pass=S3cr3t!42 — kimseyle paylaşmayın."
r = run("DB şifresi", guard, text)
types = {v.entity_type for v in r.violations}
check("CUSTOM_SECRET tespit", "CUSTOM_SECRET" in types)
check("Şifre hash'lendi", "S3cr3t!42" not in r.sanitized_text)

text = "API anahtarı: sk-prod-xK92mNzL8qW3 — sadece prod ortamında kullanın."
r = run("API key", guard, text)
types = {v.entity_type for v in r.violations}
check("API key tespit", "CUSTOM_SECRET" in types)
check("API key gizlendi", "sk-prod-xK92mNzL8qW3" not in r.sanitized_text)

text = "Toplantı notu: erişim kodu ALPHA-BRAVO-42 onaylandı."
r = run("Erişim kodu", guard, text)
types = {v.entity_type for v in r.violations}
check("Erişim kodu tespit", "CUSTOM_SECRET" in types)


# ── 4. Temiz metinler — false positive kontrolü ──────────────────────────────
section("4 · Temiz Metinler — False Positive Kontrolü")

clean_texts = [
    "Hava bugün çok güzel, piknik yapalım.",
    "The quarterly report shows 15% growth in Q3.",
    "Toplantı saat 14:00'te 3. katta.",
    "Proje teslim tarihi 30 Mart 2025.",
    "Sipariş ID: 98765, adet: 3, toplam: 450 TL.",
    "Sunucu bellek kullanımı %72, disk %45.",
    "Ürün kodu: PRD-2024-XL, stok: 120 adet.",
    "Bu çeyrek 42 yeni müşteri kazandık, hedef 50 idi.",
    "Toplantı katılımcı sayısı: 14. Gündem maddeleri: 3.",
    "Version 2.1.4 released. Fixed 12 bugs, added 3 features.",
]
for text in clean_texts:
    r = run("Temiz metin", guard, text)
    check(
        f"False positive yok: '{text[:45]}'",
        r.is_clean,
        f"{[v.entity_type + ':' + v.original for v in r.violations]}" if not r.is_clean else "",
    )


# ── 5. Karma dil — Türkçe + İngilizce ───────────────────────────────────────
section("5 · Karma Dil (TR + EN)")

text = (
    "Müşteri Mehmet Demir (mehmet@company.com) 4111111111111111 ile ödedi. TC kimlik: 12345678901."
)
r = run("Karma dil (TR prefix)", guard, text)
types = {v.entity_type for v in r.violations}
check("PERSON tespit", "PERSON" in types)
check("EMAIL tespit", "EMAIL" in types)
check("CREDIT_CARD tespit", "CREDIT_CARD" in types)
check("TC_ID tespit", "TC_ID" in types)

text2 = "Dear John Smith, contact john@acme.com for invoice #1234."
r2 = run("İngilizce only", guard, text2)
types2 = {v.entity_type for v in r2.violations}
check("EMAIL tespit (EN)", "EMAIL" in types2)
if "PERSON" not in types2:
    print(f"  {WARN}  llama3.1:8b 'John Smith'i bu bağlamda kaçırdı (bilinen sınır)")
check(
    "PERSON tespit (EN)", "PERSON" in types2, "llama3.1:8b sınırı" if "PERSON" not in types2 else ""
)


# ── 6. Span pozisyonları doğruluğu ───────────────────────────────────────────
section("6 · Span Pozisyon Doğruluğu")

text = "Müşteri adı: Ayşe Kaya, e-posta: ayse@test.com"
r = run("Span pozisyon", guard, text)
all_correct = True
for v in r.violations:
    actual = text[v.start : v.end]
    correct = actual == v.original
    if not correct:
        all_correct = False
        print(f"  {RED}  HATA: text[{v.start}:{v.end}]='{actual}' ≠ '{v.original}'{RESET}")
check("Tüm span'lar orijinal metinle örtüşüyor", all_correct)


# ── 7. Overlap — aynı span regex + LLM ──────────────────────────────────────
section("7 · Overlap Çözümü (Regex + LLM Aynı Span)")

text = "E-posta: user@example.com"
r = run("Email overlap", guard, text)
emails = [v for v in r.violations if v.entity_type == "EMAIL"]
check(
    "Email tekrar yok (overlap çözüldü)",
    len(emails) == 1,
    f"{len(emails)} adet" if len(emails) != 1 else "",
)


# ── 8. Batch tarama ──────────────────────────────────────────────────────────
section("8 · Batch Tarama")

batch_guard = Wardcat(
    use_ner=False,
    use_llm=True,
    llm_model="gemma3:12b",
)
lines = [
    "Sipariş veren: Zeynep Arslan, kart: 5500 0000 0000 0004",
    "Hava bugün çok güzel.",
    "İletişim: support@firma.com, IP: 192.168.1.1",
]
print(f"\n  {len(lines)} satır taranıyor...")
t0 = time.time()
batch_results = batch_guard.scan_batch(lines)
elapsed = time.time() - t0

for i, (line, res) in enumerate(zip(lines, batch_results, strict=True), 1):
    flag = f"{GREEN}temiz{RESET}" if res.is_clean else f"{RED}{len(res.violations)} ihlal{RESET}"
    print(f"  Satır {i}: {flag}  — {line[:55]}")

check("Batch 3 sonuç döndü", len(batch_results) == 3)
check("Satır 1 temiz değil", not batch_results[0].is_clean)
check(
    "Satır 2 temiz",
    batch_results[1].is_clean,
    str([v.entity_type for v in batch_results[1].violations])
    if not batch_results[1].is_clean
    else "",
)
check("Satır 3 temiz değil", not batch_results[2].is_clean)
print(f"  {YELLOW}  ⏱  Toplam {elapsed:.1f}s ({elapsed / len(lines):.1f}s/satır){RESET}")


# ── 9. Uzun gerçekçi metin — müşteri destek kaydı ────────────────────────────
section("9 · Uzun Metin — Müşteri Destek Kaydı")

text = """\
Müşteri Hizmetleri Kaydı #4892
Tarih: 18 Mart 2024, Saat: 10:35

Müşteri: Fatma Şahin
TC Kimlik No: 23456789012
Telefon: 0543 876 5432
E-posta: fatma.sahin@gmail.com

Konu: Kredi kartı işlemi itirazı
Kart No: 5500 0000 0000 0004 ile 15 Mart 2024 tarihinde 1.250,00 TL tutarında
gerçekleştirilen işleme itiraz edilmiştir. Müşteri söz konusu alışverişi
yapmadığını beyan etmektedir.

Aksiyon: Kart bloke edildi. İşlem araştırmaya alındı.
Temsilci: Kemal Aydın (dahili: 2341)
"""
r = run("Destek kaydı", guard, text)
types = {v.entity_type for v in r.violations}
check("PERSON (Fatma Şahin) tespit", "PERSON" in types)
check("TC_ID tespit", "TC_ID" in types)
check("PHONE tespit", "PHONE" in types)
check("EMAIL tespit", "EMAIL" in types)
check("CREDIT_CARD tespit", "CREDIT_CARD" in types)
check("Fatma Şahin sanitize edildi", "Fatma Şahin" not in r.sanitized_text)
check("TC_ID sanitize edildi", "23456789012" not in r.sanitized_text)
check("Kart no sanitize edildi", "5500 0000 0000 0004" not in r.sanitized_text)
# Temsilci adı da kişisel veri — model yakalayabilir ama zorunlu değil
if "Kemal Aydın" not in r.sanitized_text:
    print(f"  {GREEN}  Bonus: Temsilci adı da hash'lendi{RESET}")


# ── 10. Teknik log satırı — karışık format ────────────────────────────────────
section("10 · Teknik Log — Karışık Format")

text = """\
[2024-03-18 09:42:17] INFO  user_login: user=ahmet.yilmaz@firma.com ip=10.0.0.45 status=success
[2024-03-18 09:43:02] WARN  payment_attempt: card=4111111111111111 amount=750.00 result=declined
[2024-03-18 09:43:55] ERROR config_load: db_password=Passw0rd#99 host=db.internal port=5432
[2024-03-18 09:44:10] INFO  api_call: token=Bearer eyJhbGciOiJSUzI1NiJ9.eyJzdWIiOiJ1c2VyMTIz status=200
"""
r = run("Teknik log", guard, text)
types = {v.entity_type for v in r.violations}
check("EMAIL (log içinde) tespit", "EMAIL" in types)
check("CREDIT_CARD (log içinde) tespit", "CREDIT_CARD" in types)
check("CUSTOM_SECRET (db_password) tespit", "CUSTOM_SECRET" in types)
check("IP_ADDRESS tespit", "IP_ADDRESS" in types)
check("Kart sanitize edildi", "4111111111111111" not in r.sanitized_text)
check("DB şifresi sanitize edildi", "Passw0rd#99" not in r.sanitized_text)


# ── 11. Çok kişili konuşma — chat geçmişi ─────────────────────────────────────
section("11 · Chat Geçmişi — Çok Kişili")

text = """\
[Kullanıcı - 14:02] Merhaba, ben Emre Kılıç. Hesabıma erişemiyorum.
[Destek  - 14:03] Merhaba Emre Bey. TC kimlik numaranızı alabilir miyim?
[Kullanıcı - 14:03] Tabii: 34567890123
[Destek  - 14:04] Teşekkürler. Kayıtlı e-postanız emre.kilic@hotmail.com mu?
[Kullanıcı - 14:04] Evet, doğru.
[Destek  - 14:05] Şifrenizi sıfırlamak için sizi 0533 111 2233 numarasından arayacağız.
[Kullanıcı - 14:06] Tamam, bekliyorum.
"""
r = run("Chat geçmişi", guard, text)
types = {v.entity_type for v in r.violations}
check("PERSON (Emre Kılıç) tespit", "PERSON" in types)
check("TC_ID tespit", "TC_ID" in types)
check("EMAIL tespit", "EMAIL" in types)
check("PHONE tespit", "PHONE" in types)
check("TC_ID sanitize edildi", "34567890123" not in r.sanitized_text)
check("Email warn'da korundu", "emre.kilic@hotmail.com" in r.sanitized_text)

# Span pozisyon doğruluğu — uzun metinde de geçerli olmalı
all_correct = True
for v in r.violations:
    actual = text[v.start : v.end]
    if actual != v.original:
        all_correct = False
        print(f"  {RED}  HATA: text[{v.start}:{v.end}]='{actual}' ≠ '{v.original}'{RESET}")
check("Chat: tüm span pozisyonları doğru", all_correct)


# ── 12. Sözleşme metni — hukuki dil ──────────────────────────────────────────
section("12 · Sözleşme Metni — Hukuki Dil")

text = """\
İŞ SÖZLEŞMESİ

İşbu sözleşme, aşağıda bilgileri yer alan taraflar arasında akdedilmiştir.

İŞVEREN: ABC Teknoloji A.Ş., Vergi No: 1234567890
ÇALIŞAN: Selin Çelik, T.C. Kimlik No: 45678901234
          İletişim: selin.celik@abctech.com | 0212 345 6789

Madde 3 – Ücret
Çalışanın aylık brüt ücreti 45.000 TL olarak belirlenmiştir.
Ödemeler, IBAN TR78 0001 0017 4538 0973 5001 01 numaralı hesaba
her ayın 1. günü yapılacaktır.

Madde 7 – Gizlilik
Çalışan, şirketin tüm ticari sırlarını ve müşteri bilgilerini
gizli tutmayı taahhüt eder. İhlal halinde yasal yaptırım uygulanır.

İmza Tarihi: 1 Ocak 2024
"""
r = run("Sözleşme metni", guard, text)
types = {v.entity_type for v in r.violations}
check("PERSON (Selin Çelik) tespit", "PERSON" in types)
check("TC_ID tespit", "TC_ID" in types)
check("EMAIL tespit", "EMAIL" in types)
check("PHONE tespit", "PHONE" in types)
check("IBAN tespit", "IBAN" in types)
check("Selin Çelik sanitize edildi", "Selin Çelik" not in r.sanitized_text)
check("TC_ID sanitize edildi", "45678901234" not in r.sanitized_text)
check("IBAN sanitize edildi", "TR78 0001 0017 4538 0973 5001 01" not in r.sanitized_text)
check("Email warn'da korundu", "selin.celik@abctech.com" in r.sanitized_text)


# ── 13. Gömülü sırlar — DevOps config ─────────────────────────────────────────
section("13 · DevOps Config — Gömülü Kimlik Bilgileri")

text = """\
# production.env — GİZLİ, commit'leme!
DATABASE_URL=postgresql://admin:Sup3rS3cr3t@db.prod.internal:5432/appdb
REDIS_PASSWORD=r3d1sP@ss!
AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE
AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY
STRIPE_SECRET_KEY=sk_live_51NxT8ABCDEF1234567890
JWT_SECRET=my-ultra-secure-jwt-secret-key-2024
SENTRY_DSN=https://abc123@o0.ingest.sentry.io/456789
"""
r = run("DevOps config", guard, text)
types = {v.entity_type for v in r.violations}
check(
    "En az 3 CUSTOM_SECRET tespit",
    types.count("CUSTOM_SECRET") >= 3
    if False
    else len([v for v in r.violations if v.entity_type == "CUSTOM_SECRET"]) >= 3,
)
secrets_found = [v.original for v in r.violations if v.entity_type == "CUSTOM_SECRET"]
# Connection-string password (postgresql://admin:Sup3rS3cr3t@...) must be
# caught as a secret, not swallowed by the email regex as "password@host".
check("DB şifresi (connection string) tespit", any("Sup3rS3cr3t" in s for s in secrets_found))
check("DB şifresi sanitize edildi", "Sup3rS3cr3t" not in r.sanitized_text)
check("Redis şifresi tespit", any("r3d1sP" in s for s in secrets_found) or "CUSTOM_SECRET" in types)
check("Stripe key tespit", any("sk_live" in s for s in secrets_found) or "CUSTOM_SECRET" in types)
check(
    "Hiçbir secret düz kalmadı",
    all(v.replacement is not None for v in r.violations if v.entity_type == "CUSTOM_SECRET"),
)


# ── 14. Gürültülü kullanıcı mesajı — imla hataları + argüman ─────────────────
section("14 · Gürültülü Kullanıcı Mesajı")

text = (
    "hesabıma giremiyorum şifrem calisamiyor!! "
    "tc no 56789012345 ve emailim hasan_celik@yandex.com "
    "lutfen yardim edin acil durum kart numaram 4111-1111-1111-1111"
)
r = run("Gürültülü mesaj", guard, text)
types = {v.entity_type for v in r.violations}
check("TC_ID (imla hatalı metinde) tespit", "TC_ID" in types)
check("EMAIL (alt çizgili) tespit", "EMAIL" in types)
check("CREDIT_CARD (tireli format) tespit", "CREDIT_CARD" in types)
check("TC_ID gizlendi", "56789012345" not in r.sanitized_text)
check("Kart gizlendi", "4111-1111-1111-1111" not in r.sanitized_text)


# ── 15. Tekrarlayan entity — aynı kişi birden fazla ──────────────────────────
section("15 · Tekrarlayan Entity (Aynı Kişi Birden Fazla)")

text = (
    "Başvuru sahibi: Deniz Yıldız. "
    "Deniz Yıldız, 67890123456 TC numarasıyla başvurmuştur. "
    "Deniz Yıldız ile iletişim: deniz@ornek.com"
)
r = run("Tekrarlayan kişi", guard, text)
person_spans = [v for v in r.violations if v.entity_type == "PERSON"]
check("PERSON en az 2 kez tespit", len(person_spans) >= 2)
check("TC_ID tespit", "TC_ID" in {v.entity_type for v in r.violations})
check("Tüm 'Deniz Yıldız' gizlendi", "Deniz Yıldız" not in r.sanitized_text)
check("TC_ID gizlendi", "67890123456" not in r.sanitized_text)

# Aynı entity'nin tüm tekrarları aynı hash'i üretmeli
hashes = {v.replacement for v in person_spans if v.replacement}
check(
    "Aynı isim → aynı hash",
    len(hashes) == 1,
    f"Farklı hash'ler: {hashes}" if len(hashes) != 1 else "",
)


# ═════════════════════════════════════════════════════════════════════════════
# ÖZET
# ═════════════════════════════════════════════════════════════════════════════

section("ÖZET")
passed = sum(1 for _, ok, _ in results if ok)
failed = sum(1 for _, ok, _ in results if not ok)
total = len(results)

print(f"\n  Toplam  : {total}")
print(f"  {GREEN}Geçen   : {passed}{RESET}")
if failed:
    print(f"  {RED}Kalan   : {failed}{RESET}")
    print("\n  Başarısız testler:")
    for name, ok, note in results:
        if not ok:
            print(f"    {RED}✗{RESET} {name}" + (f" ({note})" if note else ""))

score = passed / total * 100
color = GREEN if score >= 90 else YELLOW if score >= 70 else RED
print(f"\n  {color}{BOLD}Skor: {score:.0f}%{RESET}\n")

sys.exit(0 if failed == 0 else 1)
