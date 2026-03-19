# ai-guard

**PII detection and anonymization for LLM inputs** — hybrid regex + NER + on-prem LLM engine.

`ai-guard` scans text for personally identifiable information (PII) before it reaches an LLM, and either warns about or masks the sensitive data using salted SHA-256 hashes. It supports Turkish and English out of the box.

```bash
pip install ai-guard                  # regex + LLM detection (no SpaCy)
pip install "ai-guard[ner]"           # + SpaCy NER (PERSON, ORG, ADDRESS)
pip install "ai-guard[all]"           # + SpaCy + Claude API backend
```

```python
from ai_guard import LLMGuard

guard = (
    LLMGuard(salt="my-secret-salt")
    .configure_entity("CREDIT_CARD", enabled=True, action="hash")
    .configure_entity("EMAIL",       enabled=True, action="warn")
    .configure_entity("TC_ID",       enabled=True, action="hash")
)

result = guard.scan("Name: Ali Veli, card: 4532 0151 1283 0366, email: ali@example.com")
print(result.sanitized_text)
# Name: Ali Veli, card: [CREDIT_CARD:ea782818], email: ali@example.com
```

**Supported entity types:** `CREDIT_CARD`, `EMAIL`, `PHONE`, `IBAN`, `TC_ID`, `IP_ADDRESS`, `ADDRESS`, `POSTAL_CODE`, `PERSON` (NER), `ORG` (NER), `CUSTOM_SECRET` (LLM)

**Actions:** `warn` (keep text, report violation) · `hash` (replace with `[TYPE:8hex]` using SHA-256 + salt)

**Backends:** Regex (built-in) · SpaCy NER (optional) · Ollama / OpenAI-compatible / Claude (optional LLM)

---

*Turkish documentation follows / Türkçe dokümantasyon aşağıda*

---

LLM girdilerindeki hassas verileri tespit eden ve anonimleştiren Python kütüphanesi.

Regex tabanlı desen eşleştirme ile SpaCy NER'ı birleştiren hibrit bir tespit motoru sunar. Kullanıcı, hangi veri tipinin taranacağını ve her biri için uygulanacak aksiyonu (`warn` veya `hash`) hem Python API'si hem de YAML dosyası üzerinden yönetir.

---

## Özellikler

- **Hibrit tespit:** Regex (CC, IBAN, TC kimlik, telefon…) + SpaCy NER (kişi, kurum, adres)
- **İki aksiyon:** `warn` (rapor et, metni koru) ve `hash` (SHA-256 + salt ile maskele)
- **Rainbow table koruması:** Kullanıcı tanımlı salt ile tuzlama
- **İki API:** Method zinciri (Programmatic) ve YAML (Declarative)
- **CLI:** `python -m ai_guard scan/batch`
- **Türkçe odaklı:** TC kimlik, IBAN, posta kodu, Türkçe adres desenleri, Türkçe SpaCy modeli desteği

---

## Kurulum

```bash
# Bağımlılıkları kur
uv sync

# İngilizce SpaCy modeli
uv pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl

# Türkçe model (opsiyonel)
# uv pip install <tr_core_news_sm wheel URL>
```

NER kullanmadan yalnızca regex ile çalışmak için SpaCy modeli gerekmez.

---

## Hızlı Başlangıç

### Programmatic API

```python
from ai_guard import LLMGuard

guard = (
    LLMGuard(salt="gizli-tuz")
    .configure_entity("CREDIT_CARD", enabled=True, action="hash")
    .configure_entity("EMAIL",       enabled=True, action="warn")
    .configure_entity("TC_ID",       enabled=True, action="hash")
)

result = guard.scan("""
  Müşteri: Ali Veli, TC: 12345678901
  Kart: 4532 0151 1283 0366
  Mail: ali.veli@example.com
""")

print(result.sanitized_text)
# Müşteri: Ali Veli, TC: [TC_ID:86349f34]
# Kart: [CREDIT_CARD:ea782818]
# Mail: ali.veli@example.com   ← warn: metin korunur

for v in result.violations:
    print(f"[{v.action.value}] {v.entity_type}: {v.original!r}")
# [hash] TC_ID: '12345678901'
# [hash] CREDIT_CARD: '4532 0151 1283 0366'
# [warn] EMAIL: 'ali.veli@example.com'
```

### Declarative API (YAML)

```python
from ai_guard import LLMGuard

guard = LLMGuard(config_path="config/my_policy.yaml")
result = guard.scan(text)
```

```yaml
# config/my_policy.yaml
salt: ""          # üretimde env değişkeninden oku
spacy_model: "en_core_web_sm"
use_ner: true

entities:
  CREDIT_CARD: { enabled: true,  action: hash }
  EMAIL:       { enabled: true,  action: warn }
  TC_ID:       { enabled: true,  action: hash }
  IBAN:        { enabled: true,  action: hash }
  PERSON:      { enabled: true,  action: hash }
  ORG:         { enabled: false, action: warn }
```

### Toplu Tarama

```python
results = guard.scan_batch([
    "ali@example.com",
    "Kart: 4111 1111 1111 1111",
    "Temiz metin.",
])

for r in results:
    print(r.is_clean, len(r.violations))
# False 1
# False 1
# True  0
```

---

## CLI

```bash
# Tek metin — text formatı
python -m ai_guard scan --text "TC: 12345678901 kart: 4111111111111111"

# Dosyadan — JSON çıktısı
python -m ai_guard scan --file girdi.txt --format json

# Salt ve NER kapalı
python -m ai_guard scan --text "..." --salt "gizli-tuz" --no-ner

# Türkçe SpaCy modeli
python -m ai_guard scan --text "..." --model tr_core_news_sm

# Toplu — her satır bağımsız metin
python -m ai_guard batch --file satirlar.txt --format json
```

#### Örnek JSON çıktısı

```json
{
  "is_clean": false,
  "sanitized_text": "kart: [CREDIT_CARD:c5a992a8]",
  "violations": [
    {
      "entity_type": "CREDIT_CARD",
      "original": "4111111111111111",
      "start": 6,
      "end": 22,
      "action": "hash",
      "replacement": "[CREDIT_CARD:c5a992a8]"
    }
  ]
}
```

---

## Desteklenen Entity Tipleri

| Entity | Dedektör | Varsayılan Aksiyon | Açıklama |
|---|---|---|---|
| `CREDIT_CARD` | Regex | `hash` | Visa, MC, Amex, Discover — bitişik ve boşluklu/çizgili format |
| `EMAIL` | Regex | `warn` | RFC uyumlu e-posta adresleri |
| `PHONE` | Regex | `warn` | Türk telefon numaraları (`0`, `+90`, `90` ön ekli) |
| `IBAN` | Regex | `hash` | Uluslararası IBAN (büyük/küçük harf duyarsız) |
| `IP_ADDRESS` | Regex | `warn` | IPv4 adresleri |
| `TC_ID` | Regex | `hash` | 11 haneli TC kimlik numarası |
| `ADDRESS` | Regex | `warn` | Türkçe adres desenleri (Cad., Sok., Mah., Blv.) |
| `POSTAL_CODE` | Regex | `warn` | Türkiye posta kodları (01000–81999) |
| `PERSON` | SpaCy NER | `hash` | Kişi adları |
| `ORG` | SpaCy NER | `warn` | Kurumsal isimler |

---

## Çıktı Yapısı

`scan()` ve `scan_batch()` her metin için bir `ScanResult` döndürür:

```python
@dataclass
class ScanResult:
    original_text:  str               # giriş metni
    sanitized_text: str               # anonimleştirilmiş metin
    violations:     List[Violation]   # bulunan ihlaller
    is_clean:       bool              # violations boş mu?

@dataclass
class Violation:
    entity_type: str          # "CREDIT_CARD", "EMAIL", …
    original:    str          # orijinal değer
    start:       int          # orijinal metindeki başlangıç konumu
    end:         int          # orijinal metindeki bitiş konumu
    action:      Action       # Action.WARN | Action.HASH
    replacement: str | None   # hash ise "[TIP:8hex]", warn ise None
```

---

## Güvenlik

### Hash'leme

Hassas veriler SHA-256 ile hash'lenir; çıktı `[TIP:8hex]` formatında yerleştirilir:

```
4532 0151 1283 0366  →  [CREDIT_CARD:ea782818]
12345678901          →  [TC_ID:86349f34]
```

### Salt (Tuzlama)

Rainbow table saldırılarını önlemek için salt desteği:

```python
guard = LLMGuard(salt="proje-spesifik-tuz")
# veya sonradan
guard.set_salt("yeni-tuz")
```

> **Üretim notu:** Salt değerini kaynak koda veya config dosyasına yazmayın.
> Ortam değişkeninden okuyun:
> ```python
> import os
> guard = LLMGuard(salt=os.environ["LLMGUARD_SALT"])
> ```

---

## Proje Yapısı

```
ai-guard/
├── src/ai_guard/
│   ├── guard.py            # LLMGuard — ana arayüz
│   ├── __main__.py         # CLI giriş noktası
│   ├── core/
│   │   ├── engine.py       # DetectionEngine — overlap çözümü, aksiyon uygulama
│   │   └── models.py       # Action, Violation, ScanResult
│   ├── detectors/
│   │   ├── base.py         # BaseDetector ABC
│   │   ├── regex_detector.py
│   │   └── ner_detector.py # SpaCy NER (İngilizce + Türkçe model desteği)
│   ├── config/
│   │   └── loader.py       # YAML yükleyici, deep-merge
│   └── utils/
│       └── hashing.py      # SHA-256 + salt
├── tests/
│   ├── unit/               # Bileşen düzeyi testler
│   └── integration/        # Senaryo ve adversarial testler
├── config/
│   └── default.yaml        # Örnek politika dosyası
└── pyproject.toml
```

---

## Testler

```bash
# Tüm testler
uv run pytest

# Yalnızca unit testler
uv run pytest tests/unit/

# Yalnızca NER testleri (SpaCy modeli gerektirir)
uv run pytest -m ner

# Kapsam raporu
uv run pytest --cov=src/ai_guard --cov-report=term-missing
```

**217 test geçiyor, 6 xfail** (belgelenmiş bilinen sınırlar).

---

## Bilinen Sınırlar

| Durum | Açıklama |
|---|---|
| `4111  1111  1111  1111` | Çift boşluk separator kart tespitini atlatır |
| `4111.1111.1111.1111` | Nokta separator desteklenmez |
| `TR...TR...` (bitişik) | Separator'suz arka arkaya iki IBAN ayrıştırılamaz |
| Kiril homoglyph | `аli@test.com` (Kiril `а`) ASCII regex'ini atlatır |
| 15 karakterlik IBAN lookalike | Regex minimum uzunluğunu karşılayan kısa diziler eşleşebilir |

---

## Türkçe Model Desteği

```yaml
# config/policy.yaml
spacy_model: "tr_core_news_sm"
```

Türkçe SpaCy modeli `PER` → `PERSON`, `LOC` → `ADDRESS` olarak eşlenir. Model kurulumu için SpaCy'nin model sayfasını inceleyin.

---

## Geliştirme

```bash
uv sync --dev
uv run pytest
```

Python 3.13+ gerektirir.
