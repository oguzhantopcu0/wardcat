"""Turkish PII corpus, hand-labelled.

Provenance matters and is stated up front: unlike the English side, which uses
presidio-research's own dataset, no public Turkish PII benchmark was available,
so this one was written by wardcat's side. That is home-field advantage and every
number drawn from it should be read with that in mind. It is written to be hard
rather than flattering — names in lower case, numbers in local formats, values
embedded in prose rather than in labelled fields.

All identifiers are synthetic. TC numbers satisfy the Nüfus İdaresi checksum,
IBANs satisfy mod-97 and cards satisfy Luhn, so a miss is a detector's and not a
fixture's. Gold spans use the same schema as synth_dataset_v2.json.
"""

from __future__ import annotations

# (text, [(entity_type, exact substring), ...])
SAMPLES: list[tuple[str, list[tuple[str, str]]]] = [
    (
        "Merhaba, ben Ahmet Yılmaz. TC kimlik numaram 62601815964, "
        "IBAN'ım TR33 0006 1005 1978 6457 8413 26. Ödeme kartım "
        "4532 0151 1283 0366 ile yaptığım işlem iki kez çekilmiş.",
        [
            ("PERSON", "Ahmet Yılmaz"),
            ("TC_ID", "62601815964"),
            ("IBAN_CODE", "TR33 0006 1005 1978 6457 8413 26"),
            ("CREDIT_CARD", "4532 0151 1283 0366"),
        ],
    ),
    (
        "Fatura adresi: Ayşe Demir, e-posta ayse.demir@example.com, "
        "telefon 0532 123 45 67. Şirket: Kuzey Lojistik A.Ş.",
        [
            ("PERSON", "Ayşe Demir"),
            ("EMAIL_ADDRESS", "ayse.demir@example.com"),
            ("PHONE_NUMBER", "0532 123 45 67"),
            ("ORGANIZATION", "Kuzey Lojistik A.Ş."),
        ],
    ),
    (
        "merhaba ben mehmet kaya, numaram +90 532 987 65 43, mail adresim mehmet@example.com",
        [
            ("PERSON", "mehmet kaya"),
            ("PHONE_NUMBER", "+90 532 987 65 43"),
            ("EMAIL_ADDRESS", "mehmet@example.com"),
        ],
    ),
    (
        "Sayın Zeynep Arslan, 15.03.1988 doğum tarihinizle kaydınız açıldı. "
        "Sunucu 10.0.0.7 üzerinden erişebilirsiniz.",
        [("PERSON", "Zeynep Arslan"), ("IP_ADDRESS", "10.0.0.7")],
    ),
    (
        "Ödeme Garanti Bankası hesabına yapıldı, IBAN "
        "TR95 0006 2000 1230 0006 2911 74, referans no 88213.",
        [("ORGANIZATION", "Garanti Bankası"), ("IBAN_CODE", "TR95 0006 2000 1230 0006 2911 74")],
    ),
    (
        "Müşteri temsilcimiz Can Öztürk sizi 0212 555 34 21 numarasından arayacak. "
        "Sorularınız için destek@example.com adresine yazabilirsiniz.",
        [
            ("PERSON", "Can Öztürk"),
            ("PHONE_NUMBER", "0212 555 34 21"),
            ("EMAIL_ADDRESS", "destek@example.com"),
        ],
    ),
    (
        "Başvuru sahibi: Elif Şahin, TC 29316138924. Çalıştığı kurum "
        "Anadolu Sigorta, kart bilgisi 5555 5555 5555 4444.",
        [
            ("PERSON", "Elif Şahin"),
            ("TC_ID", "29316138924"),
            ("ORGANIZATION", "Anadolu Sigorta"),
            ("CREDIT_CARD", "5555 5555 5555 4444"),
        ],
    ),
    (
        "kayıt: burak yıldız - tc 10000000146 - iletişim burak.yildiz@example.com",
        [
            ("PERSON", "burak yıldız"),
            ("TC_ID", "10000000146"),
            ("EMAIL_ADDRESS", "burak.yildiz@example.com"),
        ],
    ),
    (
        "Toplantı yarın saat 14:00'te. Katılımcılar Selin Kurt ve Deniz Aydın. "
        "Kişisel veri paylaşılmayacak.",
        [("PERSON", "Selin Kurt"), ("PERSON", "Deniz Aydın")],
    ),
    (
        "İade talebi için kart numarası 4111 1111 1111 1111 ve "
        "IBAN TR21 0001 0000 1234 5678 9012 34 gerekiyor.",
        [("CREDIT_CARD", "4111 1111 1111 1111"), ("IBAN_CODE", "TR21 0001 0000 1234 5678 9012 34")],
    ),
    (
        "Personel dosyası — Ad: Murat Çelik, Departman: Yazılım Geliştirme, "
        "Şirket e-postası murat.celik@example.com, dahili 0216 444 12 12.",
        [
            ("PERSON", "Murat Çelik"),
            ("EMAIL_ADDRESS", "murat.celik@example.com"),
            ("PHONE_NUMBER", "0216 444 12 12"),
        ],
    ),
    (
        "Sunucu logları: 192.168.14.77 adresinden gelen istek, "
        "kullanıcı fatma.oz@example.com oturum açtı.",
        [("IP_ADDRESS", "192.168.14.77"), ("EMAIL_ADDRESS", "fatma.oz@example.com")],
    ),
    (
        "Türk Hava Yolları ile uçuş rezervasyonu yapan Hakan Tunç, "
        "0555 111 22 33 numarasından ulaşılabilir.",
        [
            ("ORGANIZATION", "Türk Hava Yolları"),
            ("PERSON", "Hakan Tunç"),
            ("PHONE_NUMBER", "0555 111 22 33"),
        ],
    ),
    (
        "sipariş numarası 4471 ile ilgili sorun yok, kişisel bilgi içermiyor.",
        [],
    ),
    (
        "Bugün hava çok güzel, ofiste toplantı yapacağız. Herhangi bir kişisel veri paylaşılmadı.",
        [],
    ),
    (
        "Sözleşme tarafı Yıldız Teknoloji Ltd. Şti., yetkili kişi Emre Doğan, "
        "kurumsal kart 3782 822463 10005.",
        [
            ("ORGANIZATION", "Yıldız Teknoloji Ltd. Şti."),
            ("PERSON", "Emre Doğan"),
            ("CREDIT_CARD", "3782 822463 10005"),
        ],
    ),
    (
        "hasta kaydı: gülay şimşek, tc 15973514218, telefon 0533 222 33 44",
        [("PERSON", "gülay şimşek"), ("TC_ID", "15973514218"), ("PHONE_NUMBER", "0533 222 33 44")],
    ),
    (
        "Ödeme bilgisi güncellendi. Yeni kart 6011 1111 1111 1117, "
        "fatura adresi e-postası muhasebe@example.com.",
        [("CREDIT_CARD", "6011 1111 1111 1117"), ("EMAIL_ADDRESS", "muhasebe@example.com")],
    ),
    (
        "Ali Veli isimli müşterimiz Vodafone Türkiye hattını kullanıyor, "
        "numarası +90 542 000 11 22.",
        [
            ("PERSON", "Ali Veli"),
            ("ORGANIZATION", "Vodafone Türkiye"),
            ("PHONE_NUMBER", "+90 542 000 11 22"),
        ],
    ),
    (
        "Denetim kaydı — işlem yapan: Merve Kılıç, IP 172.16.0.5, "
        "hesap IBAN TR67 0001 5001 5800 7300 4444 55.",
        [
            ("PERSON", "Merve Kılıç"),
            ("IP_ADDRESS", "172.16.0.5"),
            ("IBAN_CODE", "TR67 0001 5001 5800 7300 4444 55"),
        ],
    ),
]


def as_dataset() -> list[dict]:
    """Return the corpus in the synth_dataset_v2.json schema."""
    out = []
    for text, gold in SAMPLES:
        spans = []
        for entity_type, value in gold:
            start = text.index(value)
            spans.append(
                {
                    "entity_type": entity_type,
                    "entity_value": value,
                    "start_position": start,
                    "end_position": start + len(value),
                }
            )
        out.append({"full_text": text, "spans": spans})
    return out


if __name__ == "__main__":
    import collections
    import json

    data = as_dataset()
    counts = collections.Counter(s["entity_type"] for r in data for s in r["spans"])
    print(f"{len(data)} samples, {sum(counts.values())} gold spans")
    for k, v in counts.most_common():
        print(f"   {k:<16} {v}")
    print(json.dumps(data[0], ensure_ascii=False)[:200])
